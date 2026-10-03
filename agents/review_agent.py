"""Review agent for the Telegram review gate.

This module implements the Summary_Service that generates job summaries using
the DeepSeek client with fallback to truncated excerpts when LLM calls fail.
It also provides the review coordinator that handles the review gate workflow.
"""

import hashlib
import time
from enum import Enum
from openai import OpenAI
from typing import Callable

from models.job import JobListing


class Decision(str, Enum):
    """Review decision for a job."""
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDECIDED = "undecided"


def make_summarize_fn(client: OpenAI, model: str, excerpt_chars: int = 600) -> Callable[[JobListing], str]:
    """Create a summarize function that uses DeepSeek to summarize jobs.
    
    Args:
        client: DeepSeek OpenAI client (same interface as used in document_tools.py)
        model: Model name to use for summarization
        excerpt_chars: Maximum characters for fallback excerpt (default 600)
        
    Returns:
        A summarize(job) function that returns non-empty summary text
    """
    def summarize(job: JobListing) -> str:
        """Summarize a job's responsibilities and key requirements.
        
        Args:
            job: JobListing to summarize
            
        Returns:
            Non-empty summary text. On LLM success, returns the generated summary.
            On failure or empty response, returns a bounded excerpt of the job description.
        """
        jd = job.job_description or ""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Summarise this job's responsibilities and key requirements in 3-4 short lines."},
                    {"role": "user", "content": jd[:6000]},
                ],
                max_tokens=300,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            # Any exception falls through to fallback
            pass
        
        # Fallback: bounded truncated excerpt so message always sends
        excerpt = jd.strip()[:excerpt_chars]
        return excerpt or "(no description available)"
    
    return summarize


def job_key(job: JobListing) -> str:
    """Generate a unique key for a job.
    
    Uses job.linkedin_job_id when present/non-empty, otherwise generates
    a synthetic key from URL+title+company to handle process jobs that
    may lack numeric LinkedIn IDs.
    
    Args:
        job: JobListing to generate key for
        
    Returns:
        Unique key string. LinkedIn IDs are used as-is, synthetic keys
        are prefixed with 'k' to prevent collision with numeric LinkedIn IDs.
    """
    if job.linkedin_job_id:
        return job.linkedin_job_id
    
    # Generate synthetic key for jobs without LinkedIn ID (e.g., process jobs)
    raw = f"{job.url}|{job.title}|{job.company}"
    return "k" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def render_message(title: str, company: str, summary: str) -> str:
    """Render a job review message for Telegram.
    
    Pure helper function that formats job information into a readable message.
    Always includes title, company, and summary in the message body.
    
    Args:
        title: Job title
        company: Company name
        summary: Job summary/description
        
    Returns:
        Formatted message string for Telegram
    """
    return f"**{title}** at **{company}**\n\n{summary}"


def review_jobs(
    jobs: list[JobListing],
    summarize_fn: Callable[[JobListing], str],
    telegram_client,
    timeout: int,
    now_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Decision]:
    """Review jobs via Telegram with Approve/Reject buttons.
    
    Sends one message per distinct job key, polls for button responses until
    all jobs are decided or timeout elapses. Jobs left undecided resolve to
    Decision.UNDECIDED.
    
    Args:
        jobs: List of jobs to review
        summarize_fn: Function to generate job summaries
        telegram_client: Object/module with send_review_message, poll_updates, answer_callback
        timeout: Timeout in seconds before undecided jobs resolve to UNDECIDED
        now_fn: Injectable clock function for testing (defaults to time.monotonic)
        
    Returns:
        Dict mapping job keys to Decision (APPROVED/REJECTED/UNDECIDED)
    """
    # Build keys mapping job_key -> job
    keys = {job_key(job): job for job in jobs}
    
    # Send exactly one message per distinct key
    successfully_sent = {}
    decisions = {}
    
    for key, job in keys.items():
        try:
            text = render_message(job.title, job.company, summarize_fn(job))
            telegram_client.send_review_message(text=text, job_key=key)
            successfully_sent[key] = job
        except Exception:
            # If send_review_message raises for a job, resolve that job's key to UNDECIDED
            # Never silently approve/reject and do not block others
            decisions[key] = Decision.UNDECIDED
    
    # Initialize deadline
    deadline = now_fn() + timeout
    offset = None
    
    # Loop while decisions incomplete and before deadline
    while len(decisions) < len(successfully_sent):
        current_time = now_fn()
        if current_time >= deadline:
            break
            
        try:
            # Calculate remaining time and clamp poll timeout so it never overruns deadline
            remaining_time = deadline - current_time
            poll_timeout = min(25, max(1, int(remaining_time)))
            
            updates = telegram_client.poll_updates(offset=offset, timeout=poll_timeout)
            
            if not updates:
                # No updates received, continue polling
                continue
            
            for update in updates:
                if "callback_query" in update:
                    callback_query = update["callback_query"]
                    callback_data = callback_query.get("data", "")
                    
                    try:
                        # Import decode_callback only when needed
                        from tools.telegram_tools import decode_callback
                        action, key = decode_callback(callback_data)
                        
                        # If key is in our batch and not yet decided, record decision
                        if key in successfully_sent and key not in decisions:
                            if action == "a":
                                decisions[key] = Decision.APPROVED
                            elif action == "r":
                                decisions[key] = Decision.REJECTED
                            
                            # Answer callback query to stop spinner
                            try:
                                telegram_client.answer_callback(callback_query["id"])
                            except Exception:
                                # Ack failures are swallowed, never block decision resolution
                                pass
                    except Exception:
                        # Invalid callback data, ignore
                        pass
                
                # Advance offset for next poll (always advance for each update processed)
                offset = update["update_id"] + 1
                
        except Exception:
            # Transient poll failures are caught and retried on next iteration
            pass
    
    # After the loop, any key not decided -> Decision.UNDECIDED
    for key in successfully_sent:
        if key not in decisions:
            decisions[key] = Decision.UNDECIDED
    
    return decisions