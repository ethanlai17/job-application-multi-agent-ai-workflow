"""Tests for Task 4: Add the review coordinator to agents/review_agent.py.

Feature: telegram-review-gate, Properties 3, 4, 6, 8
Tasks: 4.1 - 4.5 - Property tests and unit tests for review coordinator

This covers:
- Requirement 2.1: One message per job
- Requirement 2.2: Message body contains title, company, and summary
- Requirement 2.4: Button taps resolve to matching decision
- Requirement 6.3: Timeout resolves exactly the undecided keys
"""

import pytest
import hashlib
from hypothesis import given, strategies as st
from unittest.mock import Mock, MagicMock
from datetime import date

from agents.review_agent import Decision, job_key, render_message, review_jobs
from models.job import JobListing, ApplicationStatus


class TestDecisionEnum:
    """Test the Decision enum values."""

    def test_decision_values(self):
        """Decision enum has correct string values."""
        assert Decision.APPROVED == "approved"
        assert Decision.REJECTED == "rejected"
        assert Decision.UNDECIDED == "undecided"


class TestJobKey:
    """Test job_key function behavior."""

    def test_uses_linkedin_job_id_when_present(self):
        """When linkedin_job_id is present and non-empty, use it as the key."""
        job = JobListing(
            title="Software Engineer",
            company="Test Company",
            url="https://example.com/job/123",
            location="Remote",
            salary="$100k",
            date_found=date.today(),
            linkedin_job_id="1234567890"
        )
        
        assert job_key(job) == "1234567890"

    def test_generates_synthetic_key_when_linkedin_id_empty(self):
        """When linkedin_job_id is empty, generate synthetic key from URL+title+company."""
        job = JobListing(
            title="Software Engineer",
            company="Test Company", 
            url="https://example.com/job/abc",
            location="Remote",
            salary="$100k",
            date_found=date.today(),
            linkedin_job_id=""  # Empty
        )
        
        key = job_key(job)
        
        # Should be "k" + first 12 chars of SHA1 hash
        raw = f"{job.url}|{job.title}|{job.company}"
        expected = "k" + hashlib.sha1(raw.encode()).hexdigest()[:12]
        assert key == expected
        assert key.startswith("k")
        assert len(key) == 13  # "k" + 12 hex chars

    def test_generates_synthetic_key_when_linkedin_id_none(self):
        """When linkedin_job_id is None, generate synthetic key."""
        job = JobListing(
            title="Backend Developer",
            company="Another Company",
            url="https://example.com/job/def", 
            location="Remote",
            salary="$120k",
            date_found=date.today(),
            linkedin_job_id=None  # type: ignore
        )
        
        key = job_key(job)
        
        # Should generate synthetic key
        raw = f"{job.url}|{job.title}|{job.company}"
        expected = "k" + hashlib.sha1(raw.encode()).hexdigest()[:12]
        assert key == expected
        assert key.startswith("k")

    def test_synthetic_key_deterministic(self):
        """Synthetic keys are deterministic for same URL+title+company."""
        job1 = JobListing(
            title="Data Scientist",
            company="ML Corp",
            url="https://example.com/job/xyz",
            location="NYC",
            salary="$130k", 
            date_found=date.today(),
            linkedin_job_id=""
        )
        
        job2 = JobListing(
            title="Data Scientist",
            company="ML Corp", 
            url="https://example.com/job/xyz",
            location="SF",  # Different location, but key should be same
            salary="$140k",  # Different salary, but key should be same
            date_found=date.today(),
            linkedin_job_id=""
        )
        
        assert job_key(job1) == job_key(job2)

    def test_synthetic_key_different_for_different_jobs(self):
        """Different jobs generate different synthetic keys."""
        job1 = JobListing(
            title="Frontend Developer",
            company="Web Corp",
            url="https://example.com/job/1",
            location="Remote",
            salary="$90k",
            date_found=date.today(),
            linkedin_job_id=""
        )
        
        job2 = JobListing(
            title="Backend Developer",  # Different title
            company="Web Corp",
            url="https://example.com/job/1",
            location="Remote",
            salary="$90k",
            date_found=date.today(),
            linkedin_job_id=""
        )
        
        assert job_key(job1) != job_key(job2)


class TestRenderMessage:
    """Test render_message function."""

    @given(
        title=st.text(min_size=1, max_size=100),
        company=st.text(min_size=1, max_size=100), 
        summary=st.text(min_size=1, max_size=500)
    )
    def test_render_message_contains_all_components_property(self, title, company, summary):
        """Property 4: Message body contains title, company, and summary for arbitrary strings.
        
        Feature: telegram-review-gate, Property 4: Message body contains title, company, and summary
        Validates: Requirement 2.2
        """
        result = render_message(title, company, summary)
        
        # Must be non-empty
        assert result
        assert len(result.strip()) > 0
        
        # Must contain all three components
        assert title in result
        assert company in result
        assert summary in result

    def test_render_message_format(self):
        """Message has expected format with title and company in bold."""
        title = "Senior Software Engineer"
        company = "Tech Corp"
        summary = "This is a great job with excellent benefits."
        
        result = render_message(title, company, summary)
        
        expected = f"**{title}** at **{company}**\n\n{summary}"
        assert result == expected

    def test_render_message_handles_special_characters(self):
        """Message rendering handles special characters correctly."""
        title = "Software Engineer (Python/Django)"
        company = "AI & ML Solutions Inc."
        summary = "Looking for someone with 3+ years experience in Python, Django, and SQL databases."
        
        result = render_message(title, company, summary)
        
        assert title in result
        assert company in result
        assert summary in result
        assert result.startswith(f"**{title}** at **{company}**")


class TestReviewJobs:
    """Test review_jobs function - the main coordinator."""

    def create_mock_telegram_client(self):
        """Create a mock telegram client for testing."""
        client = Mock()
        client.send_review_message = Mock()
        client.poll_updates = Mock(return_value=[])  # Default to no updates
        client.answer_callback = Mock()
        return client

    def create_test_job(self, title="Test Job", company="Test Company", linkedin_job_id=""):
        """Create a test job for testing."""
        return JobListing(
            title=title,
            company=company,
            url=f"https://example.com/job/{title.lower().replace(' ', '_')}",
            location="Remote",
            salary="$100k",
            date_found=date.today(),
            linkedin_job_id=linkedin_job_id
        )

    @given(
        num_jobs=st.integers(min_value=1, max_value=10)
    )
    def test_one_message_per_job_property(self, num_jobs):
        """Property 3: For a batch of jobs with distinct keys, send count == number of distinct keys.
        
        Feature: telegram-review-gate, Property 3: One message per job
        Validates: Requirement 2.1
        """
        # Create jobs with distinct keys (use LinkedIn IDs to ensure distinctness)
        jobs = [
            self.create_test_job(
                title=f"Job {i}",
                company=f"Company {i}",
                linkedin_job_id=str(1000 + i)
            )
            for i in range(num_jobs)
        ]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Mock time to avoid actual waiting
        now_values = [0.0, 100.0]  # Start at 0, then jump past timeout
        now_fn = Mock(side_effect=now_values)
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=now_fn)
        
        # Should send exactly one message per distinct job key
        assert client.send_review_message.call_count == num_jobs
        
        # All jobs should be resolved (to UNDECIDED due to timeout)
        assert len(result) == num_jobs
        
        # Verify all keys are distinct
        sent_keys = {call.kwargs['job_key'] for call in client.send_review_message.call_args_list}
        assert len(sent_keys) == num_jobs

    def test_send_failure_resolves_to_undecided_unit(self):
        """Unit test: A job whose send_review_message raises resolves to UNDECIDED."""
        jobs = [self.create_test_job(linkedin_job_id="123")]
        
        client = self.create_mock_telegram_client()
        client.send_review_message.side_effect = Exception("Send failed")
        summarize_fn = Mock(return_value="Test summary")
        
        # Mock time
        now_fn = Mock(return_value=0.0)
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=now_fn)
        
        # Job should be resolved to UNDECIDED due to send failure
        assert len(result) == 1
        key = job_key(jobs[0])
        assert result[key] == Decision.UNDECIDED

    @given(
        num_tapped=st.integers(min_value=0, max_value=5),
        num_untapped=st.integers(min_value=1, max_value=5)
    )
    def test_tap_resolution_property(self, num_tapped, num_untapped):
        """Property 6: Button taps resolve to matching decision, untapped keys left for timeout.
        
        Feature: telegram-review-gate, Property 6: Button taps resolve to the matching decision
        Validates: Requirement 2.4
        """
        total_jobs = num_tapped + num_untapped
        
        # Create jobs with distinct LinkedIn IDs
        jobs = [
            self.create_test_job(
                title=f"Job {i}",
                linkedin_job_id=str(2000 + i)
            )
            for i in range(total_jobs)
        ]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Create callback updates for the first num_tapped jobs
        updates = []
        expected_tapped = {}
        
        for i in range(num_tapped):
            job = jobs[i]
            key = job_key(job)
            action = "a" if i % 2 == 0 else "r"  # Alternate approve/reject
            expected_decision = Decision.APPROVED if action == "a" else Decision.REJECTED
            expected_tapped[key] = expected_decision
            
            update = {
                "update_id": 100 + i,
                "callback_query": {
                    "id": f"callback_{i}",
                    "data": f"{action}|{key}"
                }
            }
            updates.append(update)
        
        # Mock poll_updates to return our updates, then timeout
        def mock_poll_updates(offset=None, timeout=25):
            if offset is None or offset <= 100 + num_tapped:
                return updates
            return []
        
        client.poll_updates.side_effect = mock_poll_updates
        
        # Mock time progression: first call returns 0, subsequent calls allow processing then trigger timeout
        now_calls = 0
        def mock_now():
            nonlocal now_calls
            now_calls += 1
            if now_calls == 1:
                return 0.0
            elif now_calls <= 3:  # Allow a few calls for processing
                return 10.0  # Within timeout window
            else:
                return 100.0  # Past timeout
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=mock_now)
        
        # All jobs should be resolved
        assert len(result) == total_jobs
        
        # Tapped jobs should have matching decisions
        for key, expected_decision in expected_tapped.items():
            assert result[key] == expected_decision
        
        # Untapped jobs should be UNDECIDED
        all_keys = {job_key(job) for job in jobs}
        tapped_keys = set(expected_tapped.keys())
        untapped_keys = all_keys - tapped_keys
        
        for key in untapped_keys:
            assert result[key] == Decision.UNDECIDED

    @given(
        decided_count=st.integers(min_value=1, max_value=5),
        undecided_count=st.integers(min_value=1, max_value=5)
    )
    def test_timeout_resolution_property(self, decided_count, undecided_count):
        """Property 8: With injected now_fn crossing deadline, never-decided keys -> UNDECIDED, decided keys preserved.
        
        Feature: telegram-review-gate, Property 8: Timeout resolves exactly the undecided keys
        Validates: Requirement 6.3
        """
        total_jobs = decided_count + undecided_count
        
        # Create jobs
        jobs = [
            self.create_test_job(
                title=f"Job {i}",
                linkedin_job_id=str(3000 + i)
            )
            for i in range(total_jobs)
        ]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Create updates only for the first `decided_count` jobs
        updates = []
        expected_decided = {}
        
        for i in range(decided_count):
            job = jobs[i]
            key = job_key(job)
            action = "a" if i % 2 == 0 else "r"
            expected_decision = Decision.APPROVED if action == "a" else Decision.REJECTED
            expected_decided[key] = expected_decision
            
            update = {
                "update_id": 200 + i,
                "callback_query": {
                    "id": f"callback_{i}",
                    "data": f"{action}|{key}"
                }
            }
            updates.append(update)
        
        client.poll_updates.return_value = updates
        
        # Mock deterministic time progression that crosses the deadline
        timeout_seconds = 60
        call_count = 0
        def mock_now():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0
            elif call_count == 2:
                return 30.0  # Mid-way through timeout
            else:
                return 70.0  # Past deadline
        
        result = review_jobs(jobs, summarize_fn, client, timeout=timeout_seconds, now_fn=mock_now)
        
        # All jobs should be resolved
        assert len(result) == total_jobs
        
        # Decided keys should preserve their decisions
        for key, expected_decision in expected_decided.items():
            assert result[key] == expected_decision
        
        # Never-decided keys should be UNDECIDED
        all_keys = {job_key(job) for job in jobs}
        decided_keys = set(expected_decided.keys())
        never_decided_keys = all_keys - decided_keys
        
        assert len(never_decided_keys) == undecided_count
        for key in never_decided_keys:
            assert result[key] == Decision.UNDECIDED

    def test_answer_callback_called_for_each_tap(self):
        """answer_callback is called for each button tap."""
        jobs = [
            self.create_test_job(title="Job 1", linkedin_job_id="1001"),
            self.create_test_job(title="Job 2", linkedin_job_id="1002")
        ]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Create updates for both jobs
        updates = [
            {
                "update_id": 301,
                "callback_query": {
                    "id": "callback_1",
                    "data": "a|1001"
                }
            },
            {
                "update_id": 302,
                "callback_query": {
                    "id": "callback_2", 
                    "data": "r|1002"
                }
            }
        ]
        
        client.poll_updates.return_value = updates
        
        # Mock time to let both updates be processed before timeout
        now_fn = Mock(side_effect=[0.0, 10.0, 100.0])
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=now_fn)
        
        # Both answer_callback calls should have been made
        assert client.answer_callback.call_count == 2
        client.answer_callback.assert_any_call("callback_1")
        client.answer_callback.assert_any_call("callback_2")
        
        # Results should be correct
        assert result["1001"] == Decision.APPROVED
        assert result["1002"] == Decision.REJECTED

    def test_answer_callback_failure_does_not_block_decision(self):
        """answer_callback failures are swallowed and don't block decision resolution."""
        jobs = [self.create_test_job(linkedin_job_id="123")]
        
        client = self.create_mock_telegram_client()
        client.answer_callback.side_effect = Exception("Ack failed")
        summarize_fn = Mock(return_value="Test summary")
        
        updates = [{
            "update_id": 401,
            "callback_query": {
                "id": "callback_1",
                "data": "a|123"
            }
        }]
        
        client.poll_updates.return_value = updates
        
        call_count = 0
        def mock_now():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0
            elif call_count <= 3:  # Allow processing
                return 10.0  # Within timeout
            else:
                return 100.0  # Past timeout if no decision processed
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=mock_now)
        
        # Decision should still be recorded despite ack failure
        assert result["123"] == Decision.APPROVED

    def test_invalid_callback_data_ignored(self):
        """Invalid callback data is ignored and doesn't crash."""
        jobs = [self.create_test_job(linkedin_job_id="123")]
        
        client = self.create_mock_telegram_client() 
        summarize_fn = Mock(return_value="Test summary")
        
        # Invalid callback data (missing delimiter)
        updates = [{
            "update_id": 501,
            "callback_query": {
                "id": "callback_1",
                "data": "invalid_format"
            }
        }]
        
        client.poll_updates.return_value = updates
        now_fn = Mock(side_effect=[0.0, 100.0])  # Trigger timeout
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=now_fn)
        
        # Job should timeout to UNDECIDED since invalid data was ignored
        assert result["123"] == Decision.UNDECIDED

    def test_only_processes_keys_in_current_batch(self):
        """Only processes callback updates for keys that are in the current batch."""
        jobs = [self.create_test_job(linkedin_job_id="123")]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Update for a key NOT in our batch
        updates = [{
            "update_id": 601,
            "callback_query": {
                "id": "callback_1",
                "data": "a|999"  # Different key not in batch
            }
        }]
        
        client.poll_updates.return_value = updates
        now_fn = Mock(side_effect=[0.0, 100.0])  # Trigger timeout
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=now_fn)
        
        # Job should timeout since the update was for a different key
        assert result["123"] == Decision.UNDECIDED

    def test_offset_advancement_monotonic(self):
        """Offset is advanced monotonically by update_id + 1."""
        jobs = [
            self.create_test_job(title="Job 1", linkedin_job_id="1001"),
            self.create_test_job(title="Job 2", linkedin_job_id="1002")
        ]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Multiple batches of updates with ascending update_ids
        updates_batch1 = [{
            "update_id": 701,
            "callback_query": {"id": "cb1", "data": "a|1001"}
        }]
        
        updates_batch2 = [{
            "update_id": 702, 
            "callback_query": {"id": "cb2", "data": "r|1002"}
        }]
        
        # Mock poll_updates to return different batches based on offset
        def mock_poll_updates(offset=None, timeout=25):
            if offset is None:
                return updates_batch1
            elif offset == 702:  # After first update processed
                return updates_batch2
            else:
                return []
        
        client.poll_updates.side_effect = mock_poll_updates
        now_fn = Mock(side_effect=[0.0, 10.0, 20.0, 30.0, 100.0])  # Allow more processing time
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=now_fn)
        
        # Both jobs should be decided
        assert result["1001"] == Decision.APPROVED
        assert result["1002"] == Decision.REJECTED
        
        # Verify poll_updates was called with correct offsets
        poll_calls = client.poll_updates.call_args_list
        assert len(poll_calls) >= 2
        
        # First call should be with offset=None
        assert poll_calls[0][1]["offset"] is None
        
        # Second call should be with offset=702 (701 + 1)
        assert poll_calls[1][1]["offset"] == 702

    def test_poll_timeout_clamped_to_deadline(self):
        """Poll timeout is clamped so it never overruns the deadline."""
        jobs = [self.create_test_job(linkedin_job_id="123")]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # Mock time progression where remaining time gets smaller
        timeout_seconds = 10
        call_count = 0
        def mock_now():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0.0
            elif call_count == 2:
                return 8.0  # 2 seconds remaining
            else:
                return 100.0  # Past timeout
        
        review_jobs(jobs, summarize_fn, client, timeout=timeout_seconds, now_fn=mock_now)
        
        # Check that poll_updates was called with a clamped timeout
        poll_calls = client.poll_updates.call_args_list
        assert len(poll_calls) >= 1
        
        # The poll timeout should be clamped to remaining time (not the default 25)
        called_timeout = poll_calls[0][1]["timeout"]
        assert called_timeout <= 25  # Should be clamped
        assert called_timeout >= 1   # Should be at least 1

    def test_poll_updates_exception_handling(self):
        """Transient poll_updates exceptions are caught and retried."""
        jobs = [self.create_test_job(linkedin_job_id="123")]
        
        client = self.create_mock_telegram_client()
        summarize_fn = Mock(return_value="Test summary")
        
        # First poll fails, second succeeds but no updates
        poll_call_count = 0
        def mock_poll_updates(offset=None, timeout=25):
            nonlocal poll_call_count
            poll_call_count += 1
            if poll_call_count == 1:
                raise Exception("Network error")
            else:
                return []  # No updates, will timeout
        
        client.poll_updates.side_effect = mock_poll_updates
        
        call_count = 0
        def mock_now():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:  # Allow more retries
                return 10.0  # Within timeout
            else:
                return 100.0  # Trigger timeout
        
        result = review_jobs(jobs, summarize_fn, client, timeout=50, now_fn=mock_now)
        
        # Should not crash and should timeout to UNDECIDED
        assert result["123"] == Decision.UNDECIDED
        assert poll_call_count >= 2  # Should have retried after exception