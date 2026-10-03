"""
Thin Telegram Bot API wrapper for sending review messages and polling updates.

This module provides low-level HTTP calls to the Telegram Bot API without any
business logic or gate semantics. It only handles payload construction and
HTTP requests using the `requests` library.
"""

import json
import requests
from typing import Optional


BASE_URL = "https://api.telegram.org/bot{token}/{method}"


def send_review_message(
    token: str,
    chat_id: str,
    text: str,
    job_key: str,
) -> int:
    """
    Send a message with Approve/Reject inline buttons for job review.
    
    Args:
        token: Telegram bot token
        chat_id: Target chat ID
        text: Message text (job title, company, summary)
        job_key: Unique identifier for the job
        
    Returns:
        message_id from the Telegram API response
        
    Raises:
        requests.RequestException: If the HTTP request fails
        KeyError: If the response doesn't contain expected fields
    """
    url = BASE_URL.format(token=token, method="sendMessage")
    
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": {
            "inline_keyboard": [[
                {
                    "text": "✅ Approve",
                    "callback_data": encode_callback("a", job_key)
                },
                {
                    "text": "❌ Reject", 
                    "callback_data": encode_callback("r", job_key)
                }
            ]]
        }
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    if not result.get("ok"):
        raise Exception(f"Telegram API error: {result}")
        
    return result["result"]["message_id"]


def poll_updates(
    token: str,
    offset: Optional[int] = None,
    timeout: int = 25,
) -> list[dict]:
    """
    Poll for updates using long-polling with getUpdates.
    
    Args:
        token: Telegram bot token
        offset: Optional offset for update_id (excludes older updates)
        timeout: Long-poll timeout in seconds
        
    Returns:
        List of raw update objects from Telegram API
        
    Raises:
        requests.RequestException: If the HTTP request fails
    """
    url = BASE_URL.format(token=token, method="getUpdates")
    
    params = {
        "timeout": timeout
    }
    
    # Only include offset param when not None
    if offset is not None:
        params["offset"] = offset
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    if not result.get("ok"):
        raise Exception(f"Telegram API error: {result}")
        
    return result.get("result", [])


def answer_callback(
    token: str,
    callback_query_id: str,
    text: str = "",
) -> None:
    """
    Answer a callback query to stop the client-side loading spinner.
    
    This is a best-effort operation - failures are swallowed to prevent
    blocking decision resolution.
    
    Args:
        token: Telegram bot token
        callback_query_id: ID of the callback query to answer
        text: Optional text to show to the user
    """
    url = BASE_URL.format(token=token, method="answerCallbackQuery")
    
    payload = {
        "callback_query_id": callback_query_id,
        "text": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if not result.get("ok"):
            # Log but don't raise - this is best-effort
            print(f"Warning: answerCallbackQuery failed: {result}")
            
    except Exception as e:
        # Swallow all exceptions - ack failures should never block decision resolution
        print(f"Warning: answerCallbackQuery exception: {e}")


def encode_callback(action: str, job_key: str) -> str:
    """
    Encode action and job_key into callback_data format.
    
    Args:
        action: Action type ('a' for approve, 'r' for reject)
        job_key: Job identifier
        
    Returns:
        Encoded callback_data string in format "action|job_key"
    """
    return f"{action}|{job_key}"


def decode_callback(data: str) -> tuple[str, str]:
    """
    Decode callback_data into action and job_key.
    
    Args:
        data: Callback data string in format "action|job_key"
        
    Returns:
        Tuple of (action, job_key)
    """
    action, job_key = data.split("|", 1)  # Split on first "|" only
    return action, job_key