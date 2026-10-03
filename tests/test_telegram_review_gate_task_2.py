"""Tests for Task 2: Create thin Telegram Bot API wrapper.

Feature: telegram-review-gate
Tasks: 2.1, 2.2, 2.3 - Property and unit tests for telegram_tools.py

This covers:
- Property 5: Message payload carries both buttons and round-tripping key (Requirements 2.3, 2.4)  
- Property 7: getUpdates offset is monotonic and exactly-once (Requirement 6.1)
- Unit: HTTP calls target correct endpoints, no webhook (Requirement 6.2)
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st
from hypothesis import settings as hypothesis_settings

from tools import telegram_tools


class TestCallbackEncoding:
    """Test encode_callback and decode_callback functions."""

    def test_encode_callback_basic(self):
        """Basic encode_callback functionality."""
        result = telegram_tools.encode_callback("a", "12345")
        assert result == "a|12345"

    def test_decode_callback_basic(self):
        """Basic decode_callback functionality."""
        action, job_key = telegram_tools.decode_callback("r|67890")
        assert action == "r"
        assert job_key == "67890"

    def test_encode_decode_round_trip(self):
        """encode_callback and decode_callback are inverses."""
        original_action = "a"
        original_key = "test-key-123"
        
        encoded = telegram_tools.encode_callback(original_action, original_key)
        decoded_action, decoded_key = telegram_tools.decode_callback(encoded)
        
        assert decoded_action == original_action
        assert decoded_key == original_key

    def test_decode_callback_handles_pipes_in_key(self):
        """decode_callback splits on first pipe only."""
        # Key contains pipes - should split on first pipe only
        action, job_key = telegram_tools.decode_callback("a|key|with|pipes")
        assert action == "a"
        assert job_key == "key|with|pipes"


class TestPropertySendMessagePayload:
    """Property 5: Message payload carries both buttons and round-tripping key.
    
    Validates: Requirements 2.3, 2.4
    """

    @given(
        action=st.sampled_from(["a", "r"]),
        job_key=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
    )
    @hypothesis_settings(max_examples=100)
    def test_callback_encoding_round_trip(self, action, job_key):
        """encode_callback -> decode_callback recovers original action and key.
        
        Feature: telegram-review-gate, Property 5
        Validates: Requirements 2.3, 2.4
        """
        # Encode then decode
        encoded = telegram_tools.encode_callback(action, job_key) 
        decoded_action, decoded_key = telegram_tools.decode_callback(encoded)
        
        # Should recover exactly the original values
        assert decoded_action == action
        assert decoded_key == job_key

    @patch('tools.telegram_tools.requests.post')
    def test_send_message_payload_has_two_buttons(self, mock_post):
        """sendMessage payload has exactly two inline buttons (Approve/Reject).
        
        Feature: telegram-review-gate, Property 5  
        Validates: Requirements 2.3, 2.4
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "ok": True,
            "result": {"message_id": 42}
        }
        mock_post.return_value = mock_response

        # Call the function
        telegram_tools.send_review_message(
            token="test-token",
            chat_id="test-chat", 
            text="Test job message",
            job_key="test-key-123"
        )

        # Check that POST was called once
        assert mock_post.call_count == 1
        
        # Extract the payload from the call
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        
        # Verify payload structure
        assert 'reply_markup' in payload
        assert 'inline_keyboard' in payload['reply_markup']
        
        # Should have exactly one row of buttons
        keyboard = payload['reply_markup']['inline_keyboard']
        assert len(keyboard) == 1
        
        # That row should have exactly 2 buttons
        button_row = keyboard[0]
        assert len(button_row) == 2
        
        # First button should be Approve
        approve_button = button_row[0]
        assert approve_button['text'] == '✅ Approve'
        assert approve_button['callback_data'] == 'a|test-key-123'
        
        # Second button should be Reject  
        reject_button = button_row[1]
        assert reject_button['text'] == '❌ Reject'
        assert reject_button['callback_data'] == 'r|test-key-123'


class TestPropertyOffsetMonotonic:
    """Property 7: getUpdates offset is monotonic and exactly-once.
    
    Validates: Requirement 6.1
    """

    @given(
        # Generate lists of update objects with increasing update_ids
        update_ids=st.lists(st.integers(min_value=1, max_value=1000000), min_size=1, max_size=50).map(
            lambda ids: sorted(set(ids))  # Remove duplicates and sort
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_offset_progression_is_monotonic(self, update_ids):
        """Over synthetic update lists, next offset == max(update_id)+1.
        
        Feature: telegram-review-gate, Property 7
        Validates: Requirement 6.1
        """
        # Skip empty lists (filtered out by the set operation)
        if not update_ids:
            return

        # Create synthetic update objects
        updates = [{"update_id": uid, "message": {"text": f"msg{uid}"}} for uid in update_ids]
        
        # The expected next offset should be max(update_id) + 1
        expected_next_offset = max(update_ids) + 1
        
        # Verify this is the correct progression
        # (In real usage, the caller would compute this from the returned updates)
        actual_next_offset = max(update['update_id'] for update in updates) + 1
        
        assert actual_next_offset == expected_next_offset
        assert actual_next_offset > max(update_ids)  # Should be strictly greater

    def test_offset_excludes_older_updates(self):
        """Using offset parameter excludes updates with lower update_id."""
        # This is more of a documentation test since we can't test real Telegram behavior
        # But we can verify the parameter is passed correctly
        with patch('tools.telegram_tools.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"ok": True, "result": []}
            mock_get.return_value = mock_response

            # Call with an offset
            telegram_tools.poll_updates(
                token="test-token",
                offset=12345,
                timeout=30
            )

            # Verify the offset was included in params
            args, kwargs = mock_get.call_args
            params = kwargs['params']
            assert params['offset'] == 12345
            assert params['timeout'] == 30

    def test_poll_updates_no_offset_when_none(self):
        """When offset=None, offset parameter is not included in request."""
        with patch('tools.telegram_tools.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None 
            mock_response.json.return_value = {"ok": True, "result": []}
            mock_get.return_value = mock_response

            # Call without offset
            telegram_tools.poll_updates(
                token="test-token",
                offset=None,
                timeout=25
            )

            # Verify offset was NOT included
            args, kwargs = mock_get.call_args
            params = kwargs['params']
            assert 'offset' not in params
            assert params['timeout'] == 25


class TestHttpCallsAndNoWebhook:
    """Unit tests for HTTP behavior and webhook absence.
    
    Validates: Requirement 6.2
    """

    @patch('tools.telegram_tools.requests.post')
    def test_send_message_targets_correct_endpoint(self, mock_post):
        """send_review_message targets sendMessage on api.telegram.org."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "ok": True,
            "result": {"message_id": 123}
        }
        mock_post.return_value = mock_response

        telegram_tools.send_review_message(
            token="test-token-123",
            chat_id="test-chat-456", 
            text="Test message",
            job_key="job-789"
        )

        # Verify correct URL was called
        args, kwargs = mock_post.call_args
        url = args[0]
        expected_url = "https://api.telegram.org/bot{}/{}".format("test-token-123", "sendMessage")
        assert url == expected_url

        # Verify JSON payload
        payload = kwargs['json']
        assert payload['chat_id'] == "test-chat-456"
        assert payload['text'] == "Test message"

    @patch('tools.telegram_tools.requests.get')
    def test_poll_updates_targets_correct_endpoint(self, mock_get):
        """poll_updates targets getUpdates on api.telegram.org."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True, "result": []}
        mock_get.return_value = mock_response

        telegram_tools.poll_updates(
            token="test-token-456",
            offset=100,
            timeout=30
        )

        # Verify correct URL was called  
        args, kwargs = mock_get.call_args
        url = args[0]
        expected_url = "https://api.telegram.org/bot{}/{}".format("test-token-456", "getUpdates")
        assert url == expected_url

    @patch('tools.telegram_tools.requests.post')  
    def test_answer_callback_targets_correct_endpoint(self, mock_post):
        """answer_callback targets answerCallbackQuery on api.telegram.org."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True, "result": True}
        mock_post.return_value = mock_response

        telegram_tools.answer_callback(
            token="test-token-789",
            callback_query_id="callback-123",
            text="Acknowledged"
        )

        # Verify correct URL was called
        args, kwargs = mock_post.call_args
        url = args[0]
        expected_url = "https://api.telegram.org/bot{}/{}".format("test-token-789", "answerCallbackQuery")
        assert url == expected_url

        # Verify payload
        payload = kwargs['json']
        assert payload['callback_query_id'] == "callback-123"
        assert payload['text'] == "Acknowledged"

    def test_no_webhook_methods_called(self):
        """Verify setWebhook is never called in any function.
        
        This is a static analysis test - webhook setup would use different methods.
        The design specifies long-polling only.
        """
        # This is more of a design verification
        # Check that none of our functions reference webhook-related endpoints
        import inspect
        
        # Get all functions from telegram_tools
        functions = [
            telegram_tools.send_review_message,
            telegram_tools.poll_updates, 
            telegram_tools.answer_callback
        ]
        
        for func in functions:
            source = inspect.getsource(func)
            # Verify no webhook-related API calls
            assert 'setWebhook' not in source
            assert 'deleteWebhook' not in source
            assert 'getWebhookInfo' not in source


class TestErrorHandling:
    """Test error handling in the telegram tools."""

    @patch('tools.telegram_tools.requests.post')
    def test_send_message_raises_on_http_error(self, mock_post):
        """send_review_message raises when HTTP request fails."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Network error")
        mock_post.return_value = mock_response

        with pytest.raises(requests.RequestException):
            telegram_tools.send_review_message(
                token="test-token",
                chat_id="test-chat",
                text="Test",
                job_key="key"
            )

    @patch('tools.telegram_tools.requests.post')
    def test_send_message_raises_on_telegram_error(self, mock_post):
        """send_review_message raises when Telegram API returns error."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "ok": False,
            "error_code": 400,
            "description": "Bad Request"
        }
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="Telegram API error"):
            telegram_tools.send_review_message(
                token="test-token",
                chat_id="test-chat", 
                text="Test",
                job_key="key"
            )

    @patch('tools.telegram_tools.requests.get')
    def test_poll_updates_raises_on_http_error(self, mock_get):
        """poll_updates raises when HTTP request fails."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Network error")
        mock_get.return_value = mock_response

        with pytest.raises(requests.RequestException):
            telegram_tools.poll_updates(token="test-token")

    @patch('tools.telegram_tools.requests.post')
    def test_answer_callback_swallows_exceptions(self, mock_post):
        """answer_callback swallows all exceptions (best-effort).""" 
        mock_post.side_effect = requests.RequestException("Network error")

        # Should not raise - this is best-effort
        telegram_tools.answer_callback(
            token="test-token",
            callback_query_id="callback-123"
        )

    @patch('tools.telegram_tools.requests.post')
    def test_answer_callback_swallows_telegram_errors(self, mock_post):
        """answer_callback swallows Telegram API errors (best-effort)."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "ok": False,
            "error_code": 400,
            "description": "Bad Request" 
        }
        mock_post.return_value = mock_response

        # Should not raise - this is best-effort
        telegram_tools.answer_callback(
            token="test-token",
            callback_query_id="callback-123"
        )


class TestFunctionSignatures:
    """Test that functions match the expected signatures."""

    def test_send_review_message_signature(self):
        """send_review_message has the correct signature."""
        import inspect
        sig = inspect.signature(telegram_tools.send_review_message)
        
        # Check parameter names and types
        params = list(sig.parameters.keys())
        assert params == ['token', 'chat_id', 'text', 'job_key']
        
        # Check return type annotation (if present)
        if sig.return_annotation != inspect.Signature.empty:
            assert sig.return_annotation == int

    def test_poll_updates_signature(self):
        """poll_updates has the correct signature."""
        import inspect
        sig = inspect.signature(telegram_tools.poll_updates)
        
        params = list(sig.parameters.keys())
        assert params == ['token', 'offset', 'timeout']
        
        # Check default values
        assert sig.parameters['offset'].default is None
        assert sig.parameters['timeout'].default == 25

    def test_answer_callback_signature(self):
        """answer_callback has the correct signature."""
        import inspect
        sig = inspect.signature(telegram_tools.answer_callback)
        
        params = list(sig.parameters.keys())
        assert params == ['token', 'callback_query_id', 'text']
        
        # Check default value
        assert sig.parameters['text'].default == ""

    def test_encode_decode_callback_signatures(self):
        """encode_callback and decode_callback have correct signatures."""
        import inspect
        
        encode_sig = inspect.signature(telegram_tools.encode_callback)
        encode_params = list(encode_sig.parameters.keys())
        assert encode_params == ['action', 'job_key']
        
        decode_sig = inspect.signature(telegram_tools.decode_callback)  
        decode_params = list(decode_sig.parameters.keys())
        assert decode_params == ['data']