"""Tests for Task 1: Add Undecided status, Telegram/timeout config, and HTTP dependency.

Feature: telegram-review-gate
Tasks: 1.1 - Write unit/example tests for status, config, and dependency

This covers:
- Requirement 10.1: ApplicationStatus("Undecided") round-trip
- Requirements 7.1-7.3: Config.load() reads TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID and defaults review_timeout_seconds
- Requirement 11.1: requests appears in requirements.txt
"""

import os
import pytest
from pathlib import Path

from config.settings import Config
from models.job import ApplicationStatus


# Required env vars that must be present for Config.load() to succeed
_REQUIRED_ENV = {
    "DEEPSEEK_API_KEY": "test-key",
    "GOOGLE_SHEET_URL": "https://example.com/sheet",
    "LINKEDIN_EMAIL": "test@example.com",
    "LINKEDIN_PASSWORD": "test-pass",
    "TELEGRAM_BOT_TOKEN": "test-bot-token",
    "TELEGRAM_CHAT_ID": "test-chat-id",
}


@pytest.fixture
def base_env(monkeypatch):
    """Set the required env vars and clear optional ones by default."""
    for key, val in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("REVIEW_TIMEOUT_SECONDS", raising=False)
    return monkeypatch


class TestApplicationStatusUndecided:
    """Test the new UNDECIDED status value."""

    def test_undecided_status_round_trip(self):
        """ApplicationStatus("Undecided") is ApplicationStatus.UNDECIDED round-trip.
        
        Validates: Requirement 10.1
        """
        # Create from string
        status_from_string = ApplicationStatus("Undecided")
        
        # Should equal the enum value
        assert status_from_string == ApplicationStatus.UNDECIDED
        assert status_from_string is ApplicationStatus.UNDECIDED
        
        # And the string value should be correct
        assert ApplicationStatus.UNDECIDED == "Undecided"
        assert ApplicationStatus.UNDECIDED.value == "Undecided"

    def test_undecided_status_exists_alongside_others(self):
        """Ensure UNDECIDED exists alongside all existing status values."""
        expected_statuses = {
            "Found", "CV Generated", "Pending Review", "Approved", 
            "Applying", "Applied", "Rejected", "Error", "Undecided"
        }
        
        actual_statuses = {status.value for status in ApplicationStatus}
        assert actual_statuses == expected_statuses


class TestTelegramConfig:
    """Test the new Telegram configuration fields."""

    def test_config_reads_telegram_bot_token_from_env(self, base_env):
        """Config.load() reads TELEGRAM_BOT_TOKEN from environment.
        
        Validates: Requirement 7.1
        """
        base_env.setenv("TELEGRAM_BOT_TOKEN", "my-test-bot-token")
        config = Config.load()
        assert config.telegram_bot_token == "my-test-bot-token"

    def test_config_reads_telegram_chat_id_from_env(self, base_env):
        """Config.load() reads TELEGRAM_CHAT_ID from environment.
        
        Validates: Requirement 7.2
        """
        base_env.setenv("TELEGRAM_CHAT_ID", "my-test-chat-id")
        config = Config.load()
        assert config.telegram_chat_id == "my-test-chat-id"

    def test_config_defaults_review_timeout_seconds_to_3600(self, base_env):
        """Config.load() defaults review_timeout_seconds to 3600 when REVIEW_TIMEOUT_SECONDS is unset.
        
        Validates: Requirement 7.3
        """
        # Ensure the env var is unset (fixture already does this)
        config = Config.load()
        assert config.review_timeout_seconds == 3600

    def test_config_honors_explicit_review_timeout_seconds(self, base_env):
        """Config.load() honors an explicit REVIEW_TIMEOUT_SECONDS override.
        
        Validates: Requirement 7.3
        """
        base_env.setenv("REVIEW_TIMEOUT_SECONDS", "7200")
        config = Config.load()
        assert config.review_timeout_seconds == 7200

    def test_config_load_fails_without_telegram_bot_token(self, base_env):
        """Config.load() fails when TELEGRAM_BOT_TOKEN is missing."""
        base_env.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        
        with pytest.raises(ValueError, match="Missing required environment variable: TELEGRAM_BOT_TOKEN"):
            Config.load()

    def test_config_load_fails_without_telegram_chat_id(self, base_env):
        """Config.load() fails when TELEGRAM_CHAT_ID is missing."""
        base_env.delenv("TELEGRAM_CHAT_ID", raising=False)
        
        with pytest.raises(ValueError, match="Missing required environment variable: TELEGRAM_CHAT_ID"):
            Config.load()

    def test_config_contains_all_new_fields(self, base_env):
        """Config contains all the new telegram-related fields."""
        config = Config.load()
        
        # Check that all new fields exist and are accessible
        assert hasattr(config, 'telegram_bot_token')
        assert hasattr(config, 'telegram_chat_id')
        assert hasattr(config, 'review_timeout_seconds')
        
        # Check they have the expected values
        assert config.telegram_bot_token == "test-bot-token"
        assert config.telegram_chat_id == "test-chat-id"
        assert config.review_timeout_seconds == 3600


class TestHttpDependency:
    """Test that the HTTP dependency is declared."""

    def test_requests_in_requirements_txt(self):
        """requests appears in requirements.txt.
        
        Validates: Requirement 11.1
        """
        requirements_path = Path(__file__).parent.parent / "requirements.txt"
        requirements_content = requirements_path.read_text()
        
        # Should contain requests with version constraint
        assert "requests>=2.31.0" in requirements_content

    def test_requests_can_be_imported(self):
        """Smoke test: requests can actually be imported."""
        try:
            import requests
            # Basic functionality check
            assert hasattr(requests, 'get')
            assert hasattr(requests, 'post')
        except ImportError:
            pytest.skip("requests not installed - this is expected during unit tests")