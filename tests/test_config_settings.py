"""Config loading tests for the SEARCH_EXCLUDE_KEYWORDS default.

Feature: search-company-keyword-filters

Covers Requirement 2.4: `Config.load()` reads `SEARCH_EXCLUDE_KEYWORDS` from the
environment into `search_exclude_keywords`, defaulting to an empty string when the
variable is unset.
"""

import pytest

from config.settings import Config

# Required env vars (loaded via `_require`) that must be present for Config.load()
# to succeed. Values are dummy — these tests never touch the network.
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
    """Set the required env vars and clear SEARCH_EXCLUDE_KEYWORDS by default."""
    for key, val in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("SEARCH_EXCLUDE_KEYWORDS", raising=False)
    return monkeypatch


def test_exclude_keywords_defaults_to_empty(base_env):
    config = Config.load()
    assert config.search_exclude_keywords == ""


def test_exclude_keywords_read_from_env(base_env):
    base_env.setenv("SEARCH_EXCLUDE_KEYWORDS", "clearance, senior director")
    config = Config.load()
    assert config.search_exclude_keywords == "clearance, senior director"
