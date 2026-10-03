"""Tests for Task 3: Implement Summary_Service in review_agent.

Feature: telegram-review-gate, Property 9
Tasks: 3.1 - Write property + happy-path tests for summarize_fn

This covers:
- Requirement 8.1: Summary service uses DeepSeek client and returns LLM text on success
- Requirement 8.2: Summary service falls back to truncated excerpt on failure/empty content
"""

import pytest
from hypothesis import given, strategies as st
from unittest.mock import Mock, MagicMock
from datetime import date

from agents.review_agent import make_summarize_fn
from models.job import JobListing, ApplicationStatus


class TestSummarizeFunction:
    """Test the make_summarize_fn Summary Service implementation."""

    def test_happy_path_returns_llm_text(self):
        """With a fake client returning text, assert that text is returned.
        
        Validates: Requirement 8.1
        """
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test job summary from LLM"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create the summarize function
        summarize_fn = make_summarize_fn(mock_client, "test-model", excerpt_chars=600)
        
        # Create a test job
        job = JobListing(
            title="Software Engineer",
            company="Test Company",
            url="https://example.com/job/123",
            location="Remote",
            salary="$100k",
            date_found=date.today(),
            job_description="Detailed job description here with requirements and responsibilities."
        )
        
        # Call the function
        result = summarize_fn(job)
        
        # Should return the LLM text
        assert result == "This is a test job summary from LLM"
        
        # Verify the client was called correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "Summarise this job's responsibilities and key requirements in 3-4 short lines."},
                {"role": "user", "content": "Detailed job description here with requirements and responsibilities."},
            ],
            max_tokens=300,
        )

    def test_strips_whitespace_from_llm_response(self):
        """LLM response is stripped of whitespace before being returned."""
        # Mock client returning text with whitespace
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "   \n\nSummary text with whitespace\n\n   "
        mock_client.chat.completions.create.return_value = mock_response
        
        summarize_fn = make_summarize_fn(mock_client, "test-model")
        
        job = JobListing(
            title="Test Job",
            company="Test Company", 
            url="https://example.com/job",
            location="Remote",
            salary="$50k",
            date_found=date.today(),
            job_description="Test description"
        )
        
        result = summarize_fn(job)
        assert result == "Summary text with whitespace"

    def test_truncates_job_description_to_6000_chars_for_llm(self):
        """Job description is truncated to 6000 characters when sent to LLM."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "LLM summary"
        mock_client.chat.completions.create.return_value = mock_response
        
        summarize_fn = make_summarize_fn(mock_client, "test-model")
        
        # Create very long job description (over 6000 chars)
        long_description = "A" * 7000
        job = JobListing(
            title="Test Job",
            company="Test Company",
            url="https://example.com/job",
            location="Remote", 
            salary="$50k",
            date_found=date.today(),
            job_description=long_description
        )
        
        result = summarize_fn(job)
        
        # Should get LLM response
        assert result == "LLM summary"
        
        # Check that only first 6000 chars were sent to LLM
        called_args = mock_client.chat.completions.create.call_args
        user_message = called_args[1]["messages"][1]["content"]
        assert len(user_message) == 6000
        assert user_message == "A" * 6000


class TestSummarizeFallback:
    """Test fallback behavior when LLM fails or returns empty content."""

    @pytest.mark.parametrize("failure_mode", [
        "exception",  # Client raises exception
        "empty_content",  # Client returns empty content
        "none_content",  # Client returns None content
        "whitespace_only"  # Client returns whitespace-only content
    ])
    @given(
        job_description=st.text(min_size=0, max_size=2000),
        excerpt_chars=st.integers(min_value=100, max_value=1000)
    )
    def test_fallback_bounded_excerpt_property(self, job_description, excerpt_chars, failure_mode):
        """Property 9: With a failing client, result is non-empty, len <= excerpt_chars, and is a prefix of JD.
        
        Feature: telegram-review-gate, Property 9: Summary fallback always yields a non-empty bounded excerpt
        Validates: Requirements 8.1, 8.2
        """
        # Set up mock client based on failure mode
        mock_client = Mock()
        
        if failure_mode == "exception":
            mock_client.chat.completions.create.side_effect = Exception("LLM failed")
        else:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            
            if failure_mode == "empty_content":
                mock_response.choices[0].message.content = ""
            elif failure_mode == "none_content":
                mock_response.choices[0].message.content = None
            elif failure_mode == "whitespace_only":
                mock_response.choices[0].message.content = "   \n\n   "
            
            mock_client.chat.completions.create.return_value = mock_response
        
        # Create summarize function
        summarize_fn = make_summarize_fn(mock_client, "test-model", excerpt_chars=excerpt_chars)
        
        # Create job with the generated description
        job = JobListing(
            title="Test Job",
            company="Test Company",
            url="https://example.com/job", 
            location="Remote",
            salary="$50k",
            date_found=date.today(),
            job_description=job_description
        )
        
        # Call function
        result = summarize_fn(job)
        
        # Property assertions:
        # 1. Result is non-empty
        assert result != ""
        assert len(result) > 0
        
        # 2. Result length <= excerpt_chars
        assert len(result) <= excerpt_chars
        
        # 3. Result is a prefix of the JD (when JD is non-empty)
        if job_description.strip():
            expected_excerpt = job_description.strip()[:excerpt_chars]
            assert result == expected_excerpt
            # Verify it's actually a prefix
            assert job_description.strip().startswith(result)
        else:
            # When JD is empty, should return the fallback message
            assert result == "(no description available)"

    def test_empty_job_description_fallback_message(self):
        """When job description is empty, returns fallback message."""
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM failed")
        
        summarize_fn = make_summarize_fn(mock_client, "test-model", excerpt_chars=600)
        
        # Job with empty description
        job = JobListing(
            title="Test Job",
            company="Test Company",
            url="https://example.com/job",
            location="Remote",
            salary="$50k", 
            date_found=date.today(),
            job_description=""
        )
        
        result = summarize_fn(job)
        assert result == "(no description available)"

    def test_none_job_description_fallback_message(self):
        """When job description is None, returns fallback message."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM failed")
        
        summarize_fn = make_summarize_fn(mock_client, "test-model", excerpt_chars=600)
        
        # Job with None description
        job = JobListing(
            title="Test Job",
            company="Test Company", 
            url="https://example.com/job",
            location="Remote",
            salary="$50k",
            date_found=date.today(),
            job_description=None  # type: ignore
        )
        
        result = summarize_fn(job)
        assert result == "(no description available)"

    def test_whitespace_only_job_description_fallback_message(self):
        """When job description is whitespace only, returns fallback message."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM failed")
        
        summarize_fn = make_summarize_fn(mock_client, "test-model", excerpt_chars=600)
        
        # Job with whitespace-only description
        job = JobListing(
            title="Test Job",
            company="Test Company",
            url="https://example.com/job",
            location="Remote",
            salary="$50k",
            date_found=date.today(),
            job_description="   \n\n   "
        )
        
        result = summarize_fn(job)
        assert result == "(no description available)"


class TestSummarizeFunctionExcerptCharsParameter:
    """Test that excerpt_chars parameter works correctly."""

    def test_custom_excerpt_chars_default_600(self):
        """Default excerpt_chars is 600."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM failed")
        
        summarize_fn = make_summarize_fn(mock_client, "test-model")  # No excerpt_chars specified
        
        # Job with long description
        long_description = "X" * 1000  # 1000 chars
        job = JobListing(
            title="Test Job",
            company="Test Company",
            url="https://example.com/job",
            location="Remote",
            salary="$50k",
            date_found=date.today(),
            job_description=long_description
        )
        
        result = summarize_fn(job)
        
        # Should be truncated to 600 chars (default)
        assert len(result) == 600
        assert result == "X" * 600

    def test_custom_excerpt_chars_honored(self):
        """Custom excerpt_chars parameter is honored."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM failed")
        
        summarize_fn = make_summarize_fn(mock_client, "test-model", excerpt_chars=200)
        
        # Job with long description
        long_description = "Y" * 1000
        job = JobListing(
            title="Test Job",
            company="Test Company",
            url="https://example.com/job",
            location="Remote",
            salary="$50k",
            date_found=date.today(),
            job_description=long_description
        )
        
        result = summarize_fn(job)
        
        # Should be truncated to 200 chars
        assert len(result) == 200
        assert result == "Y" * 200