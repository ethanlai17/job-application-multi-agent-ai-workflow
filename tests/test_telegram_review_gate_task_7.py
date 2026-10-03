"""
Test sub-tasks for Task 7: Wire the review gate into run_process and update docs

Tests:
- Assert run_process reviews only scraped jobs and NEVER calls get_undecided_jobs (Requirements 5.1, 5.2)
- Assert scrape errors mark ERROR before review and are excluded from batch
- Assert brand-new branches: approved → fill+generate, rejected → console only, undecided → UNDECIDED status (Requirements 1.3, 3.1, 3.2, 3.3)
- Smoke: run_full invokes run_search and inherits the gate (Requirement 1.2)
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import date
from types import SimpleNamespace

from agents.orchestrator import Orchestrator  
from agents.review_agent import Decision, job_key
from config.settings import Config
from models.job import JobListing, ApplicationStatus
from tools.playwright_tools import RecruiterJobError, JobUnavailableError


def make_test_job(linkedin_job_id: str = "", sheet_row: int = 0, **kwargs) -> JobListing:
    """Create a test JobListing with defaults."""
    defaults = {
        'title': 'Test Job',
        'company': 'Test Company', 
        'url': 'https://linkedin.com/jobs/view/12345',
        'location': 'Test Location',
        'salary': '$50k-$60k',
        'date_found': date.today(),
        'status': ApplicationStatus.FOUND,
        'cv_link': '',
        'cover_letter_link': '',
        'linkedin_job_id': linkedin_job_id,
        'job_description': 'Test job description',
        'notes': '',
        'sheet_row': sheet_row
    }
    defaults.update(kwargs)
    return JobListing(**defaults)


def make_fake_config() -> Config:
    """Create a fake config for testing."""
    return Config(
        deepseek_api_key="test_key",
        google_credentials_path="test_path", 
        google_sheet_url="test_url",
        linkedin_email="test@example.com",
        linkedin_password="test_pass",
        search_job_title="Test",
        search_location="Test Location", 
        search_work_type="test",
        search_salary_min=50000,
        search_exclude_keywords="",
        max_jobs_per_run=10,
        cv_path="test_cv.pdf",
        headless_browser=True,
        request_delay_min=1.0,
        request_delay_max=2.0,
        model="test-model",
        document_model="test-doc-model",
        telegram_bot_token="test_bot_token",
        telegram_chat_id="test_chat_id",
        review_timeout_seconds=60
    )


class TestProcessScopeBoundary:
    """Test sub-task 7.1.1: Assert run_process reviews only scraped jobs and never calls get_undecided_jobs."""
    
    @patch('agents.orchestrator.asyncio.run')
    def test_run_process_never_calls_get_undecided_jobs(self, mock_asyncio_run):
        """run_process should never call get_undecided_jobs - validates Requirements 5.1, 5.2."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            
            # Mock _process_manual_async to track method calls
            orchestrator._process_manual_async = AsyncMock()
            
            # Call run_process
            orchestrator.run_process()
            
            # Verify get_undecided_jobs was never called
            assert mock_sheets.get_undecided_jobs.call_count == 0, \
                "run_process should never call get_undecided_jobs"
            
            # Verify _process_manual_async was called
            mock_asyncio_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_manual_async_scope_boundary(self):
        """_process_manual_async should only review scraped jobs, not undecided jobs."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        # The yielded session is a passive placeholder — the awaited playwright
        # functions are patched separately, so a plain Mock avoids spurious
        # un-awaited-coroutine warnings from AsyncMock attribute access.
        mock_browser_session = Mock()
        
        # Mock successful scraping for 2 URLs
        manual_rows = [(1, 'https://linkedin.com/jobs/view/111'), (2, 'https://linkedin.com/jobs/view/222')]
        mock_sheets.get_manual_url_rows.return_value = manual_rows
        
        scraped_jobs = [
            make_test_job(linkedin_job_id="111", sheet_row=1, title="Job1", company="Company1"),
            make_test_job(linkedin_job_id="222", sheet_row=2, title="Job2", company="Company2")
        ]
        
        decisions = {
            "111": Decision.APPROVED,
            "222": Decision.REJECTED
        }
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'), \
             patch('agents.orchestrator.BrowserSession') as mock_session_class, \
             patch('agents.orchestrator.playwright_tools.linkedin_login', new_callable=AsyncMock, return_value=True), \
             patch('agents.orchestrator.playwright_tools.get_job_details', new_callable=AsyncMock) as mock_get_details, \
             patch('agents.orchestrator.review_jobs', return_value=decisions), \
             patch('agents.orchestrator.console.print'):  # Suppress output
            
            # Setup browser session mock
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_browser_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock successful job details for both URLs
            mock_get_details.side_effect = [
                {
                    "title_from_detail": "Job1", 
                    "company_from_detail": "Company1",
                    "salary": "$60k",
                    "job_description": "Job 1 description"
                },
                {
                    "title_from_detail": "Job2",
                    "company_from_detail": "Company2", 
                    "salary": "$70k",
                    "job_description": "Job 2 description"
                }
            ]
            
            orchestrator = Orchestrator(config)
            orchestrator._process_scraped = Mock()  # Mock the processing method
            
            await orchestrator._process_manual_async()
            
            # Verify get_undecided_jobs was never called during processing
            assert mock_sheets.get_undecided_jobs.call_count == 0, \
                "Process should never call get_undecided_jobs"
            
            # Verify _process_scraped was called with exactly the scraped jobs
            assert orchestrator._process_scraped.call_count == 1
            call_args = orchestrator._process_scraped.call_args
            scraped_arg = call_args[0][0]  # First argument should be scraped jobs
            decisions_arg = call_args[0][1]  # Second argument should be decisions
            
            assert len(scraped_arg) == 2
            assert scraped_arg[0].linkedin_job_id == "111"
            assert scraped_arg[1].linkedin_job_id == "222"
            assert decisions_arg == decisions


class TestScrapeErrorHandling:
    """Test sub-task 7.1.2: Assert scrape errors mark ERROR before review and are excluded from batch."""
    
    @pytest.mark.asyncio
    async def test_recruiter_job_error_excluded_from_review(self):
        """RecruiterJobError should mark row ERROR and exclude from review batch."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        mock_browser_session = Mock()
        
        # One URL that will fail with RecruiterJobError, one that succeeds
        manual_rows = [(1, 'https://linkedin.com/jobs/view/111'), (2, 'https://linkedin.com/jobs/view/222')]
        mock_sheets.get_manual_url_rows.return_value = manual_rows
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'), \
             patch('agents.orchestrator.BrowserSession') as mock_session_class, \
             patch('agents.orchestrator.playwright_tools.linkedin_login', new_callable=AsyncMock, return_value=True), \
             patch('agents.orchestrator.playwright_tools.get_job_details', new_callable=AsyncMock) as mock_get_details, \
             patch('agents.orchestrator.review_jobs') as mock_review_jobs, \
             patch('agents.orchestrator.console.print'):  # Suppress output
            
            # Setup browser session mock
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_browser_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # First URL fails with RecruiterJobError, second succeeds
            mock_get_details.side_effect = [
                RecruiterJobError("Recruiter posting"),
                {
                    "title_from_detail": "Job2",
                    "company_from_detail": "Company2",
                    "salary": "$70k", 
                    "job_description": "Job 2 description"
                }
            ]
            
            orchestrator = Orchestrator(config)
            
            await orchestrator._process_manual_async()
            
            # Verify first job was marked ERROR
            error_calls = [call for call in mock_sheets.update_job_status.call_args_list
                           if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.ERROR]
            assert len(error_calls) == 1
            error_job = error_calls[0][0][0]  # First argument of first ERROR call
            assert error_job.sheet_row == 1  # Should be the first URL
            
            # Verify review_jobs was called with only 1 job (second URL that succeeded)
            assert mock_review_jobs.call_count == 1
            reviewed_jobs = mock_review_jobs.call_args[0][0]  # First argument  
            assert len(reviewed_jobs) == 1
            assert reviewed_jobs[0].linkedin_job_id == "222"
    
    @pytest.mark.asyncio
    async def test_job_unavailable_error_excluded_from_review(self):
        """JobUnavailableError should mark row ERROR and exclude from review batch."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        mock_browser_session = Mock()
        
        manual_rows = [(1, 'https://linkedin.com/jobs/view/111')]
        mock_sheets.get_manual_url_rows.return_value = manual_rows
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'), \
             patch('agents.orchestrator.BrowserSession') as mock_session_class, \
             patch('agents.orchestrator.playwright_tools.linkedin_login', new_callable=AsyncMock, return_value=True), \
             patch('agents.orchestrator.playwright_tools.get_job_details', new_callable=AsyncMock) as mock_get_details, \
             patch('agents.orchestrator.review_jobs') as mock_review_jobs, \
             patch('agents.orchestrator.console.print'):  # Suppress output
            
            # Setup browser session mock
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_browser_session)  
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # URL fails with JobUnavailableError
            mock_get_details.side_effect = JobUnavailableError("Job removed")
            
            orchestrator = Orchestrator(config)
            
            await orchestrator._process_manual_async()
            
            # Verify job was marked ERROR
            error_calls = [call for call in mock_sheets.update_job_status.call_args_list
                           if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.ERROR]
            assert len(error_calls) == 1
            
            # Verify review_jobs was never called (no successful scrapes)
            assert mock_review_jobs.call_count == 0
    
    @pytest.mark.asyncio
    async def test_generic_exception_excluded_from_review(self):
        """Generic exceptions should mark row ERROR and exclude from review batch.""" 
        config = make_fake_config()
        
        mock_sheets = Mock()
        mock_browser_session = Mock()
        
        manual_rows = [(1, 'https://linkedin.com/jobs/view/111')]
        mock_sheets.get_manual_url_rows.return_value = manual_rows
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'), \
             patch('agents.orchestrator.BrowserSession') as mock_session_class, \
             patch('agents.orchestrator.playwright_tools.linkedin_login', new_callable=AsyncMock, return_value=True), \
             patch('agents.orchestrator.playwright_tools.get_job_details', new_callable=AsyncMock) as mock_get_details, \
             patch('agents.orchestrator.review_jobs') as mock_review_jobs, \
             patch('agents.orchestrator.console.print'):  # Suppress output
            
            # Setup browser session mock
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_browser_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # URL fails with generic exception
            mock_get_details.side_effect = Exception("Network error")
            
            orchestrator = Orchestrator(config)
            
            await orchestrator._process_manual_async()
            
            # Verify job was marked ERROR with exception message
            error_calls = [call for call in mock_sheets.update_job_status.call_args_list
                           if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.ERROR]
            assert len(error_calls) == 1
            error_call = error_calls[0]
            assert "Network error" in error_call[1]['Notes']  # Exception message in Notes
            
            # Verify review_jobs was never called (no successful scrapes)
            assert mock_review_jobs.call_count == 0


class TestBrandNewBranches:
    """Test sub-task 7.1.3: Assert brand-new branches work correctly."""
    
    def test_approved_branch_fill_and_generate(self):
        """Approved jobs should fill manual row and generate documents (Requirements 1.3, 3.1)."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            orchestrator._generate_documents_for_approved = Mock()
            
            # Create approved job
            job = make_test_job(linkedin_job_id="123", sheet_row=1, title="Approved Job")
            decisions = {"123": Decision.APPROVED}
            
            with patch('agents.orchestrator.console.print'):  # Suppress output
                orchestrator._process_scraped([job], decisions)
            
            # Verify fill_manual_row was called
            mock_sheets.fill_manual_row.assert_called_once_with(job)
            
            # Verify _generate_documents_for_approved was called
            orchestrator._generate_documents_for_approved.assert_called_once_with(job)
    
    def test_rejected_branch_console_only(self):
        """Rejected jobs should only output to console, no status change (Requirements 3.2)."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            orchestrator._generate_documents_for_approved = Mock()
            
            # Create rejected job
            job = make_test_job(linkedin_job_id="456", sheet_row=2, title="Rejected Job") 
            decisions = {"456": Decision.REJECTED}
            
            with patch('agents.orchestrator.console.print') as mock_print:
                orchestrator._process_scraped([job], decisions)
            
            # Verify console output contains rejection message
            print_calls = [str(call) for call in mock_print.call_args_list]
            rejected_message = any("Rejected" in call and "manual row left untouched" in call
                                   for call in print_calls)
            assert rejected_message, f"Expected rejection message in console output. Got: {print_calls}"
            
            # Verify no sheet operations
            mock_sheets.fill_manual_row.assert_not_called()
            mock_sheets.update_job_status.assert_not_called() 
            orchestrator._generate_documents_for_approved.assert_not_called()
    
    def test_undecided_branch_status_update(self):
        """Undecided (timeout) jobs should update status to UNDECIDED, no generation (Requirements 3.3)."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            orchestrator._generate_documents_for_approved = Mock()
            
            # Create undecided job
            job = make_test_job(linkedin_job_id="789", sheet_row=3, title="Undecided Job")
            decisions = {"789": Decision.UNDECIDED}
            
            with patch('agents.orchestrator.console.print'):  # Suppress output
                orchestrator._process_scraped([job], decisions)
            
            # Verify status was updated to UNDECIDED
            mock_sheets.update_job_status.assert_called_once_with(job, ApplicationStatus.UNDECIDED)
            
            # Verify no document generation
            orchestrator._generate_documents_for_approved.assert_not_called()
    
    def test_mixed_decisions_processed_correctly(self):
        """Multiple jobs with different decisions should be processed according to their branches."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            orchestrator._generate_documents_for_approved = Mock()
            
            # Create jobs with different decisions
            jobs = [
                make_test_job(linkedin_job_id="approved", sheet_row=1, title="Approved Job"),
                make_test_job(linkedin_job_id="rejected", sheet_row=2, title="Rejected Job"),
                make_test_job(linkedin_job_id="undecided", sheet_row=3, title="Undecided Job")
            ]
            decisions = {
                "approved": Decision.APPROVED,
                "rejected": Decision.REJECTED,
                "undecided": Decision.UNDECIDED
            }
            
            with patch('agents.orchestrator.console.print'):  # Suppress output
                orchestrator._process_scraped(jobs, decisions)
            
            # Verify approved job: fill + generate
            mock_sheets.fill_manual_row.assert_called_once_with(jobs[0])
            orchestrator._generate_documents_for_approved.assert_called_once_with(jobs[0])
            
            # Verify undecided job: status update only
            undecided_calls = [call for call in mock_sheets.update_job_status.call_args_list
                               if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.UNDECIDED]
            assert len(undecided_calls) == 1
            assert undecided_calls[0][0][0] == jobs[2]  # Third job (undecided)
            
            # Verify rejected job: no sheet operations (implicit - only 1 fill_manual_row call above)


class TestRunFullInheritance:
    """Test sub-task 7.1.4: Smoke test that run_full invokes run_search and inherits the gate."""
    
    def test_run_full_calls_run_search(self):
        """run_full should call run_search and inherit the review gate (Requirement 1.2)."""
        config = make_fake_config()
        
        with patch('agents.orchestrator.SheetsAgent'), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            
            # Mock run_search to track calls
            orchestrator.run_search = Mock()
            
            # Call run_full with some parameters
            with patch('agents.orchestrator.console.print'):  # Suppress output
                orchestrator.run_full(
                    title="Software Engineer",
                    location="San Francisco", 
                    salary_min=100000,
                    max_jobs=5
                )
            
            # Verify run_search was called with the same parameters
            orchestrator.run_search.assert_called_once_with(
                title="Software Engineer",
                location="San Francisco",
                salary_min=100000,
                max_jobs=5,
                company_include=None,
                company_exclude=None,
                exclude_keywords=None
            )
    
    def test_run_full_passes_all_parameters(self):
        """run_full should pass all parameters to run_search."""
        config = make_fake_config()
        
        with patch('agents.orchestrator.SheetsAgent'), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            orchestrator.run_search = Mock()
            
            # Call run_full with all parameters
            with patch('agents.orchestrator.console.print'):  # Suppress output
                orchestrator.run_full(
                    title="Data Scientist",
                    location="New York",
                    salary_min=120000,
                    max_jobs=10,
                    company_include=["Google", "Meta"],
                    company_exclude=["Amazon"],
                    exclude_keywords=["contract", "freelance"]
                )
            
            # Verify all parameters were passed through
            orchestrator.run_search.assert_called_once_with(
                title="Data Scientist", 
                location="New York",
                salary_min=120000,
                max_jobs=10,
                company_include=["Google", "Meta"],
                company_exclude=["Amazon"],
                exclude_keywords=["contract", "freelance"]
            )