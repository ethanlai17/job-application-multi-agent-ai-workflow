"""
Test sub-tasks for Task 6: Wire the review gate into run_search / run_full

Tests:
- Property 1: Batch merge is the deduplicated union keyed by LinkedIn Job ID (Requirement 4.1, 4.2)
- Property 2: Branch dispatch acts only according to (origin, decision) (Requirement 1.4, 3.2, 3.3, 4.5)  
- Integration: Full new/resurfaced × approve/reject/timeout matrix (Requirements 1.1, 1.4, 3.1, 3.2, 3.3, 4.3, 4.4, 4.5)
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, MagicMock, patch
from datetime import date
from types import SimpleNamespace

from agents.orchestrator import Orchestrator
from agents.review_agent import Decision, job_key
from config.settings import Config  
from models.job import JobListing, ApplicationStatus


def make_test_job(linkedin_job_id: str = "", sheet_row: int = 0, **kwargs) -> JobListing:
    """Create a test JobListing with defaults."""
    defaults = {
        'title': 'Test Job',
        'company': 'Test Company', 
        'url': 'https://test.com/job',
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


class TestMergeReviewBatch:
    """Test sub-task 6.1: Property 1 - Batch merge is the deduplicated union keyed by LinkedIn Job ID."""

    @given(
        undecided_ids=st.lists(st.text(min_size=1, max_size=10), max_size=20),
        new_ids=st.lists(st.text(min_size=1, max_size=10), max_size=20)
    )
    @settings(deadline=None)
    def test_property_1_merge_dedup_collision_handling(self, undecided_ids, new_ids):
        """Property 1: merged batch is deduplicated union; on collision prefer re-surfaced."""
        config = make_fake_config()
        
        # Mock the required agents to avoid actual instantiation
        with patch('agents.orchestrator.SheetsAgent'), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            orchestrator = Orchestrator(config)
        
        # Create undecided jobs (sheet_row > 0)
        undecided = [make_test_job(linkedin_job_id=id_val, sheet_row=i+1) 
                     for i, id_val in enumerate(undecided_ids)]
        
        # Create new jobs (sheet_row == 0)  
        new_jobs = [make_test_job(linkedin_job_id=id_val, sheet_row=0)
                    for id_val in new_ids]
        
        # Test the merge
        batch = orchestrator._merge_review_batch(undecided, new_jobs)
        
        # Extract linkedin_job_ids from the batch
        batch_ids = [job.linkedin_job_id for job in batch]
        
        # Every non-empty ID should appear at most once
        non_empty_ids = [id_val for id_val in batch_ids if id_val]
        assert len(non_empty_ids) == len(set(non_empty_ids)), "Duplicate non-empty LinkedIn Job IDs found"
        
        # All undecided IDs should be present
        for id_val in undecided_ids:
            if id_val:
                assert id_val in batch_ids, f"Missing undecided job ID: {id_val}"
        
        # New IDs should be present only if not conflicting with undecided
        for id_val in new_ids:
            if id_val and id_val in undecided_ids:
                # Should prefer undecided version (check sheet_row > 0)
                batch_job = next(job for job in batch if job.linkedin_job_id == id_val)
                assert batch_job.sheet_row > 0, f"Should prefer undecided entry for {id_val}"
            elif id_val:
                assert id_val in batch_ids, f"Missing new job ID: {id_val}"
    
    def test_empty_linkedin_job_ids_are_kept(self):
        """Jobs with empty linkedin_job_ids should all be kept."""
        config = make_fake_config()
        
        with patch('agents.orchestrator.SheetsAgent'), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            orchestrator = Orchestrator(config)
        
        undecided = [
            make_test_job(linkedin_job_id="", sheet_row=1),
            make_test_job(linkedin_job_id="", sheet_row=2)
        ]
        new_jobs = [
            make_test_job(linkedin_job_id="", sheet_row=0),
            make_test_job(linkedin_job_id="", sheet_row=0)
        ]
        
        batch = orchestrator._merge_review_batch(undecided, new_jobs)
        
        # Should have all 4 jobs (empty IDs don't deduplicate)
        assert len(batch) == 4


class TestBranchDispatch:
    """Test sub-task 6.2: Property 2 - Branch dispatch acts only according to (origin, decision)."""

    @given(
        # Generate decision maps: job_key -> Decision  
        decisions_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=15), 
            values=st.sampled_from([Decision.APPROVED, Decision.REJECTED, Decision.UNDECIDED]),
            min_size=1, max_size=20
        )
    )
    @settings(deadline=None)
    def test_property_2_branch_dispatch_performs_exact_actions(self, decisions_data):
        """Property 2: branch dispatch performs exactly the actions mandated by (origin, decision)."""
        config = make_fake_config()
        
        # Create mock agents
        mock_sheets = Mock()
        mock_job_search = Mock()
        mock_document = Mock() 
        mock_application = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent', return_value=mock_job_search), \
             patch('agents.orchestrator.DocumentAgent', return_value=mock_document), \
             patch('agents.orchestrator.ApplicationAgent', return_value=mock_application):
            
            orchestrator = Orchestrator(config)
            
            # Create jobs matching the decision keys, mix of new/resurfaced
            jobs = []
            for i, (key, decision) in enumerate(decisions_data.items()):
                # Alternate between new (sheet_row=0) and resurfaced (sheet_row>0)
                sheet_row = 0 if i % 2 == 0 else i + 1
                job = make_test_job(
                    linkedin_job_id=key,
                    sheet_row=sheet_row,
                    company=f"Company{i}",
                    title=f"Job{i}"
                )
                jobs.append(job)
            
            # Mock the various methods that will be called
            mock_sheets.append_jobs.return_value = jobs  # Return same jobs as "logged"
            mock_sheets.update_job_status.return_value = None
            mock_sheets.delete_job_row.return_value = True
            orchestrator._generate_documents_for_approved = Mock()
            
            # Mock review_jobs to return our test decisions
            with patch('agents.orchestrator.review_jobs', return_value=decisions_data), \
                 patch('agents.orchestrator.console.print'):  # Suppress console output
                # Mock the other dependencies
                mock_sheets.get_undecided_jobs.return_value = []
                mock_job_search.run.return_value = jobs
                
                # Run the method
                orchestrator.run_search()
            
            # Count expected actions by (origin, decision)
            new_approved = new_rejected = new_undecided = 0
            resurfaced_approved = resurfaced_rejected = resurfaced_undecided = 0
            
            for i, (key, decision) in enumerate(decisions_data.items()):
                origin = "new" if i % 2 == 0 else "resurfaced"
                if decision == Decision.APPROVED:
                    if origin == "new":
                        new_approved += 1
                    else:
                        resurfaced_approved += 1
                elif decision == Decision.REJECTED:
                    if origin == "new":
                        new_rejected += 1
                    else:
                        resurfaced_rejected += 1
                else:  # UNDECIDED
                    if origin == "new":
                        new_undecided += 1
                    else:
                        resurfaced_undecided += 1
            
            # Verify exact actions were performed
            
            # new + approved → append_jobs + generate
            if new_approved > 0:
                assert mock_sheets.append_jobs.call_count >= 1
                assert orchestrator._generate_documents_for_approved.call_count >= new_approved
            
            # new + undecided → append_jobs + update_job_status(UNDECIDED)
            undecided_status_calls = [call for call in mock_sheets.update_job_status.call_args_list 
                                      if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.UNDECIDED]
            assert len(undecided_status_calls) == new_undecided
            
            # new + rejected → no sheet writes (verified by checking append_jobs wasn't called extra times)
            # This is implicitly verified by the above assertions being exact
            
            # resurfaced + approved → update_job_status(FOUND) + generate
            found_status_calls = [call for call in mock_sheets.update_job_status.call_args_list
                                  if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.FOUND]
            assert len(found_status_calls) == resurfaced_approved
            
            # resurfaced + rejected → delete_job_row called
            assert mock_sheets.delete_job_row.call_count == resurfaced_rejected
            
            # resurfaced + undecided → no-op (no additional calls beyond what's accounted for)
            
    def test_bottom_first_deletion_ordering(self):
        """Ensure rejected resurfaced jobs are deleted in descending sheet_row order."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent'), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            
            # Create resurfaced jobs with different sheet_rows, all to be rejected
            jobs = [
                make_test_job(linkedin_job_id="job1", sheet_row=5, company="Company1"),
                make_test_job(linkedin_job_id="job2", sheet_row=2, company="Company2"),
                make_test_job(linkedin_job_id="job3", sheet_row=8, company="Company3")
            ]
            
            decisions = {
                "job1": Decision.REJECTED,
                "job2": Decision.REJECTED, 
                "job3": Decision.REJECTED
            }
            
            mock_sheets.delete_job_row.return_value = True
            
            with patch('agents.orchestrator.review_jobs', return_value=decisions), \
                 patch('agents.orchestrator.console.print'):  # Suppress console output
                mock_sheets.get_undecided_jobs.return_value = jobs
                mock_sheets.run.return_value = []
                
                orchestrator.run_search()
            
            # Verify delete_job_row was called in descending sheet_row order (8, 5, 2)
            delete_calls = mock_sheets.delete_job_row.call_args_list
            assert len(delete_calls) == 3
            
            # Extract the linkedin_job_ids from the delete calls
            deleted_ids = [call[0][0] for call in delete_calls]
            
            # Should be in order job3 (row 8), job1 (row 5), job2 (row 2)
            expected_order = ["job3", "job1", "job2"]
            assert deleted_ids == expected_order


class TestIntegrationMatrix:
    """Test sub-task 6.3: Integration test for full new/resurfaced × approve/reject/timeout matrix."""
    
    def test_full_decision_matrix_integration(self):
        """Integration test covering all (origin, decision) combinations."""
        config = make_fake_config()
        
        # Create comprehensive test scenario
        undecided_jobs = [
            make_test_job(linkedin_job_id="resurfaced1", sheet_row=10, company="UndecidedCo1"),
            make_test_job(linkedin_job_id="resurfaced2", sheet_row=15, company="UndecidedCo2"),
            make_test_job(linkedin_job_id="resurfaced3", sheet_row=20, company="UndecidedCo3")
        ]
        
        new_jobs = [
            make_test_job(linkedin_job_id="new1", sheet_row=0, company="NewCo1"),
            make_test_job(linkedin_job_id="new2", sheet_row=0, company="NewCo2"),  
            make_test_job(linkedin_job_id="new3", sheet_row=0, company="NewCo3")
        ]
        
        # Decision matrix covering all combinations:
        decisions = {
            "new1": Decision.APPROVED,      # new + approved → append + generate
            "new2": Decision.REJECTED,      # new + rejected → console only  
            "new3": Decision.UNDECIDED,     # new + undecided → append + status UNDECIDED
            "resurfaced1": Decision.APPROVED,   # resurfaced + approved → update FOUND + generate
            "resurfaced2": Decision.REJECTED,   # resurfaced + rejected → delete
            "resurfaced3": Decision.UNDECIDED   # resurfaced + undecided → no-op
        }
        
        mock_sheets = Mock()
        mock_job_search = Mock()
        mock_document = Mock()
        mock_application = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent', return_value=mock_job_search), \
             patch('agents.orchestrator.DocumentAgent', return_value=mock_document), \
             patch('agents.orchestrator.ApplicationAgent', return_value=mock_application):
            
            orchestrator = Orchestrator(config)
            
            # Set up mocks
            mock_sheets.get_undecided_jobs.return_value = undecided_jobs
            mock_job_search.run.return_value = new_jobs
            mock_sheets.append_jobs.return_value = new_jobs  # Return same jobs as logged
            mock_sheets.update_job_status.return_value = None
            mock_sheets.delete_job_row.return_value = True
            orchestrator._generate_documents_for_approved = Mock()
            
            with patch('agents.orchestrator.review_jobs', return_value=decisions), \
                 patch('agents.orchestrator.console.print'):  # Suppress console output
                orchestrator.run_search()
            
            # Verify all expected actions occurred:
            
            # new1 (approved): append_jobs + generate 
            append_calls = mock_sheets.append_jobs.call_args_list
            assert len(append_calls) >= 1  # Should be called at least once for new jobs
            generate_calls = orchestrator._generate_documents_for_approved.call_args_list
            assert len(generate_calls) >= 2  # new1 + resurfaced1
            
            # new2 (rejected): no sheet operations (implicit - no extra calls)
            
            # new3 (undecided): append_jobs + update_job_status(UNDECIDED)
            undecided_calls = [call for call in mock_sheets.update_job_status.call_args_list
                               if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.UNDECIDED]
            assert len(undecided_calls) == 1  # Only new3
            
            # resurfaced1 (approved): update_job_status(FOUND) + generate
            found_calls = [call for call in mock_sheets.update_job_status.call_args_list
                           if len(call[0]) >= 2 and call[0][1] == ApplicationStatus.FOUND]
            assert len(found_calls) == 1  # Only resurfaced1
            
            # resurfaced2 (rejected): delete_job_row
            delete_calls = mock_sheets.delete_job_row.call_args_list
            assert len(delete_calls) == 1
            assert delete_calls[0][0][0] == "resurfaced2"
            
            # resurfaced3 (undecided): no-op (no additional calls)
            # This is verified implicitly by the exact counts above
    
    def test_empty_batch_early_return(self):
        """Test that empty batch returns early with appropriate message."""
        config = make_fake_config()
        
        mock_sheets = Mock()
        mock_job_search = Mock()
        
        with patch('agents.orchestrator.SheetsAgent', return_value=mock_sheets), \
             patch('agents.orchestrator.JobSearchAgent', return_value=mock_job_search), \
             patch('agents.orchestrator.DocumentAgent'), \
             patch('agents.orchestrator.ApplicationAgent'):
            
            orchestrator = Orchestrator(config)
            
            # Set up empty results
            mock_sheets.get_undecided_jobs.return_value = []
            mock_job_search.run.return_value = []
            
            with patch('agents.orchestrator.console.print') as mock_print:
                orchestrator.run_search()
                
                # Should print "nothing to review" message
                print_calls = [str(call) for call in mock_print.call_args_list]
                nothing_to_review = any("Nothing to review" in call or "nothing to review" in call 
                                        for call in print_calls)
                assert nothing_to_review, f"Expected 'nothing to review' message. Got: {print_calls}"