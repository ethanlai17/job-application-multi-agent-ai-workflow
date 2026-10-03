"""Tests for Task 5: Add sheets retrieval and safe deletion.

Feature: telegram-review-gate
Tasks: 5.1, 5.2, 5.3 - Write property and unit tests for sheets operations

This covers:
- Property 10: Safe bottom-first deletion removes exactly the targets (Requirement 9.1, 9.3)
- Requirement 9.2: delete_job_row re-reads ID column immediately before deleting
- Requirement 4.1: get_undecided_jobs returns only UNDECIDED rows with sheet_row populated
"""

from datetime import date
from typing import List, Optional
import pytest
from hypothesis import given, strategies as st, assume

from models.job import ApplicationStatus, JobListing
from tools.sheets_tools import get_undecided_jobs, delete_job_row


class FakeWorksheet:
    """Fake worksheet that supports col_values(col) and delete_rows(idx) with correct index-shift semantics."""
    
    def __init__(self, data: List[List[str]], headers: List[str]):
        """
        Initialize with data rows (not including headers) and headers list.
        Row indexing is 1-based to match gspread.
        """
        self.headers = headers
        self.data = [headers] + data  # data[0] is headers, data[1:] are actual data rows
    
    def get_all_records(self) -> List[dict]:
        """Return list of dicts mapping headers to values."""
        if len(self.data) <= 1:  # Only headers or empty
            return []
        
        records = []
        for row_data in self.data[1:]:  # Skip header row
            # Pad row to match headers length
            padded_row = row_data + [''] * (len(self.headers) - len(row_data))
            record = {header: padded_row[i] for i, header in enumerate(self.headers)}
            records.append(record)
        return records
    
    def get_all_values(self) -> List[List[str]]:
        """Return all rows including headers."""
        return self.data
    
    def col_values(self, col: int) -> List[str]:
        """Return values in column (1-based), padding with empty strings as needed."""
        result = []
        for row in self.data:
            if col <= len(row):
                result.append(row[col - 1])
            else:
                result.append('')
        return result
    
    def delete_rows(self, start_row: int, end_row: Optional[int] = None) -> None:
        """Delete rows (1-based, inclusive). If end_row is None, delete only start_row."""
        if end_row is None:
            end_row = start_row
        
        # Convert to 0-based for list operations
        start_idx = start_row - 1
        end_idx = end_row - 1
        
        # Delete from end to start to avoid index shifts
        for idx in range(end_idx, start_idx - 1, -1):
            if 0 <= idx < len(self.data):
                del self.data[idx]


# Standard headers matching sheets_tools.py
HEADERS = [
    "Job Title", "Company", "URL", "Location", "Salary", "Date Found", 
    "Status", "CV Link", "Cover Letter Link", "LinkedIn Job ID", "Notes"
]


def make_job_row(
    title: str = "Test Job",
    company: str = "Test Company", 
    url: str = "https://linkedin.com/jobs/123",
    location: str = "Remote",
    salary: str = "$100k",
    date_found: str = "2024-01-01",
    status: str = "Found",
    cv_link: str = "",
    cover_letter_link: str = "",
    linkedin_job_id: str = "123",
    notes: str = ""
) -> List[str]:
    """Helper to create a job row with all required fields."""
    return [title, company, url, location, salary, date_found, status, cv_link, cover_letter_link, linkedin_job_id, notes]


class TestGetUndecidedJobs:
    """Test sub-task 5.3: unit test for get_undecided_jobs filter."""
    
    def test_get_undecided_jobs_returns_only_undecided_rows(self):
        """Assert only UNDECIDED rows are returned and sheet_row is populated.
        
        Validates: Requirement 4.1
        """
        # Create fake worksheet with mixed statuses
        data = [
            make_job_row(title="Job 1", status="Found", linkedin_job_id="111"),
            make_job_row(title="Job 2", status="Undecided", linkedin_job_id="222"),
            make_job_row(title="Job 3", status="Approved", linkedin_job_id="333"),
            make_job_row(title="Job 4", status="Undecided", linkedin_job_id="444"),
            make_job_row(title="Job 5", status="Rejected", linkedin_job_id="555"),
        ]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        # Call get_undecided_jobs
        undecided = get_undecided_jobs(fake_ws)
        
        # Should return exactly 2 jobs
        assert len(undecided) == 2
        
        # Both should be UNDECIDED status
        for job in undecided:
            assert job.status == ApplicationStatus.UNDECIDED
        
        # Should have correct titles and linkedin IDs
        titles = [job.title for job in undecided]
        ids = [job.linkedin_job_id for job in undecided]
        assert "Job 2" in titles
        assert "Job 4" in titles
        assert "222" in ids
        assert "444" in ids
        
        # sheet_row should be populated (Job 2 is 3rd row overall, Job 4 is 5th row overall)
        sheet_rows = [job.sheet_row for job in undecided]
        assert 3 in sheet_rows  # Job 2 is in data row 2, so sheet_row=3 (1-based, includes header)
        assert 5 in sheet_rows  # Job 4 is in data row 4, so sheet_row=5 (1-based, includes header)
    
    def test_get_undecided_jobs_empty_when_no_undecided(self):
        """Should return empty list when no UNDECIDED jobs exist."""
        data = [
            make_job_row(title="Job 1", status="Found", linkedin_job_id="111"),
            make_job_row(title="Job 2", status="Approved", linkedin_job_id="222"),
        ]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        undecided = get_undecided_jobs(fake_ws)
        assert len(undecided) == 0
    
    def test_get_undecided_jobs_empty_sheet(self):
        """Should return empty list for empty sheet."""
        fake_ws = FakeWorksheet([], HEADERS)
        
        undecided = get_undecided_jobs(fake_ws)
        assert len(undecided) == 0


class TestDeleteJobRowSafety:
    """Test sub-task 5.2: example test for re-resolution after row shift."""
    
    def test_delete_job_row_re_resolves_after_shift(self):
        """Assert delete_job_row re-reads ID column immediately before deleting 
        so a prior deletion's index shift doesn't delete the wrong row.
        
        Validates: Requirement 9.2
        """
        # Create worksheet with 3 jobs
        data = [
            make_job_row(title="Job A", linkedin_job_id="111"),
            make_job_row(title="Job B", linkedin_job_id="222"), 
            make_job_row(title="Job C", linkedin_job_id="333"),
        ]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        # Verify initial state
        assert len(fake_ws.get_all_values()) == 4  # header + 3 data rows
        records = fake_ws.get_all_records()
        assert len(records) == 3
        assert records[0]["Job Title"] == "Job A"
        assert records[1]["Job Title"] == "Job B"
        assert records[2]["Job Title"] == "Job C"
        
        # Delete first job (Job A with ID 111)
        result1 = delete_job_row(fake_ws, "111")
        assert result1 is True
        
        # Now Job B should have shifted to row 2, and Job C to row 3
        records_after_first = fake_ws.get_all_records()
        assert len(records_after_first) == 2
        assert records_after_first[0]["Job Title"] == "Job B"
        assert records_after_first[1]["Job Title"] == "Job C"
        
        # Delete Job C (ID 333) - this tests re-resolution
        # If delete_job_row cached the original row 4, it would try to delete 
        # beyond the sheet bounds. Instead it should re-read and find Job C is now in row 3.
        result2 = delete_job_row(fake_ws, "333")
        assert result2 is True
        
        # Should only have Job B left
        final_records = fake_ws.get_all_records()
        assert len(final_records) == 1
        assert final_records[0]["Job Title"] == "Job B"
        assert final_records[0]["LinkedIn Job ID"] == "222"
    
    def test_delete_job_row_handles_empty_id(self):
        """delete_job_row should return False for empty linkedin_job_id."""
        data = [make_job_row(title="Job A", linkedin_job_id="111")]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        result = delete_job_row(fake_ws, "")
        assert result is False
        
        # Sheet should be unchanged
        records = fake_ws.get_all_records()
        assert len(records) == 1
        assert records[0]["Job Title"] == "Job A"
    
    def test_delete_job_row_handles_missing_id(self):
        """delete_job_row should return False when linkedin_job_id not found."""
        data = [make_job_row(title="Job A", linkedin_job_id="111")]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        result = delete_job_row(fake_ws, "999")  # Non-existent ID
        assert result is False
        
        # Sheet should be unchanged
        records = fake_ws.get_all_records()
        assert len(records) == 1
        assert records[0]["Job Title"] == "Job A"


class TestSafeBottomFirstDeletion:
    """Test sub-task 5.1: property test for safe bottom-first deletion."""
    
    @given(
        # Generate arbitrary rows and a subset to delete
        all_job_ids=st.lists(
            st.text(alphabet="0123456789", min_size=1, max_size=10),
            min_size=1, max_size=20, unique=True
        ),
        target_fraction=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_property_safe_bottom_first_deletion(self, all_job_ids, target_fraction):
        """Property 10: Safe bottom-first deletion removes exactly the targets.
        
        Over a fake worksheet with arbitrary rows and an arbitrary subset of target 
        LinkedIn Job IDs, deleting those IDs (each re-resolved by id; when deleting 
        a known set, bottom-most-first) leaves exactly the non-target rows intact 
        regardless of original order.
        
        Feature: telegram-review-gate, Property 10
        Validates: Requirements 9.1, 9.3
        """
        assume(len(all_job_ids) >= 1)
        
        # Create target set - some fraction of all IDs
        num_targets = int(len(all_job_ids) * target_fraction)
        target_ids = set(all_job_ids[:num_targets])
        non_target_ids = set(all_job_ids[num_targets:])
        
        # Build fake worksheet with all jobs
        data = []
        for i, job_id in enumerate(all_job_ids):
            data.append(make_job_row(
                title=f"Job {i}",
                linkedin_job_id=job_id,
                status="Undecided"  # Make them all undecided for this test
            ))
        
        fake_ws = FakeWorksheet(data, HEADERS)
        
        # Record original state
        original_records = fake_ws.get_all_records()
        original_non_targets = [
            record for record in original_records 
            if record["LinkedIn Job ID"] in non_target_ids
        ]
        
        # Delete target IDs one by one (delete_job_row handles re-resolution)
        for target_id in target_ids:
            delete_job_row(fake_ws, target_id)
        
        # Verify final state
        final_records = fake_ws.get_all_records()
        final_job_ids = {record["LinkedIn Job ID"] for record in final_records}
        
        # Should have exactly the non-target IDs remaining
        assert final_job_ids == non_target_ids
        
        # Should have the correct number of rows
        assert len(final_records) == len(non_target_ids)
        
        # Non-target rows should have their original content intact
        # (though their row positions may have changed due to deletions)
        final_by_id = {record["LinkedIn Job ID"]: record for record in final_records}
        original_by_id = {record["LinkedIn Job ID"]: record for record in original_non_targets}
        
        for job_id in non_target_ids:
            assert job_id in final_by_id
            assert job_id in original_by_id
            
            # Content should match (excluding positional fields that might change)
            final_record = final_by_id[job_id]
            original_record = original_by_id[job_id]
            
            assert final_record["Job Title"] == original_record["Job Title"]
            assert final_record["LinkedIn Job ID"] == original_record["LinkedIn Job ID"]
            assert final_record["Status"] == original_record["Status"]
    
    def test_delete_multiple_bottom_first_ordering(self):
        """Example test showing bottom-first ordering in practice."""
        # Create worksheet with 5 jobs
        data = [
            make_job_row(title="Job 1", linkedin_job_id="111"),
            make_job_row(title="Job 2", linkedin_job_id="222"),
            make_job_row(title="Job 3", linkedin_job_id="333"),
            make_job_row(title="Job 4", linkedin_job_id="444"),
            make_job_row(title="Job 5", linkedin_job_id="555"),
        ]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        # Delete jobs 2, 4, and 5 in that logical order
        # But simulate the orchestrator doing bottom-first: delete 555, then 444, then 222
        targets_bottom_first = ["555", "444", "222"]  # Row 5, 4, 2 originally
        
        for target_id in targets_bottom_first:
            result = delete_job_row(fake_ws, target_id)
            assert result is True
        
        # Should have jobs 1 and 3 remaining
        final_records = fake_ws.get_all_records()
        assert len(final_records) == 2
        assert final_records[0]["Job Title"] == "Job 1"
        assert final_records[0]["LinkedIn Job ID"] == "111"
        assert final_records[1]["Job Title"] == "Job 3"
        assert final_records[1]["LinkedIn Job ID"] == "333"

    def test_single_row_deletion(self):
        """Edge case: deleting from a single-row sheet."""
        data = [make_job_row(title="Only Job", linkedin_job_id="123")]
        fake_ws = FakeWorksheet(data, HEADERS)
        
        result = delete_job_row(fake_ws, "123")
        assert result is True
        
        # Sheet should be empty (only headers remain)
        final_records = fake_ws.get_all_records()
        assert len(final_records) == 0
    
    def test_empty_sheet_deletion(self):
        """Edge case: attempting deletion on empty sheet."""
        fake_ws = FakeWorksheet([], HEADERS)
        
        result = delete_job_row(fake_ws, "123")
        assert result is False
        
        # Sheet should remain empty
        final_records = fake_ws.get_all_records()
        assert len(final_records) == 0