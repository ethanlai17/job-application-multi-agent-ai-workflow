from datetime import date
from pathlib import Path
from typing import Optional

import gspread
from google.oauth2.service_account import Credentials

from models.job import ApplicationStatus, JobListing

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

HEADERS = [
    "Job Title",
    "Company",
    "URL",
    "Location",
    "Salary",
    "Date Found",
    "Status",
    "CV Link",
    "Cover Letter Link",
    "LinkedIn Job ID",
    "Notes",
]


def _get_client(credentials_path: str) -> gspread.Client:
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    return gspread.authorize(creds)


def get_or_create_worksheet(credentials_path: str, sheet_url: str) -> gspread.Worksheet:
    """Open the tracking spreadsheet by URL and ensure headers exist."""
    client = _get_client(credentials_path)
    try:
        spreadsheet = client.open_by_url(sheet_url)
    except gspread.exceptions.NoValidUrlKeyFound:
        raise ValueError(
            f"Invalid Google Sheet URL: {sheet_url!r}\n"
            "Set GOOGLE_SHEET_URL in .env to the full URL of your sheet."
        )
    except gspread.exceptions.APIError as e:
        if "PERMISSION_DENIED" in str(e):
            raise PermissionError(
                "Service account does not have access to the sheet.\n"
                "Share the sheet with: job-application@job-application-494717.iam.gserviceaccount.com (Editor)"
            ) from e
        raise

    try:
        ws = spreadsheet.sheet1
    except Exception:
        ws = spreadsheet.add_worksheet(title="Jobs", rows=1000, cols=len(HEADERS))

    # Ensure header row exists
    first_row = ws.row_values(1)
    if first_row != HEADERS:
        ws.insert_row(HEADERS, index=1)
        # Format header bold
        ws.format("A1:K1", {"textFormat": {"bold": True}})

    return ws


def append_job(ws: gspread.Worksheet, job: JobListing) -> int:
    """Append a job to the sheet and return its row number."""
    row = [
        job.title,
        job.company,
        job.url,
        job.location,
        job.salary,
        job.date_found.isoformat() if isinstance(job.date_found, date) else str(job.date_found),
        job.status.value,
        job.cv_link,
        job.cover_letter_link,
        job.linkedin_job_id,
        job.notes,
    ]
    ws.append_row(row, value_input_option="USER_ENTERED")
    # Row number = total rows after append
    return len(ws.get_all_values())


def update_job_row(ws: gspread.Worksheet, row: int, **fields):
    """Update specific columns for a row. Uses column header names as keys."""
    col_map = {header: idx + 1 for idx, header in enumerate(HEADERS)}
    for field_name, value in fields.items():
        col = col_map.get(field_name)
        if col:
            ws.update_cell(row, col, str(value) if value is not None else "")


def job_already_tracked(ws: gspread.Worksheet, linkedin_job_id: str) -> bool:
    """Check if a LinkedIn job ID is already in the sheet."""
    job_id_col = HEADERS.index("LinkedIn Job ID") + 1
    existing_ids = ws.col_values(job_id_col)
    return linkedin_job_id in existing_ids


def get_all_jobs(ws: gspread.Worksheet) -> list[JobListing]:
    """Read all job rows from the sheet."""
    records = ws.get_all_records()
    jobs = []
    for idx, record in enumerate(records, start=2):
        try:
            job = JobListing(
                title=record.get("Job Title", ""),
                company=record.get("Company", ""),
                url=record.get("URL", ""),
                location=record.get("Location", ""),
                salary=record.get("Salary", ""),
                date_found=date.fromisoformat(record.get("Date Found", date.today().isoformat())),
                status=ApplicationStatus(record.get("Status", ApplicationStatus.FOUND.value)),
                cv_link=record.get("CV Link", ""),
                cover_letter_link=record.get("Cover Letter Link", ""),
                linkedin_job_id=record.get("LinkedIn Job ID", ""),
                notes=record.get("Notes", ""),
                sheet_row=idx,
            )
            jobs.append(job)
        except Exception:
            continue
    return jobs


def get_approved_jobs(ws: gspread.Worksheet) -> list[JobListing]:
    return [j for j in get_all_jobs(ws) if j.status == ApplicationStatus.APPROVED]
