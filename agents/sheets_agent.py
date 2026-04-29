import gspread

from config.settings import Config
from models.job import ApplicationStatus, JobListing
from tools import sheets_tools


class SheetsAgent:
    def __init__(self, config: Config):
        self.config = config
        self._ws: gspread.Worksheet | None = None

    @property
    def ws(self) -> gspread.Worksheet:
        if self._ws is None:
            self._ws = sheets_tools.get_or_create_worksheet(
                self.config.google_credentials_path,
                self.config.google_sheet_url,
            )
        return self._ws

    def append_jobs(self, jobs: list[JobListing]) -> list[JobListing]:
        """Append new jobs to the sheet and return them with sheet_row populated."""
        result = []
        for job in jobs:
            if not sheets_tools.job_already_tracked(self.ws, job.linkedin_job_id):
                row = sheets_tools.append_job(self.ws, job)
                job.sheet_row = row
                result.append(job)
        return result

    def update_job_status(self, job: JobListing, status: ApplicationStatus, **extra):
        job.status = status
        sheets_tools.update_job_row(self.ws, job.sheet_row, Status=status.value, **extra)

    def update_job_links(self, job: JobListing, cv_link: str, cover_letter_link: str):
        job.cv_link = cv_link
        job.cover_letter_link = cover_letter_link
        sheets_tools.update_job_row(
            self.ws,
            job.sheet_row,
            **{"CV Link": cv_link, "Cover Letter Link": cover_letter_link},
        )

    def get_approved_jobs(self) -> list[JobListing]:
        return sheets_tools.get_approved_jobs(self.ws)

    def get_all_jobs(self) -> list[JobListing]:
        return sheets_tools.get_all_jobs(self.ws)
