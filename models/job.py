from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class ApplicationStatus(str, Enum):
    FOUND = "Found"
    CV_GENERATED = "CV Generated"
    PENDING_REVIEW = "Pending Review"
    APPROVED = "Approved"
    APPLYING = "Applying"
    APPLIED = "Applied"
    REJECTED = "Rejected"
    ERROR = "Error"


@dataclass
class JobListing:
    title: str
    company: str
    url: str
    location: str
    salary: str
    date_found: date
    status: ApplicationStatus = ApplicationStatus.FOUND
    cv_link: str = ""
    cover_letter_link: str = ""
    job_description: str = ""
    sheet_row: int = 0
    linkedin_job_id: str = ""
    notes: str = ""
