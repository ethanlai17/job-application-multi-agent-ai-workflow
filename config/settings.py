import dataclasses
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val


@dataclass
class Config:
    groq_api_key: str
    google_credentials_path: str
    google_sheet_url: str
    linkedin_email: str
    linkedin_password: str
    search_job_title: str
    search_location: str
    search_work_type: str
    search_salary_min: int
    max_jobs_per_run: int
    cv_path: str
    headless_browser: bool
    request_delay_min: float
    request_delay_max: float
    model: str

    @classmethod
    def load(cls) -> "Config":
        return cls(
            groq_api_key=_require("GROQ_API_KEY"),
            google_credentials_path=os.getenv(
                "GOOGLE_CREDENTIALS_PATH",
                "credentials/google_service_account.json",
            ),
            google_sheet_url=_require("GOOGLE_SHEET_URL"),
            linkedin_email=_require("LINKEDIN_EMAIL"),
            linkedin_password=_require("LINKEDIN_PASSWORD"),
            search_job_title=os.getenv("SEARCH_JOB_TITLE", "Product Manager"),
            search_location=os.getenv("SEARCH_LOCATION", "London, United Kingdom"),
            search_work_type=os.getenv("SEARCH_WORK_TYPE", "hybrid"),
            search_salary_min=int(os.getenv("SEARCH_SALARY_MIN", "70000")),
            max_jobs_per_run=int(os.getenv("MAX_JOBS_PER_RUN", "20")),
            cv_path=os.getenv("CV_PATH", "my_cv.pdf"),
            headless_browser=os.getenv("HEADLESS_BROWSER", "false").lower() == "true",
            request_delay_min=float(os.getenv("REQUEST_DELAY_MIN", "2.0")),
            request_delay_max=float(os.getenv("REQUEST_DELAY_MAX", "5.0")),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )

    def override(self, **kwargs) -> "Config":
        return dataclasses.replace(self, **{k: v for k, v in kwargs.items() if v is not None})
