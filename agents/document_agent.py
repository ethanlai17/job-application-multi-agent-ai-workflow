import re
from pathlib import Path

from openai import OpenAI

from config.settings import Config
from models.job import JobListing
from tools.document_tools import (
    generate_cover_letter,
    generate_tailored_cv,
    render_cover_letter_pdf,
    render_cv_pdf,
)
from tools.keyword_tools import get_jd_required_keywords, get_popular_keywords_flat, update_library_from_jd


def _safe(text: str) -> str:
    """Sanitise text for use in a filename (replace non-alphanumeric with _)."""
    return re.sub(r"[^\w ]", "_", text).strip()


class DocumentAgent:
    def __init__(self, config: Config, client: OpenAI):
        self.config = config
        self.client = client

    def generate(self, job: JobListing, cv_text: str) -> tuple[str, str]:
        """Generate a tailored CV and cover letter as A4 PDFs.

        Returns (cv_path, cover_letter_path) as absolute file paths.
        """
        company = _safe(job.company)
        title   = _safe(job.title)
        out_dir = Path("output") / f"{company}_{title}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cv_file     = out_dir / f"{company}_{title}_CV_Ethan Lai.pdf"
        letter_file = out_dir / f"{company}_{title}_Cover letter_Ethan Lai.pdf"

        # Update keyword library with this JD (text-matching, no LLM tokens).
        update_library_from_jd(job.job_description)

        # Popular = appear in 2+ JDs across the market.
        # JD-required = niche keywords this employer mentions 2+ times (role-specific must-haves).
        # Both are included in the CV skills section.
        popular_keywords = get_popular_keywords_flat()
        jd_required = get_jd_required_keywords(job.job_description)
        all_approved_keywords = list(dict.fromkeys(popular_keywords + jd_required))

        # CV — LLM returns JSON with tailored sections
        cv_sections = generate_tailored_cv(
            cv_text=cv_text,
            job_description=job.job_description,
            job_title=job.title,
            company=job.company,
            client=self.client,
            model=self.config.model,
            popular_keywords=all_approved_keywords,
        )
        cv_path = render_cv_pdf(cv_sections, str(cv_file))

        # Cover letter — LLM returns plain body text
        letter_body = generate_cover_letter(
            cv_text=cv_text,
            job_description=job.job_description,
            job_title=job.title,
            company=job.company,
            client=self.client,
            model=self.config.model,
        )
        letter_path = render_cover_letter_pdf(letter_body, str(letter_file))

        return cv_path, letter_path
