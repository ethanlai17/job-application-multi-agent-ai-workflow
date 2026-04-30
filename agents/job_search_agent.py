import asyncio
from datetime import date

from browser.session import BrowserSession
from config.settings import Config
from models.job import JobListing
from tools import playwright_tools, sheets_tools
from tools.playwright_tools import JobUnavailableError, RecruiterJobError
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


class JobSearchAgent:
    """Pure-Python job search — no LLM orchestration needed.

    The search pipeline is fully deterministic: login → scrape cards →
    fetch each detail page → filter unavailable/tracked → return list.
    LLM is only used downstream for document generation.
    """

    def __init__(self, config: Config, client=None):  # noqa: ARG002
        self.config = config
        self._ws = None

    def run(
        self,
        title: str | None = None,
        location: str | None = None,
        work_type: str | None = None,
        salary_min: int | None = None,
        max_results: int | None = None,
        ws=None,
    ) -> list[JobListing]:
        title = title or self.config.search_job_title
        location = location or self.config.search_location
        work_type = work_type or self.config.search_work_type
        salary_min = salary_min or self.config.search_salary_min
        max_results = max_results or self.config.max_jobs_per_run
        self._ws = ws
        return asyncio.run(
            self._run_async(title, location, work_type, salary_min, max_results)
        )

    async def _run_async(
        self, title, location, work_type, salary_min, max_results
    ) -> list[JobListing]:
        async with BrowserSession(self.config) as session:
            # Step 1: Login
            logged_in = await playwright_tools.linkedin_login(
                session, self.config.linkedin_email, self.config.linkedin_password
            )
            if not logged_in:
                raise RuntimeError("LinkedIn login failed — check credentials in .env")

            # Step 2: Scrape job cards from search results
            print(f"  Searching LinkedIn: '{title}' in '{location}' ({work_type}, £{salary_min:,}+)...")
            cards = await playwright_tools.search_linkedin_jobs(
                session,
                title=title,
                location=location,
                work_type=work_type,
                max_results=max_results,
            )
            print(f"  Found {len(cards)} job cards on search page")

            if not cards:
                return []

            # Step 3: Fetch each job detail page, skip unavailable / already tracked
            jobs: list[JobListing] = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Fetching job details...", total=len(cards))
                for i, card in enumerate(cards):
                    job_id = card.get("linkedin_job_id", "")

                    # Skip if already in the sheet
                    if self._ws and sheets_tools.job_already_tracked(self._ws, job_id):
                        progress.console.print(f"  [dim]Already tracked — skip: {card.get('url')}[/dim]")
                        progress.advance(task)
                        continue

                    progress.update(task, description=f"[cyan]Fetching details: {card.get('company', 'Unknown')}")
                    try:
                        details = await playwright_tools.get_job_details(session, card["url"])
                    except RecruiterJobError as e:
                        progress.console.print(f"    [yellow]✗ Skipped (recruiter/agency): {e}[/yellow]")
                        progress.advance(task)
                        continue
                    except JobUnavailableError as e:
                        progress.console.print(f"    [yellow]✗ Skipped (unavailable): {e}[/yellow]")
                        progress.advance(task)
                        continue
                    except Exception as e:
                        progress.console.print(f"    [red]✗ Skipped (error): {e}[/red]")
                        progress.advance(task)
                        continue

                    # Prefer title/company from the detail page (more reliable than card)
                    title_final = details.get("title_from_detail") or card.get("title", "")
                    company_final = details.get("company_from_detail") or card.get("company", "")

                    jobs.append(JobListing(
                        title=title_final,
                        company=company_final,
                        url=card["url"],
                        location=card.get("location", ""),
                        salary=details.get("salary") or card.get("salary", ""),
                        date_found=date.today(),
                        job_description=details["job_description"],
                        linkedin_job_id=job_id,
                    ))
                    progress.console.print(f"    [green]✓[/green] {title_final} @ {company_final}")
                    progress.advance(task)

                    if len(jobs) >= max_results:
                        break

            return jobs
