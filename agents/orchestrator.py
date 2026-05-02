import asyncio
import re
from datetime import date

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from agents.application_agent import ApplicationAgent
from agents.document_agent import DocumentAgent
from agents.job_search_agent import JobSearchAgent
from agents.sheets_agent import SheetsAgent
from browser.session import BrowserSession
from config.settings import Config
from models.job import ApplicationStatus, JobListing
from tools import playwright_tools
from tools.document_tools import read_cv
from tools.playwright_tools import JobUnavailableError, RecruiterJobError


console = Console()


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        self.sheets = SheetsAgent(config)
        self.job_search = JobSearchAgent(config, self.client)
        self.document = DocumentAgent(config, self.client)
        self.application = ApplicationAgent(config, self.client)

    def run_search(
        self,
        title: str | None = None,
        location: str | None = None,
        salary_min: int | None = None,
        max_jobs: int | None = None,
    ):
        console.print("[bold blue]Phase 1: Searching LinkedIn for jobs...[/bold blue]")

        new_jobs = self.job_search.run(
            title=title,
            location=location,
            salary_min=salary_min,
            max_results=max_jobs,
            ws=self.sheets.ws,
        )

        if not new_jobs:
            console.print("[yellow]No new jobs found.[/yellow]")
            return

        console.print(f"[green]Found {len(new_jobs)} new jobs. Logging to Google Sheets...[/green]")
        logged_jobs = self.sheets.append_jobs(new_jobs)

        cv_text = self._load_cv()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Generating documents...", total=len(logged_jobs))
            for job in logged_jobs:
                progress.update(task, description=f"[cyan]Generating: {job.company}")
                self.sheets.update_job_status(job, ApplicationStatus.CV_GENERATED)
                try:
                    cv_path, letter_path = self.document.generate(job, cv_text)
                    self.sheets.update_job_links(job, cv_path, letter_path)
                    self.sheets.update_job_status(job, ApplicationStatus.PENDING_REVIEW)
                    progress.console.print(f"    [green]✓[/green] {job.company}: CV & Cover Letter created")
                except Exception as e:
                    self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
                    progress.console.print(f"    [red]✗ Error generating docs for {job.company}: {e}[/red]")
                progress.advance(task)

        self._print_summary(logged_jobs)
        console.print(
            "\n[bold]Next step:[/bold] Open your Google Sheet, review the generated documents, "
            "then set [bold green]Status → Approved[/bold green] for roles you want to apply to.\n"
            "Then run: [bold cyan]python main.py apply[/bold cyan]"
        )

    def run_apply(self):
        console.print("[bold blue]Phase 3: Applying to approved jobs...[/bold blue]")

        approved = self.sheets.get_approved_jobs()
        if not approved:
            console.print("[yellow]No jobs marked as Approved in the sheet.[/yellow]")
            return

        console.print(f"[green]{len(approved)} job(s) approved. Starting applications...[/green]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Applying to jobs...", total=len(approved))
            for job in approved:
                progress.update(task, description=f"[cyan]Applying: {job.company}")
                self.sheets.update_job_status(job, ApplicationStatus.APPLYING)
                try:
                    success = self.application.apply(job)
                    if success:
                        self.sheets.update_job_status(job, ApplicationStatus.APPLIED)
                        progress.console.print(f"  [green]✓ {job.company} — Applied successfully[/green]")
                    else:
                        self.sheets.update_job_status(
                            job, ApplicationStatus.ERROR, Notes="Application agent returned failure"
                        )
                        progress.console.print(f"  [red]✗ {job.company} — Application failed[/red]")
                except Exception as e:
                    self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
                    progress.console.print(f"  [red]✗ {job.company} — Error: {e}[/red]")
                progress.advance(task)

    def run_process(self):
        """Process LinkedIn URLs manually added to the Google Sheet."""
        console.print("[bold blue]Processing manually-added LinkedIn URLs...[/bold blue]")
        asyncio.run(self._process_manual_async())

    async def _process_manual_async(self):
        manual_rows = self.sheets.get_manual_url_rows()
        if not manual_rows:
            console.print(
                "[yellow]No pending URLs found. Add a LinkedIn job URL to the sheet "
                "(leave Status blank) then re-run.[/yellow]"
            )
            return

        console.print(f"[green]Found {len(manual_rows)} URL(s) to process.[/green]")
        cv_text = self._load_cv()

        async with BrowserSession(self.config) as session:
            logged_in = await playwright_tools.linkedin_login(
                session, self.config.linkedin_email, self.config.linkedin_password
            )
            if not logged_in:
                raise RuntimeError("LinkedIn login failed — check credentials in .env")

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            )
            def _track_manual():
                with progress:
                    task = progress.add_task("[cyan]Processing manual URLs...", total=len(manual_rows))
                    for row in manual_rows:
                        progress.update(task, description=f"[cyan]Fetching: {row[1]}")
                        yield row
                        progress.advance(task)

            for row_num, url in _track_manual():
                # Extract numeric job ID from the URL slug
                clean_url = url.split("?")[0].rstrip("/")
                last_seg = clean_url.split("/")[-1]
                if last_seg.isdigit():
                    job_id = last_seg
                else:
                    m = re.search(r"-(\d+)$", last_seg)
                    job_id = m.group(1) if m else ""

                try:
                    details = await playwright_tools.get_job_details(session, url)
                except RecruiterJobError:
                    console.print("  [red]✗ Skipped — recruiter/agency posting[/red]")
                    self.sheets.update_job_status(
                        JobListing(title="", company="", url=url, location="", salary="",
                                   date_found=date.today(), sheet_row=row_num),
                        ApplicationStatus.ERROR,
                        Notes="Recruiter/agency posting — skipped automatically",
                    )
                    continue
                except JobUnavailableError:
                    console.print("  [red]✗ Skipped — job posting unavailable or removed[/red]")
                    self.sheets.update_job_status(
                        JobListing(title="", company="", url=url, location="", salary="",
                                   date_found=date.today(), sheet_row=row_num),
                        ApplicationStatus.ERROR,
                        Notes="Job posting unavailable or removed",
                    )
                    continue
                except Exception as e:
                    console.print(f"  [red]✗ Error scraping page: {e}[/red]")
                    self.sheets.update_job_status(
                        JobListing(title="", company="", url=url, location="", salary="",
                                   date_found=date.today(), sheet_row=row_num),
                        ApplicationStatus.ERROR,
                        Notes=str(e),
                    )
                    continue

                job = JobListing(
                    title=details.get("title_from_detail", ""),
                    company=details.get("company_from_detail", ""),
                    url=url,
                    location="",
                    salary=details.get("salary", ""),
                    date_found=date.today(),
                    job_description=details["job_description"],
                    linkedin_job_id=job_id,
                    sheet_row=row_num,
                )

                self.sheets.fill_manual_row(job)
                self.sheets.update_job_status(job, ApplicationStatus.CV_GENERATED)
                console.print(f"  [cyan]→[/cyan] {job.title} @ {job.company} — generating documents…")

                try:
                    cv_path, letter_path = self.document.generate(job, cv_text)
                    self.sheets.update_job_links(job, cv_path, letter_path)
                    self.sheets.update_job_status(job, ApplicationStatus.PENDING_REVIEW)
                    console.print(f"    [green]✓[/green] CV: {cv_path}")
                    console.print(f"    [green]✓[/green] Cover letter: {letter_path}")
                except Exception as e:
                    self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
                    console.print(f"    [red]✗ Doc generation failed: {e}[/red]")

        console.print(
            "\n[bold]Done.[/bold] Open your Google Sheet, review the documents, "
            "set [bold green]Status → Approved[/bold green], then run: "
            "[bold cyan]python main.py apply[/bold cyan]"
        )

    def run_full(
        self,
        title: str | None = None,
        location: str | None = None,
        salary_min: int | None = None,
        max_jobs: int | None = None,
    ):
        self.run_search(title=title, location=location, salary_min=salary_min, max_jobs=max_jobs)
        console.print("\n[italic]Waiting for you to review and approve jobs in Google Sheets...[/italic]")
        console.print("Run [bold cyan]python main.py apply[/bold cyan] when ready.")

    def run_regenerate(self, company: str, title_filter: str | None = None, cv_only: bool = False):
        """Re-scrape and regenerate CV + cover letter for a job already tracked in the sheet."""
        console.print(f"[bold blue]Regenerating documents for:[/bold blue] {company}" +
                      (f" / {title_filter}" if title_filter else ""))

        all_jobs = self.sheets.get_all_jobs()
        matches = [
            j for j in all_jobs
            if company.lower() in j.company.lower()
            and (title_filter is None or title_filter.lower() in j.title.lower())
        ]

        if not matches:
            console.print(f"[red]No jobs found matching company '{company}'" +
                          (f" and title '{title_filter}'" if title_filter else "") + "[/red]")
            return

        if len(matches) > 1:
            console.print(f"[yellow]Multiple matches found — use --title to narrow down:[/yellow]")
            for j in matches:
                console.print(f"  • {j.title} @ {j.company} (row {j.sheet_row})")
            return

        job = matches[0]
        console.print(f"  Found: [bold]{job.title}[/bold] @ {job.company} (row {job.sheet_row})")

        if not job.url:
            console.print("[red]No LinkedIn URL stored for this job — cannot re-scrape.[/red]")
            return

        cv_text = self._load_cv()
        asyncio.run(self._regenerate_async(job, cv_text, cv_only=cv_only))

    async def _regenerate_async(self, job: JobListing, cv_text: str, cv_only: bool = False):
        async with BrowserSession(self.config) as session:
            logged_in = await playwright_tools.linkedin_login(
                session, self.config.linkedin_email, self.config.linkedin_password
            )
            if not logged_in:
                raise RuntimeError("LinkedIn login failed — check credentials in .env")

            console.print(f"  Scraping: {job.url}")
            try:
                details = await playwright_tools.get_job_details(session, job.url)
            except (JobUnavailableError, RecruiterJobError) as e:
                console.print(f"  [red]✗ Cannot scrape job: {e}[/red]")
                return
            except Exception as e:
                console.print(f"  [red]✗ Error scraping page: {e}[/red]")
                return

        job.job_description = details["job_description"]
        if not job.title:
            job.title = details.get("title_from_detail", job.title)
        if not job.company:
            job.company = details.get("company_from_detail", job.company)

        console.print("  Regenerating CV…" if cv_only else "  Regenerating CV and cover letter…")
        try:
            cv_path, letter_path = self.document.generate(job, cv_text, cv_only=cv_only)
            self.sheets.update_job_links(job, cv_path, letter_path)
            console.print(f"  [green]✓[/green] CV: {cv_path}")
            if letter_path:
                console.print(f"  [green]✓[/green] Cover letter: {letter_path}")
        except Exception as e:
            console.print(f"  [red]✗ Doc generation failed: {e}[/red]")

    def run_status(self):
        all_jobs = self.sheets.get_all_jobs()
        if not all_jobs:
            console.print("[yellow]No jobs tracked yet. Run: python main.py search[/yellow]")
            return

        table = Table(title="Job Application Tracker", show_lines=True)
        table.add_column("Title", style="bold")
        table.add_column("Company")
        table.add_column("Location")
        table.add_column("Salary")
        table.add_column("Status")
        table.add_column("Date Found")

        status_colors = {
            ApplicationStatus.FOUND: "white",
            ApplicationStatus.CV_GENERATED: "cyan",
            ApplicationStatus.PENDING_REVIEW: "yellow",
            ApplicationStatus.APPROVED: "green",
            ApplicationStatus.APPLYING: "blue",
            ApplicationStatus.APPLIED: "bold green",
            ApplicationStatus.REJECTED: "red",
            ApplicationStatus.ERROR: "bold red",
        }

        for job in all_jobs:
            color = status_colors.get(job.status, "white")
            table.add_row(
                job.title,
                job.company,
                job.location,
                job.salary or "—",
                f"[{color}]{job.status.value}[/{color}]",
                str(job.date_found),
            )

        console.print(table)

    def _load_cv(self) -> str:
        try:
            return read_cv(self.config.cv_path)
        except FileNotFoundError:
            console.print(
                f"[red]CV file not found: {self.config.cv_path}[/red]\n"
                "Set CV_PATH in .env to point to your PDF or DOCX resume."
            )
            raise

    def _print_summary(self, jobs: list[JobListing]):
        table = Table(title="Jobs Found This Run")
        table.add_column("Title", style="bold")
        table.add_column("Company")
        table.add_column("Location")
        table.add_column("Salary")
        for job in jobs:
            table.add_row(job.title, job.company, job.location, job.salary or "—")
        console.print(table)
