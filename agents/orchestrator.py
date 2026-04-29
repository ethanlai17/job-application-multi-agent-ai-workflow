from openai import OpenAI
from rich.console import Console
from rich.table import Table

from agents.application_agent import ApplicationAgent
from agents.document_agent import DocumentAgent
from agents.job_search_agent import JobSearchAgent
from agents.sheets_agent import SheetsAgent
from config.settings import Config
from models.job import ApplicationStatus, JobListing
from tools.document_tools import read_cv

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

console = Console()


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.groq_api_key,
            base_url=GROQ_BASE_URL,
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

        for job in logged_jobs:
            self.sheets.update_job_status(job, ApplicationStatus.CV_GENERATED)
            console.print(f"  Generating documents for [bold]{job.title}[/bold] @ {job.company}...")
            try:
                cv_path, letter_path = self.document.generate(job, cv_text)
                self.sheets.update_job_links(job, cv_path, letter_path)
                self.sheets.update_job_status(job, ApplicationStatus.PENDING_REVIEW)
                console.print(f"    [green]✓[/green] CV: {cv_path}")
                console.print(f"    [green]✓[/green] Cover letter: {letter_path}")
            except Exception as e:
                self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
                console.print(f"    [red]✗ Error generating docs: {e}[/red]")

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

        for job in approved:
            console.print(f"\n  Applying to [bold]{job.title}[/bold] @ {job.company}...")
            self.sheets.update_job_status(job, ApplicationStatus.APPLYING)
            try:
                success = self.application.apply(job)
                if success:
                    self.sheets.update_job_status(job, ApplicationStatus.APPLIED)
                    console.print("  [green]✓ Applied successfully[/green]")
                else:
                    self.sheets.update_job_status(
                        job, ApplicationStatus.ERROR, Notes="Application agent returned failure"
                    )
                    console.print("  [red]✗ Application failed[/red]")
            except Exception as e:
                self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
                console.print(f"  [red]✗ Error: {e}[/red]")

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
