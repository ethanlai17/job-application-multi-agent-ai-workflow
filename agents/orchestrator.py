import asyncio
import re
from datetime import date
from types import SimpleNamespace

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from agents.application_agent import ApplicationAgent
from agents.document_agent import DocumentAgent
from agents.job_search_agent import JobSearchAgent
from agents.review_agent import Decision, make_summarize_fn, review_jobs, job_key
from agents.sheets_agent import SheetsAgent
from browser.session import BrowserSession
from config.settings import Config
from models.job import ApplicationStatus, JobListing
from tools import playwright_tools, telegram_tools
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
        
        # Build the Telegram client adapter from config
        self.telegram = SimpleNamespace(
            send_review_message=lambda text, job_key: telegram_tools.send_review_message(
                config.telegram_bot_token, config.telegram_chat_id, text, job_key
            ),
            poll_updates=lambda offset=None, timeout=25: telegram_tools.poll_updates(
                config.telegram_bot_token, offset, timeout
            ),
            answer_callback=lambda callback_query_id, text="": telegram_tools.answer_callback(
                config.telegram_bot_token, callback_query_id, text
            )
        )
        
        # Build the summarize function
        self.summarize_fn = make_summarize_fn(self.client, config.model)

    def _merge_review_batch(self, undecided: list[JobListing], new_jobs: list[JobListing]) -> list[JobListing]:
        """Merge and deduplicate new jobs with undecided jobs by LinkedIn Job ID.
        
        On collision (same non-empty linkedin_job_id), prefer the undecided (sheet-backed) entry
        so origin stays 're-surfaced' and the operator's reject can delete it.
        Jobs with empty linkedin_job_ids are all kept.
        
        Args:
            undecided: Re-surfaced jobs from sheets (sheet_row > 0)  
            new_jobs: Brand-new discovered jobs (sheet_row == 0)
            
        Returns:
            Merged and deduplicated list of jobs
        """
        batch = []
        seen_ids = set()
        
        # First, add undecided jobs (deduplicating within undecided)
        for job in undecided:
            if not job.linkedin_job_id:
                # Always keep jobs with empty IDs
                batch.append(job)
            elif job.linkedin_job_id not in seen_ids:
                batch.append(job)
                seen_ids.add(job.linkedin_job_id)
        
        # Then, add new jobs that don't conflict with undecided
        for job in new_jobs:
            if not job.linkedin_job_id:
                # Always keep jobs with empty IDs
                batch.append(job)
            elif job.linkedin_job_id not in seen_ids:
                batch.append(job)
                seen_ids.add(job.linkedin_job_id)
        
        return batch

    def run_search(
        self,
        title: str | None = None,
        location: str | None = None,
        salary_min: int | None = None,
        max_jobs: int | None = None,
        company_include: list[str] | None = None,
        company_exclude: list[str] | None = None,
        exclude_keywords: list[str] | None = None,
    ):
        console.print("[bold blue]Phase 1: Searching LinkedIn for jobs...[/bold blue]")

        # Get re-surfaced undecided jobs (origin = re-surfaced, sheet_row > 0)
        undecided = self.sheets.get_undecided_jobs()
        if undecided:
            console.print(f"[yellow]Found {len(undecided)} undecided jobs from previous runs to re-review.[/yellow]")

        # Get brand-new jobs (origin = brand-new, sheet_row == 0)
        new_jobs = self.job_search.run(
            title=title,
            location=location,
            salary_min=salary_min,
            max_results=max_jobs,
            company_include=company_include,
            company_exclude=company_exclude,
            exclude_keywords=exclude_keywords,
            ws=self.sheets.ws,
        )

        if new_jobs:
            console.print(f"[green]Discovered {len(new_jobs)} new jobs.[/green]")
        else:
            console.print("[yellow]No new jobs discovered.[/yellow]")

        # Merge + dedup into one review batch
        batch = self._merge_review_batch(undecided, new_jobs)
        
        if not batch:
            console.print("[yellow]Nothing to review - no jobs found.[/yellow]")
            return

        console.print(f"[cyan]Reviewing {len(batch)} jobs via Telegram...[/cyan]")
        
        # Review jobs via Telegram gate
        decisions = review_jobs(batch, self.summarize_fn, self.telegram, self.config.review_timeout_seconds)
        
        # Track counts for summary
        approved_count = 0
        rejected_count = 0
        undecided_count = 0
        
        # Collect jobs for bottom-first deletion
        to_delete = []
        
        # Branch per (origin, decision)
        for job in batch:
            key = job_key(job)
            decision = decisions.get(key, Decision.UNDECIDED)
            origin = "resurfaced" if job.sheet_row else "new"
            
            if decision == Decision.APPROVED:
                approved_count += 1
                if origin == "new":
                    # new + approved → append_jobs([job]) + generate
                    console.print(f"[green]✓ Approved:[/green] {job.company} - {job.title} (new)")
                    logged_jobs = self.sheets.append_jobs([job])
                    if logged_jobs:
                        self._generate_documents_for_approved(logged_jobs[0])
                else:
                    # resurfaced + approved → update_job_status(FOUND) + generate  
                    console.print(f"[green]✓ Approved:[/green] {job.company} - {job.title} (re-surfaced)")
                    self.sheets.update_job_status(job, ApplicationStatus.FOUND)
                    self._generate_documents_for_approved(job)
                    
            elif decision == Decision.REJECTED:
                rejected_count += 1
                if origin == "new":
                    # new + rejected → console only, no sheet write
                    console.print(f"[red]✗ Rejected:[/red] {job.company} - {job.title} (new)")
                else:
                    # resurfaced + rejected → collect for deletion
                    console.print(f"[red]✗ Rejected:[/red] {job.company} - {job.title} (re-surfaced, will delete)")
                    to_delete.append(job)
                    
            else:  # Decision.UNDECIDED
                undecided_count += 1
                if origin == "new":
                    # new + undecided → append_jobs([job]) + update_job_status(UNDECIDED)
                    console.print(f"[yellow]? Undecided:[/yellow] {job.company} - {job.title} (new, timeout)")
                    logged_jobs = self.sheets.append_jobs([job])
                    if logged_jobs:
                        self.sheets.update_job_status(logged_jobs[0], ApplicationStatus.UNDECIDED)
                else:
                    # resurfaced + undecided → no-op (already Undecided)
                    console.print(f"[yellow]? Still undecided:[/yellow] {job.company} - {job.title} (re-surfaced, timeout)")
        
        # Delete rejected undecided jobs bottom-first for safety
        if to_delete:
            # Sort by sheet_row descending (bottom-most first)
            to_delete.sort(key=lambda j: j.sheet_row, reverse=True)
            console.print(f"[red]Deleting {len(to_delete)} rejected undecided jobs...[/red]")
            for job in to_delete:
                if self.sheets.delete_job_row(job.linkedin_job_id):
                    console.print(f"[red]  ✓ Deleted row for {job.company}[/red]")
                else:
                    console.print(f"[red]  ✗ Failed to delete row for {job.company}[/red]")
        
        # Print summary with counts
        self._print_review_summary(approved_count, rejected_count, undecided_count)
        
        if approved_count > 0:
            console.print(
                "\n[bold]Next step:[/bold] Open your Google Sheet, review the generated documents, "
                "then set [bold green]Status → Approved[/bold green] for roles you want to apply to.\n"
                "Then run: [bold cyan]python main.py apply[/bold cyan]"
            )

    def _generate_documents_for_approved(self, job: JobListing):
        """Generate CV and cover letter for an approved job with proper status updates."""
        cv_text = self._load_cv()
        
        self.sheets.update_job_status(job, ApplicationStatus.CV_GENERATED)
        try:
            cv_path, letter_path = self.document.generate(job, cv_text)
            self.sheets.update_job_links(job, cv_path, letter_path)
            self.sheets.update_job_status(job, ApplicationStatus.PENDING_REVIEW)
        except Exception as e:
            self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
            console.print(f"    [red]✗ Error generating docs for {job.company}: {e}[/red]")

    def _print_review_summary(self, approved_count: int, rejected_count: int, undecided_count: int):
        """Print a summary of review decisions."""
        total = approved_count + rejected_count + undecided_count
        console.print(f"\n[bold]Review Summary:[/bold]")
        console.print(f"  [green]Approved:[/green] {approved_count}/{total}")
        console.print(f"  [red]Rejected:[/red] {rejected_count}/{total}")
        console.print(f"  [yellow]Undecided:[/yellow] {undecided_count}/{total}")
        if undecided_count > 0:
            console.print(f"  [dim]Undecided jobs will be re-surfaced in the next search run.[/dim]")

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

        scraped = []  # Collect successfully scraped JobListing objects

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

                # Successfully scraped - add to review batch
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
                scraped.append(job)

        # After scraping completes, review the collected jobs
        if not scraped:
            console.print("[yellow]No jobs were successfully scraped for review.[/yellow]")
            return

        # Review jobs using the Telegram review gate
        decisions = review_jobs(scraped, self.summarize_fn, self.telegram, self.config.review_timeout_seconds)
        
        # Process decisions using brand-new rules only
        self._process_scraped(scraped, decisions)

        console.print(
            "\n[bold]Done.[/bold] Open your Google Sheet, review the documents, "
            "set [bold green]Status → Approved[/bold green], then run: "
            "[bold cyan]python main.py apply[/bold cyan]"
        )

    def _process_scraped(self, scraped: list[JobListing], decisions: dict[str, Decision]):
        """Process the scraped jobs based on review decisions using brand-new rules only."""
        approved_count = 0
        rejected_count = 0
        undecided_count = 0
        
        for job in scraped:
            key = job_key(job)
            decision = decisions.get(key, Decision.UNDECIDED)
            
            if decision == Decision.APPROVED:
                console.print(f"[green]✓ Approved:[/green] {job.title} @ {job.company}")
                # Fill the manual row first, then generate documents
                self.sheets.fill_manual_row(job)
                self._generate_documents_for_approved(job)
                approved_count += 1
                
            elif decision == Decision.REJECTED:
                console.print(f"[red]✗ Rejected:[/red] {job.title} @ {job.company} — manual row left untouched")
                rejected_count += 1
                
            elif decision == Decision.UNDECIDED:
                console.print(f"[yellow]? Undecided (timeout):[/yellow] {job.title} @ {job.company}")
                self.sheets.update_job_status(job, ApplicationStatus.UNDECIDED)
                undecided_count += 1
        
        self._print_review_summary(approved_count, rejected_count, undecided_count)

    def run_generate_from_jd(self, company: str, title: str, jd_text: str, cv_only: bool = False):
        """Generate a tailored CV + cover letter from a raw job description."""
        cv_text = self._load_cv()
        job = JobListing(
            title=title, company=company, url="", location="", salary="",
            date_found=date.today(), job_description=jd_text,
        )

        appended = self.sheets.append_jobs([job])
        if appended:
            job = appended[0]

        console.print(f"  [cyan]→[/cyan] {title} @ {company} — generating documents…")
        try:
            cv_path, letter_path = self.document.generate(job, cv_text, cv_only=cv_only)
            if job.sheet_row:
                self.sheets.update_job_links(job, cv_path, letter_path)
                self.sheets.update_job_status(job, ApplicationStatus.PENDING_REVIEW)
            console.print(f"    [green]✓[/green] CV: {cv_path}")
            if not cv_only:
                console.print(f"    [green]✓[/green] Cover letter: {letter_path}")
        except Exception as e:
            if job.sheet_row:
                self.sheets.update_job_status(job, ApplicationStatus.ERROR, Notes=str(e))
            console.print(f"    [red]✗ Doc generation failed: {e}[/red]")

    def run_full(
        self,
        title: str | None = None,
        location: str | None = None,
        salary_min: int | None = None,
        max_jobs: int | None = None,
        company_include: list[str] | None = None,
        company_exclude: list[str] | None = None,
        exclude_keywords: list[str] | None = None,
    ):
        self.run_search(
            title=title,
            location=location,
            salary_min=salary_min,
            max_jobs=max_jobs,
            company_include=company_include,
            company_exclude=company_exclude,
            exclude_keywords=exclude_keywords,
        )
        console.print("\n[italic]Waiting for you to review and approve jobs in Google Sheets...[/italic]")
        console.print("Run [bold cyan]python main.py apply[/bold cyan] when ready.")

    def run_regenerate(self, company: str, title_filter: str | None = None, cv_only: bool = False, letter_only: bool = False):
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
        asyncio.run(self._regenerate_async(job, cv_text, cv_only=cv_only, letter_only=letter_only))

    async def _regenerate_async(self, job: JobListing, cv_text: str, cv_only: bool = False, letter_only: bool = False):
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

        if cv_only:
            console.print("  Regenerating CV…")
        elif letter_only:
            console.print("  Regenerating cover letter…")
        else:
            console.print("  Regenerating CV and cover letter…")
        try:
            cv_path, letter_path = self.document.generate(job, cv_text, cv_only=cv_only, letter_only=letter_only)
            if letter_only:
                self.sheets.update_job_links(job, job.cv_link, letter_path)
            else:
                self.sheets.update_job_links(job, cv_path, letter_path)
            if cv_path:
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
