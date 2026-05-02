import click
from rich.console import Console

from agents.orchestrator import Orchestrator
from config.settings import Config
from tools.keyword_tools import get_library_summary

console = Console()


@click.group()
def cli():
    """Job Application AI Workflow — powered by Claude + Playwright + Google Sheets."""
    pass


@cli.command()
@click.option("--title", default=None, help="Job title to search (overrides .env default)")
@click.option("--location", default=None, help="Location (overrides .env default)")
@click.option("--salary-min", type=int, default=None, help="Minimum salary in £ (overrides .env default)")
@click.option("--max-jobs", type=int, default=None, help="Max number of jobs to find per run")
def search(title, location, salary_min, max_jobs):
    """Search LinkedIn for jobs and generate tailored CV + cover letter drafts."""
    try:
        config = Config.load()
        orchestrator = Orchestrator(config)
        orchestrator.run_search(
            title=title,
            location=location,
            salary_min=salary_min,
            max_jobs=max_jobs,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@cli.command()
def apply():
    """Apply to all jobs you've marked as Approved in Google Sheets."""
    try:
        config = Config.load()
        orchestrator = Orchestrator(config)
        orchestrator.run_apply()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@cli.command()
@click.option("--title", default=None)
@click.option("--location", default=None)
@click.option("--salary-min", type=int, default=None)
@click.option("--max-jobs", type=int, default=None)
def full(title, location, salary_min, max_jobs):
    """Run search + document generation, then wait for you to approve before applying."""
    try:
        config = Config.load()
        orchestrator = Orchestrator(config)
        orchestrator.run_full(
            title=title,
            location=location,
            salary_min=salary_min,
            max_jobs=max_jobs,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@cli.command()
def process():
    """Process LinkedIn URLs you've manually added to the Google Sheet.

    Paste a LinkedIn job URL into the URL column and leave Status blank.
    This command scrapes the job, fills in the row, and generates tailored
    CV + cover letter — then waits for your review before applying.
    """
    try:
        config = Config.load()
        orchestrator = Orchestrator(config)
        orchestrator.run_process()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@cli.command()
@click.option("--company", required=True, help="Company name to match in the sheet (case-insensitive substring)")
@click.option("--title", default=None, help="Optional job title substring to disambiguate when multiple rows match")
@click.option("--cv-only", is_flag=True, default=False, help="Regenerate CV only, skip cover letter")
def regenerate(company, title, cv_only):
    """Re-scrape and regenerate CV + cover letter for a job already in the sheet.

    Looks up the job by company (and optionally title) in Google Sheets,
    re-scrapes the LinkedIn URL stored there, and overwrites the existing PDFs.
    """
    try:
        config = Config.load()
        orchestrator = Orchestrator(config)
        orchestrator.run_regenerate(company=company, title_filter=title, cv_only=cv_only)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@cli.command()
def keywords():
    """Show the PM keyword library — popular vs. niche keywords across all tracked JDs."""
    console.print(get_library_summary())


@cli.command()
def status():
    """Show current status of all tracked job applications."""
    try:
        config = Config.load()
        orchestrator = Orchestrator(config)
        orchestrator.run_status()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
