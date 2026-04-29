import click
from rich.console import Console

from agents.orchestrator import Orchestrator
from config.settings import Config

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
