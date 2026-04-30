"""
E2E test: apply to a specific LinkedIn job using an existing CV + cover letter.
Usage: python test_apply.py
"""
from datetime import date
from config.settings import Config
from agents.application_agent import ApplicationAgent
from models.job import JobListing
from openai import OpenAI
from rich import print as rprint

config = Config.load()
client = OpenAI(api_key=config.deepseek_api_key, base_url="https://api.deepseek.com")

CV_PATH = "output/Hopper_Senior Product Manager _ Flight Connectivity/Hopper_Senior Product Manager _ Flight Connectivity_CV_Ethan Lai.pdf"
LETTER_PATH = "output/Hopper_Senior Product Manager _ Flight Connectivity/Hopper_Senior Product Manager _ Flight Connectivity_Cover letter_Ethan Lai.pdf"

job = JobListing(
    title="Senior Product Manager – Flight Connectivity",
    company="Hopper",
    url="https://www.linkedin.com/jobs/view/4402747814/",
    location="London, UK",
    salary="",
    date_found=date.today(),
    cv_link=CV_PATH,
    cover_letter_link=LETTER_PATH,
)

rprint(f"[bold blue]E2E Test: Applying to {job.title} @ {job.company}[/bold blue]")
rprint(f"  URL: {job.url}")
rprint(f"  CV: {job.cv_link}")
rprint(f"  Cover letter: {job.cover_letter_link}")
rprint("")

agent = ApplicationAgent(config, client)
success = agent.apply(job)

if success:
    rprint("\n[bold green]✓ Application submitted successfully![/bold green]")
else:
    rprint("\n[bold red]✗ Application failed — see agent output above for reason.[/bold red]")
