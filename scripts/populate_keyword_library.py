"""
One-off script: scrape all real LinkedIn JDs from the Google Sheet and
populate the keyword library from scratch.

Uses text-matching only — no LLM calls, no token cost.

Run from the project root:
    python scripts/populate_keyword_library.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from browser.session import BrowserSession
from config.settings import Config
from tools import keyword_tools, playwright_tools
from tools.keyword_tools import get_library_summary, save_library

console = Console()

# Placeholder rows added during testing — skip them
_SKIP_IDS = {"1234567890", "2345678901", "3456789012"}


async def scrape_and_index(config: Config) -> None:
    from agents.sheets_agent import SheetsAgent

    sheets = SheetsAgent(config)
    all_jobs = sheets.get_all_jobs()
    real_jobs = [j for j in all_jobs if j.linkedin_job_id not in _SKIP_IDS]
    console.print(f"[bold]Found {len(real_jobs)} real jobs to process.[/bold]")

    lib = keyword_tools._empty_library()

    async with BrowserSession(config) as session:
        for i, job in enumerate(real_jobs, 1):
            console.print(f"\n[cyan]{i}/{len(real_jobs)}[/cyan] {job.title} @ {job.company}")
            try:
                details = await playwright_tools.get_job_details(session, job.url)
                jd = details.get("job_description", "").strip()
                if not jd:
                    console.print("  [yellow]⚠ Empty JD — skipping[/yellow]")
                    continue

                console.print(f"  JD: {len(jd)} chars — matching keywords…")
                lib = keyword_tools.update_library_from_jd(
                    jd_text=jd,
                    lib=lib,
                    save=False,
                    discover=False,  # text-matching only, no LLM tokens
                )
                console.print(f"  [green]✓[/green] {len(lib['keywords'])} unique keywords so far")

                time.sleep(config.request_delay_min)

            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")

    save_library(lib)
    console.print("\n[bold green]Library saved.[/bold green]")
    console.print(get_library_summary())


if __name__ == "__main__":
    cfg = Config.load()
    asyncio.run(scrape_and_index(cfg))
