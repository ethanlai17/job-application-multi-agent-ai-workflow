# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source activate                  # Activate .venv

python main.py search            # Search LinkedIn + generate CVs/letters
python main.py process           # Process manually-added URLs in Sheets
python main.py apply             # Auto-fill forms for Approved jobs
python main.py full              # search + generate + pause for review
python main.py regenerate --company "Company"  # Re-scrape & regenerate docs
python main.py status            # Print tracking table
python main.py keywords          # View keyword library stats

python scripts/populate_keyword_library.py  # Rebuild keyword library from all JDs
```

All search/full commands accept optional flags: `--title`, `--location`, `--salary-min`, `--max-jobs`.

## Architecture

This is a Python multi-agent CLI that automates PM job applications end-to-end: LinkedIn search → tailored CV/cover letter generation → Google Sheets tracking → form auto-fill.

### Agent Roles

**`agents/orchestrator.py`** — The only entry point from `main.py`. Coordinates all agents sequentially per workflow (search → log → generate → apply). Handles status updates and error recovery.

**`agents/job_search_agent.py`** — Deterministic (no LLM). Uses Playwright to log into LinkedIn, scrape job cards, and fetch full job description pages. Filters by title keywords, salary minimum, and skips already-tracked jobs.

**`agents/document_agent.py`** — LLM-powered. Generates tailored single-page A4 CV and cover letter PDFs per job. Uses DeepSeek Pro model. Runs keyword matching (text-based, no LLM) against `data/keyword_library.json` to pick ATS keywords to inject.

**`agents/application_agent.py`** — LLM agentic loop (up to 30 steps). LLM reads page HTML, decides the next action (fill, click, navigate, upload, ask_user), Playwright executes it. Uses DeepSeek Flash model for cost efficiency.

**`agents/sheets_agent.py`** — Thin wrapper over gspread. Appends job rows, updates status/links, and queries Approved jobs.

### Data Flow

```
main.py → Orchestrator
  JobSearchAgent  → list[JobListing]
  SheetsAgent     → append rows, return sheet_row per job
  DocumentAgent   → (cv_path, cover_letter_path) per job
  SheetsAgent     → update CV/letter links, set Status = "Pending Review"

  (user sets Status = "Approved" in Sheets)

  SheetsAgent     → fetch Approved jobs
  ApplicationAgent → submit form per job
  SheetsAgent     → set Status = "Applied" or "Error"
```

### Key Configuration

**`.env`** — All secrets and defaults. Required keys: `DEEPSEEK_API_KEY`, `GOOGLE_CREDENTIALS_PATH`, `GOOGLE_SHEET_URL`, `LINKEDIN_EMAIL`, `LINKEDIN_PASSWORD`, `CV_PATH`. LLM model selection: `DEEPSEEK_MODEL` (flash, for form loop) and `DEEPSEEK_DOCUMENT_MODEL` (pro, for CV generation).

**`config/cv_data.py`** — Applicant info (name, contact, work experience, projects, education). Seeded into all LLM prompts. `WORK_EXPERIENCE_FIXED` entries appear in CV in exact order/titles.

**`config/experience_library.py`** — Curated bullet bank grouped by capability. LLM draws from these when generating CV bullets.

**`data/keyword_library.json`** — Frequency map of keywords across all scraped JDs. Keywords appearing in 2+ JDs are marked "popular" and eligible for CV injection. Updated on every new job scrape.

### Document Generation Constraints

The CV renderer in `tools/document_tools.py` enforces strict ATS rules:
- Single-page A4 (binary-searches spacing scale to fit)
- No tables — uses tab stops and hanging indents
- Helvetica ≥10pt throughout
- Bullets per company capped at 8, distributed to preserve ordering priority (TUI > Inspectorio > FPT)
- Metrics (`€`, `%`, numbers) are preserved verbatim through LLM round-trips

### Browser Automation

`browser/session.py` creates a stealth Playwright context (hides `navigator.webdriver`, sets real Chrome user-agent, locale `en-GB`, timezone `Europe/London`). LinkedIn session cookies are saved to `browser/linkedin_state.json` and reused across runs.

### LLM Calls

All LLM calls use the OpenAI-compatible DeepSeek API. `@retry` decorators (15 attempts, 2s delay) handle 503 errors. Document generation returns structured JSON; the orchestrator validates that all JD-matched keywords appear in the output before rendering.

### Models

`models/job.py` defines `JobListing` (dataclass) and `ApplicationStatus` (enum: `FOUND → CV_GENERATED → PENDING_REVIEW → APPROVED → APPLYING → APPLIED`, plus `REJECTED`/`ERROR`).
