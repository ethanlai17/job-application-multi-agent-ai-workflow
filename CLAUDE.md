# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source activate                  # Activate .venv

python main.py search            # Search LinkedIn → Telegram review → generate CVs/letters for approved
python main.py process           # Process manually-added URLs → Telegram review → generate if approved
python main.py apply             # Auto-fill forms for Approved jobs
python main.py full              # search → Telegram review → generate → pause for final review
python main.py regenerate --company "Company"  # Re-scrape & regenerate docs
python main.py status            # Print tracking table
python main.py keywords          # View keyword library stats

python scripts/populate_keyword_library.py  # Rebuild keyword library from all JDs
```

All search/full commands accept optional flags: `--title`, `--location`, `--salary-min`, `--max-jobs`.

## Architecture

This is a Python multi-agent CLI that automates PM job applications end-to-end: LinkedIn search → **Telegram review gate** → tailored CV/cover letter generation → Google Sheets tracking → form auto-fill.

**Telegram Review Gate:** Between discovery and generation, each job is sent to Telegram as a message with job summary and Approve/Reject buttons. The operator decides per job within a configurable timeout (default 1 hour). Approved jobs proceed to document generation; rejected jobs are not logged; undecided jobs are marked "Undecided" and re-presented in future runs. Uses long-polling (no webhook required).

### Agent Roles

**`agents/orchestrator.py`** — The only entry point from `main.py`. Coordinates all agents sequentially per workflow (search → **Telegram review** → log approved → generate → apply). Handles status updates and error recovery. Implements the review gate that sends jobs to Telegram and branches based on decisions.

**`agents/job_search_agent.py`** — Deterministic (no LLM). Uses Playwright to log into LinkedIn, scrape job cards, and fetch full job description pages. The search URL (built by `build_search_url` in `tools/playwright_tools.py`) restricts results to jobs posted in the last 24 hours (`f_TPR=r86400`), sorted newest-first (`sortBy=DD`). Title filtering is driven by the `SEARCH_JOB_TITLE` env var via `title_matches` in `tools/title_match.py`: a job qualifies if all words of any comma-separated configured term appear in its title (order-independent, punctuation-tolerant token-subset match). Also filters by salary minimum and skips already-tracked jobs.

**`agents/document_agent.py`** — LLM-powered. Generates tailored single-page A4 CV and cover letter PDFs per job. Uses DeepSeek Pro model. Runs keyword matching (text-based, no LLM) against `data/keyword_library.json` to pick ATS keywords to inject.

**`agents/application_agent.py`** — LLM agentic loop (up to 30 steps). LLM reads page HTML, decides the next action (fill, click, navigate, upload, ask_user), Playwright executes it. Uses DeepSeek Flash model for cost efficiency.

**`agents/review_agent.py`** — Telegram review coordinator. Sends one message per job with inline buttons, polls for decisions using long-polling, handles timeout resolution. Uses DeepSeek for job summary generation with fallback to truncated job description.

**`tools/telegram_tools.py`** — Thin Telegram Bot API wrapper. HTTP calls for sendMessage (with inline keyboard), getUpdates (long-polling), and answerCallbackQuery. No business logic.

**`agents/sheets_agent.py`** — Thin wrapper over gspread. Appends job rows, updates status/links, queries Approved jobs, retrieves/deletes Undecided jobs for re-review.

### Data Flow

```
main.py → Orchestrator
  JobSearchAgent  → list[JobListing] (brand-new jobs, sheet_row=0)
  SheetsAgent     → get undecided jobs from previous runs (sheet_row>0) 
  
  merge + dedup → Review_Batch
  
  ReviewAgent     → send each job to Telegram, poll for decisions
  Telegram        → operator presses Approve/Reject buttons
  ReviewAgent     → return decisions map
  
  Branch per (origin, decision):
    new+approved     → append to sheet + generate docs
    new+rejected     → console only, no sheet write  
    new+undecided    → append with Status="Undecided"
    old+approved     → update Status="Found" + generate docs
    old+rejected     → delete sheet row
    old+undecided    → no-op (stays "Undecided")

  DocumentAgent   → (cv_path, cover_letter_path) for approved jobs
  SheetsAgent     → update CV/letter links, set Status = "Pending Review"

  (user sets Status = "Approved" in Sheets)

  SheetsAgent     → fetch Approved jobs
  ApplicationAgent → submit form per job
  SheetsAgent     → set Status = "Applied" or "Error"
```

### Key Configuration

**`.env`** — All secrets and defaults. Required keys: `DEEPSEEK_API_KEY`, `GOOGLE_CREDENTIALS_PATH`, `GOOGLE_SHEET_URL`, `LINKEDIN_EMAIL`, `LINKEDIN_PASSWORD`, `CV_PATH`, **`TELEGRAM_BOT_TOKEN`**, **`TELEGRAM_CHAT_ID`**. Optional: **`REVIEW_TIMEOUT_SECONDS`** (default 3600). LLM model selection: `DEEPSEEK_MODEL` (flash, for form loop) and `DEEPSEEK_DOCUMENT_MODEL` (pro, for CV generation). `SEARCH_JOB_TITLE` drives flexible title matching — comma-separated terms (e.g. `platform product manager, technical product manager`); a job qualifies if all words of any term appear in its title (order-independent, punctuation-tolerant), e.g. `platform product manager` matches "Product Manager - Platform".

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

`models/job.py` defines `JobListing` (dataclass) and `ApplicationStatus` (enum: `FOUND → CV_GENERATED → PENDING_REVIEW → APPROVED → APPLYING → APPLIED`, plus `REJECTED`/`ERROR`/**`UNDECIDED`**). Undecided jobs are re-surfaced in future search runs for another review chance.
