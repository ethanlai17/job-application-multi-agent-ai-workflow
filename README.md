# Job Application AI Multi-Agent Workflow

A multi-agent Python system that automates LinkedIn job search, generates tailored CVs and cover letters, tracks applications in Google Sheets, and auto-fills application forms.

> **Note:** Beyond automation, this workflow continuously builds a keyword matching library from every job description it processes. The library tracks which skills and tools appear most frequently across the market, and uses that signal to ensure every generated CV and cover letter is ATS-friendly and precisely tailored to the requirements of each specific role.

---

## Setup

Before running any command, always navigate to the project folder and activate the virtual environment first:

```bash
cd /path/to/job-application
source activate
```

Every terminal session requires these two steps before any `python main.py ...` command will work.

---

## Commands

### 1. Search & generate documents
Searches LinkedIn for jobs matching your default criteria in `.env`, presents each job via Telegram for approval/rejection, then generates a tailored CV and cover letter for approved jobs and logs them to Google Sheets.

**Telegram Review Process:** Each discovered job is sent as a separate message to your configured Telegram chat with Approve ✅ and Reject ❌ buttons. You have a configurable timeout (default 1 hour) to decide. Jobs left undecided become Status "Undecided" and are re-presented in future search runs alongside new jobs.

The search is restricted to jobs **posted in the last 24 hours** (URL parameter `f_TPR=r86400`), sorted newest-first (`sortBy=DD`). Title matching is driven by the `SEARCH_JOB_TITLE` env var, which accepts comma-separated terms (e.g. `platform product manager, technical product manager`). Matching is order-independent and punctuation-tolerant: a job qualifies if all words of any configured term appear in its title (after normalizing case and separators like `- / , ( ) |`). For example, `SEARCH_JOB_TITLE` set to `platform product manager` matches a job titled "Product Manager - Platform".

**Decision Outcomes:**
- **Approved:** Job is logged to Google Sheets with Status "Found", then CV and cover letter are generated (Status → "CV Generated" → "Pending Review")
- **Rejected:** Console output only, job is not logged to the sheet
- **Undecided (timeout):** Job is logged with Status "Undecided" and will be re-presented in the next search run

```bash
python main.py search
```

With overrides:
```bash
python main.py search --title "Your Job Title" --location "Your Location" --salary-min <your_salary> --max-jobs <number>
```

---

### 2. Process a manually added LinkedIn URL
Paste a LinkedIn job URL into the URL column of the Google Sheet (leave Status blank), then run:

```bash
python main.py process
```

**Telegram Review Process:** Each successfully scraped job is sent to Telegram for approval/rejection (same as search command). The process command only reviews the pasted URLs - it never pulls in previously undecided jobs.

**Decision Outcomes:**
- **Approved:** Manual row is filled with job details, then CV and cover letter are generated
- **Rejected:** Console output only, manual row is left untouched (you can delete it manually)  
- **Undecided (timeout):** Row status is set to "Undecided", no document generation

Supports multiple URLs in one run. Jobs that fail to scrape (recruiter postings, unavailable jobs, errors) are marked with Status "Error" before the review step.

---

### 3. Apply to approved jobs
After reviewing documents in Google Sheets and setting Status → **Approved**, auto-fill and submit the applications:

```bash
python main.py apply
```

---

### 4. Full pipeline (search → generate → wait → apply)
Runs the search command (with Telegram review process), then pauses for your review before you manually trigger apply:

```bash
python main.py full
```

The search portion includes the same Telegram review gate as the standalone search command. Approved jobs will have documents generated and be ready for final review in the Google Sheet.

With overrides:
```bash
python main.py full --title "Your Job Title" --location "Your Location" --salary-min <your_salary> --max-jobs <number>
```

---

### 5. Regenerate CV + cover letter for an existing job
Re-scrapes the LinkedIn URL stored in the sheet and overwrites the existing PDFs. Useful after CV template improvements.

```bash
python main.py regenerate --company "Company Name"
```

With a title filter (when multiple rows match the same company):
```bash
python main.py regenerate --company "Company Name" --title "Job Title"
```

---

### 6. View application status
Prints a table of all tracked jobs and their current status.

```bash
python main.py status
```

---

### 7. View keyword library
Shows all keywords tracked across scraped job descriptions — popular (appear in 2+ JDs) vs. niche.

```bash
python main.py keywords
```

---

### 8. Rebuild keyword library from scratch
Re-scrapes all LinkedIn URLs in the sheet and rebuilds the keyword library from the actual JD text. No LLM calls — text-matching only.

```bash
python scripts/populate_keyword_library.py
```

---

### 9. Generate CV + cover letter from a raw job description
Generate documents for a job you found outside LinkedIn — no URL or Sheets row needed.
Pass the JD as a path to a `.txt` file or inline as a string. Logs the job to Google Sheets with Status = **Pending Review**.

```bash
# From a file
python main.py generate \
  --company "loveholidays" \
  --title "Head of Product, Supply Platform" \
  --jd-file /path/to/jd.txt

# Inline text
python main.py generate \
  --company "loveholidays" \
  --title "Head of Product, Supply Platform" \
  --jd "We're looking for a leader who treats suppliers..."

# CV only (skip cover letter)
python main.py generate --company "X" --title "Y" --jd-file /path/to/jd.txt --cv-only
```

---

## Typical workflow

```bash
# Step 1 — search and review via Telegram
python main.py search
# → Each job appears in Telegram with Approve/Reject buttons
# → Decide within the timeout (default 1 hour)
# → Approved jobs get documents generated automatically

# Step 2 — open Google Sheets, review generated CVs and cover letters, set Status → Approved

# Step 3 — apply
python main.py apply
```

Or for a single job found manually:

```bash
# Paste the LinkedIn URL into the sheet, then:
python main.py process
# → Job appears in Telegram for approval
# → If approved, documents are generated

# Review, approve in sheet, then:
python main.py apply
```

## Telegram Review Configuration

The system requires three environment variables for the Telegram review gate:

- `TELEGRAM_BOT_TOKEN` — Your bot token from @BotFather
- `TELEGRAM_CHAT_ID` — Your chat ID (can be personal chat or group)
- `REVIEW_TIMEOUT_SECONDS` — Optional, defaults to 3600 (1 hour)

**How it works:** Each job gets its own message with a summary (title, company, AI-generated description highlights) and inline Approve/Reject buttons. The system waits for your decision using long-polling (no webhook needed). Jobs left undecided after the timeout are marked "Undecided" and re-surface in future search runs.

**Status meanings:**
- **Found** — Recently approved, ready for document review
- **Undecided** — Timed out in review, will be re-presented next run
- **CV Generated** → **Pending Review** — Documents generated, awaiting your final approval
- **Approved** — Ready for application submission
- **Applied** — Successfully submitted
