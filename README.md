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
Searches LinkedIn for jobs matching your default criteria in `.env`, generates a tailored CV and cover letter for each, and logs them to Google Sheets.

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

Scrapes the job, fills in the row, and generates the CV + cover letter. Supports multiple URLs in one run.

---

### 3. Apply to approved jobs
After reviewing documents in Google Sheets and setting Status → **Approved**, auto-fill and submit the applications:

```bash
python main.py apply
```

---

### 4. Full pipeline (search → generate → wait → apply)
Runs search and document generation, then pauses for your review before you manually trigger apply:

```bash
python main.py full
```

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

## Typical workflow

```bash
# Step 1 — search and generate
python main.py search

# Step 2 — open Google Sheets, review CVs and cover letters, set Status → Approved

# Step 3 — apply
python main.py apply
```

Or for a single job found manually:

```bash
# Paste the LinkedIn URL into the sheet, then:
python main.py process

# Review, approve in sheet, then:
python main.py apply
```
