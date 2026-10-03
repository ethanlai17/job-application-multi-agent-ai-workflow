# Requirements Document

## Introduction

The job-application system currently discovers jobs and immediately logs every one of them to Google Sheets and generates a CV and cover letter for each. This feature inserts a human review gate between discovery and logging/generation. After jobs are discovered, each job is pushed to Telegram as its own message with Approve and Reject inline buttons. The operator decides per job. Jobs left undecided within a timeout are logged with Status `Undecided` and re-surfaced in the next search run so the operator can decide again later.

The gate applies to the `search`, `full` (which runs through `run_search`), and `process` flows. Undecided re-review (reading `Undecided` rows back from the sheet and merging them into the next review batch) applies only to `search`/`full`; `process` keeps its own gate for pasted URLs but does not pull in undecided rows. Interaction uses Telegram long-polling (`getUpdates`) with no webhook or public URL.

## Glossary

- **Review_Gate**: The component that pauses the pipeline between job discovery and logging/generation, presents each job to the operator via Telegram, and resolves each job to Approve, Reject, or Undecided.
- **Telegram_Client**: The component that sends messages with inline buttons to Telegram and receives button-press responses via long-polling (`getUpdates`).
- **Operator**: The human user who presses Approve or Reject buttons in Telegram.
- **Brand_New_Job**: A discovered job whose LinkedIn Job ID is not already present as a row in the Google Sheet.
- **Undecided_Job**: A job previously logged to the Google Sheet with Status `Undecided` and re-surfaced for a later decision.
- **Review_Batch**: The combined set of jobs presented for review in a single run.
- **Sheets_Logger**: The component that appends, updates, and deletes rows in the Google Sheet.
- **Document_Generator**: The component that generates the CV and cover letter for an approved job.
- **Summary_Service**: The component that produces the job-description-and-requirements summary used in the Telegram message, using the existing DeepSeek client.
- **DeepSeek_Client**: The existing LLM client used by Summary_Service to generate summaries.
- **LinkedIn_Job_ID**: The identifier used to match a job across discovery and Google Sheet rows and to de-duplicate the Review_Batch.
- **Review_Timeout**: The configurable number of seconds the Review_Gate waits for an Operator decision before resolving a job as Undecided; configuration key `review_timeout_seconds`, default 3600.
- **ApplicationStatus**: The enum of job statuses stored in the sheet (`Found`, `CV Generated`, `Pending Review`, `Approved`, `Applying`, `Applied`, `Rejected`, `Error`, and the new `Undecided`).

## Requirements

### Requirement 1: Review gate placement

**User Story:** As the Operator, I want a review gate inserted between job discovery and logging/generation, so that no job is written to the sheet or has documents generated without my decision.

#### Acceptance Criteria

1. WHEN the `search` flow discovers jobs, THE Review_Gate SHALL present each discovered job for review before any job is logged to the Google Sheet or passed to the Document_Generator.
2. WHEN the `full` flow runs through `run_search`, THE Review_Gate SHALL present each discovered job for review before any job is logged to the Google Sheet or passed to the Document_Generator.
3. WHEN the `process` flow receives one or more pasted URLs, THE Review_Gate SHALL present each resulting job for review before any job is logged to the Google Sheet or passed to the Document_Generator.
4. WHILE the Review_Gate has not resolved a job, THE Review_Gate SHALL NOT log that job to the Google Sheet and SHALL NOT pass that job to the Document_Generator.

### Requirement 2: Per-job Telegram message content and buttons

**User Story:** As the Operator, I want each job presented as its own Telegram message with a summary and action buttons, so that I can review and decide on jobs one at a time.

#### Acceptance Criteria

1. WHEN the Review_Gate presents a job, THE Telegram_Client SHALL send one Telegram message per job.
2. WHEN the Telegram_Client sends a job message, THE Telegram_Client SHALL include the job title, the company name, and a summary of the job description and requirements in the message body.
3. WHEN the Telegram_Client sends a job message, THE Telegram_Client SHALL attach an Approve inline button and a Reject inline button to the message.
4. WHEN the Operator presses the Approve or Reject button for a job, THE Review_Gate SHALL resolve that job to the corresponding decision.

### Requirement 3: Brand-new job outcomes

**User Story:** As the Operator, I want my decision on a brand-new job to control whether it is logged and whether documents are generated, so that only wanted jobs enter the pipeline.

#### Acceptance Criteria

1. WHEN the Operator approves a Brand_New_Job, THE Sheets_Logger SHALL append a new row for that job with Status `Found`, and THE Document_Generator SHALL generate the CV and cover letter with status progressing from `CV Generated` to `Pending Review`.
2. WHEN the Operator rejects a Brand_New_Job, THE Review_Gate SHALL emit console output for that job and SHALL NOT append a row to the Google Sheet.
3. IF the Review_Timeout elapses before the Operator decides on a Brand_New_Job, THEN THE Sheets_Logger SHALL append a new row for that job with Status `Undecided`, and THE Document_Generator SHALL NOT generate documents for that job.

### Requirement 4: Undecided re-review for search and full

**User Story:** As the Operator, I want undecided jobs re-surfaced in the next search run alongside new jobs, so that I can revisit decisions I deferred earlier.

#### Acceptance Criteria

1. WHEN a `search` or `full` run starts, THE Review_Gate SHALL read all rows with Status `Undecided` from the Google Sheet and merge them with the newly found LinkedIn jobs into a single Review_Batch.
2. WHEN the Review_Gate builds the Review_Batch, THE Review_Gate SHALL de-duplicate jobs by LinkedIn_Job_ID so that a reappearing job is presented only once.
3. WHEN the Operator approves a re-surfaced Undecided_Job, THE Sheets_Logger SHALL update the existing row Status from `Undecided` to `Found` without appending a new row, and THE Document_Generator SHALL generate the CV and cover letter for that job.
4. WHEN the Operator rejects a re-surfaced Undecided_Job, THE Sheets_Logger SHALL delete that job's existing row from the Google Sheet.
5. IF the Review_Timeout elapses again before the Operator decides on a re-surfaced Undecided_Job, THEN THE Sheets_Logger SHALL leave that job's row at Status `Undecided` so that the job re-surfaces in the next run.

### Requirement 5: Scope boundary for process

**User Story:** As the Operator, I want the process flow to review only the URLs I paste, so that pasted-URL review is not polluted by previously deferred jobs.

#### Acceptance Criteria

1. WHEN the `process` flow runs, THE Review_Gate SHALL present only the jobs derived from the pasted URLs for review.
2. WHILE the `process` flow is running, THE Review_Gate SHALL NOT read `Undecided` rows from the Google Sheet into the Review_Batch.

### Requirement 6: Long-polling and timeout behavior

**User Story:** As the Operator, I want the system to receive my button presses without exposing a public endpoint, so that no webhook or public URL is required.

#### Acceptance Criteria

1. THE Telegram_Client SHALL receive button-press responses using Telegram long-polling via `getUpdates`.
2. THE Telegram_Client SHALL operate without a webhook and without a public URL.
3. WHILE waiting for a decision on a job, THE Review_Gate SHALL wait up to Review_Timeout seconds for the Operator decision before resolving the job as Undecided.

### Requirement 7: Configuration and environment wiring

**User Story:** As the Operator, I want the Telegram credentials and timeout read from configuration, so that the review gate uses my existing environment without code changes.

#### Acceptance Criteria

1. THE configuration SHALL read `TELEGRAM_BOT_TOKEN` from the environment and make it available to the Telegram_Client.
2. THE configuration SHALL read `TELEGRAM_CHAT_ID` from the environment and make it available to the Telegram_Client.
3. THE configuration SHALL provide a `review_timeout_seconds` setting with a default value of 3600.

### Requirement 8: Summary generation with fallback

**User Story:** As the Operator, I want a readable job summary in every message even when summary generation fails, so that a review message always sends.

#### Acceptance Criteria

1. WHEN the Review_Gate prepares a job message, THE Summary_Service SHALL generate the job-description-and-requirements summary using the DeepSeek_Client.
2. IF the DeepSeek_Client fails to produce a summary, THEN THE Summary_Service SHALL provide a truncated excerpt of the job description so that the Telegram message still sends.

### Requirement 9: Safe row deletion

**User Story:** As the Operator, I want rejected undecided rows deleted reliably, so that the sheet stays accurate and gspread index shifts do not delete the wrong rows.

#### Acceptance Criteria

1. WHEN the Sheets_Logger deletes a row for a rejected Undecided_Job, THE Sheets_Logger SHALL match the row by LinkedIn_Job_ID.
2. WHEN the Sheets_Logger deletes a row, THE Sheets_Logger SHALL re-resolve the row number from the Google Sheet immediately before deleting that row.
3. WHEN the Sheets_Logger deletes multiple rows in one run, THE Sheets_Logger SHALL delete rows in bottom-most-first order so that row index shifts do not affect remaining deletions.

### Requirement 10: New application status value

**User Story:** As the Operator, I want an Undecided status recorded in the sheet, so that deferred jobs are distinguishable and can be re-surfaced.

#### Acceptance Criteria

1. THE ApplicationStatus enum SHALL include a value `UNDECIDED` with the sheet value `Undecided` alongside the existing values `Found`, `CV Generated`, `Pending Review`, `Approved`, `Applying`, `Applied`, `Rejected`, and `Error`.

### Requirement 11: HTTP library dependency

**User Story:** As a maintainer, I want the HTTP library required for Telegram calls declared as a dependency, so that the review gate installs and runs reliably.

#### Acceptance Criteria

1. THE project dependency list in `requirements.txt` SHALL include an HTTP library (`requests` or `httpx`) used by the Telegram_Client.
