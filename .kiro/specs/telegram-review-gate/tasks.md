# Implementation Plan: Telegram Review Gate

## Overview

Insert a human review gate between job discovery and logging/generation. The plan is a strict 7-task sequence; each task builds on the previous one and leaves the codebase in a working, testable state. Tasks 1–5 add foundational pieces (status/config/dependency, Telegram wrapper, summary service, review coordinator, sheets ops) that are independently testable. Task 6 wires the gate into `run_search`/`run_full` and depends on tasks 1–5. Task 7 wires the gate into `run_process` and updates docs, and depends on task 6.

The design uses Python 3.11, matching the existing codebase. Property-based tests use the already-present `hypothesis` + `pytest` toolchain, run ≥100 iterations, and are tagged `Feature: telegram-review-gate, Property {n}: {text}`. Every test in a task runs against only what has been built up to and including that task.

## Tasks

- [ ] 1. Add Undecided status, Telegram/timeout config, and HTTP dependency
  - In `models/job.py`, add `UNDECIDED = "Undecided"` to the `ApplicationStatus(str, Enum)` alongside the existing values.
  - In `config/settings.py`, add `telegram_bot_token: str`, `telegram_chat_id: str`, and `review_timeout_seconds: int` to the `Config` dataclass; in `Config.load()` read `telegram_bot_token=_require("TELEGRAM_BOT_TOKEN")`, `telegram_chat_id=_require("TELEGRAM_CHAT_ID")`, and `review_timeout_seconds=int(os.getenv("REVIEW_TIMEOUT_SECONDS", "3600"))`.
  - In `requirements.txt`, add `requests>=2.31.0`.
  - _Requirements: 7.1, 7.2, 7.3, 10.1, 11.1_

  - [~] 1.1 Write unit/example tests for status, config, and dependency
    - Example: `ApplicationStatus("Undecided") is ApplicationStatus.UNDECIDED` round-trip. _(Validates: Requirement 10.1)_
    - Example: `Config.load()` reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from env and defaults `review_timeout_seconds` to 3600 when `REVIEW_TIMEOUT_SECONDS` is unset. _(Validates: Requirements 7.1, 7.2, 7.3)_
    - Smoke: `requests` appears in `requirements.txt`. _(Validates: Requirement 11.1)_
    - _Requirements: 7.1, 7.2, 7.3, 10.1, 11.1_

- [ ] 2. Create thin Telegram Bot API wrapper
  - Create `tools/telegram_tools.py` with `BASE_URL = "https://api.telegram.org/bot{token}/{method}"` and no gate/business logic.
  - Implement `send_review_message(token, chat_id, text, job_key) -> int`: POST `sendMessage` with two inline-keyboard buttons (`✅ Approve` → `callback_data="a|<job_key>"`, `❌ Reject` → `callback_data="r|<job_key>"`); return the Telegram `message_id`.
  - Implement `poll_updates(token, offset=None, timeout=25) -> list[dict]`: long-poll `getUpdates`, return the raw `result` list; the caller advances `offset = max(update_id) + 1`.
  - Implement `answer_callback(token, callback_query_id, text="") -> None`: POST `answerCallbackQuery`, best-effort (swallow failures).
  - Implement pure helpers `encode_callback(action, job_key) -> str` and `decode_callback(data) -> tuple[str, str]` defining the `"<action>|<job_key>"` wire format.
  - _Requirements: 2.1, 2.3, 6.1, 6.2_

  - [~] 2.1 Write property test for payload buttons + callback round-trip
    - **Property 5: Message payload carries both buttons and a round-tripping key**
    - Assert the `sendMessage` payload attaches exactly two inline buttons (Approve/Reject) and that `decode_callback(encode_callback(action, key))` recovers the original action and job key for arbitrary keys.
    - **Validates: Requirements 2.3, 2.4**
    - _Requirements: 2.3, 2.4_

  - [~] 2.2 Write property test for offset monotonic/exactly-once
    - **Property 7: getUpdates offset is monotonic and exactly-once**
    - Over synthetic update lists, assert the next poll offset equals `max(update_id) + 1` and no update is processed more than once across polls.
    - **Validates: Requirement 6.1**
    - _Requirements: 6.1_

  - [~] 2.3 Write unit test for mocked HTTP with no webhook
    - Mock the HTTP layer; assert calls target `getUpdates`/`sendMessage` on `api.telegram.org` and that no webhook or public URL is configured.
    - **Validates: Requirement 6.2**
    - _Requirements: 6.2_

- [ ] 3. Implement Summary_Service in review_agent
  - Create `agents/review_agent.py` and implement `make_summarize_fn(client, model, excerpt_chars=600)` returning a `summarize(job)` closure built on the existing DeepSeek `OpenAI` client and the lightweight `config.model` (`max_tokens=300`).
  - On success, return the LLM summary text.
  - On any exception or empty content, return a bounded truncated excerpt of `job.job_description` (`[:excerpt_chars]`), or `"(no description available)"` when empty — always non-empty so a message can send.
  - _Requirements: 8.1, 8.2_

  - [~] 3.1 Write property + happy-path tests for summarize_fn
    - **Property 9: Summary fallback always yields a non-empty bounded excerpt**
    - With a failing fake client, assert the result is non-empty, has length ≤ `excerpt_chars`, and is a prefix of the JD. _(Validates: Requirements 8.1, 8.2)_
    - Happy path: with a fake client returning text, assert the LLM text is returned. _(Validates: Requirement 8.1)_
    - **Validates: Requirements 8.1, 8.2**
    - _Requirements: 8.1, 8.2_

- [ ] 4. Implement the review coordinator in review_agent
  - In `agents/review_agent.py`, add `Decision(str, Enum)` with `APPROVED = "approved"`, `REJECTED = "rejected"`, `UNDECIDED = "undecided"`.
  - Implement `job_key(job) -> str`: use `job.linkedin_job_id` when present, else `"k" + sha1(f"{job.url}|{job.title}|{job.company}").hexdigest()[:12]`.
  - Implement pure `render_message(title, company, summary) -> str` that always includes title, company, and summary.
  - Implement `review_jobs(jobs, summarize_fn, telegram_client, timeout, now_fn=time.monotonic) -> dict[str, Decision]`: send exactly one message per distinct key via `send_review_message`, long-poll until all keys decided or `deadline = now_fn() + timeout`, resolve each tapped key to the Decision matching its action, `answer_callback` on each tap, advance `offset = update_id + 1` per processed update, and resolve remaining keys to `UNDECIDED`. On `send_review_message` failure for a job, resolve that job to `UNDECIDED`.
  - _Requirements: 2.1, 2.2, 2.4, 6.3_

  - [~] 4.1 Write property test for one message per job
    - **Property 3: One message per job**
    - For a batch of jobs with distinct keys, assert send count equals the number of distinct keys.
    - **Validates: Requirement 2.1**
    - _Requirements: 2.1_

  - [~] 4.2 Write property test for message body content
    - **Property 4: Message body contains title, company, and summary**
    - For any job and summary string, assert the rendered body contains the title, company, and summary.
    - **Validates: Requirement 2.2**
    - _Requirements: 2.2_

  - [~] 4.3 Write property test for tap resolution
    - **Property 6: Button taps resolve to the matching decision**
    - Assert tapped keys resolve to the matching Decision and untapped keys are left for timeout handling.
    - **Validates: Requirement 2.4**
    - _Requirements: 2.4_

  - [~] 4.4 Write property test for timeout resolution
    - **Property 8: Timeout resolves exactly the undecided keys**
    - Use the injected `now_fn` to cross the deadline deterministically; assert never-decided keys → `UNDECIDED` and decided keys are preserved.
    - **Validates: Requirement 6.3**
    - _Requirements: 6.3_

  - [~] 4.5 Write unit test for send-failure path
    - Assert a job whose `send_review_message` raises resolves to `UNDECIDED` (never silently approved/rejected).
    - _Requirements: 6.3_

- [ ] 5. Add sheets retrieval and safe deletion
  - In `tools/sheets_tools.py`, add `get_undecided_jobs(ws)` filtering `get_all_jobs(ws)` by `ApplicationStatus.UNDECIDED` (preserving `sheet_row`).
  - In `tools/sheets_tools.py`, add `delete_job_row(ws, linkedin_job_id)`: re-resolve the row by reading the `LinkedIn Job ID` column fresh immediately before deleting, then `ws.delete_rows(row_num)`; return `True`/`False`.
  - In `agents/orchestrator.py` (batch deletion ordering), resolve all rejected-undecided targets then delete bottom-most-first (descending resolved row number) so index shifts never affect remaining deletions.
  - In `agents/sheets_agent.py`, add thin wrappers `get_undecided_jobs(self)` and `delete_job_row(self, linkedin_job_id)` delegating to the tools.
  - _Requirements: 4.1, 9.1, 9.2, 9.3_

  - [~] 5.1 Write property test for safe bottom-first deletion
    - **Property 10: Safe bottom-first deletion removes exactly the targets**
    - Over a fake worksheet, assert deleting target IDs (each re-resolved by id, bottom-most-first) leaves exactly the non-target rows intact regardless of the targets' original row order.
    - **Validates: Requirements 9.1, 9.3**
    - _Requirements: 9.1, 9.3_

  - [~] 5.2 Write example test for re-resolution after row shift
    - Assert `delete_job_row` re-reads the ID column immediately before deleting so a prior deletion's index shift does not delete the wrong row.
    - **Validates: Requirement 9.2**
    - _Requirements: 9.2_

  - [x] 5.3 Write unit test for get_undecided_jobs filter
    - Assert only `UNDECIDED` rows are returned and `sheet_row` is populated.
    - **Validates: Requirement 4.1**
    - _Requirements: 4.1_

- [ ] 6. Wire the review gate into run_search / run_full
  - In `agents/orchestrator.py` `__init__`, instantiate the Telegram client (the `telegram_tools` functions bound to `config.telegram_bot_token`/`config.telegram_chat_id`) and build `self.summarize_fn = make_summarize_fn(self.client, config.model)`.
  - In `run_search`: load `self.sheets.get_undecided_jobs()` (origin = re-surfaced, `sheet_row > 0`), run `self.job_search.run(...)` (origin = brand-new, `sheet_row == 0`), merge + dedup by LinkedIn Job ID into one batch (origin derived from `job.sheet_row`), preferring the undecided entry on collision; return early with a "nothing to review" message if the batch is empty.
  - Call `review_jobs(batch, self.summarize_fn, self.telegram, self.config.review_timeout_seconds)` and branch per `(origin, decision)` across all six branches: new+approve → `append_jobs([job])` (Found) + generate (`CV_GENERATED`→`PENDING_REVIEW`, `ERROR` on failure); new+undecided → `append_jobs([job])` then `update_job_status(UNDECIDED)`, no generation; new+reject → console only, no sheet write; resurfaced+approve → `update_job_status(FOUND)` + generate; resurfaced+reject → collect for bottom-first `delete_job_row(job.linkedin_job_id)`; resurfaced+undecided → no-op.
  - Update the end-of-run summary with approved/rejected/undecided counts.
  - Confirm `run_full` is unchanged and inherits the gate via `run_search`.
  - _Requirements: 1.1, 1.2, 1.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5_

  - [~] 6.1 Write property test for merge + dedup
    - **Property 1: Batch merge is the deduplicated union keyed by LinkedIn Job ID**
    - Assert the merged batch is the deduplicated union; every non-empty id appears at most once; on collision the re-surfaced (sheet-backed) entry is retained.
    - **Validates: Requirements 4.1, 4.2**
    - _Requirements: 4.1, 4.2_

  - [~] 6.2 Write property test for branch dispatch
    - **Property 2: Branch dispatch acts only according to (origin, decision)**
    - With fake sheets + fake Telegram client, assert each (origin, decision) performs exactly its mandated actions and no others (rejected new → zero sheet writes; new timeout → append Undecided, no generation; resurfaced timeout → zero mutations; no pending job logged/generated).
    - **Validates: Requirements 1.4, 3.2, 3.3, 4.5**
    - _Requirements: 1.4, 3.2, 3.3, 4.5_

  - [~] 6.3 Write integration test for the decision matrix
    - new/resurfaced × approve/reject/timeout matrix with fakes, asserting per-branch sheet and generation actions.
    - **Validates: Requirements 1.1, 1.4, 3.1, 3.2, 3.3, 4.3, 4.4, 4.5**
    - _Requirements: 1.1, 1.4, 3.1, 3.2, 3.3, 4.3, 4.4, 4.5_

- [ ] 7. Wire the review gate into run_process and update docs
  - In `agents/orchestrator.py` `run_process` / `_process_manual_async`: preserve the existing per-URL scraping and recruiter/unavailable/scrape-error handling that marks rows `Error` **before** review; collect successfully scraped `JobListing` objects into `scraped` (no inline append/generate). Never read undecided rows.
  - Call `review_jobs(scraped, self.summarize_fn, self.telegram, self.config.review_timeout_seconds)` and branch using **brand-new rules only**: approved → `fill_manual_row` + `update_job_status(CV_GENERATED)` + generate → `PENDING_REVIEW`; rejected → console only, leave the manual row untouched; undecided (timeout) → `update_job_status(UNDECIDED)`, no generation.
  - Confirm `run_full` inherits the gate via `run_search`.
  - Update `README.md` and `CLAUDE.md` to document the review step, undecided re-review, the new env vars (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `REVIEW_TIMEOUT_SECONDS`), the `Undecided` status, and approve/reject/timeout semantics.
  - _Requirements: 1.3, 5.1, 5.2_

  - [~] 7.1 Write integration tests for process scope boundary and branches
    - Assert `run_process` reviews only scraped jobs and never calls `get_undecided_jobs`. _(Validates: Requirements 5.1, 5.2)_
    - Assert scrape errors mark `Error` before review and are excluded from the batch.
    - Assert approve/reject/timeout branches behave per brand-new rules. _(Validates: Requirements 1.3, 3.1, 3.2, 3.3)_
    - Smoke: `run_full` invokes `run_search` and inherits the gate. _(Validates: Requirement 1.2)_
    - **Validates: Requirements 1.2, 1.3, 3.1, 3.2, 3.3, 5.1, 5.2**
    - _Requirements: 1.2, 1.3, 3.1, 3.2, 3.3, 5.1, 5.2_

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation sub-tasks are never optional.
- Each task references specific requirements for traceability and builds on prior tasks so the codebase stays working and testable at every step. Each task's tests run against only what has been built up to that point.
- Task 6 depends on tasks 1–5 (status, config, Telegram wrapper, summary service, review coordinator, sheets ops). Task 7 depends on task 6 (shared orchestrator wiring and the `review_jobs` branch pattern).
- Property tests validate the universal invariants in the design's Correctness Properties; unit/example tests cover config, wiring, and edge cases.
- Run the test suite after each task to validate incrementally; raise any questions before moving on.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "1.1", "2", "2.1", "2.2", "2.3", "3", "3.1", "5", "5.1", "5.2", "5.3"] },
    { "id": 1, "tasks": ["4", "4.1", "4.2", "4.3", "4.4", "4.5"] },
    { "id": 2, "tasks": ["6", "6.1", "6.2", "6.3"] },
    { "id": 3, "tasks": ["7", "7.1"] }
  ]
}
```
