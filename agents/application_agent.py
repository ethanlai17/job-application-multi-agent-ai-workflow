import asyncio
import json
import re
from pathlib import Path

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from openai import OpenAI
from rich import print as rprint

from browser.session import BrowserSession
from config.settings import Config
from models.job import JobListing
from tools import playwright_tools
from tools.document_tools import read_cv

_ACTION_SCHEMA = """
Return ONLY a JSON object — no markdown, no explanation — choosing one action per turn:

{"action": "navigate",     "url": "<full URL>"}
{"action": "read_page"}
{"action": "fill_field",   "selector": "<CSS selector>", "value": "<text to enter>"}
{"action": "click",        "selector": "<CSS selector or aria-label pattern>"}
{"action": "upload_file",  "selector": "<CSS selector for file input>", "path": "<file path>"}
{"action": "ask_user",     "question": "<question text to show the applicant>", "options": ["<option 1>", "<option 2>"]}
{"action": "submit"}
{"action": "done",         "success": true,  "message": "<optional note>"}
{"action": "done",         "success": false, "message": "<reason for failure>"}

Use ask_user ONLY when you cannot determine the answer from the CV, cover letter, or the applicant's
personal details. Do NOT ask for things already provided (name, email, location, GitHub, years of experience).
"""


def _parse_action(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: extract the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _is_503_error(e: Exception) -> bool:
    return "503" in str(e) or "service_unavailable" in str(e).lower()


@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(15),
    retry=retry_if_exception(_is_503_error),
    before_sleep=lambda retry_state: print(f"  [yellow]API Error. Retrying in 2 seconds... (Attempt {retry_state.attempt_number}/15)[/yellow]"),
    reraise=True
)
def _send_message_with_retry(client: OpenAI, model: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


class ApplicationAgent:
    def __init__(self, config: Config, client: OpenAI):
        self.config = config
        self.client = client

    def apply(self, job: JobListing) -> bool:
        return asyncio.run(self._apply_async(job))

    async def _apply_async(self, job: JobListing) -> bool:
        async with BrowserSession(self.config) as session:
            self._session = session
            # Must login to LinkedIn before the agent can access the application form
            logged_in = await playwright_tools.linkedin_login(
                session, self.config.linkedin_email, self.config.linkedin_password
            )
            if not logged_in:
                rprint("  [red]✗ LinkedIn login failed — check credentials in .env[/red]")
                return False
            rprint("  [green]✓ Logged into LinkedIn[/green]")
            return await self._agent_loop(job)

    async def _agent_loop(self, job: JobListing) -> bool:
        cv_text = ""
        letter_text = ""
        if job.cv_link and Path(job.cv_link).exists():
            cv_text = read_cv(job.cv_link)
        if job.cover_letter_link and Path(job.cover_letter_link).exists():
            letter_text = read_cv(job.cover_letter_link)

        system = (
            "You are an automated job application agent. "
            "Your goal is to open a LinkedIn job posting, click the Apply button, "
            "and complete the external application form.\n\n"
            "Process:\n"
            "1. Open the job URL.\n"
            "2. Read the page. Look for [APPLY_LINK ...] or [BTN] Apply entries.\n"
            "   - For [APPLY_LINK <label>] items, click using selector: a[aria-label=\"<label>\"].\n"
            "     Example: [APPLY_LINK Apply on company website] → selector: "
            "a[aria-label=\"Apply on company website\"]\n"
            "   - If clicking returns ‘ok_new_tab’, the browser is now on the new tab — "
            "     call read_page immediately.\n"
            "3. On the external application form (e.g. Ashby/Greenhouse/Lever), read the page, "
            "   then fill every field:\n"
            "   - Name field: use selector #_systemfield_name or input[name=’_systemfield_name’]. "
            "     Always enter: Ethan Lai\n"
            "   - Email field: use selector #_systemfield_email or input[name=’_systemfield_email’].\n"
            "   - Resume/CV upload: use upload_file with selector #_systemfield_resume or "
            "     input[type=file] and the CV path below.\n"
            "   - Other text inputs: fill with relevant info from the CV.\n"
            "   - Radio button groups: for each question, click the most accurate radio option "
            "     using its id attribute as selector (e.g. #<id>).\n"
            "   - Location fields with autocomplete: type the value, wait 1s, then read_page "
            "     and click the first suggestion.\n"
            "4. Answer ALL visible questions before submitting. "
            "If a question cannot be answered from the CV, cover letter, or the personal details above, "
            "use ask_user to pause and get the applicant's answer. "
            "Do NOT submit until all required fields are answered.\n"
            "5. Submit the form. Read the page to confirm. Signal success on confirmation.\n"
            "6. If unrecoverable error, signal failure with a reason.\n\n"
            "APPLICANT PERSONAL DETAILS — always use these exact values:\n"
            "  Name:     Ethan Lai\n"
            "  Email:    ethanlaipm@gmail.com\n"
            "  Location: London\n"
            "  GitHub:   https://github.com/ethanlai17\n"
            "  Years of PM experience: 6 years "
            "(for 'Less than 4 / 4-6 / 6+' radio questions, select the 6+ option)\n"
            "CV path (for upload): " + (job.cv_link or "") + "\n"
            "Only use information from the CV and cover letter below — never invent details.\n\n"
            f"{_ACTION_SCHEMA}\n"
            f"TAILORED CV:\n{cv_text}\n\n"
            f"COVER LETTER:\n{letter_text}"
        )

        max_steps = 30
        prompt = (
            f"Apply for this job on behalf of the applicant.\n"
            f"Title: {job.title}\n"
            f"Company: {job.company}\n"
            f"Job URL: {job.url}\n\n"
            "Begin by opening the job URL."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_steps):
            reply = _send_message_with_retry(self.client, self.config.model, messages)
            messages.append({"role": "assistant", "content": reply})

            action = _parse_action(reply)
            if action is None:
                messages.append({"role": "user", "content": "Invalid response. Return only a JSON action object."})
                continue

            name = action.get("action")
            rprint(f"    [dim cyan]Agent Action:[/dim cyan] {action}")

            if name == "done":
                success = bool(action.get("success", False))
                reason = action.get("message", "no reason given")
                if success:
                    rprint(f"    [green]✓ Agent reported success: {reason}[/green]")
                else:
                    rprint(f"    [red]✗ Agent gave up: {reason}[/red]")
                return success

            result = await self._execute_action(action)
            result_text = json.dumps(result)
            # Truncate large page reads to avoid filling the context window.
            if name == "read_page" and len(result_text) > 6000:
                result_text = result_text[:6000] + "...[truncated]"
                # Also prune previous read_page results from history to save tokens.
                messages = [
                    m for m in messages
                    if not (m["role"] == "user" and "read_page" in m["content"] and "...[truncated]" in m["content"])
                ]
            messages.append({"role": "user", "content": f"Result: {result_text}"})
            if name != "read_page":
                rprint(f"      [dim white]Result:[/dim white] {result_text}")
        return False

    async def _execute_action(self, action: dict) -> dict:
        name = action.get("action")
        if name == "navigate":
            title = await playwright_tools.navigate_to_url(self._session, action["url"])
            return {"page_title": title}
        if name == "read_page":
            return {"text": await playwright_tools.get_page_text(self._session)}
        if name == "fill_field":
            return await playwright_tools.fill_field(
                self._session, action["selector"], action["value"]
            )
        if name == "click":
            return await playwright_tools.click_element(self._session, action["selector"])
        if name == "upload_file":
            return await playwright_tools.upload_file(
                self._session, action.get("selector", "input[type=file]"), action["path"]
            )
        if name == "ask_user":
            return self._ask_user(action.get("question", ""), action.get("options", []))
        if name == "submit":
            return await playwright_tools.submit_form(self._session)
        return {"error": f"Unknown action: {name}"}

    def _ask_user(self, question: str, options: list) -> dict:
        rprint(f"\n  [bold yellow]❓ Agent needs your input:[/bold yellow]")
        rprint(f"  {question}")
        if options:
            for i, opt in enumerate(options):
                rprint(f"    {i + 1}. {opt}")
            rprint(f"  Enter the number of your choice, or type your answer:")
        else:
            rprint(f"  Type your answer:")
        try:
            answer = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                answer = options[idx]
        rprint(f"  [dim]Answer recorded: {answer}[/dim]")
        return {"user_answer": answer}
