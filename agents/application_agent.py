import asyncio
import json
import re
from pathlib import Path

from openai import OpenAI

from browser.session import BrowserSession
from config.settings import Config
from models.job import JobListing
from tools import playwright_tools
from tools.document_tools import read_cv

_ACTION_SCHEMA = """
Return ONLY a JSON object — no markdown, no explanation — choosing one action per turn:

{"action": "navigate",   "url": "<full URL>"}
{"action": "read_page"}
{"action": "fill_field", "selector": "<CSS selector>", "value": "<text to enter>"}
{"action": "click",      "selector": "<CSS selector>"}
{"action": "submit"}
{"action": "done",       "success": true,  "message": "<optional note>"}
{"action": "done",       "success": false, "message": "<reason for failure>"}
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


class ApplicationAgent:
    def __init__(self, config: Config, client: OpenAI):
        self.config = config
        self.client = client

    def apply(self, job: JobListing) -> bool:
        return asyncio.run(self._apply_async(job))

    async def _apply_async(self, job: JobListing) -> bool:
        async with BrowserSession(self.config) as session:
            self._session = session
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
            "Your goal is to open a job posting, find the application form, "
            "fill every field with the applicant's details, and submit it.\n\n"
            "Process:\n"
            "1. Open the job URL.\n"
            "2. Read the page to understand available form fields and buttons.\n"
            "3. Fill each field one at a time, then submit.\n"
            "4. After each submit, read the page again to check for new fields or a confirmation.\n"
            "5. When you see an application confirmation, signal success.\n"
            "6. If you hit an unrecoverable error, signal failure with a short reason.\n\n"
            "Only use information from the CV and cover letter below — never invent details.\n\n"
            f"{_ACTION_SCHEMA}\n"
            f"TAILORED CV:\n{cv_text}\n\n"
            f"COVER LETTER:\n{letter_text}"
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Apply for this job on behalf of the applicant.\n"
                    f"Title: {job.title}\n"
                    f"Company: {job.company}\n"
                    f"Job URL: {job.url}\n\n"
                    "Begin by opening the job URL."
                ),
            },
        ]

        max_steps = 30
        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=300,
                messages=messages,
            )
            reply = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": reply})

            action = _parse_action(reply)
            if action is None:
                messages.append({
                    "role": "user",
                    "content": "Invalid response. Return only a JSON action object.",
                })
                continue

            name = action.get("action")

            if name == "done":
                return bool(action.get("success", False))

            result = await self._execute_action(action)
            messages.append({"role": "user", "content": f"Result: {json.dumps(result)}"})

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
        if name == "submit":
            return await playwright_tools.submit_form(self._session)
        return {"error": f"Unknown action: {name}"}
