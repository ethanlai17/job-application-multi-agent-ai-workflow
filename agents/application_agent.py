import asyncio
import json
from pathlib import Path

from openai import OpenAI

from browser.session import BrowserSession
from config.settings import Config
from models.job import JobListing
from tools import playwright_tools

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate the browser to a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": "Return the visible text of the current page to understand form structure.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_field",
            "description": "Fill a form field by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for the field"},
                    "value": {"type": "string", "description": "Value to enter"},
                },
                "required": ["selector", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Click the primary submit/apply button on the current page.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the application has been successfully submitted or failed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                },
                "required": ["success"],
            },
        },
    },
]


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
            cv_text = Path(job.cv_link).read_text(encoding="utf-8")
        if job.cover_letter_link and Path(job.cover_letter_link).exists():
            letter_text = Path(job.cover_letter_link).read_text(encoding="utf-8")

        system = (
            "You are a job application agent. Fill in an online job application form.\n"
            "Use navigate to open the application URL, then get_page_text to read the form.\n"
            "Fill each visible field using fill_field with appropriate CSS selectors.\n"
            "After filling all visible fields, use submit to move to the next page or submit.\n"
            "Call get_page_text after each submit to check for more fields or a confirmation.\n"
            "When you see a confirmation or 'application submitted' message, call done(success=true).\n"
            "If you hit an unrecoverable error, call done(success=false, message='reason').\n"
            "IMPORTANT: Only use information from the CV and cover letter below — never invent details.\n\n"
            f"TAILORED CV:\n{cv_text}\n\n"
            f"COVER LETTER:\n{letter_text}"
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Apply for this job:\n"
                    f"Title: {job.title}\nCompany: {job.company}\nURL: {job.url}\n\n"
                    "Navigate to the application URL (look for an Apply link on the job page), "
                    "then fill in the form."
                ),
            },
        ]

        success = False

        while True:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message
            messages.append(msg)

            if response.choices[0].finish_reason == "stop" or not msg.tool_calls:
                break

            finished = False
            tool_results = []

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                inputs = json.loads(tool_call.function.arguments)
                result = await self._execute_tool(name, inputs)

                if name == "done":
                    success = inputs.get("success", False)
                    finished = True

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

            messages.extend(tool_results)

            if finished:
                break

        return success

    async def _execute_tool(self, name: str, inputs: dict):
        if name == "navigate":
            title = await playwright_tools.navigate_to_url(self._session, inputs["url"])
            return {"page_title": title}
        if name == "get_page_text":
            return await playwright_tools.get_page_text(self._session)
        if name == "fill_field":
            return await playwright_tools.fill_field(self._session, inputs["selector"], inputs["value"])
        if name == "click":
            return await playwright_tools.click_element(self._session, inputs["selector"])
        if name == "submit":
            return await playwright_tools.submit_form(self._session)
        if name == "done":
            return {"acknowledged": True}
        return {"error": f"Unknown tool: {name}"}
