import asyncio
import base64
from pathlib import Path
from urllib.parse import quote_plus

from browser.session import BrowserSession


# ── LinkedIn helpers ──────────────────────────────────────────────────────────

async def _extract_text(element, selectors: list[str]) -> str:
    """Try each CSS selector in order, return the first non-empty inner text."""
    for sel in selectors:
        try:
            el = await element.query_selector(sel)
            if el:
                text = (await el.inner_text()).strip()
                if text:
                    return text
        except Exception:
            continue
    return ""

async def linkedin_login(session: BrowserSession, email: str, password: str) -> bool:
    """Log into LinkedIn. Returns True if already logged in or login succeeded."""
    await session.page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
    await session.random_delay()

    base_url = session.page.url.split("?")[0]
    if "linkedin.com/feed" in base_url or "linkedin.com/mynetwork" in base_url:
        return True  # already logged in via saved cookies

    await session.page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
    await session.random_delay()

    # Robustly find inputs even if LinkedIn uses dynamic React IDs
    await session.page.evaluate('''() => {
        const emails = Array.from(document.querySelectorAll('input#session_key, input#username, input[type="text"], input[type="email"]'));
        const email = emails.find(e => e.offsetParent !== null); // first visible
        if (email) email.id = "injected_email_input";
        
        const passes = Array.from(document.querySelectorAll('input#session_password, input#password, input[type="password"]'));
        const pass = passes.find(e => e.offsetParent !== null); // first visible
        if (pass) pass.id = "injected_pass_input";
    }''')
    
    await session.human_type("#injected_email_input", email)
    await session.human_type("#injected_pass_input", password)
    await session.random_delay()
    # LinkedIn changed sign-in button to type="button" — press Enter instead of
    # relying on a submit selector that may break with UI updates.
    try:
        await session.page.click(
            'button[type="submit"], button[aria-label="Sign in"]',
            timeout=3000,
        )
    except Exception:
        try:
            await session.page.click(
                'button:has-text("Sign in"):not(:has-text("with"))',
                timeout=3000,
            )
        except Exception:
            await session.page.keyboard.press("Enter")
    await session.page.wait_for_load_state("domcontentloaded")
    await session.random_delay()
    await session.check_blocked()

    base_url = session.page.url.split("?")[0]
    return "linkedin.com/feed" in base_url or "linkedin.com/mynetwork" in base_url


async def search_linkedin_jobs(
    session: BrowserSession,
    title: str,
    location: str,
    work_type: str,
    max_results: int = 20,
) -> list[dict]:
    """Search LinkedIn Jobs and return a list of job dicts (id, title, company, url, location, salary)."""
    work_type_map = {
        "remote": "2",
        "hybrid": "3",
        "onsite": "1",
        "on-site": "1",
    }
    f_WT = work_type_map.get(work_type.lower(), "3")

    search_url = (
        f"https://www.linkedin.com/jobs/search/"
        f"?keywords={quote_plus(title)}"
        f"&location={quote_plus(location)}"
        f"&f_WT={f_WT}"
        f"&f_TPR=r604800"   # posted in the last 7 days
        f"&sortBy=DD"       # sort by date (newest first)
    )

    await session.page.goto(search_url, wait_until="domcontentloaded")
    await session.random_delay()
    await session.check_blocked()

    # Wait for job cards to render (member view uses li[data-occludable-job-id])
    _CARD_SEL = "li[data-occludable-job-id]"
    try:
        await session.page.wait_for_selector(_CARD_SEL, timeout=10000)
    except Exception:
        pass
    await session.human_scroll(600)
    await session.page.wait_for_timeout(1500)

    jobs: list[dict] = []
    seen_ids: set[str] = set()

    while len(jobs) < max_results:
        cards = await session.page.query_selector_all(_CARD_SEL)

        for card in cards:
            if len(jobs) >= max_results:
                break
            try:
                job_id = await card.get_attribute("data-occludable-job-id") or ""
                if not job_id or job_id in seen_ids:
                    continue
                seen_ids.add(job_id)

                link_el = await card.query_selector("a[href*='/jobs/view/']")
                clean_url = "https://www.linkedin.com/jobs/view/" + job_id + "/"

                job_title = ""
                if link_el:
                    job_title = (await link_el.get_attribute("aria-label") or "").strip()
                    if not job_title:
                        job_title = (await link_el.inner_text()).strip()
                    job_title = job_title.removesuffix(" with verification").strip()

                company = await _extract_text(card, [
                    ".artdeco-entity-lockup__subtitle",
                    ".job-card-container__company-name",
                ])
                loc = await _extract_text(card, [
                    ".artdeco-entity-lockup__caption",
                    ".job-card-container__metadata-item",
                ])
                salary = await _extract_text(card, [
                    ".job-card-container__salary-info",
                    ".compensation-text",
                ])

                jobs.append({
                    "linkedin_job_id": job_id,
                    "title": job_title,
                    "company": company,
                    "url": clean_url,
                    "location": loc,
                    "salary": salary,
                })
            except Exception:
                continue

        next_btn = await session.page.query_selector("button[aria-label='View next page']")
        if not next_btn or len(jobs) >= max_results:
            break
        await next_btn.scroll_into_view_if_needed()
        await next_btn.click()
        await session.random_delay()
        try:
            await session.page.wait_for_selector(_CARD_SEL, timeout=10000)
        except Exception:
            pass
        await session.human_scroll(400)

    return jobs


class JobUnavailableError(Exception):
    pass


class RecruiterJobError(Exception):
    pass


async def get_job_details(session: BrowserSession, url: str) -> dict:
    """Navigate to a LinkedIn job page and extract full job description + salary.

    Raises JobUnavailableError if the posting has been removed or the job ID is invalid.
    """
    await session.page.goto(url, wait_until="domcontentloaded")
    await session.random_delay()
    await session.check_blocked()

    # Scroll to trigger lazy-rendered content (LinkedIn SPA)
    await session.page.evaluate("window.scrollTo(0, 600)")
    await session.page.wait_for_timeout(2000)

    page_text = await session.page.inner_text("body")
    page_lower = page_text.lower()

    removed_signals = [
        "no longer accepting applications",
        "job may not be valid",
        "posting has been removed",
        "job has been removed",
        "this job is no longer available",
        "job posting has expired",
    ]
    if any(sig in page_lower for sig in removed_signals):
        raise JobUnavailableError(f"Job posting unavailable: {url}")

    # Extract description: everything from "About the job" to the next boundary.
    # LinkedIn's new SPA renders no stable CSS classes, so we rely on text markers.
    description = ""
    desc_start_markers = ["About the job", "About this role", "Job description", "Job Description"]
    desc_end_markers = [
        "About the company", "About Hopper", "About us\n", "Similar jobs",
        "People also viewed", "Meet the team", "How you match",
    ]
    for start_marker in desc_start_markers:
        idx = page_text.find(start_marker)
        if idx != -1:
            desc_raw = page_text[idx + len(start_marker):].strip()
            # Cut at the first end marker
            for end_marker in desc_end_markers:
                end_idx = desc_raw.find(end_marker)
                if end_idx != -1:
                    desc_raw = desc_raw[:end_idx].strip()
                    break
            if len(desc_raw) > 100:
                description = desc_raw
                break

    if not description:
        raise JobUnavailableError(f"Empty job description (removed or invalid): {url}")

    # Skip staffing/recruitment agencies
    recruiter_signals = [
        "staffing and recruiting", "staffing & recruiting", "recruitment agency",
        "recruiting firm", "executive search", "talent acquisition firm",
        "we are a recruiter", "our client is", "on behalf of our client", "our client, a",
    ]
    if any(sig in page_lower for sig in recruiter_signals):
        raise RecruiterJobError(f"Recruiter/staffing posting skipped: {url}")

    # Extract salary from the header block (first ~800 chars, before the description).
    import re as _re
    salary = ""
    header_text = page_text[:800]
    salary_match = _re.search(
        r"(?:[\$£€]\s*)?[\d,]+\s*[Kk]?\s*(?:GBP|USD|EUR)"
        r"(?:\s*/\s*(?:yr|year|mo|month))?"
        r"(?:\s*[-–]\s*(?:[\$£€]\s*)?[\d,]+\s*[Kk]?\s*(?:GBP|USD|EUR)?(?:\s*/\s*(?:yr|year|mo|month))?)?",
        header_text,
    )
    if not salary_match:
        # Fallback: £/$/€ + meaningful number (>=4 digits or K suffix) to skip "£0" nav noise
        salary_match = _re.search(r"[\$£€][\d,]{4,}(?:\s*[Kk])?(?:\s*/\s*(?:yr|year|mo|month))?", header_text)
        if not salary_match:
            salary_match = _re.search(r"[\$£€]\d+[Kk]", header_text)
    if salary_match:
        salary = salary_match.group(0).strip()

    # Extract title + company from the page <title> tag ("Job Title | Company | LinkedIn")
    page_title = await session.page.title()
    title, company = "", ""
    if "|" in page_title:
        parts = [p.strip() for p in page_title.split("|")]
        title = parts[0] if parts else ""
        company = parts[1] if len(parts) > 1 else ""
        company = company.replace(" | LinkedIn", "").strip()

    return {
        "job_description": description,
        "salary": salary,
        "title_from_detail": title,
        "company_from_detail": company,
    }


# ── Form-fill helpers ─────────────────────────────────────────────────────────

async def navigate_to_url(session: BrowserSession, url: str) -> str:
    """Navigate to a URL and return the page title."""
    await session.page.goto(url, wait_until="domcontentloaded")
    await session.random_delay()
    return await session.page.title()


async def get_page_text(session: BrowserSession) -> str:
    """Return a concise summary of the current page: headings, form labels/inputs, and buttons."""
    try:
        text = await session.page.evaluate("""() => {
            const parts = [];
            // Page title
            parts.push('URL: ' + window.location.href);
            // Headings
            document.querySelectorAll('h1,h2,h3').forEach(el => {
                const t = el.innerText.trim();
                if (t) parts.push('[H] ' + t);
            });
            // Form labels
            document.querySelectorAll('label').forEach(el => {
                const t = el.innerText.trim();
                if (t) parts.push('[LABEL] ' + t);
            });
            // Inputs and selects
            document.querySelectorAll('input,select,textarea').forEach(el => {
                const type = el.type || el.tagName.toLowerCase();
                const name = el.name || el.placeholder || el.id || '';
                const val = el.value || '';
                parts.push('[INPUT type=' + type + ' name=' + name + ' value=' + val + ']');
            });
            // Buttons and action links (skip accessibility-only ones)
            const skipTexts = new Set(['skip to main content', 'skip to content']);
            const actionKeywords = ['apply', 'submit', 'upload', 'continue', 'next', 'save', 'send', 'interested'];
            document.querySelectorAll('button,a[role=button],[type=submit]').forEach(el => {
                const t = (el.innerText || el.value || '').trim();
                if (t && !skipTexts.has(t.toLowerCase())) parts.push('[BTN] ' + t);
            });
            // Also include <a> tags that look like action links (have aria-label with action keywords)
            // Format: [APPLY_LINK <aria-label>] so the agent knows to use a[aria-label="<label>"]
            document.querySelectorAll("a[aria-label]").forEach(function(el) {
                var label = el.getAttribute("aria-label") || "";
                var t = (el.innerText || "").trim() || label;
                var lc = (label + " " + t).toLowerCase();
                if (label && actionKeywords.some(function(k) { return lc.indexOf(k) >= 0; })) {
                    parts.push("[APPLY_LINK " + label + "] " + t);
                }
            });
            // Any visible alerts or status messages
            document.querySelectorAll('[role=alert],[aria-live],.error,.success').forEach(el => {
                const t = el.innerText.trim();
                if (t) parts.push('[MSG] ' + t);
            });
            return parts.join('\\n');
        }""")
        return text[:8000]  # hard cap just in case
    except Exception as e:
        return f"Error reading page: {e}"


async def take_screenshot(session: BrowserSession, name: str = "screenshot") -> str:
    """Take a screenshot, save to output/debug/, return base64-encoded PNG."""
    debug_dir = Path("output/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = str(debug_dir / f"{name}.png")
    await session.page.screenshot(path=path, full_page=False)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def fill_field(session: BrowserSession, selector: str, value: str):
    """Fill a form field identified by CSS selector or label text."""
    try:
        el = await session.page.query_selector(selector)
        if el:
            tag = await el.evaluate("e => e.tagName.toLowerCase()")
            if tag == "select":
                await el.select_option(label=value)
            else:
                await el.fill("")
                await session.human_type(selector, value)
            return "ok"
    except Exception as e:
        return f"error: {e}"
    return "not_found"


async def click_element(session: BrowserSession, selector: str) -> str:
    """Click an element by CSS selector, :has-text() pattern, or aria-label."""
    import re as _re

    async def _do_click(el) -> str:
        """Click el; if a new tab opens within 1.5 s, switch to it."""
        pages_before = len(session._context.pages)
        await el.scroll_into_view_if_needed()
        await session.random_delay()
        await el.click()
        await asyncio.sleep(1.5)
        pages_after = session._context.pages
        if len(pages_after) > pages_before:
            new_page = pages_after[-1]
            await new_page.wait_for_load_state("domcontentloaded")
            session.page = new_page
            return "ok_new_tab"
        return "ok"

    if not selector or not selector.strip():
        return "not_found"

    try:
        el = await session.page.query_selector(selector)
        if el:
            return await _do_click(el)

        # Fallback 1: :has-text() with Unicode-safe JS text search
        m = _re.search(r':has-text\(["\'](.+?)["\']\)', selector)
        if m:
            needle = m.group(1).strip().lower()
            tag = selector.split(":")[0] or "*"
            handle = await session.page.evaluate_handle(f"""() => {{
                const needle = {repr(needle)};
                const els = Array.from(document.querySelectorAll({repr(tag)}));
                return els.find(e => e.innerText.trim().toLowerCase().includes(needle)) || null;
            }}""")
            if handle:
                js_el = handle.as_element()
                if js_el:
                    return await _do_click(js_el)

        # Fallback 2: aria-label match (e.g. 'a[aria-label="Apply on company website"]')
        m2 = _re.search(r'aria-label[=\s]*["\']([^"\']+)["\']', selector)
        if m2:
            needle = m2.group(1).strip().lower()
            handle = await session.page.evaluate_handle(f"""() => {{
                const needle = {repr(needle)};
                const els = Array.from(document.querySelectorAll('[aria-label]'));
                return els.find(e => (e.getAttribute('aria-label') || '').toLowerCase().includes(needle)) || null;
            }}""")
            if handle:
                js_el = handle.as_element()
                if js_el:
                    return await _do_click(js_el)

        return "not_found"
    except Exception as e:
        if "ok" in str(e):
            return "ok"
        return f"error: {e}"


async def upload_file(session: BrowserSession, selector: str, file_path: str) -> str:
    """Set a file on a file-input element (supports hidden inputs and drop-zone triggers)."""
    try:
        abs_path = str(Path(file_path).resolve())
        # Try the selector directly first (may be the hidden <input type=file>)
        el = await session.page.query_selector(selector)
        if el and await el.evaluate("e => e.tagName.toLowerCase()") == "input":
            await el.set_input_files(abs_path)
            return "ok"
        # Fallback: find any visible file input on the page
        inputs = await session.page.query_selector_all("input[type='file']")
        if inputs:
            await inputs[0].set_input_files(abs_path)
            return "ok"
        return "no_file_input_found"
    except Exception as e:
        return f"error: {e}"


async def submit_form(session: BrowserSession) -> str:
    """Submit the current form by clicking the primary submit button."""
    selectors = [
        "button[type='submit']",
        "input[type='submit']",
        "button:has-text('Submit')",
        "button:has-text('Apply')",
        "button:has-text('Send')",
    ]
    for sel in selectors:
        try:
            btn = await session.page.query_selector(sel)
            if btn:
                await btn.scroll_into_view_if_needed()
                await session.random_delay()
                await btn.click()
                await session.page.wait_for_load_state("domcontentloaded")
                return "submitted"
        except Exception:
            continue
    return "no_submit_button_found"


# ── Sync wrappers (called by agents via asyncio.run) ─────────────────────────

def run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
