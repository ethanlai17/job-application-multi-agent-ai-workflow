import asyncio
import base64
import re
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

    # Dismiss cookie consent banner if present (use JS click to avoid overlay timeouts)
    try:
        accept_btn = await session.page.query_selector('button[action-type="ACCEPT"]')
        if accept_btn:
            await session.page.evaluate("btn => btn.click()", accept_btn)
            await session.page.wait_for_timeout(1000)
    except Exception:
        pass

    # Wait for job cards to render (LinkedIn public page)
    try:
        await session.page.wait_for_selector("div.base-card", timeout=10000)
    except Exception:
        pass
    await session.human_scroll(600)
    await session.page.wait_for_timeout(1500)

    jobs: list[dict] = []
    seen_ids: set[str] = set()

    while len(jobs) < max_results:
        cards = await session.page.query_selector_all("div.base-card")

        for card in cards:
            if len(jobs) >= max_results:
                break
            try:
                # Job URL — on the full-link anchor
                link_el = await card.query_selector("a.base-card__full-link, a[href*='/jobs/view/']")
                href = await link_el.get_attribute("href") if link_el else ""
                if not href:
                    continue

                # Clean URL and extract numeric job ID.
                # LinkedIn URLs: /jobs/view/title-at-company-4067388618 (slug+id)
                # or legacy:     /jobs/view/4067388618 (pure id)
                clean_url = href.split("?")[0].rstrip("/")
                if not clean_url.startswith("http"):
                    clean_url = "https://www.linkedin.com" + clean_url
                last_seg = clean_url.split("/")[-1]
                if last_seg.isdigit():
                    job_id = last_seg
                else:
                    m = re.search(r"-(\d+)$", last_seg)
                    job_id = m.group(1) if m else ""
                if not job_id or job_id in seen_ids:
                    continue
                seen_ids.add(job_id)

                # Title, company, location use the public page's stable class names
                job_title = await _extract_text(card, ["h3.base-search-card__title"])
                company = await _extract_text(card, [
                    "h4.base-search-card__subtitle a",
                    "h4.base-search-card__subtitle",
                ])
                loc = await _extract_text(card, ["span.job-search-card__location"])

                jobs.append({
                    "linkedin_job_id": job_id,
                    "title": job_title,
                    "company": company,
                    "url": clean_url,
                    "location": loc,
                    "salary": "",
                })
            except Exception:
                continue

        # Next page button on public LinkedIn jobs page
        next_btn = await session.page.query_selector(
            "button[aria-label='View next page'], "
            "button[aria-label='Next'], "
            "li[data-test-pagination-page-btn].selected + li button"
        )
        if not next_btn or len(jobs) >= max_results:
            break
        await next_btn.scroll_into_view_if_needed()
        await next_btn.click()
        await session.random_delay()
        await session.page.wait_for_selector("div.base-card", timeout=10000)
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

    # Wait for job detail content to render (member view or public view selectors)
    try:
        await session.page.wait_for_selector(
            ".jobs-description-content__text, .jobs-box__html-content, "
            ".description__text, .show-more-less-html, h1",
            timeout=10000,
        )
    except Exception:
        pass

    page_text = await session.page.inner_text("body")
    removed_signals = [
        "no longer accepting applications",
        "job may not be valid",
        "posting has been removed",
        "job has been removed",
        "this job is no longer available",
        "job posting has expired",
    ]
    if any(sig in page_text.lower() for sig in removed_signals):
        raise JobUnavailableError(f"Job posting unavailable: {url}")

    # Expand "Show more" if present (member view and public view buttons)
    try:
        show_more = await session.page.query_selector(
            "button.jobs-description__footer-button, "
            "button.show-more-less-html__button"
        )
        if show_more:
            await session.page.evaluate("btn => btn.click()", show_more)
            await session.page.wait_for_timeout(500)
    except Exception:
        pass

    description = ""
    salary = ""

    try:
        desc_el = await session.page.query_selector(
            ".jobs-description-content__text, "  # member view
            ".jobs-box__html-content, "          # member view alt
            ".description__text, "              # public view
            ".show-more-less-html"              # public view alt
        )
        if desc_el:
            description = (await desc_el.inner_text()).strip()
    except Exception:
        pass

    # Skip jobs with no description — likely invalid/removed
    if not description:
        raise JobUnavailableError(f"Empty job description (removed or invalid): {url}")

    # Skip jobs posted by staffing/recruitment agencies
    recruiter_signals = [
        "staffing and recruiting",
        "staffing & recruiting",
        "recruitment agency",
        "recruiting firm",
        "executive search",
        "talent acquisition firm",
        "we are a recruiter",
        "our client is",
        "on behalf of our client",
        "our client, a",
    ]
    page_lower = page_text.lower()
    if any(sig in page_lower for sig in recruiter_signals):
        raise RecruiterJobError(f"Recruiter/staffing posting skipped: {url}")

    try:
        salary_el = await session.page.query_selector(".job-details-jobs-unified-top-card__job-insight span")
        if salary_el:
            text = (await salary_el.inner_text()).strip()
            if any(c in text for c in ["£", "$", "€", "salary", "Salary"]):
                salary = text
    except Exception:
        pass

    # Extract title + company (member view and public view selectors)
    title = await _extract_text(session.page, [
        "h1.t-24",
        "h1[class*='job-title']",
        ".jobs-unified-top-card__job-title h1",
        ".job-details-jobs-unified-top-card__job-title h1",
        ".topcard__title",   # public view
        "h1",
    ])
    company = await _extract_text(session.page, [
        ".jobs-unified-top-card__company-name a",
        ".job-details-jobs-unified-top-card__company-name a",
        ".jobs-unified-top-card__primary-description a",
        "[class*='company-name'] a",
        "[class*='company-name']",
        ".topcard__org-name-link",   # public view
        ".topcard__flavor a",        # public view alt
    ])

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
