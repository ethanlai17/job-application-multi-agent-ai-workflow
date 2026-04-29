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

    if "feed" in session.page.url:
        return True  # already logged in via saved cookies

    await session.page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
    await session.random_delay()

    await session.human_type("#username", email)
    await session.human_type("#password", password)
    await session.random_delay()
    await session.page.click('button[type="submit"]')
    await session.page.wait_for_load_state("domcontentloaded")
    await session.random_delay()
    await session.check_blocked()

    return "feed" in session.page.url or "mynetwork" in session.page.url


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
    """Return visible text of the current page (for Claude to read form structure)."""
    return await session.page.inner_text("body")


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
    """Click an element by CSS selector."""
    try:
        el = await session.page.query_selector(selector)
        if el:
            await el.scroll_into_view_if_needed()
            await session.random_delay()
            await el.click()
            return "ok"
        return "not_found"
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
