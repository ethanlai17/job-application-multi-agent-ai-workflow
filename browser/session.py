import asyncio
import random
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from config.settings import Config

LINKEDIN_STATE_PATH = "browser/linkedin_state.json"

STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-GB', 'en'] });
window.chrome = { runtime: {} };
"""


class LinkedInBlockedError(Exception):
    pass


class BrowserSession:
    def __init__(self, config: Config):
        self.config = config
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def __aenter__(self) -> "BrowserSession":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless_browser,
            args=["--disable-blink-features=AutomationControlled"],
        )

        state_path = Path(LINKEDIN_STATE_PATH)
        ctx_kwargs = dict(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-GB",
            timezone_id="Europe/London",
        )
        if state_path.exists():
            ctx_kwargs["storage_state"] = str(state_path)

        self._context = await self._browser.new_context(**ctx_kwargs)
        await self._context.add_init_script(STEALTH_SCRIPT)
        self.page = await self._context.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            state_path = Path(LINKEDIN_STATE_PATH)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=str(state_path))
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def random_delay(self):
        delay = random.uniform(
            self.config.request_delay_min, self.config.request_delay_max
        )
        await asyncio.sleep(delay)

    async def human_type(self, selector: str, text: str):
        await self.page.click(selector)
        for char in text:
            await self.page.keyboard.type(char)
            await asyncio.sleep(random.uniform(0.05, 0.15))
            if random.random() < 0.03:
                await self.page.keyboard.press("Backspace")
                await asyncio.sleep(random.uniform(0.05, 0.1))
                await self.page.keyboard.type(char)

    async def human_scroll(self, pixels: int = 600):
        scrolled = 0
        while scrolled < pixels:
            step = random.randint(100, 300)
            await self.page.mouse.wheel(0, step)
            scrolled += step
            await asyncio.sleep(random.uniform(0.1, 0.5))

    async def check_blocked(self):
        url = self.page.url
        if "/checkpoint/challenge" in url or "/authwall" in url:
            screenshot_dir = Path("output/debug")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            await self.page.screenshot(path=str(screenshot_dir / "blocked.png"))
            raise LinkedInBlockedError(
                "LinkedIn blocked the session. Screenshot saved to output/debug/blocked.png. "
                "Run again without headless mode and complete the challenge manually."
            )
