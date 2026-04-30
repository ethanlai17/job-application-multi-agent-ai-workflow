import asyncio
from config.settings import Config
from browser.session import BrowserSession
from rich import print as rprint

async def main():
    config = Config.load()
    async with BrowserSession(config) as session:
        await session.page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
        await asyncio.sleep(2)
        html = await session.page.inner_html("body")
        with open("output/login_body.html", "w") as f:
            f.write(html)
        rprint("Dumped login body to output/login_body.html")
        rprint(f"Current URL: {session.page.url}")

if __name__ == "__main__":
    asyncio.run(main())
