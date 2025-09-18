import asyncio
from dataclasses import dataclass
from typing import Optional
from crawl4ai import AsyncWebCrawler


@dataclass
class CrawlConfig:
    js_render: bool = False
    user_agent: Optional[str] = None
    timeout: int = 20000  # ms


class Crawl4AIClient:
    """Wrapper around AsyncWebCrawler to expose a sync .get()"""

    def __init__(self):
        # For MVP we create a new crawler per call via asyncio.run.
        # Later, you can optimize by keeping a long-lived crawler instance.
        pass

    def get(self, url: str, cfg: CrawlConfig) -> str:
        """Return HTML as text (synchronous wrapper)."""

        async def _run():
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                    user_agent=cfg.user_agent,
                    # Crawl4AI doesn’t have a “js_render” flag in some versions,
                    # but you can map this to args (like `js=cfg.js_render`) if supported.
                    timeout=cfg.timeout // 1000,  # convert ms → s
                )
                # choose .html for parsing with BeautifulSoup
                return result.html or ""

        return asyncio.run(_run())
