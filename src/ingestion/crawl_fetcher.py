# src/ingestion/crawl_fetcher.py
from __future__ import annotations

import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import urllib.robotparser as robotparser

from .crawl_client import Crawl4AIClient, CrawlConfig  # client wrapper you implemented


# ---------------------- Exceptions ----------------------

class FetchDisallowedByRobots(Exception):
    def __init__(self, url: str, ua: str):
        super().__init__(f"Disallowed by robots.txt for UA={ua}: {url}")
        self.url = url
        self.ua = ua


class FetchFailed(Exception):
    def __init__(self, url: str, reason: str, status: Optional[int] = None):
        msg = f"Fetch failed: {url} ({reason}{f', status={status}' if status is not None else ''})"
        super().__init__(msg)
        self.url = url
        self.reason = reason
        self.status = status


# ---------------------- Policy Config ----------------------

@dataclass
class FetchConfig:
    # Policy knobs handled by this wrapper:
    obey_robots: bool = True
    per_host_rate_per_min: int = 30
    per_host_burst: int = 6
    max_retries: int = 3
    backoff_base_ms: int = 250
    backoff_max_ms: int = 3000

    # Fetch (client) knobs we pass through:
    js_render: bool = False
    user_agent: str = "MarketNewsSummarizerBot/0.1 (+contact@example.com)"
    timeout_ms: int = 20000


# ---------------------- Robots cache ----------------------

class RobotsCache:
    def __init__(self, ttl_seconds: int = 24 * 3600):
        self._ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, robotparser.RobotFileParser]] = {}

    @staticmethod
    def _root(url: str) -> str:
        m = re.match(r"^(https?://[^/]+)", url)
        return m.group(1) if m else ""

    def get(self, url: str) -> robotparser.RobotFileParser:
        root = self._root(url)
        if not root:
            rp = robotparser.RobotFileParser()
            return rp
        now = time.time()
        ts, rp = self._cache.get(root, (0.0, None)) if root in self._cache else (0.0, None)
        if rp and (now - ts < self._ttl):
            return rp
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{root}/robots.txt")
        try:
            rp.read()
        except Exception:
            pass  # fail-open if robots fetch fails
        self._cache[root] = (now, rp)
        return rp

    def allowed(self, url: str, ua: str) -> bool:
        try:
            return self.get(url).can_fetch(ua or "*", url)
        except Exception:
            return True  # fail-open on parser errors


# ---------------------- Token bucket ----------------------

class TokenBucket:
    def __init__(self, rate_per_min: int, burst: int):
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.fill_rate = max(0.01, rate_per_min / 60.0)
        self.timestamp = time.monotonic()

    def consume(self):
        now = time.monotonic()
        elapsed = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        if self.tokens < 1.0:
            need = 1.0 - self.tokens
            sleep_s = need / self.fill_rate
            time.sleep(sleep_s + random.uniform(0, 0.05))
            self.tokens = 0.0
        self.tokens -= 1.0


# ---------------------- Fetcher ----------------------

class CrawlFetcher:
    """
    Policy wrapper around Crawl4AIClient:
      - robots.txt
      - per-host rate limiting
      - retries with backoff + jitter
    """

    def __init__(self, robots_ttl_seconds: int = 24 * 3600, client: Optional[Crawl4AIClient] = None):
        self.client = client or Crawl4AIClient()
        self.robots = RobotsCache(robots_ttl_seconds)
        self.buckets: Dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(30, 6))

    @staticmethod
    def _host(url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _bucket_for(self, url: str, cfg: FetchConfig) -> TokenBucket:
        host = self._host(url)
        if host not in self.buckets:
            self.buckets[host] = TokenBucket(cfg.per_host_rate_per_min, cfg.per_host_burst)
        return self.buckets[host]

    def _sleep_backoff(self, attempt: int, cfg: FetchConfig):
        base = cfg.backoff_base_ms * (2 ** (attempt - 1))
        delay_ms = min(base, cfg.backoff_max_ms)
        time.sleep((delay_ms + random.randint(0, delay_ms // 3 + 1)) / 1000.0)

    def get_html(self, url: str, cfg: FetchConfig) -> str:
        # robots
        if cfg.obey_robots and not self.robots.allowed(url, cfg.user_agent):
            raise FetchDisallowedByRobots(url, cfg.user_agent)

        # rate limit per host
        self._bucket_for(url, cfg).consume()

        last_exc: Optional[Exception] = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                # Map policy FetchConfig -> client CrawlConfig
                client_cfg = CrawlConfig(
                    js_render=cfg.js_render,
                    user_agent=cfg.user_agent,
                    timeout=cfg.timeout_ms,
                )
                html = self.client.get(url, client_cfg)
                if not html or not isinstance(html, str) or len(html.strip()) < 50:
                    raise FetchFailed(url, "empty_or_too_short")
                return html

            except FetchDisallowedByRobots:
                # bubble up
                raise
            except Exception as e:
                last_exc = e
                if attempt < cfg.max_retries:
                    self._sleep_backoff(attempt, cfg)
                else:
                    raise FetchFailed(url, getattr(e, "reason", e.__class__.__name__)) from e

        # Should never get here
        raise FetchFailed(url, f"unknown_error: {last_exc!r}")
