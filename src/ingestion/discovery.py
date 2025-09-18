# src/ingestion/discovery.py
from __future__ import annotations

import re
from typing import List, Dict, Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .crawl_fetcher import CrawlFetcher, FetchConfig
from .canonicalize import strip_tracking


def _as_list(value) -> List[str]:
    """Allow YAML to specify a single CSS selector or a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns or []]


def discover_urls(src_cfg: Dict, fetcher: CrawlFetcher) -> List[str]:
    """
    Discover candidate article URLs from the source's discovery pages.

    src_cfg schema (relevant parts):
      discovery:
        urls: [str, ...]
        follow_links_css: str | [str, ...]
        allow_patterns: [regex, ...]
        deny_patterns:  [regex, ...]
        js_render: bool
        max_new_urls: int
      compliance:
        obey_robots: bool
        user_agent: str
      base: str
    """
    name = src_cfg.get("name", "unknown_source")
    base = src_cfg.get("base", "")
    disc = src_cfg.get("discovery", {}) or {}
    comp = src_cfg.get("compliance", {}) or {}
    rate = src_cfg.get("rate", {}) or {}

    urls_cfg: List[str] = disc.get("urls", [])
    selectors: List[str] = _as_list(disc.get("follow_links_css"))
    allow = _compile_patterns(disc.get("allow_patterns", []))
    deny = _compile_patterns(disc.get("deny_patterns", []))
    js = bool(disc.get("js_render", False))
    obey_robots = bool(comp.get("obey_robots", True))
    ua = comp.get("user_agent") or "MarketNewsSummarizerBot/0.1 (+contact@example.com)"
    max_new = int(disc.get("max_new_urls", 50))

    if not urls_cfg:
        print(f"[WARN][{name}] No discovery URLs configured.")
        return []

    if not selectors:
        print(f"[WARN][{name}] No follow_links_css selector(s) configured.")
        return []

    discovered: List[str] = []

    # Build the fetch config for discovery pages.
    cfg = FetchConfig(
        js_render=js,
        user_agent=ua,
        obey_robots=obey_robots,
        timeout_ms=20_000,
        per_host_rate_per_min=int(rate.get("per_minute", 30)),
        per_host_burst=int(rate.get("burst", 6)),
    )

    # Fetch each discovery page and extract links
    for page in urls_cfg:
        try:
            html = fetcher.get_html(page, cfg)
        except Exception as e:
            print(f"[ERROR][{name}] discovery fetch failed: {page} ({e})")
            continue

        if not html or not isinstance(html, str) or len(html) < 50:
            print(f"[WARN][{name}] discovery page empty/short: {page}")
            continue

        soup = BeautifulSoup(html, "lxml")

        # DEBUG: if selectors find nothing, print a few candidate hrefs so we can tune selectors
        candidate_hrefs = [a.get("href") for a in soup.select("a[href]")]  # all anchors
        if not any(soup.select(sel) for sel in selectors):
            sample = []
            for href in candidate_hrefs:
                if not href:
                    continue
                abs_url = strip_tracking(urljoin(base, href))
                sample.append(abs_url)
                if len(sample) >= 15:
                    break
            print(f"[DEBUG][{name}] No matches for {selectors}. First 15 hrefs on page:")
            for u in sample:
                print("   -", u)

        found_on_page = 0
        for sel in selectors:
            for a in soup.select(sel):
                href = a.get("href")
                if not href:
                    continue
                abs_url = urljoin(base, href)
                abs_url = strip_tracking(abs_url)
                discovered.append(abs_url)
                found_on_page += 1

        print(f"[INFO][{name}] {page} → extracted {found_on_page} raw links via {len(selectors)} selector(s)")

    # Filter by allow/deny regex
    def _ok(u: str) -> bool:
        if deny and any(rx.search(u) for rx in deny):
            return False
        if allow and not any(rx.search(u) for rx in allow):
            return False
        return True

    filtered: List[str] = [u for u in discovered if _ok(u)]

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in filtered:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    # Cap to max_new_urls
    capped = uniq[:max_new]

    print(
        f"[INFO][{name}] discovered={len(discovered)} filtered={len(filtered)} "
        f"unique={len(uniq)} returning={len(capped)} (cap={max_new})"
    )

    return capped
