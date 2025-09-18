# src/ingestion/pipeline.py

from datetime import datetime, timezone
from typing import Dict, Optional
import logging

from .crawl_client import Crawl4AIClient
from .crawl_fetcher import CrawlFetcher
from .discovery import discover_urls
from .extractor import extract_article
from .canonicalize import canonical_url_from_html, strip_tracking
from .dedup import simple_key, is_duplicate
from app.models import Article
from app.db import session_scope

log = logging.getLogger(__name__)

def _safe_dt(dt_val: Optional[datetime]) -> datetime:
    if isinstance(dt_val, datetime):
        return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _make_fetcher(client: Crawl4AIClient, src_cfg: Dict):
    ua = src_cfg["compliance"].get("user_agent")
    rate = src_cfg["rate"]["per_minute"]
    burst = src_cfg["rate"].get("burst", 5)

    # Try (client, FetchConfig(...))
    try:
        from .crawl_fetcher import FetchConfig
        return CrawlFetcher(client, FetchConfig(user_agent=ua, rate_per_min=rate, burst=burst))
    except Exception:
        pass

    # Try (client, rate_per_min=..., burst=...)
    try:
        return CrawlFetcher(client, rate_per_min=rate, burst=burst)  # type: ignore
    except TypeError:
        pass

    # Fallback
    return CrawlFetcher(client)  # type: ignore

def _build_call_cfg(ua: Optional[str], obey: bool, jsr: bool):
    """Return the best per-call config object for get_html(url, cfg)."""
    # Prefer a real FetchConfig if available
    try:
        from .crawl_fetcher import FetchConfig  # type: ignore
        return FetchConfig(user_agent=ua, obey_robots=obey, js_render=jsr)
    except Exception:
        # Fallback to a simple dict; many wrappers accept a mapping as cfg
        return {"user_agent": ua, "obey_robots": obey, "js_render": jsr}

def _get_html(fetcher: CrawlFetcher, url: str, ua: Optional[str], obey: bool, jsr: bool) -> Optional[str]:
    """
    Call fetcher.get_html with maximal compatibility across different signatures:
      1) get_html(url, cfg)
      2) get_html(url, js_render=..., obey_robots=..., user_agent=...)
      3) get_html(url, user_agent=...)
      4) get_html(url)
    """
    cfg = _build_call_cfg(ua, obey, jsr)
    # (1) get_html(url, cfg)
    try:
        return fetcher.get_html(url, cfg)  # type: ignore
    except TypeError:
        pass
    # (2) keyword args
    try:
        return fetcher.get_html(url, js_render=jsr, obey_robots=obey, user_agent=ua)  # type: ignore
    except TypeError:
        pass
    # (3) minimal keyword subset
    try:
        return fetcher.get_html(url, user_agent=ua)  # type: ignore
    except TypeError:
        pass
    # (4) bare minimum
    return fetcher.get_html(url)

def run_source(src_cfg: Dict):
    client = Crawl4AIClient()
    fetcher = _make_fetcher(client, src_cfg)
    ua = src_cfg["compliance"].get("user_agent")
    obey = src_cfg["compliance"].get("obey_robots", True)
    jsr = src_cfg["article"].get("js_render", False)

    try:
        urls = discover_urls(src_cfg, fetcher)
    except Exception as e:
        log.exception("discovery failed for %s: %s", src_cfg.get("name"), e)
        return

    for url in urls:
        try:
            html = _get_html(fetcher, url, ua, obey, jsr)
            if not html:
                continue

            canon = canonical_url_from_html(url, html) or strip_tracking(url)
            data = extract_article(html, src_cfg["article"])
            title = (data.get("title") or "").strip()
            body = (data.get("body_text") or "").strip()
            if not title or not body:
                continue

            key = simple_key(title, body)

            with session_scope() as s:
                dup, _ = is_duplicate(s, strip_tracking(url), canon, key)
                if dup:
                    continue

                art = Article(
                    source=src_cfg["name"],
                    url=strip_tracking(url),
                    canonical_url=canon,
                    title=title[:512],
                    author=(data.get("author") or "").strip() or None,
                    published_at=_safe_dt(data.get("published_at")),
                    first_seen_at=datetime.now(timezone.utc),
                    lang=data.get("lang") or "en",
                    body_text=body,
                    meta={"discovered_from": (src_cfg.get("discovery", {}).get("urls") or [""])[0]},
                    status="READY_FOR_SUMMARY",
                )
                s.add(art)
                s.flush()
        except Exception as e:
            log.exception("failed to process %s: %s", url, e)
            continue
