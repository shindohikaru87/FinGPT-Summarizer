# src/ingestion/pipeline.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Set
import logging
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import select, or_
from sqlalchemy.exc import IntegrityError

from .crawl_client import Crawl4AIClient
from .crawl_fetcher import CrawlFetcher
from .discovery import discover_urls
from .extractor import extract_article
from .canonicalize import canonical_url_from_html, strip_tracking
from .dedup import simple_key, is_duplicate
from src.app.models import Article
from src.app.db import session_scope

log = logging.getLogger(__name__)

# ---------------------------- helpers ---------------------------------

def _safe_dt(dt_val: Optional[datetime]) -> datetime:
    if isinstance(dt_val, datetime):
        return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _normalize_url(u: str) -> str:
    """
    Stable normalization to reduce duplicates:
      - strip tracking query params (utm_*, gclid, fbclid, icmp, ocid)
      - lowercase host
      - drop fragment
      - keep other query params but sorted
      - remove trailing slash for non-root paths
    """
    if not u:
        return u
    # first pass: project-specific tracking cleanup
    u = strip_tracking(u)
    p = urlparse(u)
    keep_q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
              if not k.lower().startswith(("utm_", "gclid", "fbclid", "icmp", "ocid"))]
    new_q = urlencode(sorted(keep_q), doseq=True)
    path = p.path.rstrip("/") if p.path not in ("", "/") else p.path
    p = p._replace(netloc=p.netloc.lower(), query=new_q, fragment="", path=path)
    return urlunparse(p)

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
    try:
        from .crawl_fetcher import FetchConfig  # type: ignore
        return FetchConfig(user_agent=ua, obey_robots=obey, js_render=jsr)
    except Exception:
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
    try:
        return fetcher.get_html(url, cfg)  # type: ignore
    except TypeError:
        pass
    try:
        return fetcher.get_html(url, js_render=jsr, obey_robots=obey, user_agent=ua)  # type: ignore
    except TypeError:
        pass
    try:
        return fetcher.get_html(url, user_agent=ua)  # type: ignore
    except TypeError:
        pass
    return fetcher.get_html(url)


def _find_existing_article(sess, url_norm: str, canon_norm: Optional[str]):
    """
    Return an Article ORM instance if either url OR canonical_url matches.
    """
    if canon_norm:
        q = select(Article).where(or_(Article.url == url_norm,
                                      Article.canonical_url == canon_norm))
    else:
        q = select(Article).where(Article.url == url_norm)
    return sess.execute(q).scalars().first()


# ---------------------------- main ------------------------------------

def run_source(src_cfg: Dict):
    client = Crawl4AIClient()
    fetcher = _make_fetcher(client, src_cfg)
    ua = src_cfg["compliance"].get("user_agent")
    obey = src_cfg["compliance"].get("obey_robots", True)
    jsr = src_cfg["article"].get("js_render", False)

    try:
        raw_urls = discover_urls(src_cfg, fetcher)
    except Exception as e:
        log.exception("discovery failed for %s: %s", src_cfg.get("name"), e)
        return

    # Per-run guard against duplicates
    seen_urls: Set[str] = set()

    for raw in raw_urls:
        # normalize early and skip dups within this run
        url_norm = _normalize_url(raw)
        if url_norm in seen_urls:
            continue
        seen_urls.add(url_norm)

        try:
            html = _get_html(fetcher, url_norm, ua, obey, jsr)
            if not html:
                continue

            # Canonical (normalize after extraction)
            canon = canonical_url_from_html(url_norm, html) or url_norm
            canon_norm = _normalize_url(canon)

            data = extract_article(html, src_cfg["article"])
            title = (data.get("title") or "").strip()
            body = (data.get("body_text") or "").strip()
            if not title or not body:
                continue

            key = simple_key(title, body)

            with session_scope() as s:
                # Fast path: already in DB?
                existing = _find_existing_article(s, url_norm, canon_norm)
                if existing:
                    # Soft-update missing fields
                    changed = False
                    if not existing.title and title:
                        existing.title = title[:512]; changed = True
                    if not existing.body_text and body:
                        existing.body_text = body; changed = True
                    if not existing.published_at and data.get("published_at"):
                        existing.published_at = _safe_dt(data.get("published_at")); changed = True
                    if changed:
                        s.flush()
                    continue

                # Secondary dedup (hash/canonical/etc.)
                dup, _ = is_duplicate(s, url_norm, canon_norm, key)
                if dup:
                    continue

                art = Article(
                    source=src_cfg["name"],
                    url=url_norm,
                    canonical_url=canon_norm,
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
                try:
                    s.flush()
                except IntegrityError:
                    # Race: another worker inserted the same URL after our check.
                    s.rollback()
                    # Optional: fill blanks on the freshly inserted row.
                    other = _find_existing_article(s, url_norm, canon_norm)
                    if other:
                        changed = False
                        if not other.title and title:
                            other.title = title[:512]; changed = True
                        if not other.body_text and body:
                            other.body_text = body; changed = True
                        if not other.published_at and data.get("published_at"):
                            other.published_at = _safe_dt(data.get("published_at")); changed = True
                        if changed:
                            s.flush()
                    continue

        except Exception as e:
            log.exception("failed to process %s: %s", raw, e)
            continue
