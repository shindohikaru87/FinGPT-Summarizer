#!/usr/bin/env python3
# scripts/smoke_scrape.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import shorten
from typing import Dict, Any

import yaml

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add repo root to path
from src.ingestion.crawl_fetcher import CrawlFetcher, FetchConfig, FetchFailed, FetchDisallowedByRobots
from src.ingestion.discovery import discover_urls
from src.ingestion.extractor import extract_article
from src.ingestion.canonicalize import canonical_url_from_html, strip_tracking

MAX_PER_SOURCE = 3

def load_sources(yaml_path: Path) -> list[Dict[str, Any]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_source(src_cfg: Dict[str, Any], fetcher: CrawlFetcher) -> Dict[str, Any]:
    results = {"name": src_cfg["name"], "discovered": 0, "tested": 0, "ok": 0, "errors": []}
    ua = src_cfg.get("compliance", {}).get("user_agent") or "MarketNewsSummarizerBot/0.1"
    rate = src_cfg.get("rate", {}).get("per_minute", 30)
    burst = src_cfg.get("rate", {}).get("burst", 6)

    class _ShimFetcher:
        """Adapter so discovery.py can call fetcher.get_html(url, FetchConfig)."""
        def __init__(self, inner: CrawlFetcher):
            self._inner = inner

        # IMPORTANT: match discovery.py signature exactly
        def get_html(self, url, cfg):
            return self._inner.get_html(url, cfg)

    try:
        urls = discover_urls(src_cfg, fetcher=_ShimFetcher(fetcher))
    except Exception as e:
        results["errors"].append(f"discovery_error: {e}")
        return results

    results["discovered"] = len(urls)
    urls = urls[:MAX_PER_SOURCE]

    if not urls:
        print(f"\n[{src_cfg['name']}] No URLs discovered")
        return results

    print(f"\n=== {src_cfg['name']} ===")
    for url in urls:
        try:
            html = fetcher.get_html(
                url,
                FetchConfig(
                    js_render=src_cfg.get("article", {}).get("js_render", False),
                    user_agent=ua,
                    obey_robots=src_cfg.get("compliance", {}).get("obey_robots", True),
                    timeout_ms=25000,
                    per_host_rate_per_min=rate,
                    per_host_burst=burst,
                ),
            )
            canon = canonical_url_from_html(url, html)
            data = extract_article(html, src_cfg["article"])

            title = (data.get("title") or "").strip()
            body  = (data.get("body_text") or "").strip()
            ok = bool(title) and len(body) > 200

            results["tested"] += 1
            if ok:
                results["ok"] += 1
                status = "OK"
            else:
                status = "WEAK"
                preview = body[:140].replace("\n", " ")
                results["errors"].append(
                    f"weak_extract title={bool(title)} len(body)={len(body)} url={strip_tracking(url)} preview={preview}"
                )

            # Always print the article summary
            print(f"\n[{status}] {strip_tracking(url)}")
            print(f"Canonical: {canon}")
            print(f"Title    : {shorten(title, width=110)}")
            print(f"Body     : {shorten(body.replace('\\n',' '), width=200)}")

        except FetchDisallowedByRobots as e:
            msg = f"robots_disallow: {e.url}"
            results["errors"].append(msg)
            print(f"[ERROR][{src_cfg['name']}] {msg}")

        except FetchFailed as e:
            msg = f"fetch_failed: {e.url} reason={e.reason} status={e.status}"
            results["errors"].append(msg)
            print(f"[ERROR][{src_cfg['name']}] {msg}")

        except Exception as e:
            msg = f"extract_error: {strip_tracking(url)} {e.__class__.__name__}: {e}"
            results["errors"].append(msg)
            print(f"[ERROR][{src_cfg['name']}] {msg}")


    return results

def main():
    yaml_path = Path("src/ingestion/sources.yaml")
    sources = load_sources(yaml_path)
    print(f"Loaded {len(sources)} sources from {yaml_path}")
    print("Using CrawlFetcher for fetching (with rate limiting, retries, robots.txt)")
    print("Starting smoke test...\n")
    
    fetcher = CrawlFetcher()

    summary = []
    for src in sources:
        print(f"\n=== Testing source: {src['name']} ===")
        r = test_source(src, fetcher)
        summary.append(r)

    print("\n=== SUMMARY ===")
    for r in summary:
        print(f"- {r['name']}: discovered={r['discovered']} tested={r['tested']} ok={r['ok']} errors={len(r['errors'])}")
        for err in r["errors"][:3]:  # show only first 3 errors per source
            print(f"    • {err}")

    # exit non-zero if lots of failures (for CI)
    total = sum(r["tested"] for r in summary)
    oks = sum(r["ok"] for r in summary)
    if total == 0 or oks == 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
