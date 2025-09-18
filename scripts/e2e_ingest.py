#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root on sys.path so "from src.ingestion ..." imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.ingestion import pipeline  # we MUST go via pipeline.run_source

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _print_header(txt: str):
    print("\n" + "=" * 90)
    print(txt)
    print("=" * 90)

def _load_sources(sources_path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("sources", [])

def _cap_discovery(src_cfg: Dict[str, Any], cap: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(src_cfg)
    disc = cfg.setdefault("discovery", {})
    current = disc.get("max_new_urls", cap)
    try:
        disc["max_new_urls"] = min(int(current), cap)
    except Exception:
        disc["max_new_urls"] = cap
    return cfg

# ----------------------- DRY-RUN monkeypatch helpers --------------------------

class _FakeArticle:
    def __init__(self, **kw):  # mimic SQLAlchemy model init
        self.__dict__.update(kw)
        self.id = None

class _MemSession:
    def __init__(self, sink: List[dict]):
        self._sink = sink
        self._pending = None

    def add(self, obj):
        self._pending = obj

    def flush(self):
        if self._pending:
            self._pending.id = len(self._sink) + 1
            # snapshot of fields similar to DB row for printing
            snap = {
                "id": self._pending.id,
                "source": getattr(self._pending, "source", None),
                "url": getattr(self._pending, "url", None),
                "canonical_url": getattr(self._pending, "canonical_url", None),
                "title": getattr(self._pending, "title", None),
                "author": getattr(self._pending, "author", None),
                "published_at": getattr(self._pending, "published_at", None),
                "first_seen_at": getattr(self._pending, "first_seen_at", None),
                "status": getattr(self._pending, "status", None),
            }
            self._sink.append(snap)
            self._pending = None

    def close(self):  # compatibility with context manager cleanup
        pass

class _MemSessionScope:
    def __init__(self, sink: List[dict]):
        self._sink = sink

    def __call__(self):
        # allow call like a function returning a context manager
        return self

    def __enter__(self):
        return _MemSession(self._sink)

    def __exit__(self, exc_type, exc, tb):
        return False

# ---------------------------- DB printing helper ------------------------------

def _print_db_inserts_since(start_ts: datetime, source_name: str, show_n: int):
    try:
        # Import only if available
        from app.db import session_scope  # type: ignore
        from app.models import Article    # type: ignore
    except Exception:
        print("[DB] app.db/app.models not importable; cannot show DB results.")
        return

    try:
        with session_scope() as s:
            # SQLAlchemy 2.0 style Query (works with both styles if configured)
            rows = (
                s.query(Article)
                 .filter(Article.source == source_name)
                 .filter(Article.first_seen_at >= start_ts)
                 .order_by(Article.id.desc())
                 .limit(show_n)
                 .all()
            )
        if not rows:
            print(f"[DB] No rows inserted for {source_name} since {start_ts.isoformat()}.")
            return
        print(f"[DB] Showing up to {show_n} new rows for {source_name}:")
        for r in rows:
            ts = getattr(r, "published_at", None)
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else "NA"
            print(f"  • ({r.id}) {ts_str}  {r.title}  — {r.url}")
    except Exception as e:
        print(f"[DB] Query failed: {e}")

# ---------------------------------- MAIN --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E2E runner via pipeline.run_source using sources.yaml")
    ap.add_argument("--sources-file", default=str(Path("src/ingestion/sources.yaml")), help="Path to sources.yaml")
    ap.add_argument("--source", nargs="*", help="Only run these source names (exact match).")
    ap.add_argument("--limit", type=int, default=8, help="Cap discovery.max_new_urls per source during this run.")
    ap.add_argument("--dry-run", action="store_true", help="Do not touch DB; capture inserts in-memory.")
    ap.add_argument("--show", type=int, default=5, help="Show up to N results per source (DB mode or dry-run sink).")
    args = ap.parse_args()

    sources_path = Path(args.sources_file).resolve()
    if not sources_path.exists():
        print(f"sources.yaml not found at: {sources_path}")
        sys.exit(1)

    all_sources = _load_sources(sources_path)
    if not all_sources:
        print("No sources loaded from sources.yaml")
        sys.exit(1)

    if args.source:
        names = set(args.source)
        all_sources = [s for s in all_sources if s.get("name") in names]
        if not all_sources:
            print(f"No sources matched: {args.source}")
            sys.exit(1)

    _print_header("E2E INGESTION RUN (via pipeline.run_source)")
    print(f"Sources: {[s['name'] for s in all_sources]}")
    print(f"Mode   : {'DRY-RUN' if args.dry_run else 'DB'}")
    print(f"Limit  : discovery.max_new_urls = {args.limit}")

    totals = []
    for src in all_sources:
        name = src["name"]
        src_cfg = _cap_discovery(src, args.limit)

        _print_header(f"SOURCE: {name}")
        start_ts = _utc_now()

        if args.dry_run:
            # Monkey-patch pipeline’s DB touch points
            sink: List[dict] = []
            original_article = getattr(pipeline, "Article", None)
            original_session = getattr(pipeline, "session_scope", None)
            try:
                setattr(pipeline, "Article", _FakeArticle)
                setattr(pipeline, "session_scope", _MemSessionScope(sink))
                t0 = time.time()
                pipeline.run_source(src_cfg)
                dur = time.time() - t0
                print(f"[{name}] dry-run inserted: {len(sink)} rows in {dur:.1f}s")
                for row in sink[:args.show]:
                    ts = row.get("published_at")
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else "NA"
                    print(f"  • ({row['id']}) {ts_str}  {row['title']}  — {row['url']}")
                totals.append((name, len(sink), dur))
            finally:
                # Restore originals
                if original_article is not None:
                    setattr(pipeline, "Article", original_article)
                if original_session is not None:
                    setattr(pipeline, "session_scope", original_session)
        else:
            # DB mode: run the real pipeline; then query DB for new rows since start_ts
            t0 = time.time()
            pipeline.run_source(src_cfg)
            dur = time.time() - t0
            _print_db_inserts_since(start_ts, name, args.show)
            # We can’t easily count how many without a before/after delta, but prints suffice
            totals.append((name, None, dur))

    _print_header("SUMMARY")
    for (name, count, secs) in totals:
        if count is None:
            print(f"- {name}: completed in {secs:.1f}s")
        else:
            print(f"- {name}: inserted {count} (dry-run) in {secs:.1f}s")

if __name__ == "__main__":
    main()
