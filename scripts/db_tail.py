#!/usr/bin/env python3
"""
Tail the most recent Article rows.

Usage:
  python scripts/db_tail.py
  python scripts/db_tail.py --n 20
  python scripts/db_tail.py --source cnbc-markets --n 10
  python scripts/db_tail.py --status READY_FOR_SUMMARY --since-hours 6
  python scripts/db_tail.py --json

Env:
  DATABASE_URL, SQL_ECHO (same as db_init.py)
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.db import session_scope
from src.app.models import Article


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else None

def main():
    ap = argparse.ArgumentParser(description="Tail recent Article rows.")
    ap.add_argument("--n", type=int, default=10, help="How many rows to show (default: 10)")
    ap.add_argument("--source", help="Filter by source name")
    ap.add_argument("--status", help="Filter by status value")
    ap.add_argument("--since-hours", type=float, help="Only rows with first_seen_at in last N hours")
    ap.add_argument("--json", action="store_true", help="Output as JSON lines")
    args = ap.parse_args()

    since_dt = None
    if args.since_hours:
        since_dt = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)

    with session_scope() as s:
        q = s.query(Article)
        if args.source:
            q = q.filter(Article.source == args.source)
        if args.status:
            q = q.filter(Article.status == args.status)
        if since_dt:
            q = q.filter(Article.first_seen_at >= since_dt)
        q = q.order_by(Article.first_seen_at.desc()).limit(args.n)

        rows = q.all()

    if args.json:
        for r in rows:
            print(json.dumps({
                "id": r.id,
                "source": r.source,
                "title": r.title,
                "url": r.url,
                "canonical_url": r.canonical_url,
                "author": r.author,
                "published_at": iso(r.published_at),
                "first_seen_at": iso(r.first_seen_at),
                "status": r.status,
                "lang": r.lang,
            }, ensure_ascii=False))
        return

    if not rows:
        print("No rows found.")
        return

    for r in rows:
        pub = iso(r.published_at) or "NA"
        seen = iso(r.first_seen_at) or "NA"
        print(f"({r.id}) [{r.source}] {pub}")
        print(f"  {r.title}")
        print(f"  {r.url}")
        print(f"  status={r.status}  first_seen_at={seen}")
        print("-" * 80)

if __name__ == "__main__":
    main()
