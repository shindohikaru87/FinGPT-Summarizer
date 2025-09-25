#!/usr/bin/env python3
"""
Tail the most recent Article rows with summary/embedding/cluster info.

Usage:
  python scripts/db_tail.py
  python scripts/db_tail.py --n 20
  python scripts/db_tail.py --source cnbc-markets --n 10
  python scripts/db_tail.py --status READY_FOR_SUMMARY --since-hours 6
  python scripts/db_tail.py --latest-run
  python scripts/db_tail.py --run-id 1695000000
  python scripts/db_tail.py --show-body --body-chars 400 --show-summary --summary-chars 400
  python scripts/db_tail.py --json

Env:
  DATABASE_URL, SQL_ECHO
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, func, and_  # noqa: E402
from sqlalchemy.orm import aliased  # noqa: E402

from src.app.db import session_scope  # noqa: E402
from src.app.models import (  # noqa: E402
    Article,
    Summary,
    Embedding,
    Cluster,
    ArticleCluster,
)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else None


def _latest_cluster_run_id(sess) -> Optional[int]:
    """Return the most recent Cluster.run_id, or None if no clusters exist."""
    row = sess.execute(select(func.max(Cluster.run_id))).first()
    return row[0] if row and row[0] is not None else None


def _fetch_rows(
    sess,
    n: int,
    source: Optional[str],
    status: Optional[str],
    since_hours: Optional[float],
    run_id: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with article + latest summary + embedding + cluster info.
    """
    # Subquery: latest summary per article
    sq_latest_summary = (
        select(
            Summary.article_id.label("article_id"),
            func.max(Summary.created_at).label("max_created"),
        )
        .group_by(Summary.article_id)
        .subquery()
    )
    S = aliased(Summary)
    # Subquery join gives us the latest Summary row per article (if any)
    # We'll left join it in the main query.

    # Base query on Article
    q = (
        select(
            Article.id,
            Article.source,
            Article.title,
            Article.url,
            Article.canonical_url,
            Article.author,
            Article.published_at,
            Article.first_seen_at,
            Article.status,
            Article.lang,
            Article.body_text,
            # Latest summary fields (if any)
            S.model.label("summary_model"),
            S.provider.label("summary_provider"),
            S.created_at.label("summary_created_at"),
            S.summary_text.label("summary_text"),
            # Embedding fields (if any)
            Embedding.model.label("embed_model"),
            Embedding.provider.label("embed_provider"),
            Embedding.created_at.label("embed_created_at"),
            Embedding.vector.label("embed_vector"),
        )
        # Latest summary LEFT JOIN
        .join(
            sq_latest_summary,
            sq_latest_summary.c.article_id == Article.id,
            isouter=True,
        )
        .join(
            S,
            and_(
                S.article_id == sq_latest_summary.c.article_id,
                S.created_at == sq_latest_summary.c.max_created,
            ),
            isouter=True,
        )
        # Embedding LEFT JOIN (one per article by schema)
        .join(
            Embedding,
            Embedding.article_id == Article.id,
            isouter=True,
        )
    )

    # If cluster run_id is specified, bring cluster label via ArticleCluster -> Cluster
    if run_id is not None:
        q = q.add_columns(
            Cluster.run_id.label("cluster_run_id"),
            Cluster.label.label("cluster_label"),
        ).join(
            ArticleCluster,
            ArticleCluster.article_id == Article.id,
            isouter=True,
        ).join(
            Cluster,
            and_(
                Cluster.id == ArticleCluster.cluster_id,
                Cluster.run_id == run_id,
            ),
            isouter=True,
        )

    # Filters
    if source:
        q = q.where(Article.source == source)
    if status:
        q = q.where(Article.status == status)
    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        q = q.where(Article.first_seen_at >= cutoff)

    q = q.order_by(Article.first_seen_at.desc()).limit(n)

    rows = []
    for row in sess.execute(q).all():
        row = row._mapping  # name-based access

        # Derive embedding dim if vector present (vector is JSON list[float] on SQLite)
        embed_vec = row.get("embed_vector")
        embed_dim = len(embed_vec) if isinstance(embed_vec, list) else None

        payload: Dict[str, Any] = {
            "article": {
                "id": row["id"],
                "source": row["source"],
                "title": row["title"],
                "url": row["url"],
                "canonical_url": row["canonical_url"],
                "author": row["author"],
                "published_at": iso(row["published_at"]),
                "first_seen_at": iso(row["first_seen_at"]),
                "status": row["status"],
                "lang": row["lang"],
                "body_text": row["body_text"],
            },
            "summary": {
                "model": row.get("summary_model"),
                "provider": row.get("summary_provider"),
                "created_at": iso(row.get("summary_created_at")),
                "text": row.get("summary_text"),
            },
            "embedding": {
                "exists": row.get("embed_model") is not None,
                "model": row.get("embed_model"),
                "provider": row.get("embed_provider"),
                "created_at": iso(row.get("embed_created_at")),
                "dim": embed_dim,
            },
        }

        if run_id is not None:
            payload["cluster"] = {
                "run_id": row.get("cluster_run_id"),
                "label": row.get("cluster_label"),
            }
        else:
            payload["cluster"] = None

        rows.append(payload)

    return rows


def main():
    ap = argparse.ArgumentParser(description="Tail recent Article rows with summary/embedding/cluster info.")
    ap.add_argument("--n", type=int, default=10, help="How many rows to show (default: 10)")
    ap.add_argument("--source", help="Filter by source name")
    ap.add_argument("--status", help="Filter by status value")
    ap.add_argument("--since-hours", type=float, help="Only rows with first_seen_at in last N hours")
    ap.add_argument("--json", action="store_true", help="Output as JSON lines")

    # Output customization
    ap.add_argument("--show-body", action="store_true", help="Show a preview of Article.body_text")
    ap.add_argument("--body-chars", type=int, default=280, help="Chars to show from body preview (default: 280)")
    ap.add_argument("--show-summary", action="store_true", help="Show a preview of the latest summary text")
    ap.add_argument("--summary-chars", type=int, default=280, help="Chars to show from summary preview (default: 280)")

    # Cluster selection
    ap.add_argument("--latest-run", action="store_true", help="Show cluster label from the most recent run")
    ap.add_argument("--run-id", type=int, help="Show cluster label from a specific run_id")

    args = ap.parse_args()

    with session_scope() as s:
        run_id = args.run_id
        if args.latest_run and run_id is None:
            run_id = _latest_cluster_run_id(s)

        rows = _fetch_rows(
            s,
            n=args.n,
            source=args.source,
            status=args.status,
            since_hours=args.since_hours,
            run_id=run_id,
        )

    if args.json:
        for r in rows:
            # Optionally truncate previews in JSON too (keeps payload light)
            if not args.show_body and r["article"].get("body_text") is not None:
                r["article"]["body_text"] = None
            if not args.show_summary and r["summary"].get("text") is not None:
                r["summary"]["text"] = None
            print(json.dumps(r, ensure_ascii=False))
        return

    if not rows:
        print("No rows found.")
        return

    hdr = f"[cluster run_id={run_id}]" if run_id is not None else "[cluster: (none)]"
    print(f"{hdr}\n")

    for r in rows:
        a = r["article"]
        s = r["summary"]
        e = r["embedding"]
        c = r["cluster"]

        pub = a["published_at"] or "NA"
        seen = a["first_seen_at"] or "NA"

        print(f"({a['id']}) [{a['source']}] pub={pub}  seen={seen}  status={a['status']}")
        print(f"  TITLE: {a['title']}")
        print(f"  URL:   {a['url']}")
        if a.get("canonical_url"):
            print(f"  CANON: {a['canonical_url']}")

        # Summary line
        if s["model"]:
            print(f"  SUMMARY: {s['model']}@{s['provider']}  at {s['created_at']}")
            if args.show_summary and s["text"]:
                txt = s["text"].strip().replace("\n", " ")
                if len(txt) > args.summary_chars:
                    txt = txt[:args.summary_chars] + "…"
                print(f"    {txt}")
        else:
            print("  SUMMARY: (none)")

        # Embedding line
        if e["exists"]:
            dim = e["dim"] if e["dim"] is not None else "?"
            print(f"  EMBEDDING: {e['model']}@{e['provider']}  dim={dim}  at {e['created_at']}")
        else:
            print("  EMBEDDING: (none)")

        # Cluster line
        if c and c.get("run_id") is not None:
            print(f"  CLUSTER: run_id={c['run_id']}  label={c['label']}")
        else:
            print("  CLUSTER: (none)")

        # Body preview
        if args.show_body and a.get("body_text"):
            body = (a["body_text"] or "").strip().replace("\n", " ")
            if len(body) > args.body_chars:
                body = body[:args.body_chars] + "…"
            print(f"  BODY: {body}")

        print("-" * 100)


if __name__ == "__main__":
    main()
