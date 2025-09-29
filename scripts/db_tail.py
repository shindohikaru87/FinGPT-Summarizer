#!/usr/bin/env python3
"""
Tail the most recent Article rows with summary, embedding, and cluster info.

Examples:

  # === Basic article tailing ===
  python scripts/db_tail.py
  python scripts/db_tail.py --n 20
  python scripts/db_tail.py --source cnbc-markets --n 10
  python scripts/db_tail.py --status READY_FOR_SUMMARY --since-hours 6
  python scripts/db_tail.py --show-body --body-chars 400 --show-summary --summary-chars 400
  python scripts/db_tail.py --json

  # === Clusters ===
  # Show cluster label for most recent run
  python scripts/db_tail.py --latest-run

  # Show cluster label for a specific run_id
  python scripts/db_tail.py --run-id 1695000000

  # Group articles by cluster (most recent run)
  python scripts/db_tail.py --latest-run --group-by-cluster

  # Group clusters, showing only 3 articles per cluster
  python scripts/db_tail.py --latest-run --group-by-cluster --per-cluster 3

  # Group clusters, only include clusters with ≥5 members
  python scripts/db_tail.py --latest-run --group-by-cluster --min-cluster-size 5

  # Group clusters, filter to labels containing 'oil'
  python scripts/db_tail.py --latest-run --group-by-cluster --cluster-label oil

  # Group clusters for a specific run_id with summary previews
  python scripts/db_tail.py --run-id 1695000000 --group-by-cluster --show-summary --summary-chars 200

Environment variables:
  DATABASE_URL, SQL_ECHO
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, func, and_, desc  # noqa: E402
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
    row = sess.execute(select(func.max(Cluster.run_id))).first()
    return row[0] if row and row[0] is not None else None


def _latest_summary_alias():
    sq_latest_summary = (
        select(
            Summary.article_id.label("article_id"),
            func.max(Summary.created_at).label("max_created"),
        )
        .group_by(Summary.article_id)
        .subquery()
    )
    S = aliased(Summary)
    return sq_latest_summary, S


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
    sq_latest_summary, S = _latest_summary_alias()

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
            S.model.label("summary_model"),
            S.provider.label("summary_provider"),
            S.created_at.label("summary_created_at"),
            S.summary_text.label("summary_text"),
            Embedding.model.label("embed_model"),
            Embedding.provider.label("embed_provider"),
            Embedding.created_at.label("embed_created_at"),
            Embedding.vector.label("embed_vector"),
        )
        .join(sq_latest_summary, sq_latest_summary.c.article_id == Article.id, isouter=True)
        .join(
            S,
            and_(
                S.article_id == sq_latest_summary.c.article_id,
                S.created_at == sq_latest_summary.c.max_created,
            ),
            isouter=True,
        )
        .join(Embedding, Embedding.article_id == Article.id, isouter=True)
    )

    if run_id is not None:
        q = q.add_columns(
            Cluster.run_id.label("cluster_run_id"),
            Cluster.label.label("cluster_label"),
        ).join(ArticleCluster, ArticleCluster.article_id == Article.id, isouter=True
        ).join(
            Cluster,
            and_(
                Cluster.id == ArticleCluster.cluster_id,
                Cluster.run_id == run_id,
            ),
            isouter=True,
        )

    if source:
        q = q.where(Article.source == source)
    if status:
        q = q.where(Article.status == status)
    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        q = q.where(Article.first_seen_at >= cutoff)

    q = q.order_by(Article.first_seen_at.desc()).limit(n)

    rows: List[Dict[str, Any]] = []
    for row in sess.execute(q).all():
        row = row._mapping
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
            "cluster": None,
        }

        if run_id is not None:
            payload["cluster"] = {
                "run_id": row.get("cluster_run_id"),
                "label": row.get("cluster_label"),
            }

        rows.append(payload)

    return rows


def _fetch_clusters_with_articles(
    sess,
    run_id: int,
    cluster_label_filter: Optional[str],
    min_cluster_size: int,
    per_cluster: int,
    include_summary_text: bool,
    summary_chars: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return a list of clusters with their member articles.

    Structure:
    [
      {
        "label": "Earnings Season",
        "run_id": 1695...,
        "size": 12,
        "articles": [
          { "id": ..., "title": ..., "url": ..., "published_at": ..., "summary": "..." },
          ...
        ]
      },
      ...
    ]
    """
    if run_id is None:
        return [], 0

    sq_latest_summary, S = _latest_summary_alias()

    # Get all cluster memberships for this run, along with article metadata and latest summary fields
    q = (
        select(
            Cluster.label.label("cluster_label"),
            Cluster.run_id.label("run_id"),
            Article.id.label("article_id"),
            Article.title.label("title"),
            Article.url.label("url"),
            Article.published_at.label("published_at"),
            S.summary_text.label("summary_text"),
        )
        .join(ArticleCluster, Cluster.id == ArticleCluster.cluster_id)
        .join(Article, ArticleCluster.article_id == Article.id)
        .join(sq_latest_summary, sq_latest_summary.c.article_id == Article.id, isouter=True)
        .join(
            S,
            and_(
                S.article_id == sq_latest_summary.c.article_id,
                S.created_at == sq_latest_summary.c.max_created,
            ),
            isouter=True,
        )
        .where(Cluster.run_id == run_id)
        .order_by(Cluster.label.asc(), desc(Article.published_at))
    )

    if cluster_label_filter:
        like = f"%{cluster_label_filter.lower()}%"
        # case-insensitive filter; SQLite uses lower() function
        q = q.where(func.lower(Cluster.label).like(like))

    rows = sess.execute(q).all()

    # Group members by cluster
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        m = r._mapping
        grouped[m["cluster_label"]].append(
            {
                "id": m["article_id"],
                "title": m["title"],
                "url": m["url"],
                "published_at": iso(m["published_at"]),
                "summary": None if not include_summary_text else (
                    (m["summary_text"] or "").strip().replace("\n", " ")
                ),
            }
        )

    # Build cluster list with counts and apply min size & per-cluster limits
    clusters: List[Dict[str, Any]] = []
    total_articles = 0
    for label, members in sorted(grouped.items(), key=lambda kv: kv[0].lower()):
        if len(members) < min_cluster_size:
            continue
        # Limit members
        trimmed = members[: max(1, per_cluster)]
        total_articles += len(trimmed)
        # Truncate summary if needed
        if include_summary_text and summary_chars and summary_chars > 0:
            for m in trimmed:
                if m["summary"]:
                    if len(m["summary"]) > summary_chars:
                        m["summary"] = m["summary"][:summary_chars] + "…"
        clusters.append(
            {
                "label": label,
                "run_id": run_id,
                "size": len(members),
                "articles": trimmed,
            }
        )

    return clusters, total_articles


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
    ap.add_argument("--latest-run", action="store_true", help="Use the most recent clusters run")
    ap.add_argument("--run-id", type=int, help="Use a specific cluster run_id")

    # NEW: Group-by-cluster mode
    ap.add_argument("--group-by-cluster", action="store_true", help="Group output by cluster label")
    ap.add_argument("--per-cluster", type=int, default=5, help="Max articles per cluster to print (default: 5)")
    ap.add_argument("--min-cluster-size", type=int, default=1, help="Only print clusters with at least this many members")
    ap.add_argument("--cluster-label", type=str, help="Filter clusters whose label contains this substring (case-insensitive)")

    args = ap.parse_args()

    with session_scope() as s:
        # Resolve run_id if needed
        run_id = args.run_id
        if args.latest_run and run_id is None:
            run_id = _latest_cluster_run_id(s)

        if args.group_by_cluster:
            if run_id is None:
                print("⚠️  --group-by-cluster requires --latest-run or --run-id")
                return
            clusters, total = _fetch_clusters_with_articles(
                s,
                run_id=run_id,
                cluster_label_filter=args.cluster_label,
                min_cluster_size=args.min_cluster_size,
                per_cluster=args.per_cluster,
                include_summary_text=args.show_summary,
                summary_chars=args.summary_chars,
            )
            if not clusters:
                print(f"[cluster run_id={run_id}] No clusters found (check --cluster-label or --min-cluster-size).")
                return

            print(f"[cluster run_id={run_id}] Showing up to {args.per_cluster} articles per cluster\n")
            for c in clusters:
                print(f"## {c['label']}  (size={c['size']})")
                for m in c["articles"]:
                    pub = m["published_at"] or "NA"
                    print(f"- {m['title']}  [{pub}]")
                    print(f"  {m['url']}")
                    if args.show_summary and m["summary"]:
                        print(f"  SUMMARY: {m['summary']}")
                print("-" * 100)
            return

        # Default (original) flat listing
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

        if s["model"]:
            print(f"  SUMMARY: {s['model']}@{s['provider']}  at {s['created_at']}")
            if args.show_summary and s["text"]:
                txt = s["text"].strip().replace("\n", " ")
                if len(txt) > args.summary_chars:
                    txt = txt[:args.summary_chars] + "…"
                print(f"    {txt}")
        else:
            print("  SUMMARY: (none)")

        if e["exists"]:
            dim = e["dim"] if e["dim"] is not None else "?"
            print(f"  EMBEDDING: {e['model']}@{e['provider']}  dim={dim}  at {e['created_at']}")
        else:
            print("  EMBEDDING: (none)")

        if c and c.get("run_id") is not None:
            print(f"  CLUSTER: run_id={c['run_id']}  label={c['label']}")
        else:
            print("  CLUSTER: (none)")

        if args.show_body and a.get("body_text"):
            body = (a["body_text"] or "").strip().replace("\n", " ")
            if len(body) > args.body_chars:
                body = body[:args.body_chars] + "…"
            print(f"  BODY: {body}")

        print("-" * 100)


if __name__ == "__main__":
    main()
