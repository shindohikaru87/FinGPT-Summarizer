# scripts/summarize.py
from __future__ import annotations

try:
    from scripts._bootstrap_env import load_env  # tiny zero-dep loader we added earlier
    load_env()
except Exception:
    pass

import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from src.app.db import session_scope
from src.app.models import Article, Summary

# OpenAI SDK v1.x
from openai import OpenAI

PROVIDER = "OPENAI"
DEFAULT_MODEL = "gpt-4o-mini"


def _build_prompt(title: str, body: str) -> List[dict]:
    """
    System+user message pair for market-news summarization.
    Keeps it concise, structured, and useful for investors.
    """
    # Hard cap body to ~6000 chars to keep request lightweight
    body = (body or "").strip()
    if len(body) > 6000:
        body = body[:6000] + "…"

    system_msg = (
        "You are a financial news summarizer for busy investors. "
        "Write a crisp summary with 4–7 bullet points and a one-line headline. "
        "Focus on: what happened, numbers, guidance, market impact, and risks. "
        "Keep it neutral, specific, and free of hype. No extra preamble."
    )
    user_msg = f"""Title: {title or "(untitled)"}

Article:
{body}

Return JSON with fields:
- headline: one sentence
- bullets: array of 4–7 concise points
- takeaway: one sentence on likely market impact
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _summarize_text(client: OpenAI, model: str, title: str, body: str, retries: int = 3) -> str:
    """
    Call OpenAI to produce a compact JSON-like summary string.
    Returns the raw text (we store as-is in `summary_text`).
    """
    messages = _build_prompt(title, body)
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("rate limit", "429", "temporarily unavailable", "502", "503", "504")) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(15.0, delay * 1.7)
                continue
            raise

def _fetch_candidate_articles(
    sess: Session,
    limit: int,
    since_hours: Optional[int],
    model: str,
    overwrite: bool,
) -> List[Tuple[int, str, str]]:
    """
    Returns up to `limit` rows of (article_id, title, body_text) that
    either (a) have no summary for `model`, or (b) overwrite=True (pick latest articles).
    """
    q = select(Article.id, Article.title, Article.body_text)

    # Filter by published recency if provided
    if since_hours and since_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        q = q.where(Article.published_at.is_(None) | (Article.published_at >= cutoff))

    # Only articles with some body text
    q = q.where(Article.body_text.isnot(None))

    # If not overwriting, exclude articles that already have a summary for this model
    if not overwrite:
        exists_q = (
            select(func.count(Summary.id))
            .where(and_(Summary.article_id == Article.id, Summary.model == model))
            .correlate(Article)
        )
        q = q.where(exists_q.as_scalar() == 0)  # no summary for this model yet

    q = q.order_by(Article.published_at.desc().nullslast()).limit(limit)

    return [(row[0], row[1] or "", row[2] or "") for row in sess.execute(q).all()]

class Progress:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.start = time.perf_counter()
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.last_render_len = 0
        self.latencies: List[float] = []

    def tick(self, status: str, dt: Optional[float] = None):
        self.done += 1
        if status == "ok":
            self.ok += 1
        elif status == "fail":
            self.failed += 1
        elif status == "skip":
            self.skipped += 1
        if dt is not None:
            self.latencies.append(dt)
        self.render()

    def render(self, final: bool = False):
        pct = (100.0 * self.done / self.total)
        elapsed = time.perf_counter() - self.start
        avg = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        remaining = self.total - self.done
        eta = remaining * avg
        bar_len = 24
        filled = int(bar_len * self.done / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        line = f"\r[{bar}] {pct:6.2f}%  done={self.done}/{self.total}  ok={self.ok}  fail={self.failed}  skip={self.skipped}  avg={avg:0.2f}s  ETA={eta:0.0f}s"
        # Clear previous line if shorter
        pad = max(0, self.last_render_len - len(line))
        print(line + (" " * pad), end=("" if not final else "\n"), flush=True)
        self.last_render_len = len(line)

    def finish(self):
        self.render(final=True)


def _delete_existing_summaries(sess: Session, article_ids: List[int], model: str) -> None:
    if not article_ids:
        return
    sess.query(Summary).filter(
        Summary.model == model, Summary.article_id.in_(article_ids)
    ).delete(synchronize_session=False)


def summarize_batch(
    model: str,
    limit: int,
    since_hours: Optional[int],
    overwrite: bool,
) -> int:
    client = OpenAI()
    total_written = 0

    with session_scope() as sess:
        rows = _fetch_candidate_articles(
            sess, limit=limit, since_hours=since_hours, model=model, overwrite=overwrite
        )
        if not rows:
            print("No candidate articles found for summarization.")
            return 0

        print(f"Found {len(rows)} candidate articles to summarize...")
        prog = Progress(total=len(rows))

        # If overwriting, clear existing summaries for this model for the chosen set
        if overwrite:
            _delete_existing_summaries(sess, [r[0] for r in rows], model)
            print(f"\nDeleted existing summaries for {len(rows)} articles (model={model}).")

        for idx, (article_id, title, body) in enumerate(rows, start=1):
            if not body.strip():
                prog.tick("skip")
                continue

            t0 = time.perf_counter()
            try:
                text = _summarize_text(client, model=model, title=title, body=body)
                dt = time.perf_counter() - t0
            except Exception as e:
                # Brief per-item line on failure (kept short to avoid spam)
                print(f"\n  ⚠️  Error summarizing article {article_id}: {e}")
                prog.tick("fail")
                continue

            sess.add(
                Summary(
                    article_id=article_id,
                    provider=PROVIDER,
                    model=model,
                    summary_text=text,
                    highlights=None,
                    extra=None,
                    created_at=datetime.now(timezone.utc),
                )
            )
            total_written += 1
            prog.tick("ok", dt)

        prog.finish()

    return total_written


# --- inside main() ---
def main():
    ap = argparse.ArgumentParser(description="Summarize articles and store results in the 'summaries' table.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI chat model (default: {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, default=100, help="Max number of articles to summarize this run (default: 100)")
    ap.add_argument("--since-hours", type=int, default=None, help="Only consider articles published within the last N hours (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing summaries for this model (delete and re-create)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    n = summarize_batch(model=args.model, limit=args.limit, since_hours=args.since_hours, overwrite=args.overwrite)
    elapsed = time.perf_counter() - t0

    print(f"\n✅ Done. Summarized {n} articles using {PROVIDER}:{args.model}. Elapsed {elapsed:0.1f}s.\n")


if __name__ == "__main__":
    main()
