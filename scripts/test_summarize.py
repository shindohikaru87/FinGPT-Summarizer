#!/usr/bin/env python3
"""
Quick DB-based summarization test using OpenAI GPT.

- Pulls a few READY_FOR_SUMMARY articles
- Uses src/summarization/llm.py and summarize_news.j2
- Inserts into summaries table; sets article.status=SUMMARIZED

Usage:
  python scripts/test_summarize.py
"""

from __future__ import annotations
import asyncio
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import yaml
from jinja2 import Template
from sqlalchemy import select
from dotenv import load_dotenv

# --- repo root on sys.path ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --- package imports (ensure __init__.py exists in src/, src/app/, src/summarization/) ---
from src.app.db import session_scope
from src.app.models import Article, Summary
from src.summarization.llm import ModelProvider, get_model  # <-- NOTE the new import


# ----------------- helpers -----------------
def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(maybe_path: str | Path) -> Path:
    """Resolve a file path relative to the repo root if it's not absolute."""
    p = Path(maybe_path)
    return p if p.is_absolute() else (REPO_ROOT / p)

def load_prompt(path: str | Path) -> Template:
    p = resolve_path(path)
    return Template(p.read_text(encoding="utf-8"))

def render_user_prompt(tmpl: Template, a: Article) -> str:
    return tmpl.render(
        title=a.title,
        source=a.source,
        url=a.canonical_url or a.url,
        author=a.author,
        published_at=a.published_at.isoformat() if a.published_at else None,
        text=a.body_text,
    )

def extract_highlights(text: str, max_items: int = 6) -> List[str]:
    def clean(line: str) -> str:
        s = re.sub(r"^[\-\•\*\d\.\)\s]+", "", line.strip())
        return s
    lines = [clean(ln) for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if 2 <= len(ln.split()) <= 24]
    seen, out = set(), []
    for ln in lines:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
        if len(out) >= max_items:
            break
    return out

async def summarize_one(chat, sys_prompt: str, tmpl: Template, a: Article):
    user = render_user_prompt(tmpl, a)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]
    resp = await chat.ainvoke(messages)
    text = getattr(resp, "content", str(resp)).strip()
    return text, extract_highlights(text)


# ----------------- main -----------------
async def main():
    load_dotenv()  # load .env
    cfg_path = os.getenv("MNS_CONFIG", "config/config.yaml")
    cfg = load_config(resolve_path(cfg_path))

    # IMPORTANT: fix template file path in config or here
    # If your file lives at src/summarization/prompts/summarize_news.j2,
    # set in config: prompt.template_file: summarization/prompts/summarize_news.j2
    tmpl = load_prompt(cfg["prompt"]["template_file"])
    sys_prompt = cfg["prompt"]["system"]

    # Single-model path for quick test
    provider = ModelProvider[cfg["model"]["provider"].upper()]
    chat = get_model(
        cfg["model"]["name"],
        provider,
        temperature=cfg["model"].get("temperature", 0.2),
        max_tokens=cfg["model"].get("max_tokens"),
    )

    # Fetch a few READY_FOR_SUMMARY articles
    with session_scope() as s:
        articles = (
            s.query(Article)
             .filter(Article.status == "READY_FOR_SUMMARY")
             .order_by(Article.id.asc())
             .limit(int(os.getenv("TEST_BATCH_LIMIT", "3")))
             .all()
        )

    if not articles:
        print("No READY_FOR_SUMMARY articles found.")
        return

    print(f"Found {len(articles)} articles. Summarizing with {provider.value}:{cfg['model']['name']} ...")

    for a in articles:
        try:
            text, highlights = await summarize_one(chat, sys_prompt, tmpl, a)

            with session_scope() as s:
                s.add(Summary(
                    article_id=a.id,
                    provider=provider.value,
                    model=cfg["model"]["name"],
                    summary_text=text,
                    highlights={"bullets": highlights} if highlights else None,
                    created_at=datetime.now(timezone.utc),
                ))
                db_article = s.get(Article, a.id)
                if db_article:
                    db_article.status = "SUMMARIZED"

            print(f"\n--- Article {a.id}: {a.title} ---")
            print(text)
            if highlights:
                print("Highlights:", highlights)
        except Exception as e:
            print(f"[ERROR] Article {a.id} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
