# src/summarization/summarizer.py
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jinja2 import Template
from sqlalchemy import select, and_
from tqdm import tqdm

from src.app.db import session_scope
from src.app.models import Article, Summary
from llm import ModelProvider, get_model


# ---------- config / prompt ----------
def load_config(path: str) -> Dict[str, Any]:
    import pathlib
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompt(path: str) -> Template:
    from pathlib import Path
    return Template(Path(path).read_text(encoding="utf-8"))

def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

def render_user_prompt(tmpl: Template, a: Article) -> str:
    published = a.published_at.isoformat() if a.published_at else None
    return tmpl.render(
        title=a.title,
        source=a.source,
        url=a.canonical_url or a.url,
        author=a.author,
        published_at=published,
        text=a.body_text,
    )

# ---------- highlights ----------
def extract_highlights(text: str, max_items: int = 6) -> List[str]:
    def clean(line: str) -> str:
        s = line.strip()
        s = re.sub(r"^[\-\•\*\d\.\)\s]+", "", s).strip()
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

# ---------- LLM call ----------
async def call_one(chat, system_prompt: str, user_prompt: str) -> str:
    messages = build_messages(system_prompt, user_prompt)
    resp = await chat.ainvoke(messages)
    return getattr(resp, "content", None) or (resp[0].content if isinstance(resp, list) else str(resp))

# ---------- DB helpers ----------
def fetch_ready_articles(min_published_at: Optional[str], limit: int) -> List[Article]:
    with session_scope() as session:
        q = select(Article).where(Article.status == "READY_FOR_SUMMARY")
        if min_published_at:
            try:
                dt = datetime.fromisoformat(min_published_at.replace("Z", "+00:00"))
                q = q.where(Article.published_at >= dt)
            except Exception:
                pass
        q = q.order_by(Article.published_at.desc().nullslast()).limit(limit)
        return list(session.execute(q).scalars().all())

def upsert_summary(article_id: int, provider: str, model: str,
                   summary_text: str, highlights: List[str]) -> None:
    with session_scope() as session:
        existing = session.execute(
            select(Summary).where(
                and_(Summary.article_id == article_id,
                     Summary.provider == provider,
                     Summary.model == model)
            )
        ).scalar_one_or_none()
        payload = dict(
            summary_text=summary_text,
            highlights={"bullets": highlights} if highlights else None,
            created_at=datetime.now(timezone.utc),
        )
        if existing:
            for k, v in payload.items():
                setattr(existing, k, v)
        else:
            session.add(Summary(
                article_id=article_id,
                provider=provider,
                model=model,
                **payload
            ))

def mark_article_status(article_id: int, status: str) -> None:
    with session_scope() as session:
        a = session.get(Article, article_id)
        if a:
            a.status = status

# ---------- runner ----------
async def summarize_batch(cfg: Dict[str, Any]) -> None:
    # models: support single or multi
    model_section = cfg.get("model", {})
    if "models" in model_section:
        models_cfg = model_section["models"]
    else:
        models_cfg = [{
            "provider": model_section["provider"],
            "name": model_section["name"],
            "temperature": model_section.get("temperature", 0.2),
            "max_tokens": model_section.get("max_tokens"),
        }]

    system_prompt = cfg["prompt"]["system"]
    tmpl = load_prompt(cfg["prompt"]["template_file"])

    articles = fetch_ready_articles(
        cfg["io"].get("min_published_at"), int(cfg["io"].get("limit", 100))
    )
    if not articles:
        print("[OK] No READY_FOR_SUMMARY articles.")
        return

    # build chats once
    chats = []
    for m in models_cfg:
        provider = ModelProvider[m["provider"].upper()]
        chats.append((
            m["provider"].upper(),
            m["name"],
            get_model(
                model_name=m["name"],
                model_provider=provider,
                temperature=float(m.get("temperature", 0.2)),
                max_tokens=m.get("max_tokens"),
            ),
        ))

    concurrency = int(cfg["runtime"].get("concurrency", 6))
    retries = int(cfg["runtime"].get("retries", 2))
    sem = asyncio.Semaphore(concurrency)

    async def run_one_article(a: Article):
        mark_article_status(a.id, "SUMMARIZING")
        user_prompt = render_user_prompt(tmpl, a)

        async def run_one_model(provider: str, model: str, chat):
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    async with sem:
                        text = await call_one(chat, system_prompt, user_prompt)
                    return text
                except Exception as e:
                    last_exc = e
                    await asyncio.sleep(1.2 * (attempt + 1))
            raise last_exc  # type: ignore

        tasks = [run_one_model(p, m, c) for (p, m, c) in chats]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        any_success = False
        for (provider, model, _), r in zip(chats, results):
            if isinstance(r, Exception):
                upsert_summary(a.id, provider, model, f"[ERROR] {type(r).__name__}: {r}", [])
            else:
                txt = str(r).strip()
                upsert_summary(a.id, provider, model, txt, extract_highlights(txt))
                any_success = True

        mark_article_status(a.id, "SUMMARIZED" if any_success else "ERROR")

    tasks = [run_one_article(a) for a in articles]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
        await coro
    print(f"[OK] Processed {len(articles)} article(s).")

def main():
    cfg_path = os.getenv("MNS_CONFIG", "config/config.yaml")
    cfg = load_config(cfg_path)
    asyncio.run(summarize_batch(cfg))

if __name__ == "__main__":
    main()
