# src/embeddings/service.py

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional, Callable, Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from openai import OpenAI

from src.app.db import session_scope
from src.app.models import Article, Embedding


# ---------------- Types ----------------

ProgressFn = Callable[[str, Optional[float]], None]  # (status, seconds per batch)


@dataclass
class RunParams:
    limit: int = 500
    since_hours: Optional[int] = None
    progress_cb: Optional[ProgressFn] = None


@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    batch_size: int = 512
    normalize: bool = True


# ---------------- Core ----------------

def _normalize(vec: List[float]) -> List[float]:
    import math
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _fetch_candidates(sess: Session, limit: int, since_hours: Optional[int]) -> List[tuple[int, str]]:
    """Fetch up to N (id, text) that are missing embeddings."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import and_, func

    q = select(Article.id, Article.body_text).where(Article.body_text.isnot(None))

    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        q = q.where(and_(Article.published_at.is_(None) | (Article.published_at >= cutoff)))

    # exclude already embedded
    exists_q = (
        select(func.count(Embedding.id))
        .where(Embedding.article_id == Article.id)
        .correlate(Article)
    )
    q = q.where(exists_q.as_scalar() == 0)

    q = q.order_by(Article.published_at.desc().nullslast()).limit(limit)

    return [(row[0], row[1] or "") for row in sess.execute(q).all()]


def run_embeddings(ecfg: EmbeddingConfig, params: RunParams) -> int:
    """
    Generate embeddings for articles and persist to DB.
    Returns the number of new embeddings created.
    """
    client = OpenAI()
    total = 0

    with session_scope() as sess:
        rows = _fetch_candidates(sess, limit=params.limit, since_hours=params.since_hours)
        if not rows:
            print("No candidate articles for embeddings.")
            return 0

        print(f"Running embeddings for up to {len(rows)} articles...")

        batch_size = ecfg.batch_size
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]

            t0 = time.perf_counter()
            try:
                resp = client.embeddings.create(model=ecfg.model, input=texts)
                vecs = [list(d.embedding) for d in resp.data]

                if ecfg.normalize:
                    vecs = [_normalize(v) for v in vecs]

                for aid, vec in zip(ids, vecs):
                    sess.add(
                        Embedding(
                            article_id=aid,
                            provider=ecfg.provider,
                            model=ecfg.model,
                            vector=json.dumps(vec),  # stored as JSON string
                            created_at=None,  # default by DB
                        )
                    )
                sess.flush()
                dt = time.perf_counter() - t0

                total += len(ids)
                if params.progress_cb:
                    params.progress_cb("ok", dt)

            except Exception as e:
                dt = time.perf_counter() - t0
                print(f"⚠️ Error embedding batch starting at {start}: {e}")
                if params.progress_cb:
                    params.progress_cb("fail", dt)
                continue

        return total

def from_langchain_embedding(lc_embedding: Any) -> EmbeddingConfig:
    """
    Convert a LangChain embedding instance into our EmbeddingConfig.
    Currently supports OpenAIEmbeddings and HuggingFaceEmbeddings.
    """
    cls_name = lc_embedding.__class__.__name__.lower()

    if "openai" in cls_name:
        model = getattr(lc_embedding, "model", "text-embedding-3-small")
        return EmbeddingConfig(
            provider="openai",
            model=model,
            batch_size=getattr(lc_embedding, "batch_size", 512),
            normalize=True
        )

    if "huggingface" in cls_name:
        model = getattr(lc_embedding, "model_name", "sentence-transformers/all-MiniLM-L6-v2")
        return EmbeddingConfig(
            provider="huggingface",
            model=model,
            batch_size=getattr(lc_embedding, "batch_size", 32),
            normalize=True
        )

    raise ValueError(f"Unsupported LangChain embedding class: {lc_embedding.__class__.__name__}")
