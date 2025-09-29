# src/embeddings/service.py

from __future__ import annotations

import json
import math
import time
from typing import Iterable, Optional

from openai import OpenAI
from sqlalchemy import select, func, and_
from sqlalchemy.orm import aliased

from src.app.db import session_scope
from src.app.models import Article, Summary, Embedding
from src.embeddings.registry import EmbeddingConfig, RunParams

# Optional: tokenization helpers for safe truncation/chunking
try:
    import tiktoken
except Exception:
    tiktoken = None


# -----------------------------
# Utility functions
# -----------------------------
def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / s for v in vec]


def _get_tokenizer(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _truncate_text(text: str, model: str, max_tokens: int = 7500) -> str:
    enc = _get_tokenizer(model)
    if enc is None:
        max_chars = max_tokens * 4
        return text[:max_chars]
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])


def _chunk_text(
    text: str,
    model: str,
    chunk_tokens: int = 2000,
    overlap_tokens: int = 200,
) -> Iterable[str]:
    enc = _get_tokenizer(model)
    if enc is None:
        chunk_chars = chunk_tokens * 4
        overlap_chars = overlap_tokens * 4
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_chars)
            yield text[start:end]
            if end == n:
                break
            start = max(end - overlap_chars, start + 1)
        return

    toks = enc.encode(text)
    n = len(toks)
    i = 0
    step = max(1, chunk_tokens - overlap_tokens)
    while i < n:
        j = min(n, i + chunk_tokens)
        yield enc.decode(toks[i:j])
        if j == n:
            break
        i = j - overlap_tokens


def _average_vectors(vecs: list[list[float]]) -> list[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    acc = [0.0] * dim
    for v in vecs:
        if len(v) != dim:
            continue
        for k in range(dim):
            acc[k] += v[k]
    cnt = max(1, len(vecs))
    return [x / cnt for x in acc]


# -----------------------------
# Candidate fetcher
# -----------------------------
def _fetch_candidates(sess, limit: Optional[int], since_hours: Optional[float]):
    """
    Return list of (article_id, body_text) for articles that need embeddings.
    This assumes each article has at most one embedding (per schema).
    """
    q = select(Article.id, Article.body_text).outerjoin(
        Embedding, Embedding.article_id == Article.id
    ).where(Embedding.id.is_(None))

    if since_hours:
        cutoff = func.datetime("now", f"-{since_hours} hours")
        q = q.where(Article.first_seen_at >= cutoff)

    q = q.order_by(Article.first_seen_at.desc())
    if limit:
        q = q.limit(limit)

    return sess.execute(q).all()


# -----------------------------
# Main embedding function
# -----------------------------
def run_embeddings(ecfg: EmbeddingConfig, params: RunParams) -> int:
    """
    Generate embeddings for articles or summaries and persist to DB.
    Configurable behaviors:
      - params.embed_source: "article" (default) or "summary"
      - params.long_text_mode: "truncate" (default) or "chunk"
      - params.max_tokens: safety limit for truncate (default 7500)
      - params.chunk_tokens / params.overlap_tokens: for chunk mode
    """
    client = OpenAI()
    total = 0

    embed_source: str = getattr(params, "embed_source", "article")
    long_text_mode: str = getattr(params, "long_text_mode", "truncate")
    max_tokens: int = getattr(params, "max_tokens", 7500)
    chunk_tokens: int = getattr(params, "chunk_tokens", 2000)
    overlap_tokens: int = getattr(params, "overlap_tokens", 200)

    print(
        f"Embedding source={embed_source}, mode={long_text_mode}, "
        f"model={ecfg.model}, batch_size={ecfg.batch_size}, normalize={ecfg.normalize}"
    )

    with session_scope() as sess:
        rows = _fetch_candidates(sess, limit=params.limit, since_hours=params.since_hours)
        if not rows:
            print("No candidate articles for embeddings.")
            return 0

        print(f"Running embeddings for up to {len(rows)} articles...")

        # If using summaries, fetch latest per article
        latest_summary_by_article: dict[int, str] = {}
        if embed_source == "summary":
            article_ids = [r[0] for r in rows]
            S1, S2 = aliased(Summary), aliased(Summary)
            sq = (
                select(S1.article_id, func.max(S1.created_at).label("mx"))
                .where(S1.article_id.in_(article_ids))
                .group_by(S1.article_id)
                .subquery()
            )
            rs = (
                sess.execute(
                    select(S2.article_id, S2.summary_text)
                    .join(sq, and_(S2.article_id == sq.c.article_id, S2.created_at == sq.c.mx))
                )
                .all()
            )
            latest_summary_by_article = {row.article_id: row.summary_text for row in rs}

        batch_size = ecfg.batch_size
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            ids = [r[0] for r in batch]
            texts = []
            for aid, body_text in batch:
                if embed_source == "summary":
                    texts.append(latest_summary_by_article.get(aid, ""))
                else:
                    texts.append(body_text or "")

            t0 = time.perf_counter()
            try:
                vectors_per_article: list[list[float]] = []
                for text in texts:
                    if not text:
                        vectors_per_article.append([])
                        continue

                    if long_text_mode == "truncate":
                        txt = _truncate_text(text, ecfg.model, max_tokens=max_tokens)
                        resp = client.embeddings.create(model=ecfg.model, input=[txt])
                        vec = list(resp.data[0].embedding)

                    else:  # chunk
                        chunk_vecs: list[list[float]] = []
                        for chunk in _chunk_text(text, ecfg.model, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens):
                            resp = client.embeddings.create(model=ecfg.model, input=[chunk])
                            chunk_vecs.append(list(resp.data[0].embedding))
                        vec = _average_vectors(chunk_vecs)

                    if ecfg.normalize and vec:
                        vec = _l2_normalize(vec)
                    vectors_per_article.append(vec)

                # Persist
                for aid, vec in zip(ids, vectors_per_article):
                    if not vec:
                        continue
                    sess.add(
                        Embedding(
                            article_id=aid,
                            provider=ecfg.provider,
                            model=ecfg.model,
                            vector=json.dumps(vec),
                        )
                    )
                sess.flush()

                dt = time.perf_counter() - t0
                total += len([v for v in vectors_per_article if v])
                if params.progress_cb:
                    params.progress_cb("ok", dt)

            except Exception as e:
                dt = time.perf_counter() - t0
                print(f"⚠️ Error embedding batch starting at {start}: {e}")
                if params.progress_cb:
                    params.progress_cb("fail", dt)
                continue

    print(f"\nEmbedded {total} articles.")
    return total
