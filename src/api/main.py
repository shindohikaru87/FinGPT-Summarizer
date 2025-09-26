from __future__ import annotations

import os
import math
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from pydantic import BaseModel
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session, aliased

try:
    from scripts._bootstrap_env import load_env
    load_env()
except Exception:
    pass

from src.app.db import session_scope
from src.app.models import Article, Summary, Embedding, Cluster, ArticleCluster

# OpenAI embeddings
from openai import OpenAI

OPENAI_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_WINDOW_HOURS = int(os.getenv("SEARCH_WINDOW_HOURS", "720"))  # 30 days
MAX_CANDIDATES = int(os.getenv("SEARCH_MAX_CANDIDATES", "2000"))     # cap before cosine
KW_WEIGHT = float(os.getenv("SEARCH_KW_WEIGHT", "0.25"))             # 0..1 keyword weight
EMB_WEIGHT = 1.0 - KW_WEIGHT

# --- app & logging ---
app = FastAPI(title="FinGPT Summarizer Search API", version="0.1.0")
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------
class SearchHit(BaseModel):
    article_id: int
    source: str
    title: str
    url: str
    published_at: Optional[str]
    summary_text: Optional[str]
    cluster_label: Optional[str]
    score: float


class SearchResponse(BaseModel):
    query: str
    total_candidates: int
    returned: int
    page: int
    page_size: int
    hits: List[SearchHit]


class ClusterItem(BaseModel):
    run_id: int
    label: str
    cluster_id: int
    size: int
    sample_titles: List[str]


class ClustersResponse(BaseModel):
    run_id: int
    clusters: List[ClusterItem]


# ---------- helpers ----------
def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _latest_run_id(sess: Session) -> Optional[int]:
    row = sess.execute(select(func.max(Cluster.run_id))).first()
    return row[0] if row and row[0] is not None else None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _keyword_score(title: str, body: Optional[str], q_terms: List[str]) -> float:
    """Very simple term frequency score across title/body (title weighted)."""
    text_t = (title or "").lower()
    text_b = (body or "").lower()
    score = 0.0
    for t in q_terms:
        if not t:
            continue
        score += 2.0 * text_t.count(t) + 1.0 * text_b.count(t)
    denom = max(40.0, len(text_t) / 5 + len(text_b) / 50)
    return score / denom


def _embed_query(client: OpenAI, text: str) -> List[float]:
    out = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return list(out.data[0].embedding)


def _sqlite_safe_order_recent_desc(query):
    """
    Emulate ORDER BY published_at DESC NULLS LAST in a portable way:
    non-null rows first (is_(None) == False), then DESC by timestamp.
    """
    return query.order_by(Article.published_at.is_(None), desc(Article.published_at))


def _as_opt_str(x: Any) -> Optional[str]:
    """Coerce any non-None value to str; keep None as None."""
    if x is None:
        return None
    try:
        return str(x)
    except Exception:
        return None


def _as_str(x: Any, default: str = "") -> str:
    """Coerce value to str; replace None with default."""
    try:
        return default if x is None else str(x)
    except Exception:
        return default


def _fetch_candidates(
    sess: Session,
    window_hours: int,
    limit: int,
    run_id: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` recent articles joined with latest summary and embedding.
    If `run_id` given, left-join cluster label for that run.
    """
    # latest summary per article
    sq_last_sum = (
        select(Summary.article_id, func.max(Summary.created_at).label("mx"))
        .group_by(Summary.article_id)
        .subquery()
    )
    S = aliased(Summary)

    sel = (
        select(
            Article.id.label("article_id"),
            Article.source,
            Article.title,
            Article.url,
            Article.published_at,
            Article.body_text,
            S.summary_text,
            Embedding.vector,
        )
        .join(Embedding, Embedding.article_id == Article.id, isouter=False)  # need vector for cosine
        .join(sq_last_sum, sq_last_sum.c.article_id == Article.id, isouter=True)
        .join(S, and_(S.article_id == Article.id, S.created_at == sq_last_sum.c.mx), isouter=True)
    )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    sel = sel.where(or_(Article.published_at.is_(None), Article.published_at >= cutoff))

    # cluster join (optional)
    if run_id is not None:
        sel = sel.add_columns(Cluster.label.label("cluster_label"))
        sel = sel.join(ArticleCluster, ArticleCluster.article_id == Article.id, isouter=True)
        sel = sel.join(
            Cluster,
            and_(Cluster.id == ArticleCluster.cluster_id, Cluster.run_id == run_id),
            isouter=True,
        )

    sel = _sqlite_safe_order_recent_desc(sel).limit(limit)
    rows = sess.execute(sel).all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        m = r._mapping
        out.append(
            {
                "article_id": m["article_id"],
                "source": (m["source"] or "") if m.get("source") is not None else "",
                "title": (m["title"] or "") if m.get("title") is not None else "",
                "url": (m["url"] or "") if m.get("url") is not None else "",
                "published_at": m["published_at"],
                "body_text": m.get("body_text") or "",
                "summary_text": m.get("summary_text"),
                "vector": m["vector"],  # list[float] or JSON string depending on DB column type
                # <-- COERCE TO OPTIONAL STRING -->
                "cluster_label": _as_opt_str(m.get("cluster_label")),
            }
        )
    return out


# ---------- error handling ----------
@app.exception_handler(Exception)
async def unhandled_exc(_req: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


# ---------- routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": _iso(datetime.now(timezone.utc))}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="User query (keywords)"),
    k: int = Query(20, ge=1, le=100, description="Top-K ceiling before pagination"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    window_hours: int = Query(DEFAULT_WINDOW_HOURS, ge=1, le=365 * 24),
    use_latest_run: bool = Query(True, description="Attach cluster label from latest run"),
    # --- new controls ---
    min_score: float = Query(1e-6, ge=0.0, description="Filter out hits with score < min_score"),
    debug_include_zero: bool = Query(False, description="If true, do not filter low/zero score hits"),
):
    """
    Hybrid search:
      score = EMB_WEIGHT * cosine(query_emb, article_emb) + KW_WEIGHT * tf-like keyword score
    """
    try:
        # Try embeddings; fall back to keywords only if anything fails.
        q_vec: List[float] = []
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            q_vec = _embed_query(client, q)
        except Exception as e:
            log.warning("Embedding fallback (keyword-only): %s", e)
            q_vec = []

        q_terms = [t.strip().lower() for t in q.split() if t.strip()]

        with session_scope() as sess:
            run_id = _latest_run_id(sess) if use_latest_run else None
            cands = _fetch_candidates(
                sess, window_hours=window_hours, limit=MAX_CANDIDATES, run_id=run_id
            )

        # score candidates
        scored: List[Dict[str, Any]] = []
        for c in cands:
            vec = c["vector"]
            if isinstance(vec, str):
                try:
                    vec = json.loads(vec)
                except Exception:
                    vec = []
            if not isinstance(vec, list):
                vec = []

            emb_score = _cosine(q_vec, vec) if (q_vec and vec) else 0.0
            kw_score = _keyword_score(c["title"], c.get("summary_text") or c.get("body_text"), q_terms)
            score = EMB_WEIGHT * emb_score + KW_WEIGHT * kw_score

            c2 = dict(c)
            c2["score"] = float(score)
            scored.append(c2)

        # order by score desc
        scored.sort(key=lambda x: x["score"], reverse=True)

        total_candidates = len(scored)

        # --- filtering: drop 0/near-0 unless debug says otherwise ---
        if not debug_include_zero:
            filtered = [s for s in scored if s["score"] >= min_score]
        else:
            filtered = scored

        total_after_filter = len(filtered)

        # pagination *after* filtering, still honoring top-K ceiling
        start = (page - 1) * page_size
        end = start + page_size
        top_needed = max(k, end)
        window = filtered[:top_needed]
        subset = window[start:end]

        hits = [
            SearchHit(
                article_id=s["article_id"],
                source=s["source"],
                title=s["title"],
                url=s["url"],
                published_at=_iso(s["published_at"]),
                summary_text=s["summary_text"],
                cluster_label=s.get("cluster_label"),
                score=round(s["score"], 6),
            )
            for s in subset
        ]

        return SearchResponse(
            query=q,
            total_candidates=total_candidates,   # before filtering
            returned=len(hits),                  # after pagination
            page=page,
            page_size=page_size,
            hits=hits,
        )
    except Exception as e:
        log.exception("Search failed")
        return JSONResponse(
            status_code=500, content={"error": "internal_error", "where": "search", "detail": str(e)}
        )


@app.get("/clusters", response_model=ClustersResponse)
def clusters(
    run_id: Optional[int] = Query(None, description="Cluster run_id; default=latest"),
    top_titles: int = Query(3, ge=1, le=10),
):
    """
    List clusters with sizes + a few representative titles for context.
    """
    try:
        with session_scope() as sess:
            rid = run_id or _latest_run_id(sess)
            if rid is None:
                raise HTTPException(status_code=404, detail="No clustering runs found.")

            # get clusters for run
            cl_rows = sess.execute(
                select(Cluster.id, Cluster.label).where(Cluster.run_id == rid)
            ).all()

            clusters_out: List[ClusterItem] = []
            for cid, label in cl_rows:
                # size
                sz = sess.execute(
                    select(func.count(ArticleCluster.article_id)).where(ArticleCluster.cluster_id == cid)
                ).scalar_one()

                # representative titles (recent) — SQLite-safe ordering
                titles_query = (
                    select(Article.title)
                    .join(ArticleCluster, ArticleCluster.article_id == Article.id)
                    .where(ArticleCluster.cluster_id == cid)
                )
                titles_query = titles_query.order_by(
                    Article.published_at.is_(None), desc(Article.published_at)
                ).limit(top_titles)
                titles = sess.execute(titles_query).scalars().all()

                clusters_out.append(
                    ClusterItem(
                        run_id=rid,
                        # <-- COERCE label to str explicitly -->
                        label=_as_str(label, default=""),
                        cluster_id=cid,
                        size=int(sz or 0),
                        sample_titles=[t for t in titles if t],
                    )
                )

            return ClustersResponse(run_id=rid, clusters=clusters_out)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Clusters failed")
        return JSONResponse(
            status_code=500, content={"error": "internal_error", "where": "clusters", "detail": str(e)}
        )
