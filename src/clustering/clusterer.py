# src/clustering/clusterer.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.app.db import get_engine, sessionmaker_from_engine
from src.app.models import Article, Embedding, Cluster, ArticleCluster

@dataclass
class ClusterParams:
    window_hours: int = 24
    min_k: int = 5
    max_k: int = 25
    algo: str = "kmeans"
    min_cluster_size: int = 5

def _load_window_embeddings(sess: Session, hours: int) -> Tuple[np.ndarray, List[int]]:
    since = datetime.utcnow() - timedelta(hours=hours)
    q = (
        select(Embedding.article_id, Embedding.vector)
        .join(Article, Article.id == Embedding.article_id)
        .where(Article.published_at >= since)
        .order_by(Article.published_at.desc())
    )
    rows = sess.execute(q).all()
    if not rows:
        return np.zeros((0, 0), dtype="float32"), []
    ids = [r[0] for r in rows]
    X = np.array([r[1] for r in rows], dtype="float32")
    return X, ids

def _pick_k_by_silhouette(X: np.ndarray, p: ClusterParams) -> int:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    n = len(X)
    if n < max(10, p.min_k):
        return max(2, min(n // 2, p.min_k))
    best_k, best_s = p.min_k, -1.0
    for k in range(p.min_k, min(p.max_k, n - 1) + 1):
        km = MiniBatchKMeans(n_clusters=k, batch_size=512, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k

def _kmeans_cluster(X: np.ndarray, p: ClusterParams):
    from sklearn.cluster import MiniBatchKMeans
    k = _pick_k_by_silhouette(X, p)
    km = MiniBatchKMeans(n_clusters=k, batch_size=512, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels, centers

def _persist(sess: Session, ids: List[int], labels: np.ndarray, centers: np.ndarray) -> int:
    run_id = int(datetime.utcnow().timestamp())
    label_to_cluster_id: Dict[int, int] = {}
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        c = Cluster(run_id=run_id, label=int(lab), center=centers[lab].tolist())
        sess.add(c)
        sess.flush()
        label_to_cluster_id[lab] = c.id
    for i, lab in enumerate(labels):
        sess.add(ArticleCluster(article_id=ids[i], cluster_id=label_to_cluster_id[lab], score=1.0))
    sess.commit()
    return run_id

def run_clustering(p: ClusterParams) -> int:
    engine = get_engine()
    SessionLocal = sessionmaker_from_engine(engine)
    with SessionLocal() as sess:
        X, ids = _load_window_embeddings(sess, p.window_hours)
        if len(ids) == 0:
            return 0
        labels, centers = _kmeans_cluster(X, p)
        run_id = _persist(sess, ids, labels, centers)
    return run_id
