# src/app/models.py
from __future__ import annotations

import datetime as dt
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    JSON,
    Text,
    Index,
    ForeignKey,
    Float,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# -----------------------------
# Core content tables
# -----------------------------
class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Source & identity
    source = Column(String(64), index=True)
    url = Column(String(1024), unique=True, index=True)
    canonical_url = Column(String(1024), index=True, nullable=True)

    # Content & metadata
    title = Column(String(512))
    author = Column(String(256), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    first_seen_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
    )

    lang = Column(String(16), default="en")
    body_text = Column(Text)

    meta = Column(JSON, default=dict)  # free-form payload per source

    # Pipeline state: READY_FOR_SUMMARY | SUMMARIZING | SUMMARIZED | ERROR
    status = Column(String(32), default="READY_FOR_SUMMARY", index=True)

    __table_args__ = (
        Index("ix_articles_source_published", "source", "published_at"),
        Index("ix_articles_first_seen_at", "first_seen_at"),
    )

    # Relationships (optional, but handy)
    summaries = relationship("Summary", back_populates="article", cascade="all,delete-orphan")
    embeddings = relationship("Embedding", back_populates="article", cascade="all,delete-orphan")
    article_clusters = relationship("ArticleCluster", back_populates="article", cascade="all,delete-orphan")

    def __repr__(self):
        t = (self.title or "")[:40]
        return f"<Article id={self.id} source={self.source} title={t!r}>"


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Which LLM produced this
    provider = Column(String(32), nullable=False)   # e.g., OPENAI
    model = Column(String(128), nullable=False)     # e.g., gpt-4o-mini

    # Final summary text used for embeddings & UI
    summary_text = Column(Text, nullable=False)

    # Optional structured fields
    highlights = Column(JSON, nullable=True)        # {"bullets": [...]}
    extra = Column(JSON, nullable=True)             # tokens, latency, etc.

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        # Prevent duplicate (article, model) summaries
        Index("ix_summaries_article_model", "article_id", "model", unique=True),
        Index("ix_summaries_created", "created_at"),
    )

    # Relationship back to article
    article = relationship("Article", back_populates="summaries")

    # --- Compatibility alias so existing code that expects `Summary.text` keeps working ---
    @property
    def text(self) -> str:
        return self.summary_text

    def __repr__(self):
        return f"<Summary article_id={self.article_id} model={self.model}>"


# -----------------------------
# Embeddings & clustering
# -----------------------------
class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        index=True,
        unique=True,   # one embedding per article (per current pipeline)
        nullable=False,
    )

    provider = Column(String(40), nullable=False)   # 'openai'
    model = Column(String(120), nullable=False)     # e.g., 'text-embedding-3-small'
    vector = Column(JSON, nullable=False)           # list[float] (JSON for SQLite/Postgres)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_embeddings_article_id", "article_id"),
        Index("ix_embeddings_created", "created_at"),
    )

    article = relationship("Article", back_populates="embeddings")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Each clustering job/run gets a run_id (e.g., epoch seconds)
    run_id = Column(Integer, index=True, nullable=False)

    # Label index assigned by the algorithm (0..k-1)
    label = Column(Integer, index=True, nullable=False)

    # Optional cluster center vector (same dim as Embedding.vector)
    center = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("run_id", "label", name="uq_clusters_run_label"),
        Index("ix_clusters_created", "created_at"),
    )

    article_clusters = relationship("ArticleCluster", back_populates="cluster", cascade="all,delete-orphan")


class ArticleCluster(Base):
    __tablename__ = "article_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)

    article_id = Column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    cluster_id = Column(
        Integer,
        ForeignKey("clusters.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Optional: similarity to cluster center or ranking score
    score = Column(Float, default=1.0, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("article_id", "cluster_id", name="uq_article_cluster_once"),
        Index("ix_article_clusters_cluster_article", "cluster_id", "article_id"),
    )

    article = relationship("Article", back_populates="article_clusters")
    cluster = relationship("Cluster", back_populates="article_clusters")
