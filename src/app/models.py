# src/app/models.py
from sqlalchemy import Column, String, DateTime, Integer, JSON, Text, Index
from sqlalchemy.orm import declarative_base
import datetime as dt

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(64), index=True)
    url = Column(String(1024), unique=True, index=True)
    canonical_url = Column(String(1024), index=True, nullable=True)

    title = Column(String(512))
    author = Column(String(256), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    first_seen_at = Column(DateTime(timezone=True),
                           default=lambda: dt.datetime.now(dt.timezone.utc))

    lang = Column(String(16), default="en")
    body_text = Column(Text)

    meta = Column(JSON, default=dict)  # factory

    # Pipeline state: READY_FOR_SUMMARY | SUMMARIZING | SUMMARIZED | ERROR
    status = Column(String(32), default="READY_FOR_SUMMARY", index=True)

    __table_args__ = (
        Index("ix_articles_source_published", "source", "published_at"),
        Index("ix_articles_first_seen_at", "first_seen_at"),
    )

    def __repr__(self):
        return f"<Article id={self.id} source={self.source} title={self.title[:40]!r}>"


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, index=True, nullable=False)

    # Which LLM produced this
    provider = Column(String(32), nullable=False)   # e.g., OPENAI
    model = Column(String(128), nullable=False)     # e.g., gpt-4o-mini

    summary_text = Column(Text, nullable=False)
    highlights = Column(JSON, nullable=True)        # {"bullets": [...]}
    extra = Column(JSON, nullable=True)             # optional: tokens, latency, etc.

    created_at = Column(DateTime(timezone=True),
                        default=lambda: dt.datetime.now(dt.timezone.utc),
                        nullable=False)

    __table_args__ = (
        Index("ix_summaries_article_model", "article_id", "model", unique=True),
    )

    def __repr__(self):
        return f"<Summary article_id={self.article_id} model={self.model}>"
