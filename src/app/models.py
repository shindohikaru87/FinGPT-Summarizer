# src/app/models.py
from sqlalchemy import Column, String, DateTime, Integer, JSON, Text, Index
from sqlalchemy.orm import declarative_base
import datetime as dt

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(64), index=True)
    url = Column(String(1024), unique=True, index=True)           # original URL
    canonical_url = Column(String(1024), index=True, nullable=True)  # keep index=True here

    title = Column(String(512))
    author = Column(String(256), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    first_seen_at = Column(DateTime(timezone=True),
                           default=lambda: dt.datetime.now(dt.timezone.utc))

    lang = Column(String(16), default="en")
    body_text = Column(Text)

    meta = Column(JSON, default=dict)  # factory, not {}

    status = Column(String(32), default="READY_FOR_SUMMARY")

    __table_args__ = (
        Index("ix_articles_source_published", "source", "published_at"),
        Index("ix_articles_first_seen_at", "first_seen_at"),
        # NOTE: removed explicit ix_articles_canonical_url (column already has index=True)
    )

    def __repr__(self):
        return f"<Article id={self.id} source={self.source} title={self.title[:40]!r}>"
