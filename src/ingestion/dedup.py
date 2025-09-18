# src/ingestion/dedup.py
import hashlib

def simple_key(title: str, body_text: str) -> str:
    lede = (body_text or "")[:800]
    return hashlib.sha1(f"{title}\n{lede}".encode("utf-8")).hexdigest()

def is_duplicate(db, url: str, canon_url: str, content_key: str) -> tuple[bool, int | None]:
    """
    Implement with DB queries:
      1) exact URL or canonical_url match
      2) optional: near-dup via stored content_key
    """
    # pseudo:
    # row = db.find_article_by_key_or_url(canon_url, content_key)
    # return (row is not None, row.id if row else None)
    return (False, None)
