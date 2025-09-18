# src/ingestion/extractor.py
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser as dtparse
from typing import Optional, Dict

def text_from_nodes(soup, css: str) -> str:
    nodes = soup.select(css) if css else []
    return "\n".join(n.get_text(" ", strip=True) for n in nodes)

def first_attr(soup, css: str, attr: str) -> Optional[str]:
    el = soup.select_one(css) if css else None
    return el.get(attr) if el and el.has_attr(attr) else None

def extract_article(html: str, article_cfg: Dict) -> Dict:
    soup = BeautifulSoup(html, "lxml")
    title = text_from_nodes(soup, article_cfg.get("title_css", "h1"))
    body  = text_from_nodes(soup, article_cfg.get("body_css", "article p"))
    author = text_from_nodes(soup, article_cfg.get("author_css", "")) or None
    ts = first_attr(soup, article_cfg.get("time_css", ""), article_cfg.get("time_attr", "datetime"))
    published_at = None
    if ts:
        try: published_at = dtparse.parse(ts)
        except Exception: pass
    # fallbacks via meta tags
    if not published_at:
        meta = soup.find("meta", {"property": "article:published_time"}) or soup.find("meta", {"name": "date"})
        if meta and meta.get("content"):
            try: published_at = dtparse.parse(meta["content"])
            except Exception: pass
    return {"title": title, "body_text": body, "author": author, "published_at": published_at}
