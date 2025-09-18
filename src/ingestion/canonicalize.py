# src/ingestion/canonicalize.py
from urllib.parse import urlparse, parse_qsl, urlunparse, urlencode
from bs4 import BeautifulSoup

TRACK_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid"}

def strip_tracking(url: str) -> str:
    parts = list(urlparse(url))
    qs = [(k,v) for k,v in parse_qsl(parts[4], keep_blank_values=True) if k not in TRACK_PARAMS]
    parts[4] = urlencode(qs)
    return urlunparse(parts)

def canonical_url_from_html(url: str, html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        og = soup.find("meta", {"property": "og:url"})
        if og and og.get("content"):
            return og["content"]
    except Exception:
        pass
    return strip_tracking(url)
