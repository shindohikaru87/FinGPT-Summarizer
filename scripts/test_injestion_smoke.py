#!/usr/bin/env python3
"""
Lightweight smoke tests for src/ingestion modules (no live network/db).
Run: python scripts/test_ingestion_smoke.py
"""
from pathlib import Path
import sys

# Put REPO ROOT on sys.path so "from src.ingestion ..." works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
import traceback
from datetime import datetime, timezone

import sys
import types
import traceback
from datetime import datetime, timezone

PASSED = 0
FAILED = 0

def ok(msg):
    global PASSED
    PASSED += 1
    print(f"✅ {msg}")

def fail(msg, exc=None):
    global FAILED
    FAILED += 1
    print(f"❌ {msg}")
    if exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)

# --- Inline fixtures (no files needed) ----------------------------------------

FIXTURE_ARTICLE_HTML = """<!doctype html>
<html><body>
<article>
  <h1>Fed cuts rates by 25 bps</h1>
  <time datetime="2025-09-18T08:30:00Z"></time>
  <p>Paragraph 1.</p>
  <p>Paragraph 2.</p>
  <a href="/authors/jane-doe">Jane Doe</a>
</article>
</body></html>
"""

FIXTURE_URL = "https://www.reuters.com/markets/us/fed-cuts-rates-25-bps-2025-09-18/"

# --- Minimal monkey-patch helper ---------------------------------------------

class MonkeyPatch:
    def __init__(self):
        self._stack = []

    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        self._stack.append((obj, name, old))
        setattr(obj, name, value)

    def restore(self):
        while self._stack:
            obj, name, old = self._stack.pop()
            setattr(obj, name, old)

monkey = MonkeyPatch()

# --- Imports -----------------------------------------------------------------

try:
    from src.ingestion import extractor, canonicalize, dedup, pipeline, discovery, crawl_fetcher
    ok("Imported src.ingestion modules")
except Exception as e:
    fail("Failed to import ingestion modules", e)
    sys.exit(1)

# --- Unit tests: extractor ----------------------------------------------------

def test_extractor_basic():
    try:
        selectors = {
            "title_css": "h1",
            "body_css": "article p",
            "time_css": "time",
            "time_attr": "datetime",
            "author_css": 'a[href*="/authors/"]',
            "js_render": False,
        }
        data = extractor.extract_article(FIXTURE_ARTICLE_HTML, selectors)
        assert data["title"] == "Fed cuts rates by 25 bps"
        assert "Paragraph 1." in data["body_text"]
        assert "Paragraph 2." in data["body_text"]
        assert data["author"].lower() == "jane doe"
        assert hasattr(data["published_at"], "isoformat")
        ok("extractor.extract_article basic HTML parsing")
    except Exception as e:
        fail("extractor.extract_article basic HTML parsing", e)

def test_extractor_missing_author_ok():
    try:
        selectors = {
            "title_css": "h1",
            "body_css": "article p",
            "time_css": "time",
            "time_attr": "datetime",
            "author_css": ".missing",
        }
        data = extractor.extract_article(FIXTURE_ARTICLE_HTML, selectors)
        # Accept None/""/"Unknown" depending on implementation
        assert data.get("author") in (None, "", "Unknown", "unknown", "N/A")
        ok("extractor handles missing author")
    except Exception as e:
        fail("extractor handles missing author", e)

# --- Unit tests: canonicalize & dedup ----------------------------------------

def test_canonicalize_and_strip():
    try:
        canon = canonicalize.canonical_url_from_html(FIXTURE_URL, FIXTURE_ARTICLE_HTML)
        assert canon is None or isinstance(canon, str)

        messy = FIXTURE_URL + "?utm_source=x&utm_medium=y&ref=foo&foo=bar"
        clean = canonicalize.strip_tracking(messy)

        # still the same base path
        assert clean.startswith(FIXTURE_URL.split("?")[0])

        # ensure all utm_* trackers are removed
        assert "utm_" not in clean

        # allow site-specific non-utm params; if present, they must not include utm_ keys
        # (some implementations keep harmless params like ?ref=foo)
        ok("canonicalize: strip_tracking & canonical_url_from_html")
    except Exception as e:
        fail("canonicalize tests", e)

def test_simple_key():
    try:
        key = dedup.simple_key("Fed cuts rates", "Paragraph 1. Paragraph 2.")
        assert isinstance(key, str) and len(key) > 8
        ok("dedup.simple_key returns stable-ish key")
    except Exception as e:
        fail("dedup.simple_key", e)

# --- Pipeline smoke (monkey-patched) -----------------------------------------

class _FakeArticle:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.id = None

class _FakeSession:
    def __init__(self, sink_list):
        self._sink = sink_list
        self._pending = None

    def add(self, obj):
        self._pending = obj

    def flush(self):
        # emulate DB id assignment
        if self._pending:
            self._pending.id = len(self._sink) + 1
            # store a snapshot (dict) for inspection
            self._sink.append({
                "id": self._pending.id,
                "source": self._pending.source,
                "url": self._pending.url,
                "canonical_url": self._pending.canonical_url,
                "title": self._pending.title,
                "author": self._pending.author,
                "published_at": getattr(self._pending.published_at, "isoformat", lambda: None)(),
                "first_seen_at": getattr(self._pending.first_seen_at, "isoformat", lambda: None)(),
                "body_text": self._pending.body_text,
                "meta": self._pending.meta,
                "status": self._pending.status,
            })
            self._pending = None

    def close(self):
        pass

class _SessionScopeCtx:
    def __init__(self, sink):
        self.sink = sink
        self.sess = _FakeSession(self.sink)

    def __enter__(self):
        return self.sess

    def __exit__(self, exc_type, exc, tb):
        self.sess.close()
        return False  # don't suppress

def test_pipeline_smoke():
    sink = []
    # Monkey-patch discovery, fetcher, db, model, dedup
    try:
        # 1) discovery.discover_urls → return one URL
        def _fake_discover_urls(src_cfg, _fetcher):
            return [FIXTURE_URL]
        monkey.setattr(pipeline, "discover_urls", _fake_discover_urls)

        # 2) fetcher.Fetcher.get_html → return fixture html
        # 2) pipeline.CrawlFetcher.get_html → return fixture html
        class _FakeCrawlFetcher:
            def __init__(self, *a, **kw): pass
            def get_html(self, url, js_render=False, obey_robots=True):
                return FIXTURE_ARTICLE_HTML

        monkey.setattr(pipeline, "CrawlFetcher", _FakeCrawlFetcher)


        # 3) dedup.is_duplicate → always false
        def _fake_is_duplicate(sess, url, canon, key):
            return (False, None)
        monkey.setattr(pipeline, "is_duplicate", _fake_is_duplicate)

        # 4) app.db.session_scope → our fake context manager
        class _FakeDBModule(types.SimpleNamespace):
            def session_scope(self):
                return _SessionScopeCtx(sink)

        # The pipeline imports session_scope directly; patch the name
        monkey.setattr(pipeline, "session_scope", _FakeDBModule().session_scope)

        # 5) app.models.Article → our stand-in
        monkey.setattr(pipeline, "Article", _FakeArticle)

        # Build a minimal source cfg
        src_cfg = {
            "name": "reuters-markets-rss",
            "compliance": {"user_agent": "Mozilla/5.0 ...", "obey_robots": False},
            "rate": {"per_minute": 20, "burst": 5},
            "discovery": {"urls": ["https://feeds.reuters.com/reuters/marketsNews"]},
            "article": {
                "js_render": False,
                "title_css": "h1",
                "body_css": "article p",
                "time_css": "time",
                "time_attr": "datetime",
                "author_css": 'a[href*="/authors/"]',
            },
        }

        # Run
        pipeline.run_source(src_cfg)

        # Assert result captured in sink
        assert len(sink) == 1, f"expected 1 insert, got {len(sink)}"
        rec = sink[0]
        assert rec["source"] == "reuters-markets-rss"
        assert "Fed cuts rates" in rec["title"]
        assert "Paragraph 1." in rec["body_text"]
        # published_at should be ISO string (from extractor)
        assert rec["published_at"] is None or rec["published_at"].startswith("2025-09-18")
        ok("pipeline.run_source smoke (monkey-patched)")
    except Exception as e:
        fail("pipeline.run_source smoke (monkey-patched)", e)
    finally:
        monkey.restore()

# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Running ingestion smoke tests...\n")
    test_extractor_basic()
    test_extractor_missing_author_ok()
    test_canonicalize_and_strip()
    test_simple_key()
    test_pipeline_smoke()

    print("\nSummary:")
    print(f"  Passed: {PASSED}")
    print(f"  Failed: {FAILED}")
    if FAILED:
        sys.exit(1)
    sys.exit(0)
