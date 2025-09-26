#!/usr/bin/env python3
"""
Initialize (or reset) the database schema.

Usage:
  python scripts/db_init.py
  python scripts/db_init.py --drop                 # drop then create
  python scripts/db_init.py --drop --yes          # no interactive prompt
  python scripts/db_init.py --vacuum              # SQLite: VACUUM after (re)create
  python scripts/db_init.py --wal                 # SQLite: enable WAL + NORMAL sync
  python scripts/db_init.py --fk                  # SQLite: enforce foreign_keys=ON
  python scripts/db_init.py --reset-status        # set NULL/empty article.status -> READY_FOR_SUMMARY
  python scripts/db_init.py --truncate-summaries  # delete from summaries (keep articles)
  python scripts/db_init.py --reindex             # rebuild indexes (SQLite)
  python scripts/db_init.py --show-tables         # list current DB tables
  python scripts/db_init.py --show-indexes        # list current DB indexes (SQLite)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine

from src.app.db import engine, DATABASE_URL
import src.app.models as m  # import all models to register tables

Base = m.Base
Article = m.Article
Summary = m.Summary
Embedding = m.Embedding
Cluster = m.Cluster
ArticleCluster = m.ArticleCluster


def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite")

def _is_sqlite_engine(engine: Engine) -> bool:
    try:
        return engine.url.get_backend_name() == "sqlite"
    except Exception:
        return _is_sqlite_url(str(engine.url))

def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() == "y"
    except EOFError:
        return False

def _sqlite_exec(sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql))

def enable_sqlite_wal():
    if _is_sqlite_engine(engine):
        _sqlite_exec("PRAGMA journal_mode=WAL;")
        _sqlite_exec("PRAGMA synchronous=NORMAL;")
        print("SQLite: WAL enabled, synchronous=NORMAL.")

def enable_sqlite_fk():
    if _is_sqlite_engine(engine):
        _sqlite_exec("PRAGMA foreign_keys=ON;")
        print("SQLite: foreign_keys=ON.")

def vacuum_sqlite():
    if _is_sqlite_engine(engine):
        _sqlite_exec("VACUUM;")
        print("SQLite: VACUUM complete.")

def reset_article_status():
    """Set NULL/empty statuses to READY_FOR_SUMMARY (idempotent)."""
    from sqlalchemy import update, or_
    with engine.begin() as conn:
        stmt = update(Article).where(
            or_(Article.status.is_(None), Article.status == "")
        ).values(status="READY_FOR_SUMMARY")
        res = conn.execute(stmt)
        print(f"Article status reset → READY_FOR_SUMMARY (rows affected: {res.rowcount}).")

def truncate_summaries():
    """Delete all rows from summaries (keeps articles)."""
    with engine.begin() as conn:
        if _is_sqlite_engine(engine):
            conn.execute(text("DELETE FROM summaries;"))
        else:
            conn.execute(text("TRUNCATE TABLE summaries;"))
    print("Cleared table: summaries.")

def reindex_sqlite():
    if _is_sqlite_engine(engine):
        _sqlite_exec("REINDEX;")
        print("SQLite: REINDEX complete.")

def show_tables():
    insp = inspect(engine)
    tables = insp.get_table_names()
    if not tables:
        print("No tables found in current database.")
    else:
        print("Current tables:")
        for t in sorted(tables):
            print(f"  - {t}")

def show_indexes():
    if not _is_sqlite_engine(engine):
        print("Index listing is currently implemented for SQLite only.")
        return
    insp = inspect(engine)
    tables = insp.get_table_names()
    if not tables:
        print("No tables.")
        return
    print("Current indexes (SQLite):")
    for t in sorted(tables):
        idxs = insp.get_indexes(t)
        if not idxs:
            print(f"  - {t}: (none)")
            continue
        print(f"  - {t}:")
        for ix in idxs:
            unique = " UNIQUE" if ix.get("unique") else ""
            cols = ", ".join(ix.get("column_names") or [])
            print(f"      • {ix['name']}{unique} ON ({cols})")

def _declared_index_names_from_metadata():
    names = set()
    for table in Base.metadata.tables.values():
        for idx in table.indexes:
            if idx.name:
                names.add(idx.name)
    return names

def drop_existing_sqlite_indexes_from_metadata():
    """Pre-drop any indexes that our metadata declares (SQLite only, safe)."""
    if not _is_sqlite_engine(engine):
        return
    names = _declared_index_names_from_metadata()
    if not names:
        return
    print(f"SQLite: pre-dropping {len(names)} declared index(es) if they exist …")
    with engine.begin() as conn:
        for name in sorted(names):
            conn.execute(text(f'DROP INDEX IF EXISTS "{name}";'))
    print("SQLite: index pre-drop complete.")

def main():
    ap = argparse.ArgumentParser(description="Initialize DB schema.")
    ap.add_argument("--drop", action="store_true", help="Drop all tables first.")
    ap.add_argument("--yes", action="store_true", help="Assume 'yes' to destructive prompts.")
    ap.add_argument("--vacuum", action="store_true", help="SQLite: VACUUM after changes.")
    ap.add_argument("--wal", action="store_true", help="SQLite: enable WAL mode + NORMAL sync.")
    ap.add_argument("--fk", action="store_true", help="SQLite: enforce PRAGMA foreign_keys=ON.")
    ap.add_argument("--reset-status", action="store_true", help="Set NULL/empty article.status to READY_FOR_SUMMARY.")
    ap.add_argument("--truncate-summaries", action="store_true", help="Delete all rows from summaries table.")
    ap.add_argument("--reindex", action="store_true", help="SQLite: REINDEX after (re)create.")
    ap.add_argument("--show-tables", action="store_true", help="List current DB tables and exit.")
    ap.add_argument("--show-indexes", action="store_true", help="List current DB indexes (SQLite) and exit.")
    args = ap.parse_args()

    print(f"DB URL: {DATABASE_URL}")

    if args.show_tables:
        show_tables()
        sys.exit(0)

    if args.show_indexes:
        show_indexes()
        sys.exit(0)

    # Optional PRAGMAs early
    if args.fk:
        enable_sqlite_fk()
    if args.wal:
        enable_sqlite_wal()

    # Destructive operations
    if args.truncate_summaries:
        if args.yes or _confirm("This will DELETE ALL summaries. Continue?"):
            truncate_summaries()
        else:
            print("Aborted.")

    if args.drop:
        if args.yes or _confirm("This will DROP ALL tables. Continue?"):
            print("Dropping all tables…")
            Base.metadata.drop_all(engine)
        else:
            print("Aborted drop.")
    else:
        drop_existing_sqlite_indexes_from_metadata()

    print("Creating tables (if not exist)…")
    Base.metadata.create_all(engine)

    # Hygiene / maintenance
    if args.reset_status:
        reset_article_status()
    if args.reindex:
        reindex_sqlite()
    if args.vacuum:
        vacuum_sqlite()

    backend = "SQLite" if _is_sqlite_engine(engine) else "Non-SQLite"
    print(f"✅ Done. Engine backend: {backend}")
    print("Tables present:", ", ".join(sorted(t.name for t in Base.metadata.sorted_tables)))

if __name__ == "__main__":
    main()
