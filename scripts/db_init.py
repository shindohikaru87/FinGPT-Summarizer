#!/usr/bin/env python3
"""
Initialize the database schema.

Usage:
  python scripts/db_init.py
  python scripts/db_init.py --drop          # drop then create
  python scripts/db_init.py --drop --vacuum # SQLite: VACUUM after recreate

Env:
  DATABASE_URL=sqlite:///./fingpt.db (default)  | e.g. postgresql+psycopg://user:pass@host/db
  SQL_ECHO=1  (optional: echo SQL)
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from src.app.db import engine, DATABASE_URL
from src.app.models import Base


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")

def main():
    ap = argparse.ArgumentParser(description="Initialize DB schema.")
    ap.add_argument("--drop", action="store_true", help="Drop all tables first.")
    ap.add_argument("--vacuum", action="store_true", help="SQLite: VACUUM after changes.")
    args = ap.parse_args()

    print(f"DB URL: {DATABASE_URL}")
    if args.drop:
        print("Dropping all tables…")
        Base.metadata.drop_all(engine)

    print("Creating tables (if not exist)…")
    Base.metadata.create_all(engine)

    if args.vacuum and _is_sqlite(DATABASE_URL):
        print("VACUUM (SQLite)…")
        with engine.begin() as conn:
            conn.execute(text("VACUUM"))
    print("✅ Done.")

if __name__ == "__main__":
    main()
