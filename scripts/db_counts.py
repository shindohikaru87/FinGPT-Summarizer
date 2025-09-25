#!/usr/bin/env python3
"""
Show row counts for all tables in the database.
"""

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from src.app.db import engine  # reuse your project’s engine

def main():
    try:
        insp = inspect(engine)
        tables = insp.get_table_names()
        print(f"Found {len(tables)} tables.\n")

        with engine.connect() as conn:
            for t in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                    count = result.scalar()
                    print(f"{t:<25} {count}")
                except SQLAlchemyError as e:
                    print(f"{t:<25} ERROR: {e}")
    except Exception as e:
        print("Failed to inspect database:", e)

if __name__ == "__main__":
    main()
