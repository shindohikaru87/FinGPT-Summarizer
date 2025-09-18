# src/app/db.py
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fingpt.db")
SQL_ECHO = os.getenv("SQL_ECHO", "0") == "1"

engine_kwargs = dict(future=True, echo=SQL_ECHO)
if DATABASE_URL.startswith("sqlite"):
    # allow use across threads (ingestion can be chatty)
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # better resiliency for networked DBs
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(DATABASE_URL, **engine_kwargs)

# src/app/db.py
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,   # <-- add this
)

@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
