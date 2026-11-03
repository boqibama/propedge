"""Engine and session factory."""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from propedge.config import ROOT, load_config
from propedge.db.models import Base


def make_engine(url: str | None = None) -> Engine:
    url = url or load_config()["paths"]["database_url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":memory:" not in url:
        path = ROOT / url.removeprefix("sqlite:///")       # relative paths are relative to the project
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
    # SQLite allows one writer at a time; wait up to 30 s for the lock instead of failing at once
    # (the poller, backfill and dashboard can run side by side).
    engine = create_engine(url, connect_args={"timeout": 30} if url.startswith("sqlite") else {})
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _record):  # WAL lets the dashboard read while the poller writes
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    return engine


def init_db(engine: Engine) -> None:
    """Create any missing tables. (Real migrations can come with the move to Postgres.)"""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
