"""SQLAlchemy engine/session wiring plus first-run schema creation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base

logger = logging.getLogger(__name__)


#: Anchor for a relative `DATABASE_URL`. Deliberately *not* the process CWD:
#: `uvicorn app.main:app` from the repo root, a systemd unit with no
#: `WorkingDirectory`, and a container whose `WORKDIR` is `/app` would each open
#: a different file for the same configured string -- and the first two would
#: silently start with an empty roster.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sqlite_url_to_path(url: str) -> Path | None:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return None
    raw = url[len(prefix) :]
    if raw == ":memory:" or raw.startswith(":memory:"):
        return None
    return Path(raw)


def _resolved_sqlite_url(url: str) -> str:
    """Pin a relative sqlite path to :data:`PROJECT_ROOT`.

    The resolved path has to reach ``create_engine``, not just the ``mkdir``:
    resolving it locally and then handing the original relative URL onwards
    creates the directory in one place and opens the database in another.
    """
    path = _sqlite_url_to_path(url)
    if path is None:
        return url
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"{'sqlite:///'}{path.as_posix()}"


def build_engine() -> Engine:
    url = settings.database_url
    is_sqlite = url.startswith("sqlite")
    if is_sqlite:
        url = _resolved_sqlite_url(url)

    kwargs: dict = {"echo": settings.db_echo, "future": True}
    if is_sqlite:
        # FastAPI serves requests from a threadpool, so the connection must not
        # be pinned to the thread that created it.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}

    engine = create_engine(url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            # WAL lets the WebSocket loop write while request handlers read.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.close()

    return engine


engine: Engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create tables that don't exist yet.

    Deliberately not a migration tool: use Alembic once the schema stabilises.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("schema ready (%s tables)", len(Base.metadata.tables))


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one Session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for code outside the request cycle (WS, scripts)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
