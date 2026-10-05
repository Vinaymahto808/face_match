"""Database URL resolution.

A relative ``DATABASE_URL`` is a footgun in a service that gets started from
more than one place. These tests pin the behaviour: the path is resolved against
the project root, and the *resolved* path is what the engine is opened with.
"""

from __future__ import annotations

from pathlib import Path

from app.db import PROJECT_ROOT, _resolved_sqlite_url, build_engine


def test_a_relative_sqlite_path_is_pinned_to_the_project_root(monkeypatch):
    """Not the CWD. Two starts of the same app must not open two databases."""
    monkeypatch.chdir(Path("C:/"))  # a plausible systemd WorkingDirectory
    resolved = _resolved_sqlite_url("sqlite:///./data/attendance.db")
    assert Path(resolved[len("sqlite:///") :]) == PROJECT_ROOT / "data" / "attendance.db"


def test_the_resolved_path_reaches_the_engine(monkeypatch, tmp_path):
    """Regression: the old code resolved the path, made the directory, and then
    handed ``create_engine`` the *original relative* URL -- so the directory was
    created under the CWD while the database was opened somewhere else again.
    """
    from app.db import settings as db_settings

    monkeypatch.chdir(tmp_path)
    target = PROJECT_ROOT / "data" / "db_path_probe.db"
    target.unlink(missing_ok=True)
    monkeypatch.setattr(db_settings, "database_url", "sqlite:///./data/db_path_probe.db", raising=False)
    try:
        engine = build_engine()
        assert str(engine.url).endswith("data/db_path_probe.db")
        assert str(engine.url).startswith("sqlite:///")
        assert (PROJECT_ROOT / "data").is_dir()
        # Nothing was created under the CWD, which is the whole point.
        assert not (tmp_path / "data" / "db_path_probe.db").exists()
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)


def test_an_absolute_sqlite_path_is_left_alone(tmp_path):
    absolute = (tmp_path / "elsewhere.db").as_posix()
    assert _resolved_sqlite_url(f"sqlite:///{absolute}") == f"sqlite:///{absolute}"


def test_memory_and_non_sqlite_urls_pass_through():
    """In-memory is not a file, and postgres has no path to resolve."""
    for url in ("sqlite:///:memory:", "sqlite://", "postgresql+psycopg://u:p@h/db"):
        assert _resolved_sqlite_url(url) == url
