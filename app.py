"""Development entrypoint: ``uv run app.py``.

Why this file exists
--------------------
The application object is ``app.main:app``. Running the service normally means::

    uv run uvicorn app.main:app --reload

which is what you want in production and in CI. This script is the
short path for local work -- it starts the same app with the settings a
developer actually wants (reload on save, auto-reload of the env file) and
prints where to look.

It is a *launcher*, not a second application. There is exactly one app
factory, :func:`app.main.create_app`, and this calls it. Nothing is
configured here that is not already in ``app/config.py``.

Note on naming
--------------
A module named ``app.py`` sits next to the ``app/`` package. Python resolves
the package first, so ``import app`` still gets the package and this file
does not shadow it. ``uvicorn app.main:app`` is unaffected.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

# If this file is run directly, the repo root is already sys.path[0], so
# `app` imports normally. Fail loudly rather than mysteriously if not.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the face attendance API locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Bind address. Use 0.0.0.0 to expose it on the LAN.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Bind port.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes (dev only; never in production).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level.",
    )
    return parser.parse_args()


def _preflight() -> None:
    """Fail early, and legibly, on the two things that usually go wrong.

    Both of these are discovered far more confusingly on the first request
    from a customer than they are here.
    """
    from app.config import settings

    if settings.embedding_backend == "deepface":
        try:
            importlib_import_deepface()
        except Exception as exc:  # noqa: BLE001 - any module-level failure in TF counts
            print(
                f"WARNING: EMBEDDING_BACKEND=deepface but deepface will not import "
                f"({type(exc).__name__}: {exc}).\n"
                "         The API will boot, but recognition requests will fail "
                "with 503.\n"
                "         Install it with:  uv sync --extra face",
                file=sys.stderr,
            )

    if sys.version_info >= (3, 13):
        print(
            f"WARNING: running on Python {sys.version_info.major}."
            f"{sys.version_info.minor}. TensorFlow (via deepface) has no wheels "
            "for 3.13+.\n"
            "         Use Python 3.11 or 3.12 for real recognition.",
            file=sys.stderr,
        )


def importlib_import_deepface() -> None:
    """Import deepface, raising on failure. Kept separate for testability."""
    import deepface  # noqa: F401


def main() -> int:
    args = _parse_args()

    # Do this before anything imports or prints: a Windows console on cp1252
    # turns a single undecodable log glyph into a failed import. DeepFace
    # prints one on import, which would leave the app running with no face
    # backend. Harmless everywhere else.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")

    _preflight()

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn is not installed. Run:  uv sync --extra dev",
            file=sys.stderr,
        )
        return 1

    from app import __version__
    from app.config import settings

    print(f"face-attendance-api v{__version__}  env={settings.environment}")
    print(f"  docs     http://{args.host}:{args.port}/docs")
    print(f"  health   http://{args.host}:{args.port}/health/ready")
    print(f"  backend  {settings.embedding_backend} / {settings.face_model_name}")
    print(f"  liveness required={settings.require_liveness}")
    print()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        # Workers are incompatible with reload, and a face model per worker is
        # expensive. Scale with more processes/containers, not --workers here.
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
