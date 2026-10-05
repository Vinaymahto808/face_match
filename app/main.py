"""FastAPI application factory."""

from __future__ import annotations

import logging
import logging.config
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api import (
    attendance_router,
    events_router,
    health_router,
    kiosk_router,
    liveness_router,
    stream_router,
    users_router,
    verify_router,
)
from .config import settings
from .db import init_db
from .errors import AppError
from .services.face import get_analyzer
from .services.liveness.service import registry as liveness_registry

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

logger = logging.getLogger("app")


def _force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 with a lossy fallback.

    On Windows the console defaults to a legacy code page (cp1252 on most
    Western installs). DeepFace prints a warning glyph on import, which
    raises ``UnicodeEncodeError`` -- and that kills ``import deepface``
    outright, leaving the service running with no face backend and a
    readiness probe that never goes green.

    ``errors="replace"`` matters as much as the encoding: a third-party
    library printing an undecodable character should never take down the
    process that merely wanted to log about it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic stream
            logger.debug("could not reconfigure %r to utf-8", stream, exc_info=True)


def configure_logging() -> None:
    _force_utf8_stdio()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": LOG_FORMAT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
            "loggers": {
                # Uvicorn's own access log is duplicated by our timing middleware.
                "uvicorn.access": {"level": "WARNING", "handlers": ["console"], "propagate": False},
                "sqlalchemy.engine": {"level": "WARNING"},
            },
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Model warmup runs in a thread so boot is not blocked."""
    import anyio.to_thread

    configure_logging()
    logger.info("starting %s v%s (env=%s)", settings.app_name, __version__, settings.environment)

    init_db()

    # Touch the analyzer so a missing deepface install is discovered at boot
    # rather than on the first customer's request.
    try:
        await anyio.to_thread.run_sync(lambda: get_analyzer().warmup())
    except Exception as exc:  # noqa: BLE001 - never block boot on a warmup failure
        logger.error("face backend warmup failed: %s", exc)

    if settings.require_liveness:
        from .services.liveness.service import liveness_available

        availability = liveness_available()
        if not availability["available"]:
            logger.error(
                "REQUIRE_LIVENESS=true but liveness is UNAVAILABLE (%s). "
                "Attendance punches will be refused until this is fixed.",
                "; ".join(availability["reasons"]),
            )
        else:
            logger.info("active liveness ready")

    app.state.started_at = time.time()
    try:
        yield
    finally:
        liveness_registry.clear()
        logger.info("shutdown complete")


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Coerce pydantic's error payload into something ``json.dumps`` accepts.

    ``RequestValidationError.errors()`` embeds the offending ``input`` verbatim.
    For a JSON endpoint that is usually a str/int, but for a body FastAPI could
    not parse at all it is the raw ``bytes`` -- and ``JSONResponse`` then raises
    ``TypeError: Object of type bytes is not JSON serializable`` *inside the
    exception handler*. The client gets an opaque 500 for what was simply a
    malformed request, and the traceback echoes their whole payload back.

    Anything unrecognised becomes a short repr, truncated so a large blob cannot
    ride along in an error body. Depth is bounded because ``ctx`` may hold
    objects whose repr recurses.
    """
    if _depth > 6:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        # Never echo a whole upload back at the client.
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "..."


def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Attendance & Anti-Spoofing API",
        version=__version__,
        description=(
            "Face recognition attendance with active (challenge-response) and "
            "passive (texture/spectrum) anti-spoofing. Camera capture lives on "
            "the client; the server never opens a device."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request id, log the outcome, and never leak a stack trace."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logging.getLogger("app").exception(
                "unhandled error on %s %s [%s]", request.method, request.url.path, request_id
            )
            if settings.is_prod:
                response = JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "code": "internal_error",
                            "message": "internal server error",
                            "request_id": request_id,
                        }
                    },
                )
            else:
                raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if not request.url.path.startswith(("/health", "/docs", "/redoc", "/openapi")):
            logging.getLogger("app").info(
                "%s %s -> %s in %.1fms [%s]",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                request_id,
            )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                **exc.to_payload(),
                "request_id": getattr(request.state, "request_id", None),
            },
            headers={"WWW-Authenticate": "ApiKey"} if exc.status_code == 401 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "details": _json_safe(exc.errors()),
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    prefix = settings.api_prefix.rstrip("/")
    app.include_router(health_router)  # unversioned: /health must stay stable
    app.include_router(users_router, prefix=prefix)
    app.include_router(verify_router, prefix=prefix)
    app.include_router(liveness_router, prefix=prefix)
    app.include_router(attendance_router, prefix=prefix)
    app.include_router(events_router, prefix=prefix)
    app.include_router(stream_router, prefix=prefix)
    app.include_router(kiosk_router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health/ready",
            "api_prefix": prefix,
        }

    return app


app = create_app()
