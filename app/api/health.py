"""Liveness and readiness probes."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import __version__
from ..config import settings
from ..db import get_db
from ..schemas import HealthResponse
from ..services.face import analyzer_status
from ..services.liveness.geometry import eye_detector_status
from ..services.liveness.service import liveness_available, registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


def _db_check(db: Session) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:
        logger.exception("database health check failed")
        return {"ok": False, "error": str(exc)}


@router.get("/health", response_model=HealthResponse, summary="Combined health report")
def health(db: Session = Depends(get_db)) -> HealthResponse:
    db_status = _db_check(db)
    backend = analyzer_status()
    live = liveness_available()

    checks: dict[str, Any] = {
        "database": db_status,
        "face_backend": backend,
        "liveness": {
            "available": live["available"],
            "reasons": live["reasons"],
            "eye_detector": eye_detector_status(),
            "active_sessions": registry.size(),
        },
        "config": {
            "model": settings.face_model_name,
            "backend": settings.embedding_backend,
            "match_threshold": settings.match_threshold,
            "require_liveness": settings.require_liveness,
            "business_timezone": settings.business_timezone,
            "auth_required": settings.auth_required,
        },
    }

    # "ok" means the service can answer requests. Liveness being unavailable is
    # reported but not fatal for /health -- it is fatal for punching attendance,
    # which the readiness probe covers.
    healthy = db_status["ok"]
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        environment=settings.environment,
        checks=checks,
    )


@router.get("/health/live", summary="Liveness probe (process is up)")
def liveness_probe() -> dict[str, Any]:
    # Not typed as dict[str, str]: uptime_s is a float, and a str value type
    # would make FastAPI's response validation reject the whole probe.
    return {"status": "alive", "uptime_s": round(time.time() - _STARTED_AT, 1)}


@router.get(
    "/health/ready",
    summary="Readiness probe (dependencies are usable)",
    response_model=HealthResponse,
)
def readiness(
    response: Response, db: Session = Depends(get_db)
) -> HealthResponse:
    db_status = _db_check(db)
    backend = analyzer_status()
    live = liveness_available()

    reasons: list[str] = []
    if not db_status["ok"]:
        reasons.append("database_unavailable")
    if backend["is_stub"] and not settings.allow_stub_backend:
        reasons.append("stub_embedding_backend")
    if settings.require_liveness and not live["available"]:
        # Attendance is impossible without this, so we are genuinely not ready.
        reasons.extend(live["reasons"] or ["liveness_unavailable"])
    if not backend["ready"] and not backend["is_stub"]:
        reasons.append("face_backend_not_loaded")

    if reasons:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ready" if not reasons else "not_ready",
        version=__version__,
        environment=settings.environment,
        checks={
            "database": db_status,
            "face_backend": backend,
            "liveness": {**live, "active_sessions": registry.size()},
            "not_ready_reasons": reasons,
        },
    )
