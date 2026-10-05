"""Active liveness session routes."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..deps import rate_limit
from ..errors import (
    AppError,
    LivenessLockedOutError,
    NotFoundError,
    ServiceUnavailableError,
)
from ..schemas import LivenessSessionCreate, LivenessSessionOut
from ..services.liveness.active import SUPPORTED_CHALLENGES
from ..services.liveness.geometry import eye_detector_status
from ..services.liveness.service import (
    create_session,
    get_session,
    liveness_available,
    submit_frame,
)
from ..utils.images import ImageDecodeError, decode_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/liveness", tags=["liveness"])


def _session_payload(row) -> dict[str, Any]:
    challenges = row.challenges or []
    return {
        "session_id": row.id,
        "status": row.status,
        "score": float(row.active_score or 0.0),
        "passed_challenges": [c["name"] for c in challenges if c.get("completed")],
        "reason": row.failure_reason,
        "prompt": next(
            (c["instruction"] for c in challenges if not c.get("completed")), None
        ),
        "active_challenge": next(
            (c for c in challenges if not c.get("completed")), None
        ),
        "challenges": challenges,
        "required": settings.active_liveness_min_challenges,
        "passive_score": row.passive_score,
        "expires_at": row.expires_at,
        "consumed": row.consumed_at is not None,
    }


@router.get(
    "/capabilities",
    summary="What liveness checks this deployment can currently perform",
    dependencies=[Depends(rate_limit)],
)
def capabilities() -> dict[str, Any]:
    live = liveness_available()
    return {
        **live,
        "supported_challenges": list(SUPPORTED_CHALLENGES),
        "configured_challenges": settings.challenge_pool,
        "min_challenges": settings.active_liveness_min_challenges,
        "passive_min": settings.passive_liveness_min,
        "session_ttl_seconds": settings.liveness_session_ttl_seconds,
        "max_attempts": settings.liveness_max_attempts,
        "eye_detector": eye_detector_status(),
        "how_to_pass": [
            "1. POST /liveness/sessions, keep the session_id.",
            "2. Show the returned `prompt` to the user.",
            "3. POST frames to /liveness/sessions/{id}/frames at 8-15 fps.",
            # Parens required: an unparenthesised implicit concat is one
            # stray edit away from silently dropping a step.
            (
                "4. When status == 'passed', send the session_id as liveness_session_id "
                "on exactly one attendance punch, together with the verification_id "
                "from /recognition/verify. It is single-use, and it authorises a punch "
                "only for the person the verification named."
            ),
        ],
    }


@router.post(
    "/sessions",
    response_model=LivenessSessionOut,
    status_code=201,
    summary="Start a randomised challenge-response session",
    dependencies=[Depends(rate_limit)],
)
def start_session(
    payload: LivenessSessionCreate, db: Session = Depends(get_db)
) -> LivenessSessionOut:
    result = create_session(db, user_id=payload.user_id)
    if result.get("session_id") is None:
        # Two refusal modes, two statuses, and neither is a 201 with a null id:
        # that shape leaves a kiosk unable to tell a refusal from a working
        # session, and it will happily prompt a user for a challenge that can
        # never be started.
        if result.get("status") == "locked_out":
            raise LivenessLockedOutError(
                result.get("reason") or "too many failed liveness attempts"
            )
        raise ServiceUnavailableError(
            f"active liveness is unavailable: {result.get('reason')}",
            code="liveness_unavailable",
        )
    return LivenessSessionOut(**result)


@router.post(
    "/sessions/{session_id}/frames",
    summary="Submit one webcam frame towards the active challenge",
    dependencies=[Depends(rate_limit)],
)
async def submit_liveness_frame(
    session_id: str,
    image: Annotated[UploadFile, File()],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    body = await image.read()
    if not body:
        raise AppError("uploaded frame is empty", code="empty_upload")
    if len(body) > settings.max_upload_bytes:
        raise AppError(
            f"frame is {len(body)} bytes, limit is {settings.max_upload_bytes}",
            code="image_too_large",
        )
    try:
        frame = decode_image(body)
    except ImageDecodeError as exc:
        # ImageDecodeError is a ValueError, not an AppError: no .detail.
        raise AppError(str(exc), code="bad_image") from exc

    return await run_in_threadpool(submit_frame, db, session_id, frame, source="api")


@router.get(
    "/sessions/{session_id}",
    response_model=LivenessSessionOut,
    summary="Read a session's persisted state",
    dependencies=[Depends(rate_limit)],
)
def read_session(session_id: str, db: Session = Depends(get_db)) -> LivenessSessionOut:
    row = get_session(db, session_id)
    if row is None:
        raise NotFoundError(f"no liveness session {session_id!r}", code="session_not_found")
    return LivenessSessionOut(**_session_payload(row))
