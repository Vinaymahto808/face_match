"""Stateless single-frame verification."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..deps import rate_limit
from ..errors import AppError
from ..schemas import VerifyRequest, VerifyResponse
from ..services import registry as reg
from ..services import verifications
from ..services.pipeline import verify_frame
from ..utils.images import ImageDecodeError, decode_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recognition", tags=["recognition"])


def _response_payload(result, request_id: str, verification_id: str | None) -> dict[str, Any]:
    payload = result.to_dict()
    match = payload.pop("match", None)
    liveness = payload.pop("liveness", {})
    payload["request_id"] = request_id
    payload["verification_id"] = verification_id
    payload["match"] = (
        {
            **(match or {}),
            "threshold": settings.match_threshold,
            "gray_zone_threshold": settings.gray_zone_threshold,
        }
        if match
        else None
    )
    payload["liveness"] = {
        "score": liveness.get("passive"),
        "passive": liveness.get("passive"),
        "passive_verdict": liveness.get("passive_verdict"),
        "flags": liveness.get("flags", []),
        "engine": liveness.get("engine"),
        "active_required": settings.require_liveness,
    }
    return payload


async def _decode(payload: bytes):
    try:
        return decode_image(payload)
    except ImageDecodeError as exc:
        raise AppError(str(exc), code="bad_image") from exc


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Identify (1:N) or verify (1:1) the face in one image",
    dependencies=[Depends(rate_limit)],
)
async def verify_json(
    payload: VerifyRequest, db: Session = Depends(get_db)
) -> VerifyResponse:
    request_id = uuid.uuid4().hex[:12]
    # decode_image accepts a bare base64 string or a full data: URI.
    try:
        frame = decode_image(payload.image_base64)
    except ImageDecodeError as exc:
        # ImageDecodeError is a ValueError, not an AppError: it has no
        # .detail attribute, so it has to be stringified.
        raise AppError(str(exc), code="bad_image") from exc

    # Face inference is CPU/GPU heavy: keep it off the event loop.
    result = await run_in_threadpool(
        verify_frame, db, frame, expected_user_id=payload.user_id
    )
    return VerifyResponse(**_response_payload(result, request_id, _grant_id(db, result)))


def _grant_id(db: Session, result) -> str | None:
    """Mint the single-use identity proof for an eligible frame.

    Written *after* the frame has been scored, so the grant can only ever carry
    numbers this service measured. Ineligible frames (no face, review-range
    match, spoof-suspect, failed quality) get no grant at all, which is what
    makes "you may punch" a server-side statement rather than a client one.
    """
    grant = verifications.issue_from_result(db, result)
    return grant.id if grant else None


@router.post(
    "/verify/upload",
    response_model=VerifyResponse,
    summary="Same as /verify but takes multipart/form-data",
    dependencies=[Depends(rate_limit)],
)
async def verify_upload(
    image: Annotated[UploadFile, File()],
    db: Session = Depends(get_db),
    user_id: Annotated[str | None, Form(max_length=64)] = None,
) -> VerifyResponse:
    request_id = uuid.uuid4().hex[:12]
    body = await image.read()
    if not body:
        raise AppError("uploaded image is empty", code="empty_upload")
    if len(body) > settings.max_upload_bytes:
        raise AppError(
            f"image is {len(body)} bytes, limit is {settings.max_upload_bytes}",
            code="image_too_large",
        )
    frame = await _decode(body)
    result = await run_in_threadpool(verify_frame, db, frame, expected_user_id=user_id)
    return VerifyResponse(**_response_payload(result, request_id, _grant_id(db, result)))


@router.get(
    "/roster",
    summary="Current in-memory roster snapshot and matching configuration",
    dependencies=[Depends(rate_limit)],
)
def roster_info(db: Session = Depends(get_db)) -> dict[str, Any]:
    snapshot = reg.roster_cache.get(db)
    return {
        "count": snapshot.count,
        "model": snapshot.model_name,
        "embedding_dim": snapshot.dim,
        "skipped_model_mismatch": snapshot.skipped_model_mismatch,
        "match_threshold": settings.match_threshold,
        "gray_zone_threshold": settings.gray_zone_threshold,
        "require_liveness": settings.require_liveness,
        "loaded_at": snapshot.loaded_at,
    }
