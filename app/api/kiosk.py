"""Simple kiosk route: one image in, attendance in/out written.

The liveness-free counterpart of ``/attendance/punch``, for testing recognition
and the in/out bookkeeping from a browser page or a script. Everything it
writes is visible through ``/attendance`` like any other punch, tagged
``source="kiosk"`` with no liveness session, so it can be filtered out later.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..deps import rate_limit
from ..errors import AppError, ServiceUnavailableError
from ..schemas import AttendanceOut
from ..services.face import get_analyzer
from ..services.kiosk import punch_frame
from ..utils.images import ImageDecodeError, decode_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kiosk", tags=["kiosk"])


@router.post(
    "/punch",
    summary="Recognise the face in one image and punch in/out (no liveness challenge)",
    dependencies=[Depends(rate_limit)],
)
async def kiosk_punch(
    image: Annotated[UploadFile, File(description="One JPEG/PNG frame from the camera")],
    db: Session = Depends(get_db),
    mode: Annotated[Literal["auto", "in", "out"], Form()] = "auto",
    device_id: Annotated[str | None, Form(max_length=64)] = None,
    dry_run: Annotated[bool, Form()] = False,
    user_id: Annotated[str | None, Form(max_length=64)] = None,
) -> dict[str, Any]:
    """``mode=auto``: first punch of the day is *in*, later ones move the *out*.

    ``dry_run=true`` identifies without writing. ``user_id`` turns the roster
    search into a 1:1 check against that person. The response always carries
    ``action`` (``in`` | ``out`` | ``identified`` | ``refused`` | ``no_match``)
    and a human-readable ``message``.
    """
    analyzer = get_analyzer()
    if analyzer.is_stub and not settings.allow_stub_backend:
        raise ServiceUnavailableError(
            "EMBEDDING_BACKEND=stub cannot write real attendance", code="stub_backend"
        )

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
        raise AppError(str(exc), code="bad_image") from exc

    outcome = await run_in_threadpool(
        punch_frame,
        db,
        frame,
        mode=mode,
        device_id=device_id,
        dry_run=dry_run,
        expected_user_id=(user_id or "").strip() or None,
    )
    payload = outcome.to_dict()
    payload["attendance"] = (
        AttendanceOut.model_validate(outcome.record).model_dump(mode="json")
        if outcome.record is not None
        else None
    )
    return payload
