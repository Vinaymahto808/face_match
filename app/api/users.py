"""User enrolment and roster management."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..deps import rate_limit
from ..errors import AppError, NotFoundError, ServiceUnavailableError, ValidationFailedError
from ..schemas import UserList, UserOut
from ..services import embeddings as emb
from ..services import events
from ..services import registry as reg
from ..services.face import get_analyzer
from ..services.liveness.passive import passive_liveness
from ..services.quality import assess_face, to_gray
from ..utils.images import ImageDecodeError, decode_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


async def _read_image(image: UploadFile) -> bytes:
    payload = await image.read()
    if not payload:
        raise AppError("uploaded image is empty", code="empty_upload")
    if len(payload) > settings.max_upload_bytes:
        raise AppError(
            f"image is {len(payload)} bytes, limit is {settings.max_upload_bytes}",
            code="image_too_large",
        )
    return payload


def _embed_from_image(
    image_bgr: np.ndarray, *, allow_low_quality: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quality-gate, anti-spoof-check and embed one enrolment image.

    Returns ``(embedding, info)``. Rejecting a low-quality enrolment is the
    highest-leverage accuracy decision in the whole system: a bad reference
    embedding makes *every* future comparison against that person worse.
    """
    analyzer = get_analyzer()
    if analyzer.is_stub and not settings.allow_stub_backend:
        raise ServiceUnavailableError(
            "EMBEDDING_BACKEND=stub: enrolment is disabled because synthetic embeddings "
            "would poison the roster. Set ALLOW_STUB_BACKEND=true only for local testing."
        )

    observations = analyzer.analyze(image_bgr, strict=True)
    if not observations:
        raise AppError(
            "no face detected in the enrolment image",
            code="no_face_detected",
            context={"hint": "face the camera, improve lighting, remove obstructions"},
        )
    if len(observations) > 1:
        # Ambiguous: we cannot know which face is being enrolled.
        raise AppError(
            f"{len(observations)} faces detected; submit a crop containing exactly one",
            code="multiple_faces_detected",
            context={"faces": [list(o.bbox) for o in observations]},
        )

    obs = observations[0]
    crop = obs.crop(image_bgr)
    quality = assess_face(to_gray(crop), obs.w, obs.h)

    passive = passive_liveness(crop, threshold=settings.passive_liveness_min)

    info: dict[str, Any] = {
        "face_quality": quality.to_dict(),
        "passive_liveness": passive.to_dict(),
        "bbox": list(obs.bbox),
    }

    if not quality.ok and not allow_low_quality:
        raise AppError(
            f"enrolment image failed the quality gate: {', '.join(quality.reasons)}",
            code="low_quality_image",
            context=info,
        )

    if passive.verdict == "spoof_suspect" and not allow_low_quality:
        raise AppError(
            "enrolment image looks like a re-capture "
            f"(score {passive.score:.2f}; {', '.join(passive.flags) or 'no detail'}). "
            "Enrol from a live frame, not a photo or a screen.",
            code="spoof_suspect_enrolment",
            context=info,
        )

    if obs.embedding is None:
        raise AppError("could not compute an embedding for the detected face", code="no_embedding")

    return emb.coerce_vector(obs.embedding), info


@router.post(
    "",
    response_model=UserOut,
    status_code=201,
    summary="Enrol a person from a single image",
    dependencies=[Depends(rate_limit)],
)
async def create_user(
    user_id: Annotated[str, Form(alias="id", max_length=64)],
    name: Annotated[str, Form(max_length=160)],
    image: Annotated[UploadFile, File(description="Single face, front-facing, well lit")],
    db: Session = Depends(get_db),
    employee_code: Annotated[str | None, Form(max_length=64)] = None,
    notes: Annotated[str | None, Form(max_length=2000)] = None,
    allow_low_quality: Annotated[bool, Form()] = False,
) -> UserOut:
    """Register a person. Re-using an existing ``id`` replaces the embedding."""
    payload = await _read_image(image)
    try:
        frame = decode_image(payload)
    except ImageDecodeError as exc:
        raise AppError(str(exc), code="bad_image") from exc

    try:
        vector, info = _embed_from_image(frame, allow_low_quality=allow_low_quality)
    except AppError as exc:
        events.record_event(
            db,
            kind="enroll_rejected",
            severity="warning",
            user_id=user_id,
            message=f"Enrolment rejected for {user_id}: {exc.detail}",
            context=exc.context or {"code": exc.code},
        )
        raise

    try:
        user = reg.create_user(
            db,
            user_id=user_id,
            name=name,
            embedding=vector,
            sample_count=1,
            quality_score=info["face_quality"]["score"],
            employee_code=employee_code,
            notes=notes,
        )
    except ValueError as exc:
        # registry validates the id charset/name and raises ValueError. Surface
        # it as 422, not an unhandled 500.
        raise ValidationFailedError(str(exc)) from exc
    return UserOut.model_validate(user)


@router.post(
    "/enroll/start",
    status_code=201,
    summary="Start a multi-sample enrolment session (recommended)",
    dependencies=[Depends(rate_limit)],
)
def start_enrollment(
    user_id: Annotated[str, Form(alias="id", max_length=64)],
    name: Annotated[str, Form(max_length=160)],
    db: Session = Depends(get_db),
    employee_code: Annotated[str | None, Form(max_length=64)] = None,
    samples: Annotated[int, Form(ge=2, le=30)] = 5,
) -> dict[str, Any]:
    """Several live frames of the same person, averaged into one embedding.

    Cheapest real accuracy win in the system: averaging shrinks intra-class
    variance and widens the genuine/impostor gap.
    """
    try:
        session = reg.start_enrollment(
            db,
            user_id=user_id,
            user_name=name,
            employee_code=employee_code,
            samples_needed=samples,
        )
    except ValueError as exc:
        raise ValidationFailedError(str(exc)) from exc
    return {
        "enrollment_id": session.id,
        "user_id": session.user_id,
        "samples_needed": session.samples_needed,
        "status": session.status,
        "expires_at": session.expires_at,
    }


@router.post(
    "/enroll/{enrollment_id}/sample",
    summary="Add one live frame to an enrolment session",
    dependencies=[Depends(rate_limit)],
)
async def add_enrollment_sample(
    enrollment_id: str,
    image: Annotated[UploadFile, File()],
    db: Session = Depends(get_db),
    allow_low_quality: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    payload = await _read_image(image)
    try:
        frame = decode_image(payload)
    except ImageDecodeError as exc:
        raise AppError(str(exc), code="bad_image") from exc

    try:
        vector, info = _embed_from_image(frame, allow_low_quality=allow_low_quality)
    except AppError as exc:
        # A bad sample is not fatal to the session: the client can just send
        # another frame. Report the reason and keep going.
        return {
            "enrollment_id": enrollment_id,
            "accepted": False,
            "reason": exc.detail,
            "code": exc.code,
            "context": exc.context,
        }

    try:
        session = reg.add_enrollment_sample(
            db, enrollment_id, vector, quality_score=info["face_quality"]["score"]
        )
    except LookupError as exc:
        raise NotFoundError(str(exc), code="enrollment_not_found") from exc
    except ValueError as exc:
        raise AppError(str(exc), code="enrollment_closed") from exc

    captured = len(session.samples)
    return {
        "enrollment_id": enrollment_id,
        "accepted": True,
        "samples_captured": captured,
        "samples_needed": session.samples_needed,
        "ready_to_finalize": captured >= 2,
        "status": session.status,
    }


@router.post(
    "/enroll/{enrollment_id}/finalize",
    response_model=UserOut,
    summary="Average the samples and write the roster entry",
    dependencies=[Depends(rate_limit)],
)
def finalize_enrollment(enrollment_id: str, db: Session = Depends(get_db)) -> UserOut:
    try:
        user = reg.finalize_enrollment(db, enrollment_id)
    except LookupError as exc:
        raise NotFoundError(str(exc), code="enrollment_not_found") from exc
    except ValueError as exc:
        raise AppError(str(exc), code="not_enough_samples") from exc
    return UserOut.model_validate(user)


@router.get(
    "",
    response_model=UserList,
    summary="List registered people",
    dependencies=[Depends(rate_limit)],
)
def list_users(
    db: Session = Depends(get_db),
    include_inactive: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> UserList:
    users = reg.list_users(db, include_inactive=include_inactive, limit=limit, offset=offset)
    return UserList(
        items=[UserOut.model_validate(u) for u in users],
        total=len(users),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{user_id}",
    response_model=UserOut,
    summary="Fetch one person",
    dependencies=[Depends(rate_limit)],
)
def get_user(user_id: str, db: Session = Depends(get_db)) -> UserOut:
    user = reg.get_user(db, user_id)
    if user is None:
        raise NotFoundError(f"no user with id {user_id!r}", code="user_not_found")
    return UserOut.model_validate(user)


@router.delete(
    "/{user_id}",
    summary="Deactivate a person",
    dependencies=[Depends(rate_limit)],
)
def delete_user(user_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not reg.deactivate_user(db, user_id):
        raise NotFoundError(f"no user with id {user_id!r}", code="user_not_found")
    return {"id": user_id, "is_active": False}
