"""Pydantic request/response models for the public API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import settings

# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """JSON enrolment variant. Prefer the multipart route for images."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, examples=["EMP101"])
    name: str = Field(min_length=1, max_length=160, examples=["John Doe"])
    employee_code: str | None = Field(default=None, max_length=64)
    image_base64: str = Field(description="Raw base64 JPEG/PNG, or a data: URI.")
    notes: str | None = None
    allow_low_quality: bool = Field(
        default=False,
        description="Accept a face that fails the quality gate. For debugging only.",
    )

    @field_validator("id")
    @classmethod
    def _id_charset(cls, v: str) -> str:
        v = v.strip()
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("id may only contain letters, digits, '-' and '_'")
        return v


class UserOut(BaseModel):
    id: str
    name: str
    employee_code: str | None
    model_name: str
    embedding_dim: int
    sample_count: int
    quality_score: float | None
    is_active: bool
    notes: str | None
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = ConfigDict(from_attributes=True)


class UserList(BaseModel):
    items: list[UserOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Recognition / verification
# ---------------------------------------------------------------------------
Decision = Literal["match", "review", "unknown"]


class QualityOut(BaseModel):
    ok: bool
    score: float
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class MatchOut(BaseModel):
    decision: Decision
    user_id: str | None
    name: str | None
    distance: float | None
    runner_up_id: str | None
    runner_up_distance: float | None
    margin: float | None
    threshold: float
    gray_zone_threshold: float


class LivenessOut(BaseModel):
    score: float | None = None
    passive: float | None = None
    passive_verdict: str | None = None
    flags: list[str] = Field(default_factory=list)
    engine: str | None = None
    detail: dict[str, Any] | None = None
    active_required: bool = Field(
        default=False,
        description="Whether an active liveness session is needed before punching. "
        "Clients use this to decide between the fast path and the challenge flow.",
    )


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: str
    user_id: str | None = Field(
        default=None,
        description="Expected identity. If given, the response is a 1:1 verification "
        "of that person rather than a 1:N search.",
    )


class VerifyResponse(BaseModel):
    face_present: bool
    faces_detected: int
    frame_quality: QualityOut
    # Nullable: a frame can be rejected before matching (no face, failed quality
    # gate, backend error), and the response must still describe why.
    match: MatchOut | None = None
    identity_match: bool | None = Field(
        default=None,
        description="For 1:1 requests: did the face match the claimed identity?",
    )
    liveness: LivenessOut
    eligible_for_attendance: bool
    block_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    verification_id: str | None = Field(
        default=None,
        description=(
            "Single-use proof of identity for this frame, or null when the face is "
            "not eligible. Send it as `verification_id` on the attendance punch: the "
            "punch endpoint takes the person and the match distance from here, never "
            "from the client."
        ),
    )
    request_id: str


# ---------------------------------------------------------------------------
# Liveness sessions
# ---------------------------------------------------------------------------
LivenessStatus = Literal[
    "pending", "passed", "failed", "expired", "unavailable", "unknown_session", "stale_session"
]


class LivenessSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None


class LivenessChallengeOut(BaseModel):
    name: str
    index: int
    instruction: str
    revealed: bool = False
    completed: bool = False
    progress: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class LivenessSessionOut(BaseModel):
    session_id: str | None
    status: str
    score: float = 0.0
    passed_challenges: list[str] = Field(default_factory=list)
    reason: str | None = None
    prompt: str | None = None
    active_challenge: dict[str, Any] | None = None
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    required: int = settings.active_liveness_min_challenges
    passive_score: float | None = None
    signals: dict[str, Any] | None = None
    frame_ok: bool | None = None
    consumed: bool = Field(
        default=False,
        description="True once this session has authorised an attendance punch. "
        "Tokens are single-use.",
    )
    expires_at: dt.datetime | None = None
    max_challenges: int | None = None


class LivenessFrameResponse(BaseModel):
    session_id: str
    status: str
    score: float = 0.0
    passed_challenges: list[str] = Field(default_factory=list)
    reason: str | None = None
    prompt: str | None = None
    active_challenge: dict[str, Any] | None = None
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    passive_score: float | None = None
    signals: dict[str, Any] | None = None
    frame_ok: bool | None = None
    db_status: str | None = None


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
class PunchRequest(BaseModel):
    """The punch body. Note what is *not* here.

    There is no ``user_id`` and no ``match_distance``. Both used to be fields,
    and both were things the caller got to choose: the first named whoever the
    punch was for, the second decided whether the identity had been verified.
    The proof is now a server-issued ``verification_id`` that names the person
    and carries the distance the matcher measured, so the punch has nothing to
    take on trust. ``extra="forbid"`` turns a legacy client into a loud 422
    rather than a silently-ignored field.
    """

    model_config = ConfigDict(extra="forbid")

    verification_id: str = Field(
        min_length=16,
        max_length=64,
        description="Single-use proof of identity from /recognition/verify. "
        "Names the person to mark and carries the measured match distance.",
    )
    liveness_session_id: str | None = Field(
        default=None,
        description="Required when REQUIRE_LIVENESS=true. Single-use token from a "
        "passed liveness session, and only valid for the person it was started for.",
    )
    liveness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    device_id: str | None = Field(default=None, max_length=64)
    source: Literal["api", "stream", "kiosk"] = "api"


class AttendanceOut(BaseModel):
    id: int
    user_id: str
    user_name: str
    work_date: str
    first_in_at: dt.datetime
    last_seen_at: dt.datetime
    check_out_at: dt.datetime | None
    punch_count: int
    match_distance: float
    liveness_score: float
    passive_liveness: float | None
    liveness_session_id: str | None
    spoof_flags: list[str]
    source: str
    device_id: str | None

    model_config = ConfigDict(from_attributes=True)


class PunchResponse(BaseModel):
    attendance: AttendanceOut
    created: bool
    duplicate_suppressed: bool
    message: str


class AttendanceList(BaseModel):
    items: list[AttendanceOut]
    total: int
    limit: int
    offset: int
    # Mixed value types: "day" is a string, the counters are ints.
    summary: dict[str, Any] | None = None


class CheckOutResponse(BaseModel):
    attendance: AttendanceOut | None
    message: str


# ---------------------------------------------------------------------------
# Events / health
# ---------------------------------------------------------------------------
class EventOut(BaseModel):
    id: int
    created_at: dt.datetime
    severity: str
    kind: str
    user_id: str | None
    message: str
    context: dict[str, Any]
    acknowledged: bool

    model_config = ConfigDict(from_attributes=True)


class EventList(BaseModel):
    items: list[EventOut]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, Any]
