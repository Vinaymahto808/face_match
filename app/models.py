"""ORM models.

Design notes
------------
* Embeddings are stored as packed little-endian float32 BLOBs, not JSON text.
  A Facenet embedding is 512 floats: 2048 bytes as BLOB vs ~11 KB as JSON,
  and loading the whole roster becomes one ``np.frombuffer`` instead of N json
  parses.
* Every embedding records the model that produced it. Matching refuses to
  compare vectors from different models, which is the one bug that silently
  destroys accuracy when someone swaps Facenet for ArcFace.
* All timestamps go through :class:`UtcDateTime` so the API can never return a
  naive datetime and clients never have to guess a timezone.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

__all__ = [
    "Attendance",
    "Base",
    "EnrollmentSample",
    "EnrollmentSession",
    "Event",
    "LivenessSession",
    "User",
    "Verification",
    "utcnow",
]


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class UtcDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes, even on SQLite.

    SQLite has no native tz support, so SQLAlchemy would silently hand back
    naive datetimes. This coerces input to UTC and re-attaches ``tzinfo`` on
    the way out.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: dt.datetime | None, dialect) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC)

    def process_result_value(self, value: dt.datetime | None, dialect) -> dt.datetime | None:
        if value is None:
            return None
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    """A registered person. One averaged, L2-normalised embedding per person."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    employee_code: Mapped[str | None] = mapped_column(String(64), default=None)

    embedding: Mapped[bytes] = mapped_column(nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    # Number of live frames averaged into `embedding`. More samples => tighter
    # intra-class spread => a cleaner separation from impostors.
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    quality_score: Mapped[float | None] = mapped_column(Float, default=None)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        UtcDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    attendance: Mapped[list[Attendance]] = relationship(back_populates="user")


class EnrollmentSession(Base):
    """Multi-frame enrolment. Accumulates samples, then averages them."""

    __tablename__ = "enrollment_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_name: Mapped[str] = mapped_column(String(160), nullable=False)
    employee_code: Mapped[str | None] = mapped_column(String(64), default=None)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    samples_needed: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, nullable=False)

    samples: Mapped[list[EnrollmentSample]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class EnrollmentSample(Base):
    __tablename__ = "enrollment_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("enrollment_sessions.id", ondelete="CASCADE"), nullable=False
    )
    embedding: Mapped[bytes] = mapped_column(nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)

    session: Mapped[EnrollmentSession] = relationship(back_populates="samples")


class LivenessSession(Base):
    """State for an active challenge-response liveness test.

    Anti-replay rests on two things: the challenge *order* is randomised per
    session, and each challenge has a randomised ``revealed_at`` so a
    pre-recorded montage of "blink then turn left" clips cannot satisfy the
    sequence. Once passed, the session is single-use (``consumed_at``) and
    yields the token required to punch attendance.
    """

    __tablename__ = "liveness_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(64), default=None)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    challenges: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    passive_score: Mapped[float | None] = mapped_column(Float, default=None)
    active_score: Mapped[float | None] = mapped_column(Float, default=None)
    failure_reason: Mapped[str | None] = mapped_column(String(160), default=None)

    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        UtcDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
    expires_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)
    consumed_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)

    __table_args__ = (Index("ix_liveness_status_created", "status", "created_at"),)


class Verification(Base):
    """A short-lived, single-use proof that *this server* matched a face to a person.

    The punch endpoint must never be told who someone is. An identity asserted
    by the caller is an identity the caller chose, and a distance asserted by
    the caller is a number the caller wrote -- together they are enough to mark
    anybody present without showing a face. So verification ends by issuing one
    of these rows, carrying the distance the server actually measured, and the
    attendance punch consumes it.

    Single-use and short-lived on purpose: a leaked id must not become a
    standing key to somebody's attendance, and the face that earned it has to
    be the face at the desk, not one from ten minutes ago.
    """

    __tablename__ = "verifications"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    distance: Mapped[float] = mapped_column(Float, nullable=False)
    passive_liveness: Mapped[float | None] = mapped_column(Float, default=None)
    spoof_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="api")

    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    expires_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, nullable=False)
    consumed_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)

    __table_args__ = (Index("ix_verifications_expires", "expires_at"),)


class Attendance(Base):
    """One row per person per business day; re-punches update, never duplicate."""

    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    user_name: Mapped[str] = mapped_column(String(160), nullable=False)
    work_date: Mapped[str] = mapped_column(String(10), nullable=False)

    first_in_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, nullable=False)
    last_seen_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, nullable=False)
    check_out_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)
    punch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    match_distance: Mapped[float] = mapped_column(Float, nullable=False)
    liveness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    liveness_session_id: Mapped[str | None] = mapped_column(String(64), default=None)
    passive_liveness: Mapped[float | None] = mapped_column(Float, default=None)
    spoof_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="api")
    device_id: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        UtcDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="attendance")

    __table_args__ = (
        UniqueConstraint("user_id", "work_date", name="uq_attendance_user_day"),
        Index("ix_attendance_date", "work_date"),
    )


class Event(Base):
    """Alert + audit sink. Anti-spoofing hits land here and are webhooked."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64), default=None)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)

    __table_args__ = (Index("ix_events_kind_created", "kind", "created_at"),)
