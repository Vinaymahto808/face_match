"""Server-issued proof that a face was matched to a person.

Why this module exists
----------------------
The punch endpoint used to accept ``user_id`` and ``match_distance`` in the
request body. Both are *client assertions*: a caller could name anybody and
claim any distance it liked. Passing liveness does not close that hole, because
a liveness session proved that *a* live human was present -- it was never
checked against *which* human was being punched. So anyone who completed the
blink-and-turn challenge could post a punch for a colleague, and any caller
with no camera at all could invent a distance of 0.01.

The fix moves the assertion to the server side. Verification ends by issuing a
short-lived, single-use row (:class:`~app.models.Verification`) that carries the
distance this process measured, and the punch consumes that row. The punch
endpoint no longer accepts a user id or a distance at all, so there is nothing
left for a caller to lie about.

Why a database row rather than a signed token
---------------------------------------------
The same reason the liveness token is a row: a punch authorised by process-local
state stops working the moment you run a second worker, and the audit trail
("which verification authorised this row?") has to survive the request. A
HMAC-signed blob would be stateless but unreportable, and this service is
single-worker by design anyway (see ``liveness/service.py``).
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Verification, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import VerificationResult

logger = logging.getLogger(__name__)

__all__ = [
    "Grant",
    "consume",
    "issue",
    "issue_from_result",
    "lookup",
    "prune_expired",
]

#: How long a grant stays spendable. Long enough for a human to finish a
#: challenge after a face is recognised, short enough that a frame captured
#: this morning cannot authorise a punch this afternoon.
DEFAULT_TTL_SECONDS = 120

#: Failure reasons, stable strings. They end up in event context and in the
#: ``context`` of the 422 the client sees, so kiosk integrators can branch on
#: them.
REASON_UNKNOWN = "unknown_verification"
REASON_USED = "verification_already_used"
REASON_EXPIRED = "verification_expired"


@dataclass(frozen=True, slots=True)
class Grant:
    """An issued, not-yet-spent proof of identity."""

    id: str
    user_id: str
    distance: float
    passive_liveness: float | None
    spoof_flags: list[str]


def issue(
    db: Session,
    *,
    user_id: str,
    distance: float,
    passive_liveness: float | None = None,
    spoof_flags: list[str] | None = None,
    source: str = "api",
    ttl_seconds: int | None = None,
) -> Grant:
    """Write a grant. Only call this with values this process measured."""
    now = utcnow()
    ttl = ttl_seconds if ttl_seconds is not None else settings.verification_ttl_seconds
    # Prune *before* inserting: sweeping afterwards would delete the row just
    # written, whenever its TTL is at or below zero.
    _prune_expired(db, commit=True)
    row = Verification(
        id=uuid.uuid4().hex,
        user_id=user_id,
        distance=float(distance),
        passive_liveness=None if passive_liveness is None else float(passive_liveness),
        spoof_flags=list(spoof_flags or []),
        source=source,
        created_at=now,
        expires_at=now + dt.timedelta(seconds=ttl),
    )
    db.add(row)
    db.commit()
    return Grant(
        id=row.id,
        user_id=row.user_id,
        distance=row.distance,
        passive_liveness=row.passive_liveness,
        spoof_flags=list(row.spoof_flags or []),
    )


def issue_from_result(
    db: Session, result: VerificationResult, *, source: str = "api"
) -> Grant | None:
    """Issue a grant for a completed :func:`~app.services.pipeline.verify_frame`.

    Returns ``None`` -- and writes nothing -- unless the frame is eligible for
    attendance. That predicate is the same one the verify response reports as
    ``eligible_for_attendance``, so a client can never be told "yes, you may
    punch" and then find the punch refused for an unrelated reason.

    Spoof flags are carried only when the server's own passive check said
    ``spoof_suspect``. Advisory flags on a live-looking frame (mild moire, low
    chroma) are not grounds for refusing a person who is visibly present.
    """
    if result is None or not result.eligible_for_attendance or result.match is None:
        return None

    flags = list(result.passive.flags) if result.spoof_suspected else []
    return issue(
        db,
        user_id=str(result.match.user_id),
        distance=float(result.match.distance),
        passive_liveness=result.passive.score if result.passive else None,
        spoof_flags=flags,
        source=source,
    )


def lookup(db: Session, verification_id: str) -> tuple[Any | None, str]:
    """Read a grant **without** spending it. Returns ``(row_or_None, reason)``.

    Separate from :func:`consume` on purpose. The punch has to check the
    liveness token *before* it burns anything, otherwise a caller could spend a
    real verification and a real challenge on a request that was going to be
    refused for some later reason, and an honest user at the desk would have to
    start over.
    """
    if not verification_id:
        return None, REASON_UNKNOWN

    row = db.execute(
        select(Verification).where(Verification.id == verification_id)
    ).scalar_one_or_none()
    if row is None:
        return None, REASON_UNKNOWN
    if row.consumed_at is not None:
        return None, REASON_USED
    if row.expires_at <= utcnow():
        return None, REASON_EXPIRED
    return row, "ok"


def consume(db: Session, row: Any) -> Grant:
    """Spend a grant. The row is authoritative; there is nothing else to check."""
    row.consumed_at = utcnow()
    db.commit()
    return Grant(
        id=row.id,
        user_id=row.user_id,
        distance=row.distance,
        passive_liveness=row.passive_liveness,
        spoof_flags=list(row.spoof_flags or []),
    )


def prune_expired(db: Session) -> int:
    """Housekeeping for deployments that never call :func:`issue`."""
    removed, _ = _prune_expired(db, commit=True)
    return removed


def _prune_expired(db: Session, *, commit: bool) -> tuple[int, Any]:
    result = db.execute(delete(Verification).where(Verification.expires_at <= utcnow()))
    if commit:
        db.commit()
    return int(result.rowcount or 0), result
