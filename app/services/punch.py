"""The attendance punch rules, in one place.

Extracted from the HTTP route so the WebSocket path uses byte-identical
security semantics. A guard that exists on one transport and not the other is
a guard that does not exist.

What authorises a punch
-----------------------
Exactly two things, neither of them supplied by the caller:

1. A **verification grant** -- a short-lived, single-use row written by the
   server's own face pipeline, carrying the distance it measured for the person
   who is being punched. It also names the user, so the punch endpoint has no
   ``user_id`` field to lie about.
2. A **liveness token** -- a passed, unconsumed challenge session, *bound to
   the same person*. The binding is the point: passing a challenge proves a
   live human was there, so it must not authorise a punch for a colleague.

A caller holding someone else's grant plus their own liveness session gets
``identity_mismatch``, not attendance.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ..config import settings
from ..errors import (
    IdentityMismatchError,
    LivenessRequiredError,
    NotFoundError,
    SpoofSuspectedError,
    ValidationFailedError,
)
from . import attendance as att
from . import events, verifications
from . import registry as reg
from .liveness.service import consume_token

logger = logging.getLogger(__name__)

__all__ = ["punch_attendance"]

#: consume_token reasons that mean "this session exists and is valid, it is just
#: not yours" rather than "no usable liveness here". They get a critical alert.
_CRITICAL_REASONS = frozenset({"liveness_token_already_used", "liveness_session_user_mismatch"})


def punch_attendance(
    db: Session,
    *,
    verification_id: str = "",
    liveness_session_id: str | None = None,
    liveness_score: float = 0.0,
    device_id: str | None = None,
    source: str = "api",
) -> tuple[att.PunchResult, str]:
    """Mark attendance. Returns ``(result, message)``.

    Gate order is deliberate:

    1. **the grant resolves, and the person exists and is active** -- no
       database work and no state spent before the cheap checks. A missing
       ``verification_id`` lands here too, with the same 422 as a spent or
       unknown one: there is nothing to punch with.
    2. **the grant's distance is still within threshold**. The grant was issued
       by our own matcher, so this is belt-and-braces: the threshold can have
       been retuned since, or the roster re-enrolled. It is checked *before*
       anything is consumed, so a refused punch never burns a liveness session
       an honest user just spent a minute earning.
    3. **liveness token**, when required -- consumed here, and refused if it
       was issued for a different person.
    4. **no spoof flags** on the grant. Deliberately after the liveness check:
       a print of somebody's face should burn the attacker's own challenge, and
       it costs an honest user nothing because the session is bound to them.
    5. **the grant is spent** -- last, so that every refusal above leaves it
       spendable. An honest user who mistypes their id on the punch gets to use
       the grant their own face earned.

    Steps 2 and 4 exist because the most common real-world bypass of a liveness
    system is not beating the liveness check at all, it is calling the write
    endpoint directly with a plausible-looking payload.
    """
    row, reason = verifications.lookup(db, verification_id)
    if row is None:
        events.record_event(
            db,
            kind="attendance_blocked",
            severity="warning",
            message=f"Punch refused: {reason}",
            context={"verification_id": verification_id, "reason": reason, "source": source},
        )
        # 422, not the 400 default: the request is well-formed but the
        # identity it points at does not exist, is spent, or is stale. Same
        # status the old "distance above threshold" refusal used, so clients
        # that already branch on 422 keep working. See README §3.
        raise ValidationFailedError(
            f"no usable identity verification ({reason}); run /recognition/verify "
            "and send the verification_id it returns",
            code="unverified_identity",
            context={"verification_id": verification_id, "reason": reason},
        )

    user = reg.get_user(db, row.user_id)
    if user is None or not user.is_active:
        raise NotFoundError(
            f"unknown or inactive user {row.user_id!r}", code="user_not_found"
        )

    flags = list(row.spoof_flags or [])

    if flags:
        events.record_event(
            db,
            kind="spoof_detected",
            severity="critical",
            user_id=user.id,
            message=f"Attendance punch attempted with spoof flags: {', '.join(flags)}",
            context={"flags": flags, "liveness_session_id": liveness_session_id, "source": source},
        )

    if row.distance > settings.match_threshold:
        events.record_event(
            db,
            kind="attendance_blocked",
            severity="critical",
            user_id=user.id,
            message=(
                f"Punch rejected: match distance {row.distance:.3f} exceeds "
                f"threshold {settings.match_threshold:.3f}"
            ),
            context={"distance": row.distance, "threshold": settings.match_threshold},
        )
        raise ValidationFailedError(
            f"match distance {row.distance:.3f} is above the threshold "
            f"{settings.match_threshold:.3f}; the identity was not verified",
            code="unverified_identity",
            context={"distance": row.distance, "threshold": settings.match_threshold},
        )

    if settings.require_liveness:
        if not liveness_session_id:
            events.record_event(
                db,
                kind="attendance_blocked",
                severity="warning",
                user_id=user.id,
                message="Punch attempted with no liveness session",
                context={"source": source},
            )
            raise LivenessRequiredError(
                "liveness is required: pass liveness_session_id from a passed "
                "challenge-response session",
                context={"require_liveness": True},
            )

        ok, reason, token_score = consume_token(db, liveness_session_id, user_id=user.id)
        if not ok:
            severity = "critical" if reason in _CRITICAL_REASONS else "warning"
            events.record_event(
                db,
                kind="attendance_blocked",
                severity=severity,
                user_id=user.id,
                message=f"Punch blocked: {reason}",
                context={
                    "liveness_session_id": liveness_session_id,
                    "expected_user_id": user.id,
                    "source": source,
                },
            )
            if reason == "liveness_session_user_mismatch":
                # Not a liveness failure: the challenge was passed honestly, by
                # somebody. It is just not this person, which is either a
                # confused kiosk or an attempt to punch a colleague.
                raise IdentityMismatchError(
                    "this liveness session was not issued for the verified identity; "
                    "run the challenge again with this person in front of the camera",
                    code="identity_mismatch",
                    context={"verified_user_id": user.id},
                )
            raise LivenessRequiredError(
                f"liveness check failed: {reason}", context={"reason": reason}
            )

        # The token's own score is authoritative; the client's claim is a floor.
        liveness_score = max(liveness_score, token_score)

    if flags and not settings.debug:
        raise SpoofSuspectedError(
            f"anti-spoofing flagged this frame: {', '.join(flags)}",
            context={"flags": flags},
        )

    grant = verifications.consume(db, row)

    result = att.punch(
        db,
        user,
        match_distance=grant.distance,
        liveness_score=liveness_score,
        passive_liveness=grant.passive_liveness,
        spoof_flags=flags,
        liveness_session_id=liveness_session_id,
        source=source,
        device_id=device_id,
    )

    if result.created:
        events.record_event(
            db,
            kind="attendance_marked",
            severity="info",
            user_id=user.id,
            message=f"{user.name} marked present for {result.record.work_date}",
            context={
                "attendance_id": result.record_id,
                "match_distance": grant.distance,
                "liveness_score": liveness_score,
                "liveness_session_id": liveness_session_id,
                "verification_id": grant.id,
            },
        )
        message = "attendance marked"
    else:
        message = (
            f"already marked present for {result.record.work_date} "
            f"(punch #{result.record.punch_count}, last seen updated)"
        )

    return result, message
