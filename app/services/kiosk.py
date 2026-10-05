"""The simple punch: one frame in, an *in* or *out* written.

Shared by the webcam CLI (``kiosk.py``) and ``POST /kiosk/punch`` so both
transports apply the same rule. There is deliberately no liveness challenge on
this path -- it exists to test recognition and the in/out bookkeeping with the
least machinery possible. A frame the passive check flags as a photo or a
screen is still refused, so the cheap anti-spoof signal is not lost.

In/out follows the one-row-per-business-day model in :mod:`attendance`: the
first punch of the day is the *in*, every later punch moves the *out*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sqlalchemy.orm import Session

from . import attendance as att
from . import events
from . import registry as reg
from .pipeline import VerificationResult, verify_frame

__all__ = ["KioskOutcome", "Mode", "apply_punch", "punch_frame"]

Mode = Literal["auto", "in", "out"]
Action = Literal["in", "out", "identified", "refused", "no_match"]

SOURCE = "kiosk"


@dataclass(slots=True)
class KioskOutcome:
    action: Action
    message: str
    user_id: str | None = None
    name: str | None = None
    distance: float | None = None
    decision: str | None = None
    block_reasons: list[str] = field(default_factory=list)
    passive_score: float | None = None
    passive_flags: list[str] = field(default_factory=list)
    # Largest detected face as [x, y, w, h] in the analysed frame, plus that
    # frame's [width, height], so a client can draw the box over its preview
    # whatever size it displays the video at.
    face_box: list[int] | None = None
    frame_size: list[int] | None = None
    record: Any = None  # Attendance row when something was written

    @property
    def written(self) -> bool:
        return self.action in ("in", "out")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "message": self.message,
            "user_id": self.user_id,
            "name": self.name,
            "distance": round(self.distance, 5) if self.distance is not None else None,
            "decision": self.decision,
            "block_reasons": list(self.block_reasons),
            "passive_score": round(self.passive_score, 4) if self.passive_score is not None else None,
            "passive_flags": list(self.passive_flags),
            "face_box": self.face_box,
            "frame_size": self.frame_size,
        }


def _stamp(moment: Any) -> str:
    return att.local_now(moment).strftime("%H:%M:%S")


def _outcome_from(result: VerificationResult, action: Action, message: str) -> KioskOutcome:
    match = result.match
    return KioskOutcome(
        action=action,
        message=message,
        user_id=match.user_id if match else None,
        name=match.name if match else None,
        distance=match.distance if match and np.isfinite(match.distance) else None,
        decision=match.decision if match else None,
        block_reasons=list(result.block_reasons),
        passive_score=result.passive.score if result.passive else None,
        passive_flags=list(result.passive.flags) if result.passive else [],
    )


def punch_frame(
    db: Session,
    frame_bgr: np.ndarray,
    *,
    mode: Mode = "auto",
    device_id: str | None = None,
    dry_run: bool = False,
    expected_user_id: str | None = None,
    source: str = SOURCE,
) -> KioskOutcome:
    """Recognise the face in ``frame_bgr`` and, unless ``dry_run``, punch it.

    With ``expected_user_id`` (the person typed their id first) the check is
    1:1 against that enrolment, which is both faster and stricter than a
    roster search.
    """
    result = verify_frame(
        db, frame_bgr, expected_user_id=expected_user_id, record_events=False
    )
    if not result.matched:
        match = result.match
        if match is not None and match.decision == "review":
            message = (
                f"borderline: {match.name} at distance {match.distance:.3f}; "
                "move closer and face the camera"
            )
        else:
            message = "no recognised face: " + (", ".join(result.block_reasons) or "no match")
        outcome = _outcome_from(result, "no_match", message)
    elif dry_run:
        match = result.match
        outcome = _outcome_from(
            result,
            "identified",
            f"{match.name} ({match.user_id}) at distance {match.distance:.3f}; nothing written",
        )
    else:
        outcome = apply_punch(db, result, mode=mode, device_id=device_id, source=source)

    if result.observations:
        primary = max(result.observations, key=lambda o: o.area)
        outcome.face_box = [int(v) for v in primary.bbox]
    outcome.frame_size = [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])]
    return outcome


def apply_punch(
    db: Session,
    result: VerificationResult,
    *,
    mode: Mode,
    device_id: str | None,
    source: str = SOURCE,
) -> KioskOutcome:
    """Write the in/out for the matched person in ``result``."""
    match = result.match
    assert match is not None and match.user_id is not None
    user = reg.get_user(db, match.user_id)
    if user is None or not user.is_active:
        return _outcome_from(result, "refused", f"{match.user_id} is not an active user")

    flags = list(result.passive.flags) if result.passive else []
    passive = result.passive.score if result.passive else None
    if result.spoof_suspected:
        events.record_event(
            db,
            kind="spoof_detected",
            severity="critical",
            user_id=user.id,
            message=f"Kiosk punch refused for {user.name}: {', '.join(flags) or 'spoof suspect'}",
            context={"passive_score": passive, "flags": flags, "source": source},
        )
        return _outcome_from(
            result,
            "refused",
            f"frame of {user.name} looks like a photo or screen ({', '.join(flags)})",
        )

    today = att.business_day()
    rows, _ = att.list_attendance(db, day=today, user_id=user.id, limit=1)
    existing = rows[0] if rows else None
    if mode == "auto":
        mode = "in" if existing is None else "out"

    def _punch_in() -> att.PunchResult:
        return att.punch(
            db,
            user,
            match_distance=match.distance,
            passive_liveness=passive,
            spoof_flags=flags,
            source=source,
            device_id=device_id,
        )

    if mode == "in":
        punched = _punch_in()
        if punched.created:
            events.record_event(
                db,
                kind="attendance_marked",
                severity="info",
                user_id=user.id,
                message=f"{user.name} punched in at the kiosk for {today}",
                context={"attendance_id": punched.record_id, "distance": match.distance},
            )
            message = f"{user.name} punched in at {_stamp(punched.record.first_in_at)}"
        else:
            message = (
                f"{user.name} already in since {_stamp(punched.record.first_in_at)}; "
                f"last seen updated (punch #{punched.record.punch_count})"
            )
        out = _outcome_from(result, "in", message)
        out.record = punched.record
        return out

    # mode == "out". A forced "out" with no "in" today records both, so the
    # day is not lost just because the morning punch was missed.
    if existing is None:
        _punch_in()
    record = att.check_out(db, user)
    assert record is not None
    note = " (no earlier in today; in and out recorded together)" if existing is None else ""
    out = _outcome_from(result, "out", f"{user.name} punched out at {_stamp(record.check_out_at)}{note}")
    out.record = record
    return out
