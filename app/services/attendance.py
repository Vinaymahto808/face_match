"""Attendance marking: one row per person per business day, idempotently.

The notebook solved duplicate suppression with ``except sqlite3.IntegrityError:
pass`` around a bare INSERT. That silently swallows *every* constraint
violation, not just the daily duplicate, and it re-inserts on every matching
frame for the whole day. This upserts instead: the first punch records
``first_in_at``, every subsequent punch updates ``last_seen_at`` and increments
``punch_count``, and check-out is explicit.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Attendance, User, utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "PunchResult",
    "business_day",
    "business_zone",
    "check_out",
    "day_summary",
    "list_attendance",
    "local_now",
    "punch",
]


def business_zone() -> ZoneInfo:
    return ZoneInfo(settings.business_timezone)


def local_now(moment: dt.datetime | None = None) -> dt.datetime:
    moment = moment or utcnow()
    return moment.astimezone(business_zone())


def business_day(moment: dt.datetime | None = None) -> str:
    """Which *business* day a UTC instant belongs to (YYYY-MM-DD local)."""
    return local_now(moment).strftime("%Y-%m-%d")


@dataclass(slots=True)
class PunchResult:
    record: Attendance
    created: bool
    duplicate_suppressed: bool

    @property
    def record_id(self) -> int:
        return int(self.record.id)


def _get_for_day(db: Session, user_id: str, day: str) -> Attendance | None:
    return db.execute(
        select(Attendance).where(Attendance.user_id == user_id, Attendance.work_date == day)
    ).scalar_one_or_none()


def punch(
    db: Session,
    user: User,
    *,
    match_distance: float,
    liveness_score: float = 0.0,
    passive_liveness: float | None = None,
    spoof_flags: list[str] | None = None,
    liveness_session_id: str | None = None,
    source: str = "api",
    device_id: str | None = None,
    moment: dt.datetime | None = None,
) -> PunchResult:
    now = moment or utcnow()
    day = business_day(now)
    record = _get_for_day(db, user.id, day)
    flags = list(spoof_flags or [])

    if record is None:
        record = Attendance(
            user_id=user.id,
            user_name=user.name,
            work_date=day,
            first_in_at=now,
            last_seen_at=now,
            punch_count=1,
            match_distance=match_distance,
            liveness_score=liveness_score,
            passive_liveness=passive_liveness,
            liveness_session_id=liveness_session_id,
            spoof_flags=flags,
            source=source,
            device_id=device_id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return PunchResult(record=record, created=True, duplicate_suppressed=False)

    # Same person, same day: update the existing row.
    record.user_name = user.name
    record.last_seen_at = now
    record.punch_count += 1
    # Keep the best (lowest) distance of the day -- it is the least noisy
    # measurement of the day and the one worth auditing.
    record.match_distance = min(record.match_distance, match_distance)
    record.liveness_score = max(record.liveness_score, liveness_score)
    if passive_liveness is not None:
        record.passive_liveness = max(record.passive_liveness or 0.0, passive_liveness)
    record.spoof_flags = sorted(set(record.spoof_flags or []) | set(flags))
    record.source = source
    if device_id:
        record.device_id = device_id
    if liveness_session_id:
        record.liveness_session_id = liveness_session_id
    record.updated_at = now
    db.commit()
    db.refresh(record)
    return PunchResult(record=record, created=False, duplicate_suppressed=True)


def check_out(
    db: Session, user: User, *, moment: dt.datetime | None = None
) -> Attendance | None:
    now = moment or utcnow()
    record = _get_for_day(db, user.id, business_day(now))
    if record is None:
        return None
    record.check_out_at = now
    record.last_seen_at = now
    record.updated_at = now
    db.commit()
    db.refresh(record)
    return record


def list_attendance(
    db: Session,
    *,
    day: str | None = None,
    user_id: str | None = None,
    since: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Attendance], int]:
    """Returns ``(rows, total)`` for the requested page."""
    filters = []
    if day:
        filters.append(Attendance.work_date == day)
    if since:
        filters.append(Attendance.work_date >= since)
    if user_id:
        filters.append(Attendance.user_id == user_id)

    total = db.execute(
        select(func.count()).select_from(Attendance).where(*filters)
    ).scalar_one()

    rows = (
        db.execute(
            select(Attendance)
            .where(*filters)
            .order_by(Attendance.work_date.desc(), Attendance.first_in_at.asc())
            .limit(max(1, min(limit, 500)))
            .offset(max(0, offset))
        )
        .scalars()
        .all()
    )
    return list(rows), int(total)


def day_summary(db: Session, day: str | None = None) -> dict[str, Any]:
    day = day or business_day()
    row = db.execute(
        select(
            func.count(Attendance.id),
            func.coalesce(func.sum(Attendance.punch_count), 0),
        ).where(Attendance.work_date == day)
    ).one()
    return {"day": day, "present": int(row[0]), "punches": int(row[1])}
