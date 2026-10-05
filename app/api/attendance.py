"""Attendance punch, queries and export.

The punch endpoint is the security boundary. Everything else in the service
exists to produce trustworthy inputs for it.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import rate_limit
from ..errors import NotFoundError
from ..schemas import (
    AttendanceList,
    AttendanceOut,
    CheckOutResponse,
    PunchRequest,
    PunchResponse,
)
from ..services import attendance as att
from ..services import registry as reg
from ..services.punch import punch_attendance

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.post(
    "/punch",
    response_model=PunchResponse,
    summary="Mark attendance (requires a passed liveness session)",
    dependencies=[Depends(rate_limit)],
)
def punch(
    payload: PunchRequest, db: Session = Depends(get_db)
) -> PunchResponse:
    """Record attendance for one person for today.

    The body carries no user id and no match distance on purpose: both come
    from the ``verification_id``, which the server issued when it recognised
    the face. All the security rules live in
    :func:`app.services.punch.punch_attendance` so the WebSocket path enforces
    exactly the same checks.
    """
    result, message = punch_attendance(
        db,
        verification_id=payload.verification_id,
        liveness_session_id=payload.liveness_session_id,
        liveness_score=payload.liveness_score,
        device_id=payload.device_id,
        source=payload.source,
    )

    return PunchResponse(
        attendance=AttendanceOut.model_validate(result.record),
        created=result.created,
        duplicate_suppressed=result.duplicate_suppressed,
        message=message,
    )


@router.post(
    "/check-out",
    response_model=CheckOutResponse,
    summary="Close out the current business day for a person",
    dependencies=[Depends(rate_limit)],
)
def check_out(
    user_id: Annotated[str, Query(max_length=64)], db: Session = Depends(get_db)
) -> CheckOutResponse:
    user = reg.get_user(db, user_id)
    if user is None:
        raise NotFoundError(f"unknown user {user_id!r}", code="user_not_found")
    record = att.check_out(db, user)
    if record is None:
        return CheckOutResponse(
            attendance=None, message="no attendance recorded for today; nothing to close"
        )
    return CheckOutResponse(
        attendance=AttendanceOut.model_validate(record),
        message=f"checked out at {record.check_out_at.isoformat() if record.check_out_at else ''}",
    )


@router.get(
    "",
    response_model=AttendanceList,
    summary="Query attendance records",
    dependencies=[Depends(rate_limit)],
)
def query_attendance(
    db: Session = Depends(get_db),
    day: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    since: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    user_id: Annotated[str | None, Query(max_length=64)] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    with_summary: bool = Query(default=False),
) -> AttendanceList:
    rows, total = att.list_attendance(
        db, day=day, since=since, user_id=user_id, limit=limit, offset=offset
    )
    return AttendanceList(
        items=[AttendanceOut.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
        summary=att.day_summary(db, day) if with_summary else None,
    )


@router.get(
    "/export.csv",
    summary="Export attendance as CSV",
    dependencies=[Depends(rate_limit)],
)
def export_csv(
    db: Session = Depends(get_db),
    day: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    since: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    limit: int = Query(default=5000, ge=1, le=20000),
) -> Response:
    rows, _ = att.list_attendance(db, day=day, since=since, limit=limit)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "user_id",
            "user_name",
            "work_date",
            "first_in_at",
            "last_seen_at",
            "check_out_at",
            "punch_count",
            "match_distance",
            "liveness_score",
            "passive_liveness",
            "spoof_flags",
            "source",
            "device_id",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.user_id,
                r.user_name,
                r.work_date,
                r.first_in_at.isoformat() if r.first_in_at else "",
                r.last_seen_at.isoformat() if r.last_seen_at else "",
                r.check_out_at.isoformat() if r.check_out_at else "",
                r.punch_count,
                f"{r.match_distance:.5f}",
                f"{r.liveness_score:.4f}",
                f"{r.passive_liveness:.4f}" if r.passive_liveness is not None else "",
                "|".join(r.spoof_flags or []),
                r.source,
                r.device_id or "",
            ]
        )

    filename = f"attendance_{day or 'all'}.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
