"""Security event / alert feed."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..deps import rate_limit
from ..models import Event, utcnow
from ..schemas import EventList, EventOut
from ..services.events import SEVERITY_ORDER

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get(
    "",
    response_model=EventList,
    summary="List anti-spoof and system events",
    dependencies=[Depends(rate_limit)],
)
def list_events(
    db: Session = Depends(get_db),
    kind: Annotated[str | None, Query(max_length=48)] = None,
    severity: Annotated[str | None, Query(pattern="^(info|warning|critical)$")] = None,
    user_id: Annotated[str | None, Query(max_length=64)] = None,
    since_hours: int = Query(default=24, ge=1, le=24 * 90),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> EventList:
    import datetime as dt

    filters = [Event.created_at >= utcnow() - dt.timedelta(hours=since_hours)]
    if kind:
        filters.append(Event.kind == kind)
    if severity:
        filters.append(Event.severity == severity)
    if user_id:
        filters.append(Event.user_id == user_id)

    total = db.execute(select(func.count()).select_from(Event).where(*filters)).scalar_one()
    rows = (
        db.execute(
            select(Event)
            .where(*filters)
            .order_by(Event.created_at.desc(), Event.id.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return EventList(
        items=[EventOut.model_validate(e) for e in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/summary",
    summary="Counts by severity and kind, for a dashboard",
    dependencies=[Depends(rate_limit)],
)
def events_summary(
    db: Session = Depends(get_db), since_hours: int = Query(default=24, ge=1, le=24 * 90)
) -> dict[str, Any]:
    import datetime as dt

    since = utcnow() - dt.timedelta(hours=since_hours)
    rows = db.execute(
        select(Event.severity, Event.kind, func.count())
        .where(Event.created_at >= since)
        .group_by(Event.severity, Event.kind)
    ).all()

    by_severity: dict[str, int] = dict.fromkeys(SEVERITY_ORDER, 0)
    by_kind: dict[str, int] = {}
    for severity, kind, count in rows:
        by_severity[severity] = by_severity.get(severity, 0) + int(count)
        by_kind[kind] = by_kind.get(kind, 0) + int(count)

    return {
        "window_hours": since_hours,
        "since": since,
        "total": sum(by_severity.values()),
        "by_severity": by_severity,
        "by_kind": by_kind,
        "webhook_configured": bool(settings.alert_webhook_url),
    }
