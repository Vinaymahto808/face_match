"""Event log + alert dispatch.

Anti-spoofing detections are useless if nobody is told about them, so every
rejection lands in the ``events`` table *and* (optionally) gets POSTed to a
webhook for Slack/PagerDuty/email. Delivery is fire-and-forget on a daemon
thread: a dead webhook must never slow down or fail an attendance request.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Event, utcnow

logger = logging.getLogger(__name__)

__all__ = ["KINDS", "SEVERITY_ORDER", "dispatch_alert", "record_event"]

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

KINDS = {
    "spoof_detected",
    "liveness_failed",
    "liveness_passed",
    "liveness_unavailable",
    "enroll_rejected",
    "match_ambiguous",
    "low_quality_frame",
    "attendance_blocked",
    "backend_degraded",
}


def record_event(
    db: Session,
    *,
    kind: str,
    message: str,
    severity: str = "info",
    user_id: str | None = None,
    context: dict[str, Any] | None = None,
    commit: bool = True,
) -> Event:
    """Persist an event. Returns the (possibly uncommitted) ORM object."""
    if severity not in SEVERITY_ORDER:
        severity = "info"
    event = Event(
        kind=kind,
        severity=severity,
        message=message,
        user_id=user_id,
        context=context or {},
        created_at=utcnow(),
    )
    db.add(event)
    if commit:
        db.commit()
        db.refresh(event)
    dispatch_alert(event)
    return event


def _post(url: str, body: bytes) -> None:
    # Scheme is constrained to http(s) by the ALERT_WEBHOOK_URL validator in
    # app/config.py, which runs before this can be reached.
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "face-attendance-api/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
        resp.read()


def dispatch_alert(event: Event) -> None:
    url = settings.alert_webhook_url.strip()
    if not url or event.severity == "info":
        return

    payload = {
        "id": event.id,
        "severity": event.severity,
        "kind": event.kind,
        "user_id": event.user_id,
        "message": event.message,
        "context": event.context,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }
    body = json.dumps(payload, default=str).encode("utf-8")

    def _run() -> None:
        try:
            _post(url, body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("alert webhook delivery failed: %s", exc)

    threading.Thread(target=_run, name="alert-webhook", daemon=True).start()
