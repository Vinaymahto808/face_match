"""API routes."""

from .attendance import router as attendance_router
from .events import router as events_router
from .health import router as health_router
from .kiosk import router as kiosk_router
from .liveness import router as liveness_router
from .stream import router as stream_router
from .users import router as users_router
from .verify import router as verify_router

__all__ = [
    "attendance_router",
    "events_router",
    "health_router",
    "kiosk_router",
    "liveness_router",
    "stream_router",
    "users_router",
    "verify_router",
]
