"""Typed application errors and their HTTP mapping."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class. ``code`` is a stable machine-readable string."""

    status_code = 400
    code = "bad_request"

    def __init__(self, detail: str, *, code: str | None = None, context: dict[str, Any] | None = None):
        super().__init__(detail)
        self.detail = detail
        if code:
            self.code = code
        self.context = context or {}

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.detail, **({"context": self.context} if self.context else {})}}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationFailedError(AppError):
    """Client sent a semantically invalid value.

    Distinct from FastAPI's own 422, which only covers shape/type errors in the
    request body. A service-layer check like "user id may only contain letters,
    digits, '-' and '_'" is a 422 too -- and must not surface as a 500, which
    would tell the caller the server broke rather than that their input was bad.
    """

    status_code = 422
    code = "validation_failed"


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["error"]["hint"] = "send the credential in the X-API-Key header"
        return payload


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"


class BackendUnavailableError(ServiceUnavailableError):
    code = "face_backend_unavailable"


class LivenessRequiredError(AppError):
    """Raised when attendance is attempted without passing liveness.

    403 rather than 401: the caller is authenticated, they just have not
    proven they are physically present.
    """

    status_code = 403
    code = "liveness_required"


class IdentityMismatchError(AppError):
    """The proof and the person disagree: a liveness session issued for
    somebody else, or a verification that does not name the punched user.

    403 like :class:`LivenessRequiredError` -- the caller is authenticated and
    their own liveness may well be valid, they are just punching the wrong
    person. Distinct code because the client action differs: retry the
    challenge for *this* person rather than re-verify the face.
    """

    status_code = 403
    code = "identity_mismatch"


class SpoofSuspectedError(AppError):
    status_code = 403
    code = "spoof_suspected"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"


class LivenessLockedOutError(AppError):
    """Too many failed liveness sessions for one identity, too recently.

    A 429, not a 403 or a 503: the request is not forbidden and the service is
    not broken, it is asking this caller to come back later. The distinction
    matters to a kiosk -- 503 means "nobody can pass right now", which is not
    something retrying will fix, while this one carries a countdown.
    """

    status_code = 429
    code = "liveness_locked_out"
