"""Request dependencies: API-key auth and per-caller rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request

from .config import settings
from .errors import RateLimitedError, ServiceUnavailableError, UnauthorizedError

logger = logging.getLogger(__name__)

__all__ = ["RateLimiter", "client_identity", "rate_limiter", "require_api_key"]


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def client_identity(request: Request, x_api_key: str | None) -> str:
    """Stable caller id for rate limiting: the key if present, else the IP.

    ``X-Forwarded-For`` is only consulted when ``TRUST_PROXY_HEADERS=true``.
    It is a client-writable header: trusting it by default means every caller
    can pick a new value per request and never hit the limit again, so the
    limiter silently stops limiting. Behind a reverse proxy that *overwrites*
    the header, set the flag and the bucket follows the real client.
    """
    if x_api_key:
        return "key:" + hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return "ip:" + forwarded.split(",")[0].strip()
    return "ip:" + (request.client.host if request.client else "unknown")


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Constant-time API-key check.

    Auth is enforced when keys are configured *or* in prod. If prod has no keys
    configured that is a deployment mistake, so it fails loudly with 503 rather
    than quietly serving an unauthenticated recognition endpoint.
    """
    if not settings.auth_required:
        return client_identity(request, x_api_key)

    keys = settings.api_key_list
    if not keys:
        logger.error("auth is required but API_KEYS is empty")
        raise ServiceUnavailableError(
            "server is misconfigured: API_KEYS is empty but authentication is required"
        )

    if not x_api_key:
        raise UnauthorizedError("missing X-API-Key header")

    presented = _digest(x_api_key)
    for candidate in keys:
        if hmac.compare_digest(presented, _digest(candidate)):
            return client_identity(request, x_api_key)

    # Constant-time across all candidates: the loop above always runs to the
    # end, so a wrong key's response time does not reveal a prefix match.
    raise UnauthorizedError("invalid API key")


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class RateLimiter:
    """In-process token bucket, keyed by API key or client IP.

    Per-process, so a multi-worker deployment multiplies the effective limit by
    the worker count. That is the right trade for a CPU-bound face service:
    a shared Redis counter would add a network hop to the hot path to save
    nothing meaningful. Front it with nginx/ALB limits if you need a global cap.
    """

    def __init__(self, per_minute: int, burst: int | None = None) -> None:
        self.rate = max(1, per_minute) / 60.0
        self.capacity = float(burst if burst is not None else max(1, per_minute // 4))
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, cost: float = 1.0) -> float:
        """Consume tokens. Returns seconds until the next token (0 if allowed)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = now - bucket.updated_at
                if elapsed > 0:
                    bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
                bucket.updated_at = now

            if self._buckets and len(self._buckets) > 10_000:
                # Cheap bound on memory for a long-running process.
                cutoff = now - 300
                for k in [k for k, b in self._buckets.items() if b.updated_at < cutoff]:
                    self._buckets.pop(k, None)

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return 0.0
            return (cost - bucket.tokens) / self.rate

    def enforce(self, key: str, *, cost: float = 1.0) -> None:
        wait = self.check(key, cost=cost)
        if wait > 0:
            raise RateLimitedError(
                f"rate limit exceeded; retry in {wait:.1f}s",
                context={"retry_after_s": round(wait, 2)},
            )


rate_limiter = RateLimiter(settings.rate_limit_per_minute)


def rate_limit(identity: Annotated[str, Depends(require_api_key)]) -> str:
    """Dependency form: apply the default per-caller budget."""
    rate_limiter.enforce(identity)
    return identity
