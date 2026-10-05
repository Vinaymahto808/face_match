"""Registered-person roster: enrolment, consensus averaging, cosine matching."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import EnrollmentSample, EnrollmentSession, User, utcnow
from . import embeddings as emb

logger = logging.getLogger(__name__)

__all__ = [
    "MatchResult",
    "RosterCache",
    "RosterSnapshot",
    "add_enrollment_sample",
    "consensus_embedding",
    "create_user",
    "current_model_name",
    "deactivate_user",
    "finalize_enrollment",
    "get_user",
    "list_users",
    "roster_cache",
    "start_enrollment",
    "upsert_user_embedding",
]

Decision = Literal["match", "review", "unknown"]


def current_model_name() -> str:
    return settings.face_model_name


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RosterSnapshot:
    model_name: str
    ids: tuple[str, ...]
    names: tuple[str, ...]
    matrix: np.ndarray  # (n, d) float32, L2-normalised rows
    skipped_model_mismatch: int = 0
    loaded_at: float = 0.0

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def empty(self) -> bool:
        return self.count == 0

    @property
    def dim(self) -> int | None:
        return int(self.matrix.shape[1]) if self.matrix.size else None


@dataclass(frozen=True)
class MatchResult:
    decision: Decision
    distance: float
    user_id: str | None
    name: str | None
    runner_up_id: str | None = None
    runner_up_distance: float | None = None
    margin: float | None = None

    @property
    def matched(self) -> bool:
        return self.decision == "match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "user_id": self.user_id,
            "name": self.name,
            "distance": round(self.distance, 5) if np.isfinite(self.distance) else None,
            "runner_up_id": self.runner_up_id,
            "runner_up_distance": (
                round(self.runner_up_distance, 5)
                if self.runner_up_distance is not None and np.isfinite(self.runner_up_distance)
                else None
            ),
            "margin": round(self.margin, 5) if self.margin is not None else None,
        }


class RosterCache:
    """Process-local snapshot of the roster, with version + TTL invalidation.

    Rebuilding the matrix on every frame is wasteful once the roster is large;
    a 5-second TTL keeps it fresh enough to pick up enrolments made by other
    workers while still collapsing a burst of frames into one query.
    """

    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self._lock = threading.RLock()
        self._snapshot: RosterSnapshot | None = None
        self._loaded_at = 0.0
        self._ttl = ttl_seconds
        self._version = 0
        self._loaded_version = -1

    def invalidate(self) -> None:
        with self._lock:
            self._version += 1

    def get(self, db: Session, *, force: bool = False) -> RosterSnapshot:
        with self._lock:
            fresh = (
                self._snapshot is not None
                and self._loaded_version == self._version
                and (time.monotonic() - self._loaded_at) < self._ttl
            )
            if fresh and not force:
                return self._snapshot  # type: ignore[return-value]
            snapshot = self._load(db)
            self._snapshot = snapshot
            self._loaded_at = time.monotonic()
            self._loaded_version = self._version
            return snapshot

    def _load(self, db: Session) -> RosterSnapshot:
        model = current_model_name()
        rows = db.execute(
            select(User).where(User.is_active.is_(True)).order_by(User.id)
        ).scalars().all()

        ids: list[str] = []
        names: list[str] = []
        vectors: list[np.ndarray] = []
        skipped = 0

        for user in rows:
            if user.model_name != model:
                # Comparing a Facenet vector to an ArcFace vector produces a
                # meaningless number. Skip rather than silently mis-match.
                skipped += 1
                continue
            try:
                vectors.append(emb.unpack(user.embedding, user.embedding_dim))
            except ValueError as exc:
                logger.warning("skipping user %s: %s", user.id, exc)
                skipped += 1
                continue
            ids.append(user.id)
            names.append(user.name)

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)

        if skipped:
            logger.warning(
                "roster: skipped %d entr%s (embedding model != %s)",
                skipped,
                "y" if skipped == 1 else "ies",
                model,
            )
        return RosterSnapshot(
            model_name=model,
            ids=tuple(ids),
            names=tuple(names),
            matrix=matrix,
            skipped_model_mismatch=skipped,
            loaded_at=time.monotonic(),
        )


roster_cache = RosterCache()


def match_embedding(
    snapshot: RosterSnapshot,
    query: np.ndarray,
    *,
    threshold: float | None = None,
    gray_factor: float | None = None,
) -> MatchResult:
    """Nearest-neighbour match with a three-way decision.

    ``review`` exists so borderline cases are visible to a human instead of
    being silently reported as "unknown" (or worse, a confident false match).
    """
    thr = settings.match_threshold if threshold is None else threshold
    gray = thr * (settings.match_gray_zone_factor if gray_factor is None else gray_factor)

    if snapshot.empty:
        return MatchResult(decision="unknown", distance=float("inf"), user_id=None, name=None)

    distances = emb.cosine_similarity_matrix(query, snapshot.matrix)
    order = np.argsort(distances)
    best_i = int(order[0])
    best_d = float(distances[best_i])

    runner_id: str | None = None
    runner_d: float | None = None
    margin: float | None = None
    if distances.size > 1:
        second_i = int(order[1])
        runner_id = snapshot.ids[second_i]
        runner_d = float(distances[second_i])
        # How much clearer the winner is than the next-best candidate. Small
        # margins mean twins, siblings or a genuinely ambiguous frame.
        margin = runner_d - best_d

    if best_d <= thr:
        decision: Decision = "match"
    elif best_d <= gray:
        decision = "review"
    else:
        decision = "unknown"

    return MatchResult(
        decision=decision,
        distance=best_d,
        user_id=snapshot.ids[best_i],
        name=snapshot.names[best_i],
        runner_up_id=runner_id,
        runner_up_distance=runner_d,
        margin=margin,
    )


def consensus_embedding(samples: list[np.ndarray]) -> np.ndarray:
    """Average normalised samples, then renormalise.

    Averaging several live frames of the same person shrinks intra-class
    variance, which widens the gap between genuine and impostor distances --
    it is the single cheapest accuracy win available at enrolment time. It also
    smooths out the noisy embedding from one blink or half-turn.
    """
    if not samples:
        raise ValueError("no samples to average")
    stacked = np.vstack([emb.l2_normalize(s) for s in samples]).astype(np.float64)
    return emb.l2_normalize(stacked.mean(axis=0))


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def _validate_id(user_id: str) -> str:
    cleaned = (user_id or "").strip()
    if not cleaned or len(cleaned) > 64:
        raise ValueError("user id must be 1..64 characters")
    if not cleaned.replace("_", "").replace("-", "").isalnum():
        raise ValueError("user id may only contain letters, digits, '-' and '_'")
    return cleaned


def create_user(
    db: Session,
    *,
    user_id: str,
    name: str,
    embedding: np.ndarray,
    sample_count: int = 1,
    quality_score: float | None = None,
    employee_code: str | None = None,
    notes: str | None = None,
    model_name: str | None = None,
) -> User:
    clean_id = _validate_id(user_id)
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("name is required")

    vector = emb.coerce_vector(embedding)
    packed = emb.pack(vector)

    existing = db.get(User, clean_id)
    if existing is not None:
        existing.name = clean_name
        existing.employee_code = employee_code
        existing.embedding = packed
        existing.embedding_dim = int(vector.size)
        existing.model_name = model_name or current_model_name()
        existing.sample_count = max(1, int(sample_count))
        existing.quality_score = quality_score
        existing.notes = notes
        existing.is_active = True
        existing.updated_at = utcnow()
        user = existing
    else:
        user = User(
            id=clean_id,
            name=clean_name,
            employee_code=employee_code,
            embedding=packed,
            embedding_dim=int(vector.size),
            model_name=model_name or current_model_name(),
            sample_count=max(1, int(sample_count)),
            quality_score=quality_score,
            notes=notes,
            is_active=True,
        )
        db.add(user)

    db.commit()
    db.refresh(user)
    roster_cache.invalidate()
    logger.info("enrolled user %s (%s) with %d sample(s)", user.id, user.name, user.sample_count)
    return user


def upsert_user_embedding(
    db: Session, user_id: str, embedding: np.ndarray, *, model_name: str | None = None
) -> User:
    user = get_user(db, user_id)
    if user is None:
        raise LookupError(f"unknown user {user_id!r}")
    vector = emb.coerce_vector(embedding)
    if user.embedding_dim and vector.size != user.embedding_dim:
        raise ValueError(
            f"embedding dim {vector.size} does not match stored dim {user.embedding_dim}"
        )
    user.embedding = emb.pack(vector)
    user.model_name = model_name or user.model_name
    user.updated_at = utcnow()
    db.commit()
    db.refresh(user)
    roster_cache.invalidate()
    return user


def get_user(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)


def list_users(
    db: Session, *, include_inactive: bool = False, limit: int = 100, offset: int = 0
) -> list[User]:
    stmt = select(User).order_by(User.id)
    if not include_inactive:
        stmt = stmt.where(User.is_active.is_(True))
    stmt = stmt.limit(max(1, min(limit, 1000))).offset(max(0, offset))
    return list(db.execute(stmt).scalars().all())


def deactivate_user(db: Session, user_id: str) -> bool:
    user = get_user(db, user_id)
    if user is None:
        return False
    user.is_active = False
    user.updated_at = utcnow()
    db.commit()
    roster_cache.invalidate()
    return True


# ---------------------------------------------------------------------------
# Multi-sample enrolment sessions
# ---------------------------------------------------------------------------
def start_enrollment(
    db: Session,
    *,
    user_id: str,
    user_name: str,
    employee_code: str | None = None,
    samples_needed: int = 5,
    ttl_seconds: int = 300,
) -> EnrollmentSession:
    import datetime as dt

    clean_id = _validate_id(user_id)
    clean_name = (user_name or "").strip()
    if not clean_name:
        raise ValueError("name is required")

    session = EnrollmentSession(
        id=uuid.uuid4().hex,
        user_id=clean_id,
        user_name=clean_name,
        employee_code=employee_code,
        status="pending",
        samples_needed=max(1, min(int(samples_needed), 30)),
        created_at=utcnow(),
        expires_at=utcnow() + dt.timedelta(seconds=ttl_seconds),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def add_enrollment_sample(
    db: Session, session_id: str, embedding: np.ndarray, quality_score: float
) -> EnrollmentSession:
    session = db.get(EnrollmentSession, session_id)
    if session is None:
        raise LookupError("unknown enrollment session")
    if session.status != "pending":
        raise ValueError(f"enrollment session is {session.status}")
    if session.expires_at <= utcnow():
        session.status = "expired"
        db.commit()
        raise ValueError("enrollment session expired")

    vector = emb.coerce_vector(embedding)
    sample = EnrollmentSample(
        session_id=session.id,
        embedding=vector.astype(emb.DTYPE).tobytes(),
        quality_score=float(quality_score),
        created_at=utcnow(),
    )
    db.add(sample)
    db.commit()
    db.refresh(session)
    return session


def finalize_enrollment(db: Session, session_id: str, *, min_samples: int = 2) -> User:
    """Average the collected samples and write the roster entry."""
    session = db.get(EnrollmentSession, session_id)
    if session is None:
        raise LookupError("unknown enrollment session")
    if session.status != "pending":
        raise ValueError(f"enrollment session is {session.status}")

    samples = (
        db.execute(
            select(EnrollmentSample)
            .where(EnrollmentSample.session_id == session_id)
            .order_by(EnrollmentSample.id)
        )
        .scalars()
        .all()
    )
    if len(samples) < max(2, min_samples):
        raise ValueError(
            f"need at least {max(2, min_samples)} good samples, have {len(samples)}"
        )

    vectors = [emb.unpack(s.embedding) for s in samples]
    mean_quality = sum(s.quality_score for s in samples) / len(samples)

    user = create_user(
        db,
        user_id=session.user_id,
        name=session.user_name,
        embedding=consensus_embedding(vectors),
        sample_count=len(vectors),
        quality_score=mean_quality,
        employee_code=session.employee_code,
    )

    session.status = "complete"
    # The averaged vector is now canonical; the raw samples are transient.
    db.execute(delete(EnrollmentSample).where(EnrollmentSample.session_id == session_id))
    db.commit()
    return user
