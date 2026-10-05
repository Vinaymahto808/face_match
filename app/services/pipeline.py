"""The single frame -> decision pipeline.

Both the HTTP ``/verify`` route and the WebSocket stream call
:func:`verify_frame`, so the gating rules exist in exactly one place. Adding a
second, slightly different rule to each transport is how a system ends up
"protected" on one path and wide open on the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ..config import settings
from . import embeddings as emb
from . import registry as reg
from .face import FaceObservation, get_analyzer
from .liveness.passive import PassiveVerdict, passive_liveness
from .quality import QualityReport, assess_face, assess_frame, to_gray

logger = logging.getLogger(__name__)

__all__ = ["VerificationResult", "annotate", "verify_frame"]


@dataclass
class VerificationResult:
    frame_quality: QualityReport
    face_quality: QualityReport | None = None
    observations: list[FaceObservation] = field(default_factory=list)
    match: reg.MatchResult | None = None
    passive: PassiveVerdict | None = None
    identity_match: bool | None = None
    block_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    roster_size: int = 0

    @property
    def face_present(self) -> bool:
        return bool(self.observations)

    @property
    def matched(self) -> bool:
        return self.match is not None and self.match.decision == "match"

    @property
    def spoof_suspected(self) -> bool:
        return self.passive is not None and self.passive.verdict == "spoof_suspect"

    @property
    def eligible_for_attendance(self) -> bool:
        """Identity side of the gate.

        Liveness is enforced separately, at punch time, by consuming a
        single-use liveness token. So this flag means "the face is good enough
        and the identity is trusted" -- not "you may punch yet".
        """
        return self.matched and not self.block_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_present": self.face_present,
            "faces_detected": len(self.observations),
            "frame_quality": self.frame_quality.to_dict(),
            "face_quality": self.face_quality.to_dict() if self.face_quality else None,
            "match": self.match.to_dict() if self.match else None,
            "identity_match": self.identity_match,
            "liveness": {
                "passive": self.passive.score if self.passive else None,
                "passive_verdict": self.passive.verdict if self.passive else None,
                "flags": self.passive.flags if self.passive else [],
                "engine": self.passive.engine if self.passive else None,
            },
            "eligible_for_attendance": self.eligible_for_attendance,
            "block_reasons": list(self.block_reasons),
            "warnings": list(self.warnings),
            "roster_size": self.roster_size,
        }


def verify_frame(
    db: Session,
    frame_bgr: np.ndarray,
    *,
    expected_user_id: str | None = None,
    record_events: bool = True,
) -> VerificationResult:
    """Run the full gate chain on one frame."""
    from .events import record_event

    analyzer = get_analyzer()
    frame_quality = assess_frame(frame_bgr)
    result = VerificationResult(frame_quality=frame_quality)

    if not frame_quality.ok:
        result.block_reasons.extend(frame_quality.reasons)
        if record_events and frame_bgr.size:
            record_event(
                db,
                kind="low_quality_frame",
                severity="info",
                message=f"Frame rejected by quality gate: {', '.join(frame_quality.reasons)}",
                context=frame_quality.to_dict(),
                commit=False,
            )
        # A frame rejected as blurry/flat is exactly what a re-imaged print
        # looks like, so the quality gate is already an anti-spoof signal.
        # Score it anyway so the caller gets a number instead of `passive:
        # null` -- which is indistinguishable from "no liveness available".
        # This only ever *adds* information: every block reason above is
        # already recorded, so the allow/deny outcome is unchanged.
        result.passive = _passive_for_reporting(analyzer, frame_bgr)
        return result

    if analyzer.is_stub and not settings.allow_stub_backend:
        # Never let fake vectors produce a confident match that someone then
        # trusts. Fail loudly instead.
        result.block_reasons.append("stub_embedding_backend")
        result.warnings.append(
            "EMBEDDING_BACKEND=stub: recognition results are synthetic and must not "
            "be used to mark real attendance"
        )
        return result

    try:
        observations = analyzer.analyze(frame_bgr, strict=False)
    except Exception as exc:  # noqa: BLE001 - surfaced as a block reason, not a 500
        logger.warning("analysis failed: %s", exc)
        result.block_reasons.append("analyzer_error")
        result.warnings.append(str(exc))
        return result

    result.observations = observations
    if not observations:
        result.block_reasons.append("no_face_detected")
        return result

    face = max(observations, key=lambda o: o.area)
    crop = face.crop(frame_bgr)
    face_quality = assess_face(to_gray(crop), face.w, face.h)
    result.face_quality = face_quality

    if not face_quality.ok:
        result.block_reasons.extend(f"face:{r}" for r in face_quality.reasons)
        return result

    passive = passive_liveness(crop, threshold=settings.passive_liveness_min)
    result.passive = passive

    if face.embedding is None:
        result.block_reasons.append("no_embedding")
        return result

    snapshot = reg.roster_cache.get(db)
    result.roster_size = snapshot.count
    if snapshot.skipped_model_mismatch:
        result.warnings.append(
            f"{snapshot.skipped_model_mismatch} roster entr"
            f"{'y was' if snapshot.skipped_model_mismatch == 1 else 'ies were'} skipped: "
            f"stored with a different model than '{snapshot.model_name}'"
        )

    if snapshot.empty:
        result.match = reg.MatchResult(
            decision="unknown", distance=float("inf"), user_id=None, name=None
        )
        result.block_reasons.append("empty_roster")
        return result

    # 1:1 verification (claimed identity) or 1:N search.
    if expected_user_id:
        result.match, result.identity_match = _verify_against_one(
            snapshot, face.embedding, expected_user_id
        )
    else:
        result.match = reg.match_embedding(snapshot, face.embedding)

    if result.match is not None:
        if result.match.decision == "unknown":
            result.block_reasons.append("no_match")
        elif result.match.decision == "review":
            result.block_reasons.append("ambiguous_match")
            if record_events:
                record_event(
                    db,
                    kind="match_ambiguous",
                    severity="warning",
                    user_id=result.match.user_id,
                    message=(
                        f"Borderline match for {result.match.user_id} "
                        f"(distance {result.match.distance:.3f})"
                    ),
                    context=result.match.to_dict(),
                    commit=False,
                )

        # A small margin means the frame matched two people almost equally well.
        if (
            result.match.margin is not None
            and result.match.decision == "match"
            and result.match.margin < 0.02
        ):
            result.warnings.append(
                f"weak separation from runner-up (margin {result.match.margin:.4f})"
            )

    if result.spoof_suspected:
        result.block_reasons.append("passive_spoof_suspect")
        if record_events:
            record_event(
                db,
                kind="spoof_detected",
                severity="critical",
                user_id=result.match.user_id if result.match else None,
                message=(
                    "Passive anti-spoofing flagged a re-capture artefact "
                    f"(score {passive.score:.2f}): {', '.join(passive.flags) or 'no detail'}"
                ),
                context=passive.to_dict(),
                commit=False,
            )

    return result


def _passive_for_reporting(analyzer, frame_bgr: np.ndarray) -> PassiveVerdict | None:
    """Best-effort passive score for a frame that already failed the gate.

    Never raises: this is advisory output on a path that is already returning a
    block, so a detection failure must not turn into a 500.
    """
    try:
        observations = analyzer.analyze(frame_bgr, strict=False)
    except Exception as exc:  # noqa: BLE001 - advisory path, documented as never raising
        logger.debug("reporting-only analysis failed: %s", exc)
        return None
    if not observations:
        return None
    try:
        face = max(observations, key=lambda o: o.area)
        return passive_liveness(face.crop(frame_bgr), threshold=settings.passive_liveness_min)
    except Exception as exc:  # noqa: BLE001 - advisory path, documented as never raising
        logger.debug("reporting-only liveness failed: %s", exc)
        return None


def _verify_against_one(
    snapshot: reg.RosterSnapshot, query: np.ndarray, expected_user_id: str
) -> tuple[reg.MatchResult, bool]:
    """1:1 check against a claimed identity, with the impostor set subtracted.

    Two outcomes are kept distinct on purpose:
      * claimed person is the nearest -> ``match`` (proves it is them)
      * claimed person is absent or further away than someone else
        -> ``unknown`` and the real nearest is reported, so the caller can see
        *who* actually matched. Swallowing that would hide a genuine
        impersonation attempt.
    """
    ids = list(snapshot.ids)
    try:
        idx = ids.index(expected_user_id)
    except ValueError:
        return (
            reg.MatchResult(
                decision="unknown",
                distance=float("inf"),
                user_id=None,
                name=None,
            ),
            False,
        )

    claimed_distance = float(
        emb.cosine_similarity_matrix(query, snapshot.matrix[idx : idx + 1])[0]
    )
    result = reg.match_embedding(snapshot, query)
    is_claim = result.user_id == expected_user_id

    if claimed_distance <= settings.match_threshold and is_claim:
        out = reg.MatchResult(
            decision="match",
            distance=claimed_distance,
            user_id=expected_user_id,
            name=snapshot.names[idx],
            runner_up_id=result.runner_up_id,
            runner_up_distance=result.runner_up_distance,
            margin=result.margin,
        )
    elif claimed_distance <= settings.gray_zone_threshold:
        out = reg.MatchResult(
            decision="review",
            distance=claimed_distance,
            user_id=expected_user_id,
            name=snapshot.names[idx],
            runner_up_id=result.user_id if not is_claim else result.runner_up_id,
            runner_up_distance=result.distance if not is_claim else result.runner_up_distance,
        )
    else:
        out = reg.MatchResult(
            decision="unknown",
            distance=claimed_distance,
            user_id=expected_user_id,
            name=snapshot.names[idx],
            runner_up_id=result.user_id,
            runner_up_distance=result.distance,
        )
    return out, out.decision == "match"


def annotate(
    frame_bgr: np.ndarray,
    result: VerificationResult,
    *,
    caption: str | None = None,
) -> np.ndarray:
    """Draw the boxes/labels returned to the client.

    OpenCV-optional: without it the frame is returned untouched rather than
    raising, because drawing is a convenience, not a security control.
    """
    try:
        import cv2
    except ImportError:
        return frame_bgr

    out = frame_bgr
    for obs in result.observations:
        color = (0, 0, 255)
        if result.match is not None and result.spoof_suspected:
            color = (0, 165, 255)  # orange: suspected replay
        elif result.matched:
            color = (0, 200, 0)
        elif result.match is not None and result.match.decision == "review":
            color = (0, 215, 255)

        cv2.rectangle(out, (obs.x, obs.y), (obs.x + obs.w, obs.y + obs.h), color, 2)

        name = "no face"
        if result.match is not None:
            if result.match.decision == "unknown":
                name = "Unknown"
            else:
                name = f"{result.match.name} {result.match.distance:.2f}"
        if result.spoof_suspected:
            name = f"SPOOF? {name}"

        text_y = max(18, obs.y - 8)
        cv2.putText(
            out, name, (obs.x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    if result.passive is not None:
        cv2.putText(
            out,
            f"passive {result.passive.score:.2f} [{result.passive.verdict}]",
            (10, out.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if caption:
        cv2.putText(
            out, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return out
