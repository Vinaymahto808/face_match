"""Anti-spoofing: passive texture/colour analysis + active challenge-response."""

from .active import SUPPORTED_CHALLENGES, EngineVerdict, LivenessEngine
from .geometry import FrameSignals, eye_detector_status
from .passive import PassiveVerdict, passive_liveness, score_features

__all__ = [
    "SUPPORTED_CHALLENGES",
    "EngineVerdict",
    "FrameSignals",
    "LivenessEngine",
    "PassiveVerdict",
    "eye_detector_status",
    "passive_liveness",
    "score_features",
]
