"""Contract tests for the real DeepFace backend.

Why this file exists
--------------------
Every other test in the suite runs on :class:`StubAnalyzer`, which means the
DeepFace call in :meth:`DeepFaceAnalyzer.analyze` was never actually executed
by a test. That let a real production-breaking bug ship: the call passed
``detector=`` where DeepFace's ``represent()`` takes ``detector_backend=``. The
server booted, health went green, every test passed -- and then *every single
recognition request* failed with ``represent() got an unexpected keyword
argument 'detector'``.

The asymmetry is the point: ``warmup()`` only builds the model, and the model
building path does not go through ``represent()``'s public signature, so even
the readiness probe could not have caught it. These tests bind the exact
keyword arguments the code sends against the *installed* DeepFace signature, so
a rename on either side fails here instead of in production.

They need no model weights and no real face: the analyzer is handed a recording
double that only records its kwargs.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from app.services.face import DeepFaceAnalyzer

deepface_representation = pytest.importorskip(
    "deepface.modules.representation",
    reason="deepface not installed; the production backend cannot be exercised",
)


def test_represent_is_called_with_kwargs_the_installed_deepface_accepts():
    """The exact call in analyze() must bind against the real signature.

    ``Signature.bind`` raises TypeError on an unknown keyword -- which is the
    same failure the server used to raise on every request, but here it is a
    test failure instead of a 503 at 9am.
    """
    analyzer = DeepFaceAnalyzer(model_name="Facenet", detector="opencv")
    signature = inspect.signature(deepface_representation.represent)

    captured: dict[str, object] = {}

    def recording_represent(**kwargs):
        signature.bind(**kwargs)  # raises TypeError if the call is malformed
        captured.update(kwargs)
        return []

    analyzer._deepface = type("_FakeDeepFace", (), {"represent": staticmethod(recording_represent)})
    analyzer._ready = True

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert analyzer.analyze(frame) == []

    assert captured["model_name"] == "Facenet"
    assert captured["enforce_detection"] is True
    # The bug: this was `detector`, which no DeepFace release accepts.
    assert captured["detector_backend"] == "opencv"
    assert "detector" not in captured


def test_detector_setting_reaches_deepface_under_the_right_kwarg():
    """settings.face_detector must not be silently dropped or misnamed."""
    analyzer = DeepFaceAnalyzer(model_name="Facenet", detector="retinaface")
    signature = inspect.signature(deepface_representation.represent)
    captured: dict[str, object] = {}

    def recording_represent(**kwargs):
        signature.bind(**kwargs)
        captured.update(kwargs)
        return []

    analyzer._deepface = type("_FakeDeepFace", (), {"represent": staticmethod(recording_represent)})
    analyzer._ready = True
    analyzer.analyze(np.zeros((64, 64, 3), dtype=np.uint8))

    assert captured["detector_backend"] == "retinaface"


def test_warmup_inference_path_also_binds():
    """The lazy-load fallback calls represent() too -- same guarantee."""
    signature = inspect.signature(deepface_representation.represent)
    signature.bind(
        img_path=np.zeros((224, 224, 3), dtype=np.uint8),
        model_name="Facenet",
        enforce_detection=False,
    )


def test_facial_area_shape_is_parsed():
    """A represent() result must become a usable FaceObservation.

    Guards the field names we read back off DeepFace's output dict; a rename
    there would otherwise silently yield empty crops.
    """
    analyzer = DeepFaceAnalyzer()
    analyzer._deepface = type(
        "_FakeDeepFace",
        (),
        {
            "represent": staticmethod(
                lambda **kw: [
                    {
                        "embedding": [0.1] * 128,
                        "facial_area": {"x": 10, "y": 20, "w": 64, "h": 64},
                        "face_confidence": 0.98,
                    }
                ]
            )
        },
    )
    analyzer._ready = True

    obs = analyzer.analyze(np.zeros((200, 200, 3), dtype=np.uint8))

    assert len(obs) == 1
    assert obs[0].bbox == (10, 20, 64, 64)
    assert obs[0].confidence == pytest.approx(0.98)
    assert obs[0].embedding is not None and obs[0].embedding.shape == (128,)


def test_face_not_detected_raises_are_translated_into_an_empty_result():
    """``analyze`` must keep its "no face -> no observations" contract.

    A second production-breaking bug of the same family as the kwarg one above:
    with ``enforce_detection=True`` (i.e. ``strict=True``) DeepFace *raises*
    ``FaceNotDetected`` rather than returning ``[]``. That escape made
    ``POST /users/enroll/{id}/sample`` answer an opaque **500** for a photo with
    no face in it, instead of the intended 400 ``no_face_detected`` -- a user
    uploading a blurry shot got an error that told them nothing and looked like
    a server fault.

    Every caller is written against the empty-list shape, so the raise has to be
    translated back into it here. StubAnalyzer never raised, which is why the
    stub-based suite could not see it.
    """
    face_not_detected = pytest.importorskip(
        "deepface.modules.exceptions",
        reason="deepface not installed",
    ).FaceNotDetected

    analyzer = DeepFaceAnalyzer(model_name="Facenet", detector="opencv")

    def raising_represent(**kwargs):
        raise face_not_detected("Face could not be detected in numpy array.")

    analyzer._deepface = type("_FakeDeepFace", (), {"represent": staticmethod(raising_represent)})
    analyzer._ready = True

    # strict=True is the enrolment path that hit the 500.
    assert analyzer.analyze(np.zeros((64, 64, 3), dtype=np.uint8), strict=True) == []


def test_other_deepface_errors_are_not_swallowed_by_the_translation():
    """Only FaceNotDetected becomes "no face".

    A blanket ``except Exception -> []`` here would turn a genuine backend
    failure (weights missing, detector backend absent, OOM) into a confident
    "there is nobody in this photo", which is exactly the wrong answer to give
    a liveness system.
    """
    analyzer = DeepFaceAnalyzer(model_name="Facenet", detector="opencv")

    def broken_represent(**kwargs):
        raise RuntimeError("no such detector backend: 'nonsense'")

    analyzer._deepface = type("_FakeDeepFace", (), {"represent": staticmethod(broken_represent)})
    analyzer._ready = True

    with pytest.raises(RuntimeError, match="no such detector backend"):
        analyzer.analyze(np.zeros((64, 64, 3), dtype=np.uint8), strict=True)
