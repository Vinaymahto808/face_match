"""Streamlit port of ``Face_detect.ipynb``: capture, describe, capture, match.

    uv run streamlit run streamlit_app.py

The notebook runs four cells in a fixed order and prints one block per step.
This app keeps that order and shows every block, one after another:

    1. photo 1  ->  DeepFace.analyze(age, gender, race, emotion) + annotated frame
    2. photo 2  ->  DeepFace.verify                            + side-by-side frames

Both photos live in the Streamlit session and are dropped when the tab closes.
Nothing is written to disk, which is what the notebook did with ``photo.jpg``.

The notebook's ``DeepFace.verify`` used the library defaults (VGG-Face,
euclidean). The sidebar here defaults to Facenet + cosine, which is a cheaper
model at the same accuracy for a two-photo comparison; pick VGG-Face to
reproduce the notebook's exact numbers. Every other model listed downloads its
weights on first use.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from typing import Any

import cv2
import numpy as np
import streamlit as st
from streamlit.runtime import exists as _runtime_exists

if not _runtime_exists():
    # `python streamlit_app.py` runs the file as a plain script. Every widget
    # then fails deep inside Streamlit with "Cursor is not set", which reads like
    # a broken install rather than a missing subcommand. Fail with the fix.
    sys.exit(
        "This file has to be run by Streamlit, not by Python:\n"
        "\n"
        "    uv run streamlit run streamlit_app.py\n"
        "\n"
        "or, without uv:\n"
        "\n"
        "    .venv\\Scripts\\python.exe -m streamlit run streamlit_app.py\n"
    )

MODEL_CHOICES = ("Facenet", "VGG-Face", "ArcFace")
DETECTOR_CHOICES = ("opencv", "yunet", "ssd", "mtcnn", "retinaface")
METRIC_CHOICES = ("cosine", "euclidean", "euclidean_l2")
ACTIONS = ("age", "gender", "race", "emotion")
ATTRIBUTE_MODELS = ("Age", "Gender", "Race", "Emotion")

# DeepFace/TensorFlow is not safe to call from two threads at once, and Streamlit
# runs each session on its own thread. Without this, two open tabs interleave two
# inference calls inside the same Keras model.
_inference_lock = threading.Lock()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the face model…")
def load_models(model_name: str) -> dict[str, Any]:
    """Build the recognition model plus the four attribute models, once.

    Returns the attribute models keyed by lowercase name, with the failure text
    in place of the model when one cannot be built. A missing attribute model
    only drops that column from the analysis; it must not stop the app, because
    the notebook's step 1 is the part people wait on.
    """
    from deepface import DeepFace
    from deepface.modules import modeling

    DeepFace.build_model(model_name)

    built: dict[str, Any] = {}
    for label in ATTRIBUTE_MODELS:
        try:
            built[label.lower()] = modeling.build_model(
                task="facial_attribute", model_name=label
            )
        except Exception as exc:  # noqa: BLE001 - one missing model must not kill step 1
            built[label.lower()] = f"unavailable: {type(exc).__name__}: {exc}"
    return built


def analyze_faces(
    frame: np.ndarray, model_name: str, detector: str
) -> list[dict[str, Any]]:
    """``DeepFace.analyze`` with the notebook's four actions.

    ``enforce_detection=False`` makes DeepFace return an empty list instead of
    raising, which is what the notebook's ``if analysis_results:`` branch expects.
    """
    from deepface import DeepFace

    with _inference_lock:
        results = DeepFace.analyze(
            img_path=frame,
            actions=list(ACTIONS),
            detector_backend=detector,
            enforce_detection=False,
            silent=True,
        )
    return [
        r for r in (results or []) if float(r.get("face_confidence") or 0) > 0
    ]


def verify_faces(
    first: np.ndarray,
    second: np.ndarray,
    model_name: str,
    detector: str,
    metric: str,
) -> dict[str, Any]:
    """``DeepFace.verify``; raises ValueError when a photo has no face."""
    from deepface import DeepFace

    with _inference_lock:
        return DeepFace.verify(
            img1_path=first,
            img2_path=second,
            model_name=model_name,
            detector_backend=detector,
            distance_metric=metric,
            enforce_detection=True,
            silent=True,
        )


# --------------------------------------------------------------------------
# decoding and drawing
# --------------------------------------------------------------------------
def decode(payload: bytes) -> np.ndarray | None:
    buf = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, 3-channel
    return None if frame is None or frame.size == 0 else frame


def _label(out: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Text on a white plate, so it stays readable over any photo."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.5, 1)
    top = max(0, y - th - base)
    cv2.rectangle(out, (x, top), (x + tw + 6, top + th + base + 4), (255, 255, 255), -1)
    cv2.putText(out, text, (x + 3, top + th + 2), font, 0.5, color, 1, cv2.LINE_AA)


def draw_analysis(frame: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
    """The notebook's cell-9 overlay: red box, attributes above, emotion below."""
    out = frame.copy()
    for result in results:
        region = result.get("region") or {}
        x, y = int(region.get("x", 0)), int(region.get("y", 0))
        w, h = int(region.get("w", 0)), int(region.get("h", 0))
        color = (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        _label(out, f"Age: {result.get('age')}, {result.get('dominant_gender')}", x, y - 6, color)
        _label(
            out,
            f"Emotion: {result.get('dominant_emotion')}",
            x,
            y + h + 20,
            color,
        )
    return out


def draw_match(frame: np.ndarray, area: dict[str, Any], verified: bool, caption: str) -> np.ndarray:
    """One face box coloured by the verdict, like simple_app.py's canvas overlay."""
    out = frame.copy()
    color = (0, 200, 0) if verified else (0, 0, 255)
    if area:
        x, y = int(area.get("x", 0)), int(area.get("y", 0))
        w, h = int(area.get("w", 0)), int(area.get("h", 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        _label(out, caption, x, y - 6, color)
    return out


# --------------------------------------------------------------------------
# notebook-style reports
# --------------------------------------------------------------------------
def _py(value: Any) -> Any:
    """numpy scalars/arrays -> plain python, so the text block is printable."""
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_py(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    return value


def _scores(scores: dict[str, Any] | None) -> dict[str, float]:
    return {k: round(float(v), 2) for k, v in (scores or {}).items()}


def _region_text(region: dict[str, Any]) -> str:
    """The notebook prints the raw region dict, eye coordinates included."""
    parts = [f"'{k}': {_py(v)}" for k, v in region.items()]
    return "{" + ", ".join(parts) + "}"


def analysis_report(results: list[dict[str, Any]]) -> str:
    """Byte-for-byte the shape of the notebook's `print` block from cell 9."""
    if not results:
        return "No faces detected in the image."

    lines = ["Analysis Results:"]
    for result in results:
        race, emotion = _scores(result.get("race")), _scores(result.get("emotion"))
        lines += [
            "-----------------",
            f"Face Detected at: {_region_text(result.get('region') or {})}",
            f"Face confidence: {round(float(result.get('face_confidence') or 0), 4)}",
            f"Age: {result.get('age')}",
            f"Gender: {result.get('dominant_gender')} ({_scores(result.get('gender'))})",
            f"Race: {result.get('dominant_race')} ({race})",
            f"Emotion: {result.get('dominant_emotion')} ({emotion})",
        ]
    return "\n".join(lines)


def verify_report(result: dict[str, Any]) -> str:
    """The notebook's `print(verification_result)` plus its one-line conclusion."""
    areas = _py(result.get("facial_areas") or {})
    verdict = (
        "The two images belong to the same person."
        if result.get("verified")
        else "The two images belong to different persons."
    )
    return (
        "Verification Result:\n"
        f"{result.get('verified')=}, {round(float(result.get('distance', 0)), 4)=} "
        f"(cosine distance, below the threshold means the same person), "
        f"{round(float(result.get('threshold', 0)), 2)=}, {result.get('model')=}, "
        f"{result.get('detector_backend')=}, {result.get('similarity_metric')=}, "
        f"{round(float(result.get('time', 0)), 2)=}s\n"
        f"facial_areas: {areas}\n\n{verdict}"
    )


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------
def _score_table(scores: dict[str, float]) -> str:
    if not scores:
        return "not computed"
    return "  ".join(f"{name} {value}%" for name, value in sorted(
        scores.items(), key=lambda kv: kv[1], reverse=True
    ))


def _sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Recognition model",
            MODEL_CHOICES,
            index=0,
            help="Facenet is small and cached locally. VGG-Face is the notebook's "
            "default but is roughly 580 MB of extra weights.",
        )
        detector = st.selectbox(
            "Detector",
            DETECTOR_CHOICES,
            index=0,
            help="opencv is the notebook's detector. retinaface and mtcnn find "
            "small or angled faces better but are slower.",
        )
        metric = st.selectbox(
            "Distance metric",
            METRIC_CHOICES,
            index=0,
            help="cosine compares embedding direction, so it ignores brightness.",
        )
        if st.button("Reset", use_container_width=True):
            for key in ("photo1", "analysis", "analysis_key", "photo2", "verification",
                        "verify_key", "verify_error"):
                st.session_state.pop(key, None)
            st.rerun()
    return model_name, detector, metric


def _shot(key: str, label: str, **kwargs: Any) -> bytes | None:
    """Read a camera widget, ignoring the shot it is still holding from last run."""
    photo = st.camera_input(label, key=key, **kwargs)
    if photo is None:
        return None
    return photo.getvalue()


def _fingerprint(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def main() -> None:
    st.set_page_config(page_title="Face match", page_icon="🙂", layout="wide")
    model_name, detector, metric = _sidebar()

    load_models(model_name)

    st.title("Same person?")
    st.caption(
        "The four notebook cells in order: describe the first face, take a second "
        "photo, then compare them with DeepFace."
    )

    # -- step 1: photo one, described -------------------------------------
    st.subheader("Step 1 — take a photo and describe the face")
    first_payload = _shot("camera1", "Camera 1")

    if first_payload is None:
        st.info("Take the first photo to see its age, gender, race and emotion.")
    else:
        key = (first_payload, model_name, detector)
        if st.session_state.get("analysis_key") != key:
            frame = decode(first_payload)
            if frame is None:
                st.error("That file is not a readable image.")
            else:
                with st.spinner("Analysing the face…"):
                    st.session_state["analysis"] = analyze_faces(frame, model_name, detector)
                st.session_state["analysis_key"] = key

        results = st.session_state.get("analysis") or []
        frame = decode(first_payload)

        if not results:
            st.warning("No faces detected in the image.")
        else:
            best = max(results, key=lambda r: r["region"]["w"] * r["region"]["h"])
            headline = (
                f"{best.get('dominant_gender')}, about {best.get('age')}, "
                f"{best.get('dominant_emotion')}"
            )
            st.success(
                f"{len(results)} face(s) detected — largest one: {headline}."
            )

            annotated, report = st.columns([3, 2])
            annotated.image(
                draw_analysis(frame, results),
                caption="Detected Faces and Attributes",
                channels="BGR",
            )
            with report:
                st.markdown("**Analysis Results**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Age", best.get("age"))
                m2.metric("Gender", best.get("dominant_gender"))
                m3.metric("Emotion", best.get("dominant_emotion"))
                st.caption("**Race scores** — " + _score_table(_scores(best.get("race"))))
                st.caption("**Emotion scores** — " + _score_table(_scores(best.get("emotion"))))
                st.caption("**Gender scores** — " + _score_table(_scores(best.get("gender"))))

            with st.expander("Notebook output", expanded=False):
                st.code(analysis_report(results), language=None)

    # -- step 2: photo two, compared --------------------------------------
    st.subheader("Step 2 — take a second photo and verify")
    if st.session_state.get("photo1") is None and first_payload is not None:
        st.session_state["photo1"] = first_payload

    if st.session_state.get("photo1") is None:
        st.info("Take the first photo above before verifying.")
        return

    second_payload = _shot("camera2", "Camera 2")

    if second_payload is None:
        st.info("Now take the photo to compare.")
        return
    if second_payload == st.session_state["photo1"]:
        st.info("Step 2 — clear the second camera, then take the photo to compare.")
        return

    key = (st.session_state["photo1"], second_payload, model_name, detector, metric)
    if st.session_state.get("verify_key") != key:
        first_frame = decode(st.session_state["photo1"])
        second_frame = decode(second_payload)
        if first_frame is None or second_frame is None:
            st.session_state["verify_error"] = "one of the photos is not a readable image"
        else:
            with st.spinner("Comparing…"):
                try:
                    st.session_state["verification"] = verify_faces(
                        first_frame, second_frame, model_name, detector, metric
                    )
                    st.session_state["verify_error"] = None
                except ValueError as exc:
                    st.session_state["verification"] = None
                    st.session_state["verify_error"] = str(exc).split(".")[0]
        st.session_state["verify_key"] = key

    error = st.session_state.get("verify_error")
    result = st.session_state.get("verification")

    if error and result is None:
        st.error(f"Error during DeepFace verification: {error}")
        return
    if result is None:
        return

    verified = bool(result.get("verified"))
    if verified:
        st.success("## ✅ Verification Result: Same Person")
    else:
        st.error("## ❌ Verification Result: Different Persons")

    left, right = st.columns(2)
    areas = result.get("facial_areas") or {}
    caption = (
        f"{result.get('model')} {round(float(result.get('distance', 0)), 3)}"
        f" / {round(float(result.get('threshold', 0)), 2)}"
    )
    left.image(
        draw_match(decode(st.session_state["photo1"]), areas.get("img1") or {}, verified, caption),
        caption="First Image",
        channels="BGR",
    )
    right.image(
        draw_match(decode(second_payload), areas.get("img2") or {}, verified, caption),
        caption="Second Image",
        channels="BGR",
    )

    a, b, c = st.columns(3)
    a.metric("Distance", f"{float(result.get('distance', 0)):.3f}")
    b.metric("Threshold", f"{float(result.get('threshold', 0)):.2f}")
    c.metric("Time", f"{float(result.get('time', 0)):.2f}s")
    st.caption(
        f"{result.get('model')} · {result.get('detector_backend')} · "
        f"{result.get('similarity_metric')} — distance below the threshold "
        "means the same person."
    )

    with st.expander("Notebook output", expanded=False):
        st.code(verify_report(result), language=None)


main()