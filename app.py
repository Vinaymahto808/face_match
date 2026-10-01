import os

import requests
import streamlit as st

API_URL = os.getenv("FACE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Face Match", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2.2rem; }
    .eyebrow { color: #176b5b; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }
    div[data-testid="stFileUploader"] section { border: 1px solid #d7ded9; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">FACE VERIFICATION / DEEPFACE</div>', unsafe_allow_html=True)
st.title("Compare two faces")
st.caption("Capture two photos in sequence. For phone camera access, open this app over HTTPS and allow permission.")
st.caption("DeepFace compares the captured pair; `pyproject.toml` is project configuration, not a training-image folder.")
for key, default in {
    "reference_bytes": None,
    "reference_type": "image/jpeg",
    "candidate_bytes": None,
    "candidate_type": "image/jpeg",
    "reference_round": 0,
    "candidate_round": 0,
    "match_result": None,
}.items():
    st.session_state.setdefault(key, default)

if st.session_state.reference_bytes is None:
    st.subheader("Step 1 · Capture reference photo")
    reference_capture = st.camera_input(
        "Open camera for the first photo",
        key=f"reference_capture_{st.session_state.reference_round}",
    )
    if reference_capture:
        st.session_state.reference_bytes = reference_capture.getvalue()
        st.session_state.reference_type = reference_capture.type or "image/jpeg"
        st.rerun()
    st.info("Take the first photo. It will be kept in this session while you capture the second.")
elif st.session_state.candidate_bytes is None:
    st.subheader("Step 2 · Capture comparison photo")
    st.image(st.session_state.reference_bytes, caption="Saved reference photo", width="stretch")
    st.caption("Reference photo is stored in session memory only; it is not saved to disk.")

    if st.button("Retake reference photo"):
        st.session_state.reference_round += 1
        st.session_state.candidate_round += 1
        st.session_state.reference_bytes = None
        st.session_state.candidate_bytes = None
        st.session_state.match_result = None
        st.rerun()

    candidate_capture = st.camera_input(
        "Open camera for the second photo",
        key=f"candidate_capture_{st.session_state.candidate_round}",
    )
    if candidate_capture:
        st.session_state.candidate_bytes = candidate_capture.getvalue()
        st.session_state.candidate_type = candidate_capture.type or "image/jpeg"
        st.rerun()
else:
    st.subheader("Step 3 · Compare photos")
    reference_column, candidate_column = st.columns(2, gap="large")
    with reference_column:
        st.image(st.session_state.reference_bytes, caption="Reference photo", width="stretch")
    with candidate_column:
        st.image(st.session_state.candidate_bytes, caption="Second photo", width="stretch")

    reset_column, retake_column, verify_column = st.columns([1, 1, 2])
    with reset_column:
        if st.button("Start over"):
            st.session_state.reference_round += 1
            st.session_state.candidate_round += 1
            st.session_state.reference_bytes = None
            st.session_state.candidate_bytes = None
            st.session_state.match_result = None
            st.rerun()
    with retake_column:
        if st.button("Retake second photo"):
            st.session_state.candidate_round += 1
            st.session_state.candidate_bytes = None
            st.session_state.match_result = None
            st.rerun()
    with verify_column:
        verify_clicked = st.button("Verify faces", type="primary")

    if verify_clicked:
        try:
            with st.spinner("Checking the images… the first request may take longer while models initialize."):
                response = requests.post(
                    f"{API_URL.rstrip('/')}/verify",
                    files={
                        "reference": (
                            "reference.jpg",
                            st.session_state.reference_bytes,
                            st.session_state.reference_type,
                        ),
                        "candidate": (
                            "candidate.jpg",
                            st.session_state.candidate_bytes,
                            st.session_state.candidate_type,
                        ),
                    },
                    timeout=300,
                )
            response.raise_for_status()
            st.session_state.match_result = response.json()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                try:
                    detail = exc.response.json().get("detail", "")
                except ValueError:
                    detail = exc.response.text
            st.error(detail or f"Could not reach the face API at {API_URL}. Start it with uvicorn api:app --reload.")
        except (KeyError, ValueError) as exc:
            st.error(f"The API returned an unexpected response: {exc}")

    result = st.session_state.match_result
    if result is not None:
        st.divider()
        if result["verified"]:
            st.success("Match found")
            face_details = result.get("face_details") or {}
            detail_left, detail_right = st.columns(2, gap="large")
            for column, label, key in (
                (detail_left, "Reference face details", "reference"),
                (detail_right, "Candidate face details", "candidate"),
            ):
                with column:
                    st.subheader(label)
                    details = face_details.get(key, [])
                    if details:
                        for index, face in enumerate(details, start=1):
                            with st.expander(f"Face {index}", expanded=True):
                                st.json(face)
                    else:
                        st.info("No face attributes were returned.")
        else:
            st.warning("No match found")
            st.caption("Face attributes are only displayed for a verified match.")

        with st.expander("Verification details", expanded=True):
            st.json(result.get("verification", {}))
