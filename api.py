from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="Face Match API", version="1.0.0")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ANALYSIS_ACTIONS = ["age", "gender", "race", "emotion"]


def _decode_image(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{filename} exceeds the 10 MB limit.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"{filename} is not a supported image.")
    return image


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_face_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    return value or []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verify")
async def verify_faces(
    reference: UploadFile = File(...),
    candidate: UploadFile = File(...),
) -> dict[str, Any]:
    reference_image = _decode_image(await reference.read(), reference.filename or "Reference image")
    candidate_image = _decode_image(await candidate.read(), candidate.filename or "Candidate image")

    try:
        from deepface import DeepFace

        verification = DeepFace.verify(
            img1_path=reference_image,
            img2_path=candidate_image,
        )
        result: dict[str, Any] = {
            "verified": bool(verification.get("verified", False)),
            "verification": _json_safe(verification),
            "face_details": None,
        }

        if result["verified"]:
            reference_details = DeepFace.analyze(
                img_path=reference_image,
                actions=ANALYSIS_ACTIONS,
            )
            candidate_details = DeepFace.analyze(
                img_path=candidate_image,
                actions=ANALYSIS_ACTIONS,
            )
            result["face_details"] = {
                "reference": _json_safe(_as_face_list(reference_details)),
                "candidate": _json_safe(_as_face_list(candidate_details)),
            }
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face analysis failed: {exc}") from exc
