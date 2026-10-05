"""Shared test configuration.

Environment variables are set at *import* time, before any ``app.*`` module is
loaded, because :mod:`app.config` builds a settings singleton at import.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest

_TEST_DB = Path("data") / f"pytest_{uuid.uuid4().hex[:8]}.db"
_TEST_DB.parent.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("ENVIRONMENT", "dev")
os.environ["EMBEDDING_BACKEND"] = "stub"
os.environ["ALLOW_STUB_BACKEND"] = "true"
os.environ["DATABASE_URL"] = f"sqlite:///./{_TEST_DB.as_posix()}"
os.environ["REQUIRE_LIVENESS"] = "true"
os.environ["API_KEYS"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ["BUSINESS_TIMEZONE"] = "UTC"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.main import create_app  # noqa: E402


def pytest_sessionfinish(session, exitstatus) -> None:
    """Drop the scratch SQLite file.

    Windows refuses to unlink a file that still has an open handle, so the
    engine (and therefore every pooled connection) must be disposed first.
    """
    from app.db import engine

    engine.dispose()
    for suffix in ("", "-wal", "-shm"):
        Path(str(_TEST_DB) + suffix).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Synthetic imagery
# ---------------------------------------------------------------------------
def make_face_image(
    seed: int = 0,
    *,
    width: int = 640,
    height: int = 480,
    texture: float = 0.035,
    noise: float = 1.6,
    blur: int = 0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Synthesise a face-like BGR frame.

    Deliberately parameterised along the axes the passive liveness cues
    measure (texture energy, sensor noise, blur, chroma) so tests can build a
    "live capture" and a "re-captured print" from the same code path.
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 96, dtype=np.float32)

    # Background vignette so the frame is not a flat field.
    yy = np.linspace(-1.0, 1.0, height)[:, None]
    xx = np.linspace(-1.0, 1.0, width)[None, :]
    radial = np.hypot(xx, yy)
    img -= (radial * 26.0)[..., None]

    # Skin-toned head.
    cy, cx = int(height * 0.46), width // 2
    ry, rx = int(height * 0.28), int(width * 0.17)
    gy, gx = np.ogrid[:height, :width]
    head = ((gy - cy) / ry) ** 2 + ((gx - cx) / rx) ** 2 <= 1.0
    skin = np.array([150.0, 172.0, 205.0])  # BGR
    img[head] = skin

    # Cheek shading.
    shade = 1.0 - 0.16 * np.clip(radial, 0, 1)[..., None]
    img[head] *= shade[head]

    # Hair band on top of the head.
    hair = head & (gy < cy - ry * 0.55)
    img[hair] = np.array([46.0, 52.0, 61.0])

    # Eyes.
    for sign in (-1, 1):
        ex = cx + sign * int(rx * 0.42)
        ey = cy - int(ry * 0.18)
        eye = ((gy - ey) / (ry * 0.13)) ** 2 + ((gx - ex) / (rx * 0.24)) ** 2 <= 1.0
        img[eye] = np.array([28.0, 30.0, 34.0])

    # Nose shadow + mouth.
    nose = ((gy - cy + int(ry * 0.10)) / (ry * 0.22)) ** 2 + (
        (gx - cx) / (rx * 0.16)
    ) ** 2 <= 1.0
    img[nose] = skin * 0.90
    mouth = ((gy - (cy + int(ry * 0.52))) / (ry * 0.10)) ** 2 + (
        (gx - cx) / (rx * 0.45)
    ) ** 2 <= 1.0
    img[mouth] = np.array([92.0, 96.0, 130.0])

    # Pores / micro-texture: high-frequency, what a print+dust pass destroys.
    img += rng.normal(0.0, texture * 255.0, size=img.shape)

    # Sensor noise floor.
    if noise > 0:
        img += rng.normal(0.0, noise, size=img.shape)

    # Desaturate toward the print/screenshot look.
    if saturation != 1.0:
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * saturation

    img = np.clip(img, 0, 255).astype(np.uint8)

    if blur > 0:
        for _ in range(blur):
            img = _box_blur(img, 3)
    return img


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur in NumPy (no OpenCV dependency in tests).

    Dimension-preserving: a running-sum implementation must prepend a zero to
    the cumulative sum, otherwise ``cum[k:] - cum[:-k]`` silently drops a row
    and column per pass. That matters here because the fixture is compared
    against size-dependent quality thresholds.
    """
    arr = img.astype(np.float32)
    k = 2 * radius + 1

    def _pass(plane: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0)] * plane.ndim
        pad_width[axis] = (radius, radius)
        pad = np.pad(plane, pad_width, mode="edge")
        zero = np.zeros_like(np.take(pad, [0], axis=axis))
        cum = np.concatenate([zero, np.cumsum(pad, axis=axis)], axis=axis)
        return (np.take(cum, np.arange(k, cum.shape[axis]), axis=axis)
                - np.take(cum, np.arange(0, cum.shape[axis] - k), axis=axis)) / k

    out = _pass(arr, 0)
    return np.clip(_pass(out, 1), 0, 255).astype(np.uint8)


def make_spoof_image(seed: int = 7) -> np.ndarray:
    """A 'printed photo held up to the lens' analogue.

    Smooth (no pores), noiseless (print has no sensor), blurred by re-imaging
    and slightly desaturated by the print/ICC path.
    """
    img = make_face_image(seed, texture=0.002, noise=0.0, blur=3, saturation=0.55)
    return img


def make_blank_image(width: int = 640, height: int = 480, value: int = 118) -> np.ndarray:
    """A featureless frame: no face, so nothing can be matched against it."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def to_jpeg_bytes(image: np.ndarray, quality: int = 85) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image[:, :, ::-1]).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def schema():
    """Create the tables once per session.

    ``init_db`` is idempotent, so a test file that touches the database
    directly does not have to depend on the HTTP client fixture running first.
    """
    init_db()
    yield


@pytest.fixture(scope="session")
def client(schema):
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_state():
    """Reset process-local caches between tests."""
    from app.services.face import reset_analyzer
    from app.services.liveness.service import reset_registry
    from app.services.registry import roster_cache

    roster_cache.invalidate()
    reset_registry()
    reset_analyzer()
    yield
    reset_registry()
    roster_cache.invalidate()


@pytest.fixture
def db(schema):
    from app.db import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def settings_override():
    """Temporarily patch settings attributes."""

    class _Override:
        def __init__(self) -> None:
            self._saved: list[tuple[str, object]] = []

        def set(self, **kwargs) -> None:
            for key, value in kwargs.items():
                self._saved.append((key, getattr(settings, key)))
                setattr(settings, key, value)

        def restore(self) -> None:
            for key, value in reversed(self._saved):
                setattr(settings, key, value)
            self._saved.clear()

    override = _Override()
    yield override
    override.restore()
