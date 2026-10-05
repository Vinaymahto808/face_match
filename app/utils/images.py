"""Image ingest helpers: decoding, validation, resizing, encoding.

Handles the real-world upload hazards: EXIF-rotated phone photos, base64
data-URI prefixes, alpha channels, and 12-megapixel originals.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "ImageDecodeError",
    "as_bgr_uint8",
    "decode_image",
    "encode_jpeg",
    "resize_bilinear",
    "resize_max_edge",
]


class ImageDecodeError(ValueError):
    """Raised for anything that is not a decodable image."""


def _decode_with_cv2(payload: bytes) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    buf = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, 3-channel
    return img


def _decode_with_pil(payload: bytes) -> np.ndarray:
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(payload)) as im:
            # Phone cameras store orientation in EXIF; without this, portrait
            # photos arrive sideways and every detection is degraded.
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError(f"unreadable image: {exc}") from exc
    return arr[:, :, ::-1].copy()  # RGB -> BGR


def decode_image(source: bytes | bytearray | str) -> np.ndarray:
    """Decode bytes or a base64/data-URI string into a BGR uint8 array."""
    if isinstance(source, str):
        payload_str = source.strip()
        if payload_str.startswith("data:"):
            _, _, payload_str = payload_str.partition(",")
        try:
            source = base64.b64decode(payload_str, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageDecodeError("value is not valid base64") from exc

    payload = bytes(source)
    if not payload:
        raise ImageDecodeError("empty image payload")
    if len(payload) > settings.max_upload_bytes:
        raise ImageDecodeError(
            f"image is {len(payload)} bytes, limit is {settings.max_upload_bytes}"
        )

    img = _decode_with_cv2(payload)
    if img is None or img.size == 0:
        img = _decode_with_pil(payload)
    if img is None or img.size == 0:
        raise ImageDecodeError("image decoded to zero pixels")

    img = as_bgr_uint8(img)
    return resize_max_edge(img, settings.max_frame_edge)


def as_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise dtype/channel count: grayscale -> BGR, RGBA/RGB -> BGR, float -> uint8."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ImageDecodeError(f"unsupported image shape {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] != 3:
        raise ImageDecodeError(f"unsupported channel count {arr.shape[2]}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        # Assume 0..1 for float input, 0..255 otherwise.
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize without OpenCV (used by tests and the liveness path)."""
    src = np.asarray(image, dtype=np.float32)
    h, w = src.shape[:2]
    if h == height and w == width:
        return src.copy()

    def _axis(idx: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        pos = (idx + 0.5) * (size / (idx.size + 1)) - 0.5
        pos = np.clip(pos, 0, size - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, size - 1)
        return pos - lo, (lo, hi)

    ys, y0, y1 = _axis(np.arange(height, dtype=np.float64), h)
    xs, x0, x1 = _axis(np.arange(width, dtype=np.float64), w)

    if src.ndim == 2:
        top = src[y0][:, x0] * (1 - xs) + src[y0][:, x1] * xs
        bot = src[y1][:, x0] * (1 - xs) + src[y1][:, x1] * xs
        return (top * (1 - ys) + bot * ys).astype(np.float32)

    chans = []
    for c in range(src.shape[2]):
        plane = src[..., c]
        top = plane[y0][:, x0] * (1 - xs) + plane[y0][:, x1] * xs
        bot = plane[y1][:, x0] * (1 - xs) + plane[y1][:, x1] * xs
        chans.append((top * (1 - ys) + bot * ys).astype(np.float32))
    return np.stack(chans, axis=2)


def resize_max_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the longest edge is at most ``max_edge``. Never upscales."""
    if max_edge <= 0:
        return image
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return arr
    scale = max_edge / float(longest)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    try:
        import cv2

        interp = cv2.INTER_AREA  # correct downsampling filter for decimation
        return cv2.resize(arr, (new_w, new_h), interpolation=interp)
    except ImportError:
        resized = resize_bilinear(arr, new_w, new_h)
        if resized.dtype == np.float32 and arr.dtype == np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        return resized


def encode_jpeg(image: np.ndarray, quality: int = 70) -> bytes:
    """Encode a frame for streaming to clients. Falls back to PIL if needed."""
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", as_bgr_uint8(image), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass
    from PIL import Image

    rgb = as_bgr_uint8(image)[:, :, ::-1]
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def jpeg_dimensions(payload: bytes) -> tuple[int, int] | None:
    """Read JPEG dimensions from the SOF marker without decoding pixels."""
    i = 2
    n = len(payload)
    while i + 9 < n:
        if payload[i] != 0xFF:
            i += 1
            continue
        marker = payload[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
            height = int.from_bytes(payload[i + 5 : i + 7], "big")
            width = int.from_bytes(payload[i + 7 : i + 9], "big")
            return width, height
        seg_len = int.from_bytes(payload[i + 2 : i + 4], "big")
        if seg_len <= 0:
            return None
        i += 2 + seg_len
    return None


def ceil_to(value: float, step: int) -> int:
    return math.ceil(value / step) * step
