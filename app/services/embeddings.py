"""Embedding (de)serialisation and distance maths.

Vectors are stored as little-endian float32 BLOBs and L2-normalised once, on
write. Every read path then gets a unit vector for free, so a match is a
single ``matmul`` instead of a per-pair dot/norm/norm loop, and numerical
drift from repeated normalisation can't accumulate in the database.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DTYPE",
    "best_match",
    "coerce_vector",
    "cosine_similarity_matrix",
    "l2_normalize",
    "pack",
    "unpack",
]

DTYPE = np.dtype("<f4")  # explicit endianness: blobs outlive this machine


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-length copy. Zero vectors are returned unchanged (as zeros)."""
    v = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v.astype(DTYPE)
    return (v / norm).astype(DTYPE)


def pack(vec: np.ndarray) -> bytes:
    """Normalise and pack to a float32 BLOB."""
    return l2_normalize(vec).tobytes()


def unpack(blob: bytes, dim: int | None = None) -> np.ndarray:
    """Unpack a float32 BLOB. Raises if the length is not a multiple of 4."""
    arr = np.frombuffer(blob, dtype=DTYPE)
    if dim is not None and arr.size != dim:
        raise ValueError(f"embedding dim mismatch: stored {arr.size}, expected {dim}")
    if arr.size == 0:
        raise ValueError("empty embedding blob")
    return arr


def coerce_vector(values) -> np.ndarray:
    """Accept a list/tuple/np.ndarray/bytes and return a validated 1-D array."""
    if isinstance(values, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(values), dtype=DTYPE).astype(np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D embedding, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding contains NaN or Inf")
    return arr


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against a stack of unit vectors.

    Returns ``distance = 1 - similarity`` to match DeepFace's convention, so
    thresholds carry over from the notebook and from DeepFace's own defaults.
    """
    q = l2_normalize(np.asarray(query, dtype=np.float64)).astype(np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    sims = m @ q
    return 1.0 - np.clip(sims, -1.0, 1.0)


def best_match(
    query: np.ndarray, ids: list[str], names: list[str], matrix: np.ndarray
) -> tuple[str | None, str | None, float]:
    """Return ``(id, name, distance)`` for the nearest stored vector.

    ``(None, None, inf)`` when the roster is empty.
    """
    distances = cosine_similarity_matrix(query, matrix)
    if distances.size == 0:
        return None, None, float("inf")
    idx = int(np.argmin(distances))
    return ids[idx], names[idx], float(distances[idx])
