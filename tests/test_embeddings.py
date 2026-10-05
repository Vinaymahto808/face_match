"""Embedding codec, consensus averaging and cosine matching."""

from __future__ import annotations

import numpy as np
import pytest

from app.services import embeddings as emb
from app.services.registry import RosterSnapshot, consensus_embedding, match_embedding


def test_pack_unpack_roundtrip_is_lossless_enough():
    rng = np.random.default_rng(1)
    vec = rng.normal(size=512)
    blob = emb.pack(vec)
    out = emb.unpack(blob, 512)
    assert out.size == 512
    # float32 storage of a unit vector: cosine similarity must be ~1.0
    assert emb.cosine_similarity_matrix(vec, out[None, :])[0] < 1e-5


def test_pack_produces_compact_blob():
    """512 float32 = 2048 bytes. JSON would be ~11 KB."""
    rng = np.random.default_rng(2)
    assert emb.pack(rng.normal(size=512)).__len__() == 2048


def test_pack_normalises_to_unit_length():
    vec = np.array([3.0, 4.0])
    out = emb.unpack(emb.pack(vec), 2)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)


def test_zero_vector_does_not_divide_by_zero():
    out = emb.unpack(emb.pack(np.zeros(8)), 8)
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


def test_unpack_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="dim mismatch"):
        emb.unpack(emb.pack(np.ones(16)), 512)


def test_unpack_rejects_empty_blob():
    with pytest.raises(ValueError, match="empty"):
        emb.unpack(b"", None)


def test_coerce_vector_accepts_nested_single_row():
    out = emb.coerce_vector(np.array([[1.0, 2.0, 3.0]]))
    assert out.shape == (3,)


def test_coerce_vector_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        emb.coerce_vector([1.0, float("nan")])


def test_cosine_distance_matches_deepface_convention():
    """Identical vectors -> distance 0; opposite -> distance 2."""
    v = np.array([1.0, 2.0, 3.0])
    same = emb.cosine_similarity_matrix(v, emb.l2_normalize(v)[None, :])
    opposite = emb.cosine_similarity_matrix(v, -emb.l2_normalize(v)[None, :])
    assert same[0] == pytest.approx(0.0, abs=1e-6)
    assert opposite[0] == pytest.approx(2.0, abs=1e-6)


def test_consensus_reduces_noise():
    """Averaging noisy samples of one vector must land closer than any sample."""
    rng = np.random.default_rng(3)
    true = rng.normal(size=512)
    noisy = [true + rng.normal(0, 0.5, size=512) for _ in range(8)]

    avg = consensus_embedding(noisy)
    err_avg = emb.cosine_similarity_matrix(true, avg[None, :])[0]
    err_each = [emb.cosine_similarity_matrix(true, emb.l2_normalize(n)[None, :])[0] for n in noisy]

    assert err_avg < min(err_each)
    assert np.isclose(np.linalg.norm(avg), 1.0, atol=1e-5)


def test_consensus_requires_samples():
    with pytest.raises(ValueError):
        consensus_embedding([])


def _roster(ids_and_names, seed=4) -> RosterSnapshot:
    rng = np.random.default_rng(seed)
    vecs = [emb.l2_normalize(rng.normal(size=128)) for _ in ids_and_names]
    return RosterSnapshot(
        model_name="Facenet",
        ids=tuple(i for i, _ in ids_and_names),
        names=tuple(n for _, n in ids_and_names),
        matrix=np.vstack(vecs).astype(np.float32),
    )


def test_match_finds_the_right_person():
    roster = _roster([("A", "Alice"), ("B", "Bob"), ("C", "Carol")])
    query = roster.matrix[1]  # Bob
    result = match_embedding(roster, query, threshold=0.4, gray_factor=1.25)
    assert result.decision == "match"
    assert result.user_id == "B"
    assert result.name == "Bob"
    assert result.distance < 0.01


def test_impostor_falls_into_gray_zone_or_unknown():
    roster = _roster([("A", "Alice")], seed=9)
    rng = np.random.default_rng(99)
    result = match_embedding(roster, rng.normal(size=128), threshold=0.4, gray_factor=1.25)
    assert result.decision in ("review", "unknown")
    assert not result.matched


def test_empty_roster_returns_unknown_not_a_crash():
    empty = RosterSnapshot(model_name="Facenet", ids=(), names=(), matrix=np.zeros((0, 0), np.float32))
    result = match_embedding(empty, np.ones(128), threshold=0.4, gray_factor=1.25)
    assert result.decision == "unknown"
    assert result.user_id is None
    assert result.distance == float("inf")


def test_gray_zone_is_reported_as_review_not_unknown():
    """A borderline match must be visible to a human, not silently 'unknown'.

    eps=0.11 lands the cosine distance just past the 0.40 threshold but inside
    the 0.50 gray-zone band.
    """
    roster = _roster([("A", "Alice")], seed=5)
    query = roster.matrix[0] + 0.11 * np.random.default_rng(6).normal(size=128)
    result = match_embedding(roster, query, threshold=0.40, gray_factor=1.25)
    assert result.decision == "review"
    assert 0.40 < result.distance <= 0.50
