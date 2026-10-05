"""Attendance punch rules, server-issued identity, single-use liveness tokens.

These are unit tests against the service layer rather than the HTTP surface,
because the HTTP tests already prove the routes call this -- what needs proving
*here* is that the security ordering is right. Three properties carry the weight:

1. A punch is authorised **only** by a database ``liveness_sessions`` row that
   reads ``passed`` and has ``consumed_at IS NULL`` -- **and** that row was
   issued for the person being punched.
2. The same token cannot authorise two punches, and neither can a
   ``verifications`` grant.
3. Nothing the caller sends decides *who* is punched or *how well* they
   matched. There is no such field any more; the grant carries both.

If any of these regresses, a replayed photo -- or a colleague's face -- becomes
a valid attendance mark and the test suite still goes green everywhere else.
"""

from __future__ import annotations

import datetime as dt
import typing
import uuid

import numpy as np
import pytest

from app.config import settings
from app.errors import (
    AppError,
    IdentityMismatchError,
    LivenessRequiredError,
    NotFoundError,
    SpoofSuspectedError,
    ValidationFailedError,
)
from app.models import LivenessSession, Verification, utcnow
from app.services import embeddings as emb
from app.services import registry as reg
from app.services import verifications
from app.services.liveness.service import consume_token
from app.services.punch import punch_attendance


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def user(db):
    user_id = f"punch-{uuid.uuid4().hex[:8]}"
    vector = emb.coerce_vector(np.arange(512, dtype=np.float32) / 512.0)
    return reg.create_user(db, user_id=user_id, name="Punch Tester", embedding=vector)


@pytest.fixture
def other_user(db):
    """A second real person, for the "somebody else's session" tests."""
    user_id = f"other-{uuid.uuid4().hex[:8]}"
    vector = emb.coerce_vector(np.arange(512, dtype=np.float32)[::-1] / 512.0)
    return reg.create_user(db, user_id=user_id, name="Somebody Else", embedding=vector)


@pytest.fixture
def make_session(db):
    """Insert a ``liveness_sessions`` row directly, bypassing the engine."""

    def _make(
        *,
        status: str = "passed",
        user_id: str | None = None,
        active_score: float | None = 0.93,
        passive_score: float | None = None,
        consumed: bool = False,
        expired: bool = False,
    ) -> LivenessSession:
        now = utcnow()
        row = LivenessSession(
            id=uuid.uuid4().hex,
            user_id=user_id,
            status=status,
            challenges=[],
            attempts=1,
            max_attempts=3,
            passive_score=passive_score,
            active_score=active_score,
            created_at=now,
            updated_at=now,
            expires_at=now - dt.timedelta(seconds=5) if expired else now + dt.timedelta(minutes=5),
            completed_at=now if status in ("passed", "failed") else None,
            consumed_at=now if consumed else None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    return _make


@pytest.fixture
def make_grant(db):
    """Insert a ``verifications`` row, as the face pipeline would have."""

    def _make(
        *,
        user_id: str,
        distance: float = 0.1,
        passive_liveness: float | None = 0.8,
        spoof_flags: list[str] | None = None,
        consumed: bool = False,
        expired: bool = False,
    ) -> Verification:
        now = utcnow()
        row = Verification(
            id=uuid.uuid4().hex,
            user_id=user_id,
            distance=distance,
            passive_liveness=passive_liveness,
            spoof_flags=list(spoof_flags or []),
            source="unit-test",
            created_at=now,
            expires_at=now - dt.timedelta(seconds=5) if expired else now + dt.timedelta(minutes=2),
            consumed_at=now if consumed else None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    return _make


def _punch(db, user=None, **kwargs):
    """Punch ``user``, minting a grant for them unless one is supplied."""
    params = {"source": "unit-test"}
    if "verification_id" not in kwargs and user is not None:
        params["verification_id"] = verifications.issue(
            db, user_id=user.id, distance=0.1
        ).id
    params.update(kwargs)
    return punch_attendance(db, **params)


# ---------------------------------------------------------------------------
# the grant is the identity -- there is nothing else to trust
# ---------------------------------------------------------------------------
def test_punch_without_a_verification_is_refused(db, user, make_session):
    session = make_session(status="passed", user_id=user.id)
    with pytest.raises(ValidationFailedError) as exc:
        punch_attendance(db, liveness_session_id=session.id, source="unit-test")
    assert exc.value.code == "unverified_identity"
    assert exc.value.status_code == 422


def test_unknown_verification_id_is_refused(db, user, make_session):
    session = make_session(status="passed", user_id=user.id)
    with pytest.raises(ValidationFailedError, match="unknown_verification"):
        _punch(db, user, verification_id="f" * 32, liveness_session_id=session.id)


def test_expired_verification_is_refused(db, user, make_session, make_grant):
    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id, expired=True)
    with pytest.raises(ValidationFailedError, match="verification_expired"):
        _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)


def test_a_grant_authorises_exactly_one_punch(db, user, make_session, make_grant):
    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id)
    _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)

    second_session = make_session(status="passed", user_id=user.id)
    with pytest.raises(ValidationFailedError, match="verification_already_used"):
        _punch(db, user, verification_id=grant.id, liveness_session_id=second_session)


def test_the_person_punched_is_the_person_the_server_matched(db, user, other_user, make_session):
    """The bug this whole module exists to prevent.

    A caller with a valid liveness session used to name any ``user_id`` they
    liked. Now the session and the grant must agree on who that is.
    """
    alice, bob = user, other_user
    alice_session = make_session(status="passed", user_id=alice.id)
    bob_grant = verifications.issue(db, user_id=bob.id, distance=0.05)

    with pytest.raises(IdentityMismatchError) as exc:
        _punch(db, bob, verification_id=bob_grant.id, liveness_session_id=alice_session.id)
    assert exc.value.code == "identity_mismatch"
    assert exc.value.status_code == 403

    # And nothing was written for Bob.
    from app.models import Attendance

    assert db.query(Attendance).filter(Attendance.user_id == bob.id).count() == 0


def test_a_mismatched_session_is_left_usable_for_its_own_user(
    db, user, other_user, make_session, make_grant
):
    """A refused punch must not cost the rightful owner their passed session."""
    session = make_session(status="passed", user_id=user.id)

    with pytest.raises(IdentityMismatchError):
        _punch(
            db,
            other_user,
            verification_id=verifications.issue(db, user_id=other_user.id, distance=0.1).id,
            liveness_session_id=session.id,
        )

    db.expire_all()
    assert consume_token(db, session.id, user_id=user.id)[0] is True


def test_an_anonymous_session_binds_to_the_first_person_who_spends_it(
    db, user, other_user, make_session, make_grant
):
    """A session started without a claimed identity still cannot be shared.

    It is adopted *and* consumed in one transaction, so there is exactly one
    winner and no window in which two people hold the same token.
    """
    session = make_session(status="passed", user_id=None)
    _punch(db, user, verification_id=make_grant(user_id=user.id).id, liveness_session_id=session.id)

    db.expire_all()
    stored = db.get(LivenessSession, session.id)
    assert stored.user_id == user.id

    with pytest.raises(LivenessRequiredError, match="liveness_token_already_used"):
        _punch(
            db,
            other_user,
            verification_id=make_grant(user_id=other_user.id).id,
            liveness_session_id=session.id,
        )


def test_unknown_user_is_rejected(db, make_grant, make_session):
    """A grant can outlive the person: deactivated, or purged, after verifying."""
    ghost = make_grant(user_id="nobody", distance=0.05)
    with pytest.raises(NotFoundError):
        _punch(db, None, verification_id=ghost.id)


def test_inactive_user_is_rejected(db, user, make_grant):
    reg.deactivate_user(db, user.id)
    db.refresh(user)
    assert not user.is_active
    with pytest.raises(NotFoundError):
        _punch(db, user, verification_id=make_grant(user_id=user.id).id)


def test_unknown_user_check_runs_before_the_liveness_check(db, make_grant):
    """Ordering matters: don't burn or leak liveness state for a bogus id."""
    ghost = make_grant(user_id="nobody", distance=0.05)
    with pytest.raises(NotFoundError) as exc:
        _punch(db, None, verification_id=ghost.id, liveness_session_id="nope")
    assert exc.value.code == "user_not_found"


# ---------------------------------------------------------------------------
# liveness token requirement
# ---------------------------------------------------------------------------
def test_punch_without_a_session_is_refused_when_liveness_is_required(db, user):
    assert settings.require_liveness is True  # conftest sets REQUIRE_LIVENESS=true
    with pytest.raises(LivenessRequiredError) as exc:
        _punch(db, user)
    assert exc.value.context == {"require_liveness": True}


def test_punch_succeeds_with_a_passed_session(db, user, make_session):
    session = make_session(status="passed", user_id=user.id)
    result, message = _punch(db, user, liveness_session_id=session.id)
    assert result.created is True
    assert message == "attendance marked"
    assert result.record.liveness_session_id == session.id
    assert result.record.match_distance == pytest.approx(0.1)


def test_pending_session_is_refused(db, user, make_session):
    session = make_session(status="pending", user_id=user.id)
    with pytest.raises(LivenessRequiredError, match="liveness_pending"):
        _punch(db, user, liveness_session_id=session.id)


def test_failed_session_is_refused(db, user, make_session):
    session = make_session(status="failed", user_id=user.id)
    with pytest.raises(LivenessRequiredError, match="liveness_failed"):
        _punch(db, user, liveness_session_id=session.id)


def test_unknown_session_id_is_refused(db, user):
    with pytest.raises(LivenessRequiredError, match="unknown_liveness_session"):
        _punch(db, user, liveness_session_id="deadbeef" * 4)


def test_expired_session_is_refused(db, user, make_session):
    session = make_session(status="passed", user_id=user.id, expired=True)
    with pytest.raises(LivenessRequiredError, match="liveness_token_expired"):
        _punch(db, user, liveness_session_id=session.id)


# ---------------------------------------------------------------------------
# single use -- the anti-replay invariant
# ---------------------------------------------------------------------------
def test_a_token_authorises_exactly_one_punch(db, user, make_session):
    session = make_session(status="passed", user_id=user.id)
    _punch(db, user, liveness_session_id=session.id)
    with pytest.raises(LivenessRequiredError, match="liveness_token_already_used"):
        _punch(db, user, liveness_session_id=session.id)


def test_a_consumed_token_is_refused_even_if_never_used_by_this_path(db, user, make_session):
    """``consumed_at`` in the database is the authority, not process memory."""
    session = make_session(status="passed", user_id=user.id, consumed=True)
    with pytest.raises(LivenessRequiredError, match="liveness_token_already_used"):
        _punch(db, user, liveness_session_id=session.id)


def test_rejected_identity_does_not_consume_the_token(db, user, make_session, make_grant):
    """A caller must not be able to burn a good session with a bad verification.

    The token is consumed *after* the identity checks precisely so a failed
    verification leaves the session usable for an honest retry.
    """
    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id, distance=settings.match_threshold + 0.1)
    with pytest.raises(AppError):
        _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)

    db.expire_all()
    assert consume_token(db, session.id, user_id=user.id)[0] is True, "token should still be usable"


def test_a_rejected_punch_leaves_the_grant_spendable(db, user, make_session, make_grant):
    grant = make_grant(user_id=user.id)
    with pytest.raises(LivenessRequiredError):
        _punch(db, user, verification_id=grant.id)  # no liveness session

    db.expire_all()
    result, _ = _punch(
        db, user, verification_id=grant.id,
        liveness_session_id=make_session(status="passed", user_id=user.id).id,
    )
    assert result.created is True


def test_two_punches_need_two_tokens(db, user, make_session):
    first = make_session(status="passed", user_id=user.id)
    second = make_session(status="passed", user_id=user.id)
    _punch(db, user, liveness_session_id=first.id)
    result, message = _punch(db, user, liveness_session_id=second.id)
    # Second punch of the day updates the same row rather than duplicating it.
    assert result.created is False
    assert result.record.punch_count == 2
    assert "already marked present" in message


def test_token_score_is_authoritative_over_a_smaller_client_claim(db, user, make_session):
    """The client may raise its claimed score; it may never lower the real one."""
    session = make_session(status="passed", user_id=user.id, active_score=0.91)
    result, _ = _punch(db, user, liveness_session_id=session.id, liveness_score=0.05)
    assert result.record.liveness_score == pytest.approx(0.91)


def test_a_higher_client_claim_is_taken_as_a_floor(db, user, make_session):
    session = make_session(status="passed", user_id=user.id, active_score=0.60)
    result, _ = _punch(db, user, liveness_session_id=session.id, liveness_score=0.95)
    assert result.record.liveness_score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# spoof flags, carried by the grant because the server measured them
# ---------------------------------------------------------------------------
def test_spoof_flags_block_the_punch(db, user, make_session, make_grant):
    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id, spoof_flags=["cnn_spoof_probability"])
    with pytest.raises(SpoofSuspectedError) as exc:
        _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)
    assert exc.value.context["flags"] == ["cnn_spoof_probability"]


def test_spoof_flagging_is_recorded_as_a_critical_event(db, user, make_session, make_grant):
    from app.models import Event

    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id, spoof_flags=["smoothed_texture"])
    with pytest.raises(SpoofSuspectedError):
        _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)

    logged = db.query(Event).filter(Event.kind == "spoof_detected").all()
    assert any(e.user_id == user.id and e.severity == "critical" for e in logged)


def test_an_identity_mismatch_raises_a_critical_alert(db, user, other_user, make_session):
    """A transferable session is a spoofing signal, not a support ticket."""
    from app.models import Event

    session = make_session(status="passed", user_id=user.id)
    with pytest.raises(IdentityMismatchError):
        _punch(
            db,
            other_user,
            verification_id=verifications.issue(db, user_id=other_user.id, distance=0.1).id,
            liveness_session_id=session.id,
        )

    logged = db.query(Event).filter(Event.kind == "attendance_blocked").all()
    assert any(
        e.user_id == other_user.id
        and e.severity == "critical"
        and e.context.get("liveness_session_id") == session.id
        for e in logged
    )


# ---------------------------------------------------------------------------
# match distance, now the server's number
# ---------------------------------------------------------------------------
def test_distance_above_threshold_is_refused(db, user, make_session, make_grant):
    """Defence in depth: the grant came from our matcher, but the bar can move.

    ``MATCH_THRESHOLD`` may have been retuned, or the roster re-enrolled,
    between the frame and the punch. The grant is re-checked anyway.
    """
    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id, distance=settings.match_threshold + 0.1)
    with pytest.raises(AppError) as exc:
        _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)
    assert exc.value.code == "unverified_identity"
    # Pinned because README §3 publishes this status to kiosk integrators.
    # A well-formed request whose business rule failed is a 422, not a 400.
    assert exc.value.status_code == 422


def test_distance_exactly_at_the_threshold_is_accepted(db, user, make_session, make_grant):
    session = make_session(status="passed", user_id=user.id)
    grant = make_grant(user_id=user.id, distance=settings.match_threshold)
    result, _ = _punch(db, user, verification_id=grant.id, liveness_session_id=session.id)
    assert result.created is True


# ---------------------------------------------------------------------------
# liveness disabled (local/dev mode)
# ---------------------------------------------------------------------------
def test_punch_without_a_session_is_allowed_when_liveness_is_off(
    db, user, settings_override
):
    settings_override.set(require_liveness=False)
    result, _ = _punch(db, user)
    assert result.created is True


def test_a_bogus_session_is_ignored_when_liveness_is_off(db, user, settings_override):
    settings_override.set(require_liveness=False)
    result, _ = _punch(db, user, liveness_session_id="nonexistent")
    assert result.created is True
    assert result.record.liveness_session_id == "nonexistent"


# ---------------------------------------------------------------------------
# consume_token contract
# ---------------------------------------------------------------------------
def test_consume_token_reports_each_failure_reason(db, make_session):
    assert consume_token(db, "nope")[1] == "unknown_liveness_session"
    assert consume_token(db, make_session(status="pending").id)[1] == "liveness_pending"
    assert consume_token(db, make_session(status="failed").id)[1] == "liveness_failed"
    assert consume_token(db, make_session(status="passed", consumed=True).id)[1] == (
        "liveness_token_already_used"
    )
    assert consume_token(db, make_session(status="passed", expired=True).id)[1] == (
        "liveness_token_expired"
    )


def test_consume_token_marks_the_row_consumed(db, make_session):
    session = make_session(status="passed", active_score=0.9)
    ok, reason, score = consume_token(db, session.id)
    assert (ok, reason) == (True, "ok")
    assert score == pytest.approx(0.9)
    db.expire_all()
    assert consume_token(db, session.id)[1] == "liveness_token_already_used"


def test_consume_token_binds_an_anonymous_session(db, user, make_session):
    session = make_session(status="passed", user_id=None)
    assert consume_token(db, session.id, user_id=user.id)[0] is True
    db.expire_all()
    assert db.get(LivenessSession, session.id).user_id == user.id


def test_consume_token_refuses_another_users_session(db, user, make_session):
    session = make_session(status="passed", user_id=user.id)
    ok, reason, _ = consume_token(db, session.id, user_id="somebody-else")
    assert (ok, reason) == (False, "liveness_session_user_mismatch")
    db.expire_all()
    assert db.get(LivenessSession, session.id).consumed_at is None


def test_consume_token_corroborates_with_passive_without_averaging_down(
    db, make_session
):
    """A strong active score must not be dragged down by a weak passive cue."""
    session = make_session(status="passed", active_score=0.9, passive_score=0.4)
    _, _, score = consume_token(db, session.id)
    assert score == pytest.approx(0.9)


def test_consume_token_can_be_strengthened_by_passive(db, make_session):
    session = make_session(status="passed", active_score=0.3, passive_score=0.9)
    _, _, score = consume_token(db, session.id)
    assert score == pytest.approx(0.45)


def test_consume_token_with_no_scores_is_zero(db, make_session):
    session = make_session(status="passed", active_score=None, passive_score=None)
    assert consume_token(db, session.id)[2] == 0.0


# ---------------------------------------------------------------------------
# the grant service
# ---------------------------------------------------------------------------
def test_issue_returns_the_values_it_was_given(db, user):
    grant = verifications.issue(db, user_id=user.id, distance=0.31, passive_liveness=0.7)
    assert grant.distance == pytest.approx(0.31)
    assert grant.passive_liveness == pytest.approx(0.7)
    assert grant.spoof_flags == []


def test_lookup_does_not_spend_the_grant(db, user):
    grant = verifications.issue(db, user_id=user.id, distance=0.2)
    row, reason = verifications.lookup(db, grant.id)
    assert reason == "ok"
    assert row is not None
    assert verifications.lookup(db, grant.id)[0] is not None, "lookup must not consume"


def test_issue_honours_an_explicit_ttl(db, user):
    grant = verifications.issue(db, user_id=user.id, distance=0.1, ttl_seconds=-1)
    assert verifications.lookup(db, grant.id)[1] == "verification_expired"


def test_expired_grants_are_pruned(db, user):
    """Each issue sweeps the rows that can no longer be spent."""
    kept = verifications.issue(db, user_id=user.id, distance=0.1)
    verifications.issue(db, user_id=user.id, distance=0.1, ttl_seconds=-5)
    assert verifications.prune_expired(db) >= 1
    assert verifications.lookup(db, kept.id)[0] is not None


def test_issue_from_result_refuses_an_ineligible_frame(db, user):
    """No grant without a match: ``eligible_for_attendance`` is the gate."""

    class _Match:
        decision = "unknown"
        user_id = user.id
        distance = 0.1

    class _Result:
        eligible_for_attendance = False
        match = _Match()
        passive = None
        spoof_suspected = False

    assert verifications.issue_from_result(db, _Result()) is None


def test_issue_from_result_carries_flags_only_on_a_spoof_verdict(db, user):
    class _Passive:
        score = 0.2
        flags: typing.ClassVar[list[str]] = ["moire", "flat_texture"]

    class _Match:
        decision = "match"
        user_id = user.id
        distance = 0.2

    class _Result:
        eligible_for_attendance = True
        match = _Match()
        passive = _Passive()
        spoof_suspected = False

    benign = verifications.issue_from_result(db, _Result())
    assert benign is not None and benign.spoof_flags == [], "advisory flags are not a verdict"

    _Result.spoof_suspected = True
    flagged = verifications.issue_from_result(db, _Result())
    assert flagged is not None and flagged.spoof_flags == ["moire", "flat_texture"]
