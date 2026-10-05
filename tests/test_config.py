"""Startup validation of :class:`app.config.Settings`.

Every validator here exists so that a misconfiguration surfaces as a loud
refusal to boot rather than as a wrong punch time, a loosened match bar, or a
webhook that reads a local file three weeks into production.
"""

from __future__ import annotations

import pytest

from app.config import Settings


def _settings(**overrides) -> Settings:
    """Build a fresh Settings, bypassing the import-time singleton.

    Init kwargs outrank anything in the environment, so this is deterministic
    regardless of what conftest exported.
    """
    return Settings(**overrides)


# ---------------------------------------------------------------------------
# ALERT_WEBHOOK_URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.example.com/abc",
        "https://hooks.example.com/abc",
        "HTTPS://HOOKS.EXAMPLE.COM/ABC",
    ],
)
def test_webhook_accepts_http_and_https(url):
    assert _settings(alert_webhook_url=url).alert_webhook_url.startswith(("http", "HTTPS"))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/hook",
        "gopher://example.com/hook",
        "data:text/plain,hi",
    ],
)
def test_webhook_rejects_non_http_schemes(url):
    # events._post hands this to urllib.request.urlopen unfiltered, so a
    # non-HTTP scheme is an operator footgun, not a feature.
    with pytest.raises(ValueError, match="must be http"):
        _settings(alert_webhook_url=url)


@pytest.mark.parametrize("url", ["", "   "])
def test_empty_webhook_is_allowed(url):
    assert _settings(alert_webhook_url=url).alert_webhook_url == ""


def test_webhook_is_stripped():
    assert _settings(alert_webhook_url="  https://example.com/h  ").alert_webhook_url == (
        "https://example.com/h"
    )


# ---------------------------------------------------------------------------
# BUSINESS_TIMEZONE
# ---------------------------------------------------------------------------
def test_unknown_timezone_is_rejected():
    with pytest.raises(ValueError, match="BUSINESS_TIMEZONE"):
        _settings(business_timezone="Mars/Olympus_Mons")


@pytest.mark.parametrize("tz", ["UTC", "Asia/Kolkata", "Europe/Berlin"])
def test_known_timezone_is_accepted(tz):
    assert _settings(business_timezone=tz).business_timezone == tz


# ---------------------------------------------------------------------------
# Numeric bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, -0.1, 1.5, 2.0])
def test_match_threshold_must_be_in_range(value):
    with pytest.raises(ValueError, match="MATCH_THRESHOLD"):
        _settings(match_threshold=value)


@pytest.mark.parametrize("value", [0.0, 0.3, 1.0])
def test_passive_liveness_min_accepts_its_closed_range(value):
    # 0.0 is legal: it disables the passive bar without deleting the feature.
    assert _settings(passive_liveness_min=value).passive_liveness_min == value


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_passive_liveness_min_outside_range_is_rejected(value):
    with pytest.raises(ValueError, match="PASSIVE_LIVENESS_MIN"):
        _settings(passive_liveness_min=value)


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def test_cors_origin_list_splits_and_drops_blanks():
    assert _settings(cors_origins=" https://a.example , ,https://b.example ").cors_origin_list == [
        "https://a.example",
        "https://b.example",
    ]


def test_api_key_list_splits_and_drops_blanks():
    parsed = _settings(api_keys="k1, k2 ,,k3").api_key_list
    assert parsed == ["k1", "k2", "k3"]
    assert len(set(parsed)) == 3, "duplicate keys must not weaken the comparison set"