"""Application settings, loaded from environment / .env file."""

from __future__ import annotations

import functools
from typing import Literal
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ----------------------------------------------------------------
    app_name: str = "face-attendance-api"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: str = "*"

    # --- Database -----------------------------------------------------------
    database_url: str = "sqlite:///./data/attendance.db"
    db_echo: bool = False
    business_timezone: str = "Asia/Kolkata"

    # --- Ingest limits ------------------------------------------------------
    max_upload_bytes: int = 5 * 1024 * 1024
    max_frame_edge: int = 1280

    # --- Face recognition ---------------------------------------------------
    embedding_backend: Literal["deepface", "stub"] = "deepface"
    face_model_name: str = "Facenet"
    face_detector: str = "opencv"
    match_threshold: float = 0.40
    match_gray_zone_factor: float = 1.25
    allow_stub_backend: bool = False
    # How long a server-issued identity proof stays spendable by a punch. The
    # face that earned it should be the face at the desk, so this is a kiosk
    # flow budget (recognise -> challenge -> punch), not a session lifetime.
    verification_ttl_seconds: int = 120

    # --- Frame / face quality gate -----------------------------------------
    # A blurry, tiny, blown-out or heavily compressed crop produces an embedding
    # that matches nobody well and pollutes the enrolled roster. Rejecting it
    # upstream is cheaper than debugging bad matches later.
    min_face_edge_px: int = 64
    min_frame_edge_px: int = 160
    min_sharpness: float = 6.0
    min_brightness: float = 30.0
    max_brightness: float = 235.0
    min_contrast: float = 12.0
    max_flat_region_fraction: float = 0.55

    # --- Anti-spoofing ------------------------------------------------------
    require_liveness: bool = True
    passive_liveness_min: float = 0.55
    active_liveness_min_challenges: int = 2
    liveness_challenges: str = "blink,head_turn_left,head_turn_right,move_closer"
    liveness_session_ttl_seconds: int = 90
    liveness_max_attempts: int = 3
    liveness_lockout_seconds: int = 300
    liveness_fasnet_model: str = ""

    # --- Security / ops -----------------------------------------------------
    api_keys: str = ""
    rate_limit_per_minute: int = 120
    # Off by default. When on, the rate limiter buckets by the first
    # X-Forwarded-For entry -- correct only behind a proxy that *overwrites*
    # that header. Left on without one, every caller mints a fresh bucket per
    # request by changing the header, and the limit stops existing.
    trust_proxy_headers: bool = False
    alert_webhook_url: str = ""

    @field_validator("alert_webhook_url")
    @classmethod
    def _webhook_must_be_http(cls, v: str) -> str:
        """Reject non-HTTP schemes up front.

        events._post hands this string straight to urllib.request.urlopen, so a
        `file:` value here would turn an ops convenience into a local-file read.
        Validate once at startup rather than at the moment an alert fires.
        """
        v = v.strip()
        if not v:
            return v
        scheme = urlsplit(v).scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(f"ALERT_WEBHOOK_URL must be http(s), got {scheme!r}")
        return v

    @field_validator("business_timezone")
    @classmethod
    def _tz_must_exist(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:  # pragma: no cover
            raise ValueError(f"unknown BUSINESS_TIMEZONE: {v!r}") from exc
        return v

    @field_validator("match_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not 0.0 < v < 1.5:
            raise ValueError("MATCH_THRESHOLD must be in (0, 1.5)")
        return v

    @field_validator("passive_liveness_min")
    @classmethod
    def _passive_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("PASSIVE_LIVENESS_MIN must be in [0, 1]")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        raw = [p.strip() for p in self.cors_origins.split(",") if p.strip()]
        return raw or ["*"]

    @property
    def api_key_list(self) -> list[str]:
        return [p.strip() for p in self.api_keys.split(",") if p.strip()]

    @property
    def challenge_pool(self) -> list[str]:
        return [p.strip().lower() for p in self.liveness_challenges.split(",") if p.strip()]

    @property
    def gray_zone_threshold(self) -> float:
        return self.match_threshold * self.match_gray_zone_factor

    @property
    def auth_required(self) -> bool:
        """Auth is enforced only once keys are configured, so dev stays frictionless."""
        return self.environment == "prod" or bool(self.api_key_list)

    @property
    def fasnet_path(self) -> str | None:
        return self.liveness_fasnet_model.strip() or None

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
