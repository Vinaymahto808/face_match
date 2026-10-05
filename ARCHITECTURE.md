# Face Attendance System — Architecture

Production-ready target for the notebook prototype in this repo: OpenCV capture, DeepFace
embeddings, SQLite persistence, wrapped in a stateless REST API with anti-spoofing.

Visual diagrams: [`docs/flowcharts.txt`](docs/flowcharts.txt) (8 ASCII diagrams).

Current repo state — the API is complete and tested. See `README.md` for how to run it.

| Area | Files |
|---|---|
| Config, persistence, ORM | `app/config.py`, `app/db.py`, `app/models.py` |
| Vector maths | `app/services/embeddings.py` |
| Face engine + model lifecycle | `app/services/face.py` |
| Frame/face quality gate | `app/services/quality.py` |
| Roster cache + matching | `app/services/registry.py` |
| One-frame → one-decision pipeline | `app/services/pipeline.py` |
| Single-use identity grants | `app/services/verifications.py` |
| Attendance business rules | `app/services/attendance.py`, `app/services/punch.py` |
| Audit + alerting | `app/services/events.py` |
| Active liveness FSM | `app/services/liveness/active.py`, `service.py`, `geometry.py` |
| Passive anti-spoofing + optional mini-FASNet | `app/services/liveness/passive.py` |
| Transport: auth, errors, routers, app factory | `app/deps.py`, `app/errors.py`, `app/schemas.py`, `app/api/*`, `app/main.py` |
| Image decode / resize / guards | `app/utils/images.py` |
| Tests (218) | `tests/` |
| Deployment | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |

Two structural notes that the layering below does not show:

- `app/deps.py` and `app/errors.py` live at the package root, not under `api/`, because
  `app/services/punch.py` needs the error types and depending on `api/` from a service
  would invert the dependency direction.
- The WebSocket route (`app/api/stream.py`) is not a thin wrapper over the REST route. Both
  call `verify_frame` and `punch_attendance`, but the socket holds liveness state per
  connection and mints its own `verification_id` from the match it already computed — the
  client never sends an identity, a distance, or a `liveness_session_id` on either
  transport.

---

## 1. What changes from the notebook, and why

The notebook works as a demo. Five things break in production, and each one has a fix
that is already reflected in the design.

| Notebook behaviour | Failure in production | Design fix |
|---|---|---|
| `DeepFace.represent()` per frame | Reloads the TF graph every call — 1–2 s per frame | Model loaded once per worker at startup, cached |
| Writes `temp_live.jpg`, then deletes it | Disk churn, races between concurrent requests, and the bytes hit disk in cleartext | Decode to a numpy array, pass the array to the model. No temp files |
| `except Exception: pass` around the whole frame | A real bug — wrong model, corrupt frame, DB down — is indistinguishable from "nothing happened" | Typed errors, each mapped to a status code, all logged |
| `distance_threshold=0.40` hard-coded | One number for every site, light condition, and camera | Config-driven, with a gray zone so borderline cases go to review instead of being silently called a stranger |
| JSON embedding text, one embedding per person | 11 KB/row, N `json.loads` at startup, and no way to re-enrol without overwriting | `float32` BLOB, model name recorded, multi-frame averaging stored as samples |
| Single `PRIMARY KEY (id, date)` and a bare `except IntegrityError: pass` | Duplicate punches are invisible; you cannot tell "already punched" from "insert failed" | `UNIQUE(user_id, work_date)` + upsert that distinguishes the two |

Two correctness bugs worth naming explicitly, because both are silent:

- **Model swap.** Facenet and ArcFace embeddings live in different vector spaces. Comparing
  across them returns plausible numbers that never match. `users.model_name` is stored, and
  the matcher refuses to compare vectors from different models.
- **Naive timestamps.** SQLite has no timezone type, so SQLAlchemy hands back naive datetimes
  and `first_in_at` silently lands in the wrong business day. All timestamps go through
  `UtcDateTime`, and the business day key is computed in `BUSINESS_TIMEZONE`.

---

## 2. Layering

```
Client  ->  Transport (TLS, auth, rate limit)  ->  API (routers + schemas)
        ->  Services (recognition, liveness, attendance, events)
        ->  Persistence (SQLite WAL / Postgres)
        ->  ML runtime (DeepFace, optional mini-FASNet)
```

Dependencies point down only. Routers do no recognition work, services do no HTTP, and
nothing outside `services/` imports DeepFace. That is what makes the recognition engine
swappable (ArcFace, InsightFace, a vendor API) without touching a single route.

`config.py` is the only source of tunable values. No threshold, model name, or limit is
written as a literal in a service.

---

## 3. Component responsibilities

| Component | Owns | Does not own |
|---|---|---|
| `api/routers/` | HTTP shape, status codes, dependency injection | recognition logic, thresholds |
| `schemas/` | request/response contracts, validation | persistence |
| `services/face.py` | analyzer protocol, DeepFace/Stub impls, model load lifecycle | thresholds, SQL |
| `services/recognition.py` | detect + embed, threshold, gray zone, model guard | SQL |
| `services/quality.py` | frame/face quality gate (sharpness, brightness, contrast, flatness) | pass/fail policy |
| `services/liveness/passive.py` | passive spoof score, optional mini-FASNet fusion | active challenges |
| `services/liveness.py` | challenge FSM, token issue/consume, user binding | attendance writes |
| `services/verifications.py` | issue/spend single-use identity grants | deciding whether a frame is eligible |
| `services/punch.py` | every gate a punch must pass, in one ordered list | HTTP status codes |
| `services/attendance.py` | business-day key, idempotent punch, first/last seen | embedding maths |
| `services/events.py` | audit rows + alert webhook dispatch | deciding whether an alert is warranted |
| `services/embeddings.py` | float32 pack/unpack, L2 norm, cosine matmul | model inference |
| `services/roster.py` | cached active-user matrix, invalidated on enrolment change | matching policy |
| `utils/images.py` | decode, resize, size guards, JPEG encode | domain decisions |

`roster.py` is worth its own line. Loading every embedding per punch is a database round
trip plus N unpacks for a comparison that is one `matmul` against a cached `(N, 512)` unit
matrix. Cache it, invalidate on user create/update/delete.

---

## 4. API surface

Base path `/api/v1`. Auth is a static `X-API-Key` from `API_KEYS` (comma separated), and
it is only enforced when keys are configured *or* `ENVIRONMENT=prod` — so dev stays
frictionless but a prod deploy cannot accidentally ship unauthenticated. The dependency is
attached to **every** router, not per-route: a new route inherits the requirement instead
of having to remember it. `/health` and `/health/live` stay open for probes, and the
WebSocket authenticates during the handshake.

Rate-limit bucketing uses the socket peer address. `X-Forwarded-For` is **not** trusted
unless `TRUST_PROXY_HEADERS=true`, because it is client-writable and an unauthenticated
caller would otherwise mint a fresh bucket per request by rotating it. Set that flag only
when a proxy in front of the app is known to overwrite the header.

### Enrollment

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/users` | Register from one image (multipart) |
| `POST` | `/users/enroll/start` | Open a session, returns `enrollment_id` + `samples_needed` |
| `POST` | `/users/enroll/{id}/sample` | Add one frame's embedding (multipart image) |
| `POST` | `/users/enroll/{id}/finalize` | Average samples, L2-normalise, upsert `users` |

Registration is multi-frame on purpose. One snapshot gives a vector with wide intra-class
spread, which widens the impostor tail. Averaging 5 samples tightens the same person's
spread and buys back margin at the threshold — the single highest-value accuracy change in
this system. The single-image `POST /users` exists and works, but is materially worse.

### Liveness

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/liveness/capabilities` | What this deployment can actually enforce |
| `POST` | `/liveness/sessions` | Randomised challenge order + randomised reveal times |
| `POST` | `/liveness/sessions/{id}/frames` | Score a frame, advance the FSM |
| `GET` | `/liveness/sessions/{id}` | Poll state for the kiosk UI |

There is no `/consume` endpoint. The session id **is** the token: it is passed to
`/attendance/punch` as `liveness_session_id`, and `punch_attendance` consumes it inside
the same transaction as the attendance write. A separate exchange step would add a
round trip without adding a guarantee.

A token is also **bound to a person**. `POST /liveness/sessions` takes an optional
`user_id`, stored on the row; `consume_token` compares it against the identity the punch
resolved and refuses a mismatch *without* consuming, so a challenge passed honestly by
one employee cannot authorise a punch for a colleague. A session started with no
`user_id` is anonymous and adopts the first person who spends it — convenient for an
open kiosk, and still only once.

Anti-replay rests on two things, not on "ask for a blink": the challenge **order** is
randomised per session, and each challenge has a randomised `revealed_at` (0.6–2.2 s base
delay, 0.4–1.4 s between challenges). A pre-recorded "blink then turn left" montage cannot
satisfy the sequence, because it does not know when the challenges will be asked or in what
order. Frames are additionally checked for plausibility (a gap below 1/20 s is a replay),
sessions are TTL-bounded (`LIVENESS_SESSION_TTL_SECONDS`), and attempts are capped twice:
by the engine within one session, and by `create_session` across them
(`LIVENESS_MAX_ATTEMPTS` failures for an identity inside `LIVENESS_LOCKOUT_SECONDS` →
`429 liveness_locked_out`, for a further `LIVENESS_LOCKOUT_SECONDS` counted from the last
failure, so the countdown in the body is the real one). The second cap is the one that
matters — a per-session cap alone means a failed session simply ends and the next one
starts free, so a target could be worked on indefinitely. It needs a named identity; an
anonymous session has nothing to count failures against, and inventing one from the socket
address would be a guess.

The engine's `score` is persisted to `liveness_sessions.active_score` on pass.
`consume_token` returns it as the authoritative liveness score for the punch (the
client's claim is only a floor), so a NULL column would mean the `liveness_score` on the
attendance row was half the passive score — a number this service never measured.

### The identity proof

Liveness answers "was a live person present?". It does not answer "which one" — so on its
own it cannot authorise a write, and an earlier draft of this service let the punch
request supply that half. `user_id` and `match_distance` were both client assertions:
passing a challenge and then posting a colleague's id with an invented distance of `0.01`
was a working attendance forgery, and a caller with no camera at all could write rows.

`app/services/verifications.py` removes both fields. Recognition ends by issuing a
short-lived, single-use `verifications` row carrying **the distance this process
measured** and **the user it matched**; the punch spends that row and nothing else. There
is no longer any input on the punch that decides who is marked or how well they matched.

Why a table and not a signed token: the same reason the liveness token is a row. A punch
authorised by process-local state stops working the moment a second worker exists, and the
audit trail ("which verification authorised this row?") has to outlive the request. The id
is written into the `attendance_marked` event context.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/attendance/punch` | Spend one verification grant + one liveness token, and record. Never takes a `user_id` |
| `POST` | `/attendance/check-out` | Close out the current business day for a person |
| `GET` | `/attendance` | Filter by `work_date`, `user_id`, `device_id`, paginated |
| `GET` | `/attendance/export.csv` | CSV stream |

Recognition returns one of three **decisions**, not a boolean — `match.decision` is
`match` / `review` / `unknown`:

- `match` — within `MATCH_THRESHOLD`, eligible to punch
- `review` — distance in the gray zone (`MATCH_THRESHOLD` … `MATCH_THRESHOLD ×
  MATCH_GRAY_ZONE_FACTOR`), written to `events` for a human, **not** silently collapsed
  into "unknown"
- `unknown` — no roster match

Spoofing is a **separate** signal, not a fourth decision. It surfaces as
`liveness.passive_verdict` / `liveness.flags` on the verify response, and the *punch* is
what refuses — `403 spoof_suspected`.

The gray zone exists because the dangerous failure mode of a face system is not rejecting
a stranger, it is confidently writing a wrong identity. Borderline cases should reach a
human.

### Users & events

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/users` | Roster, paginated, never returns embedding bytes |
| `GET`/`DELETE` | `/users/{id}` | Read / deactivate a person |
| `GET` | `/alerts` | Audit + alert feed, filter by `kind`/`severity` |
| `GET` | `/alerts/summary` | Counts by kind and severity |

`DELETE /users/{id}` deactivates rather than deleting, so attendance history keeps its
foreign key.

### Streaming

`WS /stream/verify` — see `README.md` for the wire protocol. One analysis per frame, fanned
out to matching and liveness; punch authorisation is server-side.

### Operations

`GET /health` (full report), `GET /health/live` (process up), `GET /health/ready` (DB
reachable **and** face backend resident — 503 until then, which is what makes it usable as
a container readiness probe). `/metrics` is **not** implemented; see §11.

---

## 5. Anti-spoofing

Two independent signals, plus one optional model. They are **not** fused into a single
score: active and passive are separate gates, and passive is additionally a 0-1 score of
its own. Single signals are weak; requiring both is what makes printed photos and phone
replays unreliable.

1. **Passive, per frame** — texture and colour statistics, specular highlight distribution,
  flatness, edge density, blur ratio, colour histogram distance. Catches printed photos
  and flat screen replays. Cheap, runs on every frame, never trusted alone.
   Secondary defence.
2. **Active, challenge-response** — randomised order, randomised reveal times, single-use
   token, TTL, attempt cap, lockout, frame-rate plausibility, return-to-neutral between
   challenges. Defeats pre-recorded video. Primary defence.
3. **Deep, optional** — mini-FASNet ONNX (1.8 MB) texture-spectrum CNN, enabled via
  `LIVENESS_FASNET_MODEL`. Blank in `.env` = pure-numpy heuristics. When present it fuses
   into the *passive* score at 0.65 CNN / 0.35 heuristics; it does not replace the active
   gate.

Note the ordering in `punch_attendance`: the grant is resolved, its user checked and its
distance compared **before** the liveness token is consumed. A stale id or a
deactivated employee therefore cannot burn a session the user just spent a minute
earning. Passive flags are checked last, so a caller without a liveness token learns
nothing from the passive path.

Rules that must hold in code, not in convention:

- **No attendance row is written without a consumed, unexpired verification grant *and*,
  when `REQUIRE_LIVENESS=true`, a consumed, unexpired liveness token belonging to the
  same person.** Neither token can be spent twice. Backed by tests.
- A `punch` on the WebSocket mints its grant from the match the server already computed
  and goes through the same `punch_attendance`, so there is one code path, not two that
  happen to agree. A refused punch leaves the socket's state intact.
- A low score goes to the spare classifier, never to auto-punishment. A liveness model
  that is wrong about a real employee is a worse failure than a missed spoof.
- Spoof flags are recorded on the attendance row even on success, so historical punches
  can be re-scored later without keeping images.

One rule that is *not* enforced, contrary to an earlier draft of this document: the 403
from `SpoofSuspectedError` names the flags that fired
(`anti-spoofing flagged this frame: cnn_spoof_probability, moire_detected`). That is a
deliberate trade, not an oversight — reaching that branch requires either
`REQUIRE_LIVENESS=false` or an already-passed liveness token, so the caller has defeated
the active challenge before the flags are disclosed, and a named flag is the difference
between a kiosk bug that is fixable in an afternoon and one that is not.

If that trade is ever revisited, the change is one line in `app/services/punch.py` —
replace the `flags` interpolation with a fixed string. Keep `context={"flags": flags}`
out of the response too; it is currently returned alongside.

---

## 6. Data model

Seven tables, defined in `app/models.py`.

- **`users`** — one averaged, L2-normalised embedding per person. `model_name`,
  `sample_count`, `quality_score`, `is_active`.
- **`enrollment_sessions` / `enrollment_samples`** — accumulation, then average. CASCADE on
  delete.
- **`liveness_sessions`** — the FSM: `challenges` (JSON, randomised), `attempts`,
  `status`, `passive_score`, `active_score`, `user_id`, `expires_at`, `consumed_at`.
  `user_id` is the binding described in §5; NULL means anonymous.
- **`verifications`** — the identity grant: `user_id`, `distance`, `passive_liveness`,
  `spoof_flags`, `source`, `created_at`, `expires_at`, `consumed_at`. Indexed on
  `expires_at` because expired rows are swept on every issue. Single-use.
- **`attendance`** — `UNIQUE(user_id, work_date)`. `first_in_at` set once, `last_seen_at`
  bumped, `punch_count` incremented. Carries `match_distance`, `liveness_score`,
  `passive_liveness`, `spoof_flags` for audit.
- **`events`** — audit + alert sink: `severity`, `kind`, `context` JSON, `acknowledged`.

Embeddings are packed little-endian `float32` BLOBs. Facenet is 512 floats = 2048 bytes,
versus roughly 11 KB as JSON text, and loading the roster becomes one `np.frombuffer`
instead of N JSON parses. Explicit `<f4` endianness so the blobs outlive the machine that
wrote them.

---

## 7. Build order

Built in this order, bottom-up by dependency. Kept here because the layering is the point:
each layer was finished and tested before the one above it existed.

| # | Layer | Landed as |
|---|---|---|
| 1 | Config, DB, ORM | `app/config.py`, `app/db.py`, `app/models.py` |
| 2 | Vector maths | `app/services/embeddings.py` |
| 3 | Face engine, model lifecycle | `app/services/face.py` |
| 4 | Quality gate | `app/services/quality.py` |
| 5 | Passive anti-spoofing | `app/services/liveness/passive.py` |
| 6 | Image decode / resize | `app/utils/images.py` |
| 7 | Roster cache + matching | `app/services/registry.py` |
| 8 | Active liveness FSM | `app/services/liveness/{active,geometry,service}.py` |
| 9 | Identity grants | `app/services/verifications.py` |
| 10 | Attendance + punch rules | `app/services/{attendance,punch}.py` |
| 11 | Audit + alerting | `app/services/events.py` |
| 12 | Contracts, transport, routers, app factory | `app/schemas.py`, `app/deps.py`, `app/errors.py`, `app/api/*`, `app/main.py` |
| 13 | Tests (218) | `tests/` |
| 14 | Deployment | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |

Two things were deliberately **not** built:

- **A background worker / scheduler.** `dispatch_alert` fires inline from the request
  thread with a 5 s timeout, so a slow webhook costs latency rather than correctness. Once
  alerts need retry or fan-out, move it to a queue — do not grow the inline call.
- **`/metrics`.** The `events` table already carries the domain counters and
  `GET /alerts/summary` aggregates them. Prometheus metrics are a scrape-format problem
  layered on top, and adding a second source of truth for the same counts is how dashboards
  start disagreeing with each other. See §11.

### Runtime constraints

- **`opencv-python-headless` in the server image, pinned below 5.0.** `cv2.imshow`/`waitKey`
  have no display in a container and hard-crash. The camera belongs to the kiosk; the server
  takes stills or short bursts. No `VideoCapture` server-side. OpenCV 5 removed
  `cv2.CascadeClassifier`, which is the eye detector behind the blink challenge — losing it
  does not crash, it makes liveness unavailable and `REQUIRE_LIVENESS` then refuses every
  punch, which is the correct fail-closed outcome but an annoying one to debug.
- **Python 3.11 or 3.12.** TensorFlow via DeepFace has no 3.13/3.14 wheels. `pyproject.toml`
  declares `>=3.11,<3.14` and the image pins `python:3.11-slim`.
- **One model load per worker**, at startup, behind a lazy singleton.
- **Migrations:** `init_db()` creates missing tables only — it will add a table but will not
  alter one. Use Alembic (already in `requirements.txt`) before the first schema change in a
  live deployment.
- **`tzdata` is a hard dependency, not an extra.** `zoneinfo` reads `/usr/share/zoneinfo` on
  Linux and has no database at all on Windows, and `BUSINESS_TIMEZONE` is validated at
  startup — without the package the app fails to *import*, not merely to default oddly.

---

## 8. Configuration

Every knob lives in `.env` (see `.env.example`) and is validated at startup by
`pydantic-settings`. A bad value fails the boot, not the first punch.

| Group | Keys |
|---|---|
| App | `APP_NAME`, `ENVIRONMENT`, `DEBUG`, `API_PREFIX`, `CORS_ORIGINS` |
| Database | `DATABASE_URL`, `DB_ECHO`, `BUSINESS_TIMEZONE` |
| Ingest | `MAX_UPLOAD_BYTES`, `MAX_FRAME_EDGE` |
| Recognition | `EMBEDDING_BACKEND`, `FACE_MODEL_NAME`, `FACE_DETECTOR`, `MATCH_THRESHOLD`, `MATCH_GRAY_ZONE_FACTOR` |
| Quality gate | `MIN_FACE_EDGE_PX`, `MIN_FRAME_EDGE_PX`, `MIN_SHARPNESS`, `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`, `MIN_CONTRAST`, `MAX_FLAT_REGION_FRACTION` |
| Anti-spoofing | `REQUIRE_LIVENESS`, `PASSIVE_LIVENESS_MIN`, `ACTIVE_LIVENESS_MIN_CHALLENGES`, `LIVENESS_CHALLENGES`, `LIVENESS_SESSION_TTL_SECONDS`, `LIVENESS_MAX_ATTEMPTS`, `LIVENESS_LOCKOUT_SECONDS`, `LIVENESS_FASNET_MODEL` |
| Security | `API_KEYS`, `RATE_LIMIT_PER_MINUTE` |
| Alerting | `ALERT_WEBHOOK_URL` |

`EMBEDDING_BACKEND=stub` is a deterministic no-op for local dev and tests. It is
hard-blocked from writing attendance unless `ALLOW_STUB_BACKEND=true`, so a test
configuration can never leak into a real punch.

---

## 9. Scale path

SQLite in WAL mode with a busy timeout handles one gate site comfortably and keeps the
deployment to a single file. Beyond that, the switch is one env var, not a rewrite:

- `DATABASE_URL=postgresql://…` — SQLAlchemy already abstracts it. The `UtcDateTime` type
  decorator exists partly so Postgres is a drop-in.
- Multiple API workers, each with its own model load and its own roster cache.
- Multi-site: add `site_id` to `users` and `attendance`, partition the roster cache.
- Embeddings stay in the main DB. At 10k+ users, move them to an ANN index
  (FAISS/pgvector) and keep only the row pointers in SQL.

---

## 10. Security & data handling

- API keys via header, constant-time comparison, required in prod.
- Rate limit per key (`RATE_LIMIT_PER_MINUTE`).
- Uploads capped in bytes and downscaled to `MAX_FRAME_EDGE` before any decode.
- Embeddings never leave the API. They are not returned by any endpoint.
- Frames are processed in memory and discarded. Nothing is written to disk by default.
- Anti-spoof failures are the security-sensitive event, so they are logged, counted, and
  pushed to the webhook; successes are not.
- Retention: attendance records are kept per policy; raw images are not stored at all.
  Embedding vectors are biometric data — treat a database dump as PII.

---

## 11. Observability

What exists today:

- **Structured logs** with `request_id`, method, path, status and latency, emitted by the
  middleware in `app/main.py`. `X-Request-ID` and `X-Process-Time-Ms` come back on every
  response, so a kiosk's error report can be tied to exact server logs.
- **Domain audit in `events`**, which survives restarts, unlike logs. Filterable by kind,
  severity and user, with `GET /alerts/summary` for counts.
- **Per-frame metrics inline** in the quality report — sharpness, brightness, contrast,
  flat-region fraction — returned to the client with every verify and WebSocket frame. This
  is deliberate: the operator diagnosing "the gate keeps rejecting this camera" needs the
  numbers, and putting them behind a metrics scrape means they will not have them.
- **Liveness availability is public**, via `GET /liveness/capabilities`, including *why* it
  is unavailable (`eye_detector_unavailable: …`). An attendance system that silently
  degrades to "no liveness" is worse than one that is down.

Not implemented: `/metrics` in Prometheus format. When it is added, source it from the
counters already in `events` rather than from new in-process gauges, so the two cannot
disagree.

Three questions the dashboards must answer: Is it up? Is it accurate (review/unknown
drift per site)? Is it under attack (spoof rate, lockouts, devices)?

The accuracy one matters most in practice. A rise in `review` rate usually means a camera
moved, lighting changed, or the roster went stale — and it shows up days before anyone
complains that the system "isn't working".
