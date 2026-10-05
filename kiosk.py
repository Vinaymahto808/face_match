"""Webcam kiosk: capture a frame, recognise the face, punch in or out.

The small path through this project. No HTTP, no WebSocket, no liveness
challenge -- the camera is on this machine, the frame goes straight to the
same Facenet model, and the punch lands in the same SQLite database the API
serves. Anything recorded here shows up in ``GET /api/v1/attendance`` and in
the attendance export.

    python kiosk.py enroll EMP101 "Ada Lovelace"   # 5 webcam samples, averaged
    python kiosk.py punch                          # one person: in or out
    python kiosk.py watch                          # keep punching whoever appears
    python kiosk.py today                          # today's sheet

Punch semantics match the API's one-row-per-day model: the first punch of the
business day is the *in*, every later punch moves the *out*. Force either side
with ``--mode in`` / ``--mode out``.

What it deliberately does NOT do: anti-spoofing challenges. A printed photo
still has to beat the passive texture check, and a flagged frame is refused,
but there is no blink / head-turn test. Use the API + browser UI for that.

Reuses ``app.services`` rather than re-implementing them, so the matching
threshold, quality gate and roster cache behave exactly as in the service.
The same in/out rule is exposed over HTTP as ``POST /api/v1/kiosk/punch`` for
the browser test page at ``/kiosk`` in ``face-attendance-ui``.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Run from anywhere: the repo root must be importable as the `app` package.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera:
    """Thin context manager over ``cv2.VideoCapture``.

    OpenCV here is the *headless* build (no ``imshow``), which is why the kiosk
    talks through the terminal instead of a preview window. Capture itself is
    unaffected.
    """

    def __init__(self, index: int, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Any = None

    def __enter__(self) -> Camera:
        import cv2

        # DirectShow opens in well under a second on Windows; the default
        # MSMF backend can take 10+ s to negotiate a format.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
        self._cap = cv2.VideoCapture(self.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {self.index} could not be opened")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Let auto-exposure settle; the first frames are usually dark.
        for _ in range(5):
            self._cap.read()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._cap is not None:
            self._cap.release()

    def read(self) -> Any:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("camera returned no frame")
        return frame


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
def _force_utf8_stdio() -> None:
    # Same reason as app.py: DeepFace prints a glyph on import that a cp1252
    # console turns into a failed import.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def _boot() -> None:
    """Validate settings, create tables, load the face model. Exits on failure."""
    from app.config import settings
    from app.db import init_db
    from app.services.face import get_analyzer

    if settings.embedding_backend == "stub" and not settings.allow_stub_backend:
        sys.exit(
            "EMBEDDING_BACKEND=stub cannot write real attendance. "
            "Set EMBEDDING_BACKEND=deepface in .env."
        )
    init_db()
    print("loading face model ...", end="", flush=True)
    analyzer = get_analyzer()
    analyzer.warmup()
    if not analyzer.ready:
        sys.exit(f"\nface backend failed to load: {getattr(analyzer, 'last_error', 'unknown')}")
    print(f" ready ({getattr(analyzer, 'load_seconds', '?')}s, {settings.face_model_name})")


def _device_id(explicit: str | None) -> str:
    return (explicit or f"kiosk-{platform.node()}")[:64]


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def recognise(db: Any, cam: Camera, *, timeout_s: float, min_hits: int = 2) -> Any | None:
    """Read frames until one person is matched ``min_hits`` times in a row.

    Returns the :class:`~app.services.pipeline.VerificationResult` of the
    confirming frame, or ``None`` on timeout. Two consecutive hits on the same
    id cost under half a second and remove most single-frame flukes.
    """
    from app.services.pipeline import verify_frame

    deadline = time.monotonic() + timeout_s
    hits: Counter[str] = Counter()
    last_status = ""

    while time.monotonic() < deadline:
        frame = cam.read()
        result = verify_frame(db, frame, record_events=False)

        if result.matched:
            uid = result.match.user_id
            hits[uid] += 1
            if hits[uid] >= min_hits:
                return result
            status = f"seeing {result.match.name} ({result.match.distance:.3f}) ..."
        else:
            hits.clear()
            if result.match is not None and result.match.decision == "review":
                status = (
                    f"borderline: {result.match.name} at {result.match.distance:.3f} "
                    "-- move closer / face the camera"
                )
            else:
                status = ", ".join(result.block_reasons) or "no match"

        if status != last_status:
            print(f"  {status}")
            last_status = status
    return None


# ---------------------------------------------------------------------------
# Punch
# ---------------------------------------------------------------------------
def apply_punch(db: Any, result: Any, *, mode: str, device_id: str) -> str:
    """Write the in/out for the person in ``result``. Returns a one-line message.

    The rule itself lives in :mod:`app.services.kiosk` so ``POST /kiosk/punch``
    and this CLI cannot drift apart.
    """
    from app.services.kiosk import apply_punch as _apply

    outcome = _apply(db, result, mode=mode, device_id=device_id)  # type: ignore[arg-type]
    label = {"in": "IN  ", "out": "OUT ", "refused": "REFUSED:"}[outcome.action]
    return f"{label} {outcome.message}"


def _save_frame(path: str, frame: Any, result: Any) -> None:
    import cv2

    from app.services.pipeline import annotate

    cv2.imwrite(path, annotate(frame.copy(), result))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_enroll(args: argparse.Namespace) -> int:
    from app.api.users import _embed_from_image  # same gate the API applies
    from app.db import session_scope
    from app.errors import AppError
    from app.services import registry as reg

    _boot()
    vectors: list[Any] = []
    qualities: list[float] = []
    print(f"enrolling {args.user_id!r} as {args.name!r}: need {args.samples} good frames")
    print("look at the camera; move your head slightly between samples")

    with Camera(args.camera) as cam:
        deadline = time.monotonic() + args.timeout
        last_reason = ""
        while len(vectors) < args.samples and time.monotonic() < deadline:
            frame = cam.read()
            try:
                vector, info = _embed_from_image(frame, allow_low_quality=False)
            except AppError as exc:
                if exc.detail != last_reason:
                    print(f"  skipped: {exc.detail}")
                    last_reason = exc.detail
                continue
            vectors.append(vector)
            qualities.append(float(info["face_quality"]["score"]))
            print(f"  sample {len(vectors)}/{args.samples} (quality {qualities[-1]:.2f})")
            last_reason = ""
            time.sleep(args.gap)  # distinct frames, not five copies of one

    if len(vectors) < 2:
        print(f"only {len(vectors)} usable frame(s); need at least 2. Not enrolled.")
        return 2

    with session_scope() as db:
        try:
            user = reg.create_user(
                db,
                user_id=args.user_id,
                name=args.name,
                embedding=reg.consensus_embedding(vectors),
                sample_count=len(vectors),
                quality_score=sum(qualities) / len(qualities),
                employee_code=args.employee_code,
            )
        except ValueError as exc:
            print(f"not enrolled: {exc}")
            return 2
    print(f"enrolled {user.name} ({user.id}) from {user.sample_count} samples")
    return 0


def cmd_punch(args: argparse.Namespace) -> int:
    from app.db import session_scope

    _boot()
    device = _device_id(args.device_id)
    with Camera(args.camera) as cam, session_scope() as db:
        print(f"look at the camera ({args.timeout:.0f}s) ...")
        result = recognise(db, cam, timeout_s=args.timeout)
        if result is None:
            print("no recognised face. Nothing recorded.")
            return 2
        if args.dry_run:
            m = result.match
            print(f"DRY  would punch {m.name} ({m.user_id}) at distance {m.distance:.3f}; nothing written")
            return 0
        message = apply_punch(db, result, mode=args.mode, device_id=device)
        print(message)
        if args.save_frame:
            _save_frame(args.save_frame, cam.read(), result)
    return 0 if not message.startswith("REFUSED") else 3


def cmd_watch(args: argparse.Namespace) -> int:
    from app.db import session_scope

    _boot()
    device = _device_id(args.device_id)
    cooldown_until: dict[str, float] = {}
    print(
        f"watching camera {args.camera}; mode={args.mode}, "
        f"{args.cooldown:.0f}s per-person cooldown. Ctrl+C to stop."
    )
    try:
        with Camera(args.camera) as cam:
            while True:
                with session_scope() as db:
                    result = recognise(db, cam, timeout_s=args.timeout)
                    if result is None:
                        continue
                    uid = result.match.user_id
                    if cooldown_until.get(uid, 0.0) > time.monotonic():
                        continue
                    cooldown_until[uid] = time.monotonic() + args.cooldown
                    print(f"[{time.strftime('%H:%M:%S')}] {apply_punch(db, result, mode=args.mode, device_id=device)}")
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_today(args: argparse.Namespace) -> int:
    from app.db import init_db, session_scope
    from app.services import attendance as att

    init_db()
    day = args.day or att.business_day()
    with session_scope() as db:
        rows, total = att.list_attendance(db, day=day, limit=500)
        print(f"{day}: {total} present")
        if rows:
            print(f"{'id':<12} {'name':<24} {'in':>8} {'out':>8} {'punches':>7} {'dist':>6}")
        for r in rows:
            out = att.local_now(r.check_out_at).strftime("%H:%M:%S") if r.check_out_at else "-"
            print(
                f"{r.user_id:<12} {r.user_name[:24]:<24} "
                f"{att.local_now(r.first_in_at).strftime('%H:%M:%S'):>8} {out:>8} "
                f"{r.punch_count:>7} {r.match_distance:>6.3f}"
            )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Webcam attendance kiosk: capture, recognise, punch in/out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def camera_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
        p.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for a face")

    p = sub.add_parser("enroll", help="register a person from webcam samples")
    p.add_argument("user_id")
    p.add_argument("name")
    p.add_argument("--samples", type=int, default=5, help="good frames to average")
    p.add_argument("--gap", type=float, default=0.7, help="seconds between samples")
    p.add_argument("--employee-code", default=None)
    camera_opts(p)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("punch", help="recognise one person and punch in or out")
    p.add_argument("--mode", choices=["auto", "in", "out"], default="auto",
                   help="auto = first punch today is in, later ones are out")
    p.add_argument("--device-id", default=None, help="recorded on the attendance row")
    p.add_argument("--save-frame", default=None, metavar="PATH",
                   help="write the annotated frame as JPEG (off by default; frames stay in memory)")
    p.add_argument("--dry-run", action="store_true", help="recognise only; write nothing")
    camera_opts(p)
    p.set_defaults(func=cmd_punch)

    p = sub.add_parser("watch", help="run continuously, punching whoever is recognised")
    p.add_argument("--mode", choices=["auto", "in", "out"], default="auto")
    p.add_argument("--cooldown", type=float, default=60.0,
                   help="seconds before the same person can punch again")
    p.add_argument("--device-id", default=None)
    camera_opts(p)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("today", help="print the attendance sheet")
    p.add_argument("--day", default=None, help="YYYY-MM-DD (default: today)")
    p.set_defaults(func=cmd_today)
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:  # camera problems
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
