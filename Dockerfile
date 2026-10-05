# syntax=docker/dockerfile:1

# Face attendance API image.
#
# Two things this image is deliberately built around:
#
#   1. One model load per worker, at startup. DeepFace's Facenet weights are
#      baked into the image (see the DEEPFACE_HOME step) so the first request
#      does not pay a ~90 s download inside the function/request timeout.
#
#   2. opencv-python-headless only. The GUI build needs libGL, which is absent
#      on slim images and hard-crashes on import. cv2.imshow/waitKey have no
#      display in a container anyway -- the camera belongs to the kiosk, and
#      this service never opens a device.

# ---------------------------------------------------------------------------
# Builder: resolve and install dependencies into a self-contained venv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# deepface prints a warning glyph during import, which crashes on a non-UTF8
# console. Harmless on Linux with a UTF-8 locale, but cheap insurance.
ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

WORKDIR /app

# Bake the Facenet weights into the build so they are never fetched at runtime.
# DEEPFACE_HOME is the documented override for the ~/.deepface cache location.
ENV DEEPFACE_HOME=/opt/deepface
RUN mkdir -p "$DEEPFACE_HOME"

# Dependency layer first: code changes then rebuild in seconds instead of
# re-resolving TensorFlow. --frozen makes the build fail rather than silently
# re-lock, so uv.lock and pyproject.toml can never drift apart in an image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra face --extra liveness --no-install-project

# Now the application code. The editable install of the project itself is cheap.
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra face --extra liveness && \
    python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

# ---------------------------------------------------------------------------
# Runtime: no compilers, no build cache, no uv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEEPFACE_HOME=/opt/deepface \
    PATH="/opt/venv/bin:$PATH" \
    BUSINESS_TIMEZONE=Asia/Kolkata

# ENVIRONMENT is deliberately not baked in. It defaults to "dev", which leaves
# auth off, so a real deploy must set ENVIRONMENT=prod (or at minimum
# API_KEYS) on the platform -- auth_required in app/config.py keys off exactly
# those two. Baking "prod" here would instead make a bare `docker run` demand an
# API key, which is a confusing first experience for local use.

# libgl1/libglib2.0 are intentionally absent: they are only needed by the
# GUI build of OpenCV, which must never be installed here.
RUN groupadd --system --gid 1001 app && \
    useradd --system --uid 1001 --gid app --create-home --home-dir /home/app app

WORKDIR /app

# uv sync installs into UV_PROJECT_ENVIRONMENT (/opt/venv), not the default
# ./.venv, so this path is real. The venv hardcodes its own prefix, which is why
# it must be copied to the identical path in the runtime stage below.
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/deepface /opt/deepface
COPY --chown=app:app app ./app

# SQLite lives here. Mount a volume or point DATABASE_URL at an external
# Postgres -- see ARCHITECTURE.md section 9. Without one of those, attendance
# rows live only as long as this container.
RUN mkdir -p /app/data && chown app:app /app/data
VOLUME ["/app/data"]

USER app
EXPOSE 8000

# No curl in slim images, so probe with the interpreter that is already here.
# /health/ready is the meaningful one: it reports 503 until the face model is
# actually loaded, which /health/live does not check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"]

# One worker on purpose: each worker holds its own ~500 MB TensorFlow model and
# its own roster cache. Scale with more containers, not more uvicorn workers,
# until the model is moved out of process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
