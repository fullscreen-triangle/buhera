# syntax=docker/dockerfile:1
#
# Multi-stage build on the official uv image (as the brief requests).
#   stage 1 (builder) — resolve+install deps into a venv, build the KG artefacts
#   stage 2 (runtime) — copy the venv + app, run uvicorn as a non-root user
#
# The knowledge graph is materialised at build time, so the container starts
# instantly and needs no network at runtime.

# --- builder ---------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# uv configuration: compile bytecode, copy (not link) into a self-contained venv.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first, using only the lock + manifest, for layer caching.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev 2>/dev/null || \
    uv sync --no-install-project --no-dev

# Now install the project itself and build the RDF artefacts.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev 2>/dev/null || uv sync --no-dev
RUN uv run build-kg

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Bring over the resolved environment and the application (incl. built graph).
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').read()==b'ok' else 1)"

CMD ["uvicorn", "transaminase_kg.api:app", "--host", "0.0.0.0", "--port", "8000"]
