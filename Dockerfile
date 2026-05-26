# Stage 1: Build — install deps that need gcc/libcurl headers
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends libcurl4-openssl-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime — only runtime libs, no build tools
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libcurl4 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app/ app/

ENV HOST=0.0.0.0
ENV PORT=8011

EXPOSE ${PORT}

CMD uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8011}