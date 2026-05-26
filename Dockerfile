FROM python:3.12-slim

WORKDIR /app

# Install system deps for curl_cffi (needs libcurl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcurl4-openssl-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY app/ app/

ENV HOST=0.0.0.0
ENV PORT=8011

EXPOSE ${PORT}

CMD uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8011}