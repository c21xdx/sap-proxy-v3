# sap-proxy-v3

OpenAI-compatible proxy for SAP AI Launchpad's private LLM API.

Single endpoint (`/v1/chat/completions`) — no Anthropic layer. Simpler to maintain.

## What changed from v2

- **Removed** `/v1/messages` (Anthropic Messages API endpoint)
- **Removed** `anthropic_api.py` (~840 lines of format translation)
- **Removed** cookie/auth research endpoints
- Core OpenAI proxy logic unchanged

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /debug/session` | Session cache info |
| `GET /debug/models` | Model access info |

## Features

- Model alias: `gpt-5.4` → `openai--gpt-5.4`, `claude-4.5-sonnet` → `anthropic--claude-4.5-sonnet`
- Reasoning effort via suffix: `gpt-5.4:high`, `claude-4.6-opus:high`
- Claude adaptive thinking passthrough (4.6+ / 4.7+)
- Tool calls (native + `<function_call>` tag fallback)
- Image support (base64 image_url in messages)
- History truncation with tool adjacency repair

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Configure .env with SAP credentials
cp .env.example .env

# Run
.venv/bin/uvicorn app.main:app --port 8013
```
