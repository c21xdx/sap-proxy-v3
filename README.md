# SAP Proxy V3

OpenAI-compatible proxy that bridges [SAP AI Launchpad](https://help.sap.com/docs/ai-launchpad)'s private LLM API to the standard OpenAI `/v1/chat/completions` format.

Point any OpenAI SDK, agent framework, or curl command at this proxy and use SAP-hosted models — GPT-5.4, Claude 4.7 Opus, Gemini 3.1 Flash, and more — as if they were native OpenAI models.

## Features

- **Drop-in OpenAI compatible** — works with OpenAI SDKs, LangChain, AutoGen, Shelley, etc.
- **Model aliases** — `gpt-5.4` → `openai--gpt-5.4`, `claude-4.5-sonnet` → `anthropic--claude-4.5-sonnet`
- **Reasoning effort via suffix** — `gpt-5.4:high`, `claude-4.6-opus:high`
- **Claude thinking passthrough** — adaptive thinking for Claude 4.6+, budget_tokens for 4.5
- **Tool calls** — native `tool_calls` + `<function_call>` tag fallback
- **Image support** — base64 `image_url` in messages
- **Smart history truncation** — with tool adjacency repair and missing tool result synthesis
- **Anti-empty-response** — auto user hint after tool results to prevent Claude empty replies

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /debug/session` | SAP session cache info |
| `GET /debug/models` | Model access details |

## Supported Models

| Alias | SAP Canonical ID | Provider |
|---|---|---|
| `gpt-5.4`, `gpt5.4`, `gpt-5.4-turbo` | `openai--gpt-5.4` | OpenAI |
| `gpt-5.4-nano` | `openai--gpt-5.4-nano` | OpenAI |
| `gpt-5.3-codex` | `openai--gpt-5.3-codex` | OpenAI |
| `gpt-5.2` | `openai--gpt-5.2` | OpenAI |
| `o4-mini` | `openai--o4-mini` | OpenAI |
| `o3` | `openai--o3` | OpenAI |
| `claude-opus-4-7`, `claude-4.7-opus` | `anthropic--claude-4.7-opus` | Anthropic |
| `claude-opus-4-6`, `claude-4.6-opus` | `anthropic--claude-4.6-opus` | Anthropic |
| `claude-sonnet-4-6`, `claude-4.6-sonnet` | `anthropic--claude-4.6-sonnet` | Anthropic |
| `claude-sonnet-4-5`, `claude-4.5-sonnet` | `anthropic--claude-4.5-sonnet` | Anthropic |
| `claude-haiku-4-5`, `claude-4.5-haiku` | `anthropic--claude-4.5-haiku` | Anthropic |
| `gemini-3.1-flash-lite` | `google--gemini-3.1-flash-lite` | Google |

Use the alias or the canonical ID — both work.

### Reasoning Effort

Append `:effort` to any model name:

```
gpt-5.4:high
claude-4.6-opus:medium
o3:xhigh
```

Valid values: `low`, `medium`, `high`, `xhigh`.

For OpenAI models → `reasoning_effort` parameter.
For Claude 4.6+ → adaptive thinking with `output_config.effort`.
For Claude 4.5 → `thinking` with `budget_tokens` (low=2048, medium=8192, high=16000, xhigh=32000).

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. Configure

Create `.env` with your SAP AI Launchpad credentials:

```env
SAP_USER=<your-sap-user>
SAP_PASS=<your-sap-password>
API_KEY=<your-api-key-for-proxy-auth>
SAP_BASE_URL=YOUR_SAP_BASE_URL
SAP_DEPLOYMENT_ID=<your-deployment-id>
SAP_RESOURCE_GROUP_ID=<your-resource-group>
SAP_FORCE_NON_STREAM=true
```

### 3. Run

```bash
.venv/bin/uvicorn app.main:app --port 8013
```
Or install as a systemd service:

```bash
sudo cp srv.service /etc/systemd/system/sap-proxy-v3.service
sudo systemctl daemon-reload && sudo systemctl enable --now sap-proxy-v3
```

### 4. Use

```bash
curl http://localhost:8013/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or with any OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8013/v1", api_key="YOUR_API_KEY")
resp = client.chat.completions.create(
    model="claude-4.6-sonnet:high",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(resp.choices[0].message.content)
```

## Architecture

```
Client (OpenAI format)
  → /v1/chat/completions
  → resolve_model_cached()          # alias → SAP canonical ID
  → _to_completion_request()        # build CompletionRequest
     ├── _build_template_messages()  # current turn → SAP template
     ├── _build_messages_history()   # prior turns → SAP messages_history
     ├── _build_model_params()       # effort/thinking params
     └── _build_native_tools()       # tools format conversion
  → _build_completion_payload()     # assemble SAP completionV2 payload
  → SAP login + completion request   # via curl_cffi (TLS fingerprint)
  → OpenAI format response          # streaming or non-streaming
```

## Project Structure

```
app/
  main.py          # FastAPI routes (~280 lines)
  openai_api.py    # Core logic: model resolution, message building,
                   # turn boundaries, tool repair, SSE streaming (~1640 lines)
  curl_login.py    # SAP auth, session cache, payload assembly, curl_cffi (~1100 lines)
  config.py        # Pydantic Settings from .env (~120 lines)
tests/             # pytest suite
srv.service        # systemd unit file
Dockerfile         # container build
docker-compose.yml # container orchestration
```

## License

See [LICENSE](LICENSE).
