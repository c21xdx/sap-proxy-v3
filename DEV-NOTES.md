# sap-proxy-v3 开发笔记

## 项目概述

OpenAI-only 代理，将 SAP AI Launchpad 的私有 LLM API 翻译为 OpenAI `/v1/chat/completions` 格式。
从 v2 衍生，去掉了 Anthropic Messages API 层（~840 行），只维护 OpenAI 格式端点。

**仓库**: https://github.com/c21xdx/sap-proxy-v3
**服务**: `sap-proxy-v3.service`，端口 8011（与 v2 共存可改为 8013）
**v2 仓库**: https://github.com/c21xdx/sap-proxy-v2 （保留，端口 8011，含 Anthropic 端点）

---

## 环境与部署

### 目录
- 项目: `/home/exedev/107sapv3`
- venv: `.venv/bin/python` / `.venv/bin/uvicorn`
- .env: SAP 凭证 + `SAP_FORCE_NON_STREAM=true`

### .env 关键配置
```
SAP_USER=YOUR_SAP_USER
SAP_PASS=YOUR_SAP_PASSWORD
API_KEY=YOUR_API_KEY
SAP_BASE_URL=YOUR_SAP_BASE_URL
SAP_DEPLOYMENT_ID=YOUR_DEPLOYMENT_ID
SAP_RESOURCE_GROUP_ID=doc-grounding
SAP_FORCE_NON_STREAM=true
```

### 部署命令
```bash
sudo cp srv.service /etc/systemd/system/sap-proxy-v3.service
sudo systemctl daemon-reload && sudo systemctl enable --now sap-proxy-v3
```
管理: `systemctl status/restart sap-proxy-v3`，日志: `journalctl -u sap-proxy-v3 -f`

### 测试
```bash
.venv/bin/python -m pytest tests/ -q \
  --deselect tests/test_openai_api.py::test_to_completion_request_uses_request_defaults \
  --deselect tests/test_openai_api.py::test_text_mode_keeps_stream
```
2 个 deselected 是 `SAP_FORCE_NON_STREAM=true` 导致的已知问题。

---

## 架构

### 端点
| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions（唯一 LLM 端点）|
| `GET /v1/models` | 列出可用模型 |
| `GET /health` | 健康检查 |
| `GET /debug/session` | SAP 会话缓存 |
| `GET /debug/models` | 模型访问详情 |

### 请求流程
```
Client (OpenAI format)
  → /v1/chat/completions
  → resolve_model_cached()  # 模型名 → SAP canonical ID
  → _to_completion_request()  # 构建 CompletionRequest
     ├── _build_template_messages()  # 当前 turn → SAP template
     ├── _build_messages_history()   # 历史对话 → SAP messages_history
     ├── _build_model_params()       # 模型参数（含 effort/thinking）
     └── _build_native_tools()       # tools 格式转换
  → _build_completion_payload()  # 组装 SAP completionV2 payload
  → execute_completion_with_password_curl_cffi()  # SAP 登录+请求
  → 返回 OpenAI 格式响应（streaming 或 non-streaming）
```

### SAP payload 结构
```
{
  "config": {
    "modules": {
      "prompt_templating": {
        "prompt": {
          "template": [...],     # 当前 turn 消息
          "tools": [...]          # 原生 tools
        },
        "model": {                # 模型规格 + 参数
          "name": "openai--gpt-5.4",
          "params": {...},
          "version": "latest"
        }
      }
    },
    "stream": {"enabled": false}  # SAP_FORCE_NON_STREAM=true 时始终 false
  },
  "messages_history": [...],      # 历史 turn
  "placeholder_values": {}
}
```

SAP 上游端点:
- `/aic/llm/api/v1/metadataV2` — 模型列表
- `/aic/llm/api/v1/completionV2` — 补全请求

---

## 关键文件

### `app/main.py` (~280 行)
FastAPI 路由定义。仅包含:
- `/v1/chat/completions` — 核心端点
- `/v1/models` — 模型列表
- `/health` — 健康检查
- `/debug/session`, `/debug/models` — 调试

**去掉了 v2 中的**:
- `/v1/messages` (Anthropic 端点)
- `/research/auth`, `/research/import-cookies`, `/research/validate-cookies`
- `auth_research.py`, `cookie_import.py` 模块引用

### `app/openai_api.py` (~1640 行)
核心逻辑，也是 v2 的核心文件。包含:
- 模型解析: `resolve_model_cached()`, `MODEL_ALIASES`, `_parse_model_effort()`
- Template/History 构建: `_build_template_messages()`, `_build_messages_history()`
- Turn 边界计算: `turn_start` 回溯算法
- Tool 结果补全: `_ensure_tool_results_complete()`
- Tool 邻接修复: `_repair_tool_adjacency()`
- Effort/Thinking 参数: `_build_model_params()`, `_filter_model_params()`
- Claude thinking 支持: `_claude_supports_adaptive_thinking()` 等
- SSE 流式处理: `iter_openai_sse()`
- Tool call 解析: `parse_sap_sse_tool_calls()`, `parse_tool_calls()`

### `app/curl_login.py` (~1100 行)
SAP 登录和请求执行:
- `execute_completion_with_password_curl_cffi()` — 主入口
- `_build_completion_payload()` — payload 组装
- `_build_template_entry()` — 单条 template 消息格式转换
- `_build_model_params()` — 模型参数（含 Claude thinking/effort 映射）
- `_filter_model_params()` — 按 SAP 支持的参数过滤
- `_log_payload_structure()` — 调试日志
- Session 缓存、CSRF token、auto-discover deployment

### `app/config.py` (~120 行)
Pydantic Settings，从 .env 加载:
- SAP 凭证: `SAP_USER`, `SAP_PASS`, `API_KEY`
- SAP 端点: `SAP_BASE_URL`, `SAP_DEPLOYMENT_ID`, `SAP_RESOURCE_GROUP_ID`
- `SAP_FORCE_NON_STREAM`: 强制 SAP 侧非流式（客户端仍得 SSE 格式）
- `SAP_STREAM_ENABLED`, `max_history_turns`, `max_history_tokens`
- 多用户支持: `SAP_CREDENTIALS` (JSON)

---

## 已修复的关键 Bug

### §52 Turn boundary — assistant(tool_calls) 孤立在 history

**症状**: agent 对话在第一轮 tool_use → tool_result 后 502。SAP 400:
"assistant message with 'tool_calls' must be followed by tool messages"

**根因**: `_build_template_messages` 和 `_build_messages_history` 用 `last_user_idx`
（最后一个 user/tool 消息）作为分界。当最后一轮含 `assistant(tc) → tool_result` 时:
- 模板有 tool_result 但没有前面的 assistant(tc) → SAP 看到孤立 tool_result
- 历史有 assistant(tc) 但没有后面的 tool_result → SAP 看到孤立 tool_calls
- 双重孤立，SAP 两头报错

**修复**: 引入 `turn_start` 回溯，从 `last_user_idx` 向前找到当前 turn 的真正起点。
模板从 `turn_start` 开始，历史只含 `turn_start` 之前的消息。

### §52b 并行 tool_results 回溯 + 缺失 tool_result 补全

**症状**: GPT-5.4 发出 3 个并行 tool_calls，Shelley 只返回 2 个 tool_result，
第三次请求 502。SAP: "tool_call_ids did not have response messages"

**根因**: 两个问题叠加:
1. turn_start 回溯不处理连续 tool 消息（并行结果）。旧算法只匹配
   `assistant(tc) → user/tool`，遇到 `tool → tool` 直接 break。
2. 客户端部分 tool_result：3 个并行 call 只完成 2 个，第 3 个无 result。

**修复**:
1. 回溯算法重写——持续回溯 `tool`/`assistant(tc)`/`user`，直到遇到
   `system` 或 `assistant(无 tc)` 才停止:
   ```
   turn_start = last_user_idx
   while turn_start > 0:
       prev = messages[turn_start - 1]
       if prev.role == "tool":       turn_start -= 1; continue
       elif prev.role == "assistant" and prev.tool_calls: turn_start -= 1; continue
       elif prev.role == "user":     turn_start -= 1; continue
       else: break  # boundary
   ```

2. `_ensure_tool_results_complete(template)`: 扫描所有 assistant(tc) 声明的
   tool_call_id，对比已有 tool 消息，为缺失的自动补全占位结果:
   `"[tool result pending — no result available]"`
   在 template 和 history 的构建末尾都调用。

---

## 模型系统

### 可用模型（§50 范围内）
| SAP canonical ID | 别名 | 说明 |
|---|---|---|
| `openai--gpt-5.4` | `gpt5.4`, `gpt-5.4-turbo` | 旗舰 |
| `openai--gpt-5.4-nano` | | 轻量 |
| `openai--gpt-5.3-codex` | | 代码 |
| `openai--gpt-5.2` | | |
| `openai--o4-mini` | | 推理 |
| `openai--o3` | | 推理 |
| `anthropic--claude-4.7-opus` | `claude-opus-4-7`, `claude-4.7-opus` | 4.7 新 API |
| `anthropic--claude-4.6-opus` | `claude-opus-4-6`, `claude-4.6-opus` | |
| `anthropic--claude-4.6-sonnet` | `claude-sonnet-4-6`, `claude-4.6-sonnet` | |
| `anthropic--claude-4.5-sonnet` | `claude-sonnet-4-5`, `claude-4.5-sonnet` | |
| `anthropic--claude-4.5-haiku` | `claude-haiku-4-5`, `claude-4.5-haiku` | |
| `google--gemini-3.1-flash-lite` | | |

### Effort 后缀 (`model:effort`)
- 格式: `gpt-5.4:high`, `claude-4.6-opus:high`
- 有效值: `low`, `medium`, `high`, `xhigh`
- `_parse_model_effort()` 解析，`_to_completion_request()` 传递

**OpenAI 模型** (gpt-5.x / o-series): → `reasoning_effort` 参数
**Claude 模型**: → thinking/output_config 映射:
| 版本 | effort 映射 | temperature |
|---|---|---|
| 4.7+ | `thinking={type:"adaptive"}, output_config={effort:"high"}` | 始终移除（已废弃）|
| 4.6+ | `thinking={type:"adaptive"}, output_config={effort:"high"}` | thinking 开启时移除 |
| 4.5 | `thinking={type:"enabled", budget_tokens=16000}` | 允许 |

4.5 effort→budget_tokens 映射: low=2048, medium=8192, high=16000, xhigh=32000
4.7 无 effort 后缀时: 无 temperature（已废弃），无 thinking

### SAP 参数过滤
```python
_SAP_SUPPORTED_PARAMS = {
    "anthropic": {"max_tokens", "temperature", "thinking", "output_config"},
    "openai":    {"max_completion_tokens", "reasoning_effort"},
    "_default":  {"max_tokens", "max_completion_tokens", "temperature"},
}
```
`_filter_model_params()` 按 owner 过滤，移除不支持参数并 warning log。

---

## Anti-empty-response 机制

Claude-4.6-opus 在收到 tool_result 后常返回空内容。两个防御:
1. **user hint**: 模板以 tool_result 结尾时，自动追加 user 消息:
   "Please continue with the next step based on the tool results above."
2. **空响应处理**: Anthropic 流式空响应现在发空 text content_block（非 `content=[]`）

---

## Tool call 机制

两套 tool call 路径:
1. **原生 tool_calls**: SAP 在 `final_result.llm.choices[].message.tool_calls` 返回
2. **标签解析回退**: 文本内容含 `<function_call>` 标签时，`parse_tool_calls()` 解析

`_build_native_tools()` 将 OpenAI tools 转为 SAP 格式。SAP 模板中
`assistant(tool_calls) → tool(tool_call_id)` 必须成对出现。

---

## Thinking 标签

OpenAI 流式路径中 `<thinking>...</thinking>` 标签被 `strip_thinking()` 移除。
Agent 不需要 thinking 块，无影响。

---

## 已知问题

1. `SAP_FORCE_NON_STREAM=true` 覆盖 stream 参数 → 2 个测试 deselected
2. GPT-5.4 在 tool_result 后有时仍返回空内容（模型行为，非代理 bug）
3. Claude 4.5 不支持 adaptive thinking（仅 `enabled` + `budget_tokens`）
4. GitHub token: 见 `.env` 或 GitHub Settings → Tokens

---

## 与 v2 的关系

v2 = v3 + Anthropic Messages API 层。核心逻辑（openai_api.py, curl_login.py, config.py）
完全相同，bug 修复同步。
v2 的 Anthropic 端点 `/v1/messages` 实质是 OpenAI 管道上的格式翻译层:
请求 → `anthropic_to_openai_request()` → 走 OpenAI 管道 → `openai_response_to_anthropic()` 返回

Claude 通过 v2 Anthropic 端点"正常工作"不是格式优势，而是 Claude 的单 tool_call 模式
刚好绕过了 turn boundary bug。GPT-5.4 的并行 tool_call 暴露了问题。
