from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
from app.curl_login import CompletionHTTPResult, CompletionRequest, SAPMetadataError, check_model_access_with_password_curl_cffi, get_cached_session_curl_cffi
from app.model_registry import (
    MODEL_ALIASES,
    ModelDebugEntry,
    SAPMetadataResponse,
    SAPModelEntry,
    SupportedModel,
    _is_supported_model,
    _model_owned_by,
    _parse_model_effort,
    _resolve_alias,
    _supports_reasoning_effort,
    extract_supported_models,
    fetch_supported_models,
    inspect_supported_models,
    invalidate_metadata_cache,
    resolve_model,
    resolve_model_cached,
)


class OpenAIFunction(BaseModel):
    name: str
    description: str = ""
    parameters: Any | None = None


class OpenAITool(BaseModel):
    type: str
    function: OpenAIFunction


class OpenAIToolFunctionCall(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: str = "function"
    function: OpenAIToolFunctionCall


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[OpenAIToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    stream: bool = False
    tools: list[OpenAITool] = Field(default_factory=list)


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]




class ParsedToolCall(BaseModel):
    id: str
    name: str
    arguments: str


TOOL_CALL_REGEX = re.compile(r"(?s)<function_call>\s*\n\s*([a-zA-Z0-9_.-]+)\s*\n(.*?)\n\s*</function_call>")



def _message_text(message: OpenAIMessage) -> str:
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for part in message.content:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts)



def _build_tool_system_prompt(tools: list[OpenAITool]) -> str:
    if not tools:
        return ""
    lines = [
        "In this environment you have access to a set of tools you can use to answer the user's question.",
        "",
        "You may call them by outputting a block with the following syntax:",
        "",
        "<function_call>",
        "function_name",
        '{"arguments": "value"}',
        "</function_call>",
        "",
        "IMPORTANT RULES:",
        "1. When you need to call a function, output ONLY the <function_call> block(s) and NOTHING ELSE.",
        "2. Do NOT make up or fabricate function results.",
        "3. Do NOT continue generating text after a function call. Stop immediately after the </function_call> tag.",
        "4. The user will provide the function result in the next message.",
        "5. If you do NOT need to call any function, just respond normally with text.",
        "",
        "Here are the tools available:",
        "",
    ]
    for tool in tools:
        lines.append(f"{tool.function.name}:")
        lines.append(f"  {tool.function.description}")
        if tool.function.parameters is not None:
            params_json = json.dumps(tool.function.parameters, ensure_ascii=False, indent=2)
            lines.append(f"  Parameters: {params_json}")
        lines.append("")
    return "\n".join(lines).strip()



def _build_tool_result_message(message: OpenAIMessage) -> str:
    return f"[tool result for {message.tool_call_id or ''}]: {_message_text(message)}"



def parse_tool_calls(text: str) -> tuple[list[ParsedToolCall], str]:
    matches = TOOL_CALL_REGEX.findall(text)
    if not matches:
        return [], text.strip()
    calls: list[ParsedToolCall] = []
    for i, match in enumerate(matches):
        name = match[0].strip()
        args = match[1].strip()
        calls.append(ParsedToolCall(id=f"call_{name}_{i}", name=name, arguments=args))
    # Split text around <function_call> blocks to find final reply text.
    # In prompt_templating mode, the model often outputs English "thinking" 
    # preamble between tool calls. Only text AFTER the last tool call
    # is the actual final reply to the user.
    parts = TOOL_CALL_REGEX.split(text)
    # parts layout: [pre1, name1, args1, mid1, name2, args2, ..., post_last]
    # After the last tool call group, the remaining text is the final reply.
    # Each match produces 3 groups (name, args) but split gives: pre, name, args, mid, name, args, ..., post
    # For N matches: len(parts) = 1 + 3*N
    # The last element is always the text after the final tool call.
    if len(parts) > 1:
        remaining = parts[-1].strip()
    else:
        remaining = text.strip()
    return calls, remaining



def _build_current_prompt(messages: list[OpenAIMessage], tools: list[OpenAITool] | None = None) -> str:
    """Build the current prompt text for the SAP template.

    Contains: system prompt (with tool definitions), and messages
    from the last user/tool turn onward. Earlier messages go into messages_history.
    """
    system_parts: list[str] = []
    current_parts: list[str] = []

    tool_prompt = _build_tool_system_prompt(tools or [])
    if tool_prompt:
        system_parts.append(tool_prompt)

    # Find the last user/tool message index - everything before it is history
    last_user_idx = -1
    for i, message in enumerate(messages):
        if message.role in ("user", "tool"):
            last_user_idx = i

    for i, message in enumerate(messages):
        text = _message_text(message).strip()

        if message.role == "system":
            if text:
                system_parts.append(text)
            continue

        # Only include messages from the last user turn onward
        if i < last_user_idx:
            continue

        if message.role == "tool":
            current_parts.append(f"User: {_build_tool_result_message(message)}")
            continue

        if message.role == "assistant" and message.tool_calls:
            parts: list[str] = []
            if text:
                parts.append(text)
            for tool_call in message.tool_calls:
                parts.extend([
                    "<function_call>",
                    tool_call.function.name,
                    tool_call.function.arguments,
                    "</function_call>",
                ])
            current_parts.append("Assistant: " + "\n".join(parts))
            continue

        if not text:
            continue
        if message.role == "assistant":
            current_parts.append(f"Assistant: {text}")
        elif message.role == "user":
            current_parts.append(f"User: {text}")
        else:
            current_parts.append(f"{message.role}: {text}")

    if system_parts:
        current_parts.insert(0, "System:\n" + "\n\n".join(system_parts))
    return "\n\n".join(current_parts)


def _repair_tool_adjacency(history: list[dict]) -> list[dict]:
    """Ensure every tool result has a preceding assistant with matching tool_use.

    After truncation, an assistant message containing tool_use may be removed
    while its corresponding tool result survives. SAP's upstream LLM validates
    that each tool_result's tool_use_id exists in the immediately preceding
    assistant message. Orphaned tool_results cause 400 errors like:
      "unexpected `tool_use_id` found in `tool_result` blocks"

    This function removes orphaned tool results and any dangling messages
    that would create an invalid sequence (e.g. user→tool without an
    intervening assistant with tool_use).
    """
    if not history:
        return history

    # Scan forward: track which tool_use_ids have been declared by
    # assistant messages. When we encounter a tool result whose
    # tool_call_id is not in the known set, remove it.
    known_tool_ids: set[str] = set()
    repaired: list[dict] = []

    for entry in history:
        role = entry.get("role", "")

        if role == "assistant":
            # Register tool_use IDs from this assistant message
            for tc in entry.get("tool_calls", []):
                tc_id = tc.get("id", "")
                if tc_id:
                    known_tool_ids.add(tc_id)
            repaired.append(entry)

        elif role == "tool":
            tool_call_id = entry.get("tool_call_id", "")
            if tool_call_id and tool_call_id not in known_tool_ids:
                # Orphaned tool result — skip it
                logger.info(
                    "history repair: dropping orphaned tool result (tool_call_id=%s not in known_tool_ids)",
                    tool_call_id[:40],
                )
                continue
            repaired.append(entry)

        else:
            # user or other — keep as-is
            repaired.append(entry)

    # Also check that the history doesn't START with a tool message
    # (which would have no preceding assistant). Remove leading tool msgs.
    while repaired and repaired[0].get("role") == "tool":
        orphan = repaired.pop(0)
        logger.info(
            "history repair: dropping leading tool entry (tool_call_id=%s)",
            orphan.get("tool_call_id", "?")[:40],
        )

    return repaired


def _find_turn_start(messages: list[OpenAIMessage]) -> int:
    """Find the index where the current turn begins.

    Walks backwards from the last user/tool message to find the start
    of the current turn. A turn includes: user message, any preceding
    assistant(tool_calls), and tool results in a chain.
    Returns 0 if the entire conversation is one turn.
    """
    last_user_idx = -1
    for i, message in enumerate(messages):
        if message.role in ("user", "tool"):
            last_user_idx = i

    turn_start = last_user_idx
    while turn_start > 0:
        prev = messages[turn_start - 1]
        if prev.role == "tool":
            # Parallel tool result — belongs to same turn
            turn_start -= 1
            continue
        elif prev.role == "assistant" and prev.tool_calls:
            # assistant(tool_calls) — its tool results are in this turn
            turn_start -= 1
            continue
        elif prev.role == "user":
            # User message that started this turn
            turn_start -= 1
            continue
        else:
            # Boundary: system or assistant without tool_calls
            break
    return turn_start


def _build_messages_history(messages: list[OpenAIMessage]) -> list[dict]:
    """Build messages_history for SAP completionV2.

    Includes all messages BEFORE the current turn.
    Uses native OpenAI format: tool_calls in assistant messages,
    role=tool with tool_call_id for tool results.

    The current turn (including assistant(tool_calls) → tool_result chains)
    goes in the template. History must NOT contain orphaned assistant(tool_calls)
    whose tool_result is in the template, or SAP rejects with 400.

    Truncates oldest history if it exceeds MAX_HISTORY_TURNS
    or MAX_HISTORY_TOKENS limits.
    """
    turn_start = _find_turn_start(messages)

    if turn_start <= 0:
        return []

    # Collect all history messages (before the current turn)
    history_msgs: list[tuple[int, OpenAIMessage]] = []
    for i, message in enumerate(messages):
        if i >= turn_start:
            break
        if message.role == "system":
            continue
        history_msgs.append((i, message))

    # Truncate from the front if too many turns
    max_turns = settings.max_history_turns
    # A "turn" = one user message + its assistant/tool responses
    # Count user messages as turn boundaries
    user_indices = [idx for idx, (i, m) in enumerate(history_msgs) if m.role == "user"]
    if len(user_indices) > max_turns:
        # Keep only the last max_turns user messages and everything after
        cutoff_idx = user_indices[-max_turns]
        history_msgs = history_msgs[cutoff_idx:]

    # Build history entries
    history: list[dict] = []
    total_chars = 0
    for i, message in history_msgs:
        text = _message_text(message).strip()

        if message.role == "user":
            if text:
                entry = {"role": "user", "content": text}
                history.append(entry)
                total_chars += len(text)
        elif message.role == "assistant":
            if message.tool_calls:
                entry: dict = {"role": "assistant", "content": text or ""}
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
                history.append(entry)
                total_chars += len(text) + sum(len(tc.function.arguments) for tc in message.tool_calls)
            elif text:
                entry = {"role": "assistant", "content": text}
                history.append(entry)
                total_chars += len(text)
        elif message.role == "tool":
            content = text or _build_tool_result_message(message)
            entry = {"role": "tool", "content": content}
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            history.append(entry)
            total_chars += len(content)

    # Truncate from the front if token budget exceeded.
    # Previously this was done inline with a `break` after each budget check,
    # which caused a critical bug: when a large entry (e.g. image_url) pushed
    # the budget over the limit mid-loop, the break dropped ALL subsequent
    # messages — including the assistant with tool_use whose tool_result
    # was in the template, causing SAP 400 "unexpected tool_use_id" errors.
    # Now we collect all entries first, then truncate from the front once.
    if total_chars // 4 > settings.max_history_tokens:
        while history and total_chars // 4 > settings.max_history_tokens:
            removed = history.pop(0)
            removed_chars = len(str(removed.get("content", "")))
            for tc in removed.get("tool_calls", []):
                removed_chars += len(tc.get("function", {}).get("arguments", ""))
            total_chars -= removed_chars

    # Validate tool_use→tool_result adjacency after truncation.
    # If truncation removed an assistant message containing tool_use,
    # any subsequent tool result referencing that tool_use_id becomes orphaned.
    # SAP's upstream LLM rejects orphaned tool_results with 400.
    history = _repair_tool_adjacency(history)
    _ensure_tool_results_complete(history)

    return history


def _build_prompt_text(messages: list[OpenAIMessage], tools: list[OpenAITool] | None = None) -> str:
    """Legacy: build full prompt text with all messages concatenated."""
    system_parts: list[str] = []
    dialogue_parts: list[str] = []

    tool_prompt = _build_tool_system_prompt(tools or [])
    if tool_prompt:
        system_parts.append(tool_prompt)

    found_user = False
    for message in messages:
        text = _message_text(message).strip()
        if message.role == "system" and not found_user:
            if text:
                system_parts.append(text)
            continue

        if message.role != "system":
            found_user = True

        if message.role == "tool":
            dialogue_parts.append(f"User: {_build_tool_result_message(message)}")
            continue

        if message.role == "assistant" and message.tool_calls:
            parts: list[str] = []
            if text:
                parts.append(text)
            for tool_call in message.tool_calls:
                parts.extend([
                    "<function_call>",
                    tool_call.function.name,
                    tool_call.function.arguments,
                    "</function_call>",
                ])
            dialogue_parts.append("Assistant: " + "\n".join(parts))
            continue

        if not text:
            continue
        if message.role == "assistant":
            dialogue_parts.append(f"Assistant: {text}")
        elif message.role == "user":
            dialogue_parts.append(f"User: {text}")
        elif message.role == "system":
            dialogue_parts.append(f"System: {text}")
        else:
            dialogue_parts.append(f"{message.role}: {text}")

    if system_parts:
        dialogue_parts.insert(0, "System:\n" + "\n\n".join(system_parts))
    return "\n\n".join(dialogue_parts)


def _build_messages_history_with_images(messages: list[OpenAIMessage]) -> list[dict]:
    """Build messages_history for SAP, preserving image_url blocks.

    Unlike _build_messages_history which drops image_url (only keeps text),
    this preserves image_url content blocks in user messages. This allows
    SAP to see the complete conversation including images in messages_history,
    which is critical for multi-round agent vision: the model correctly
    reasons about tool use (e.g., calling read_image for new images)
    when it sees the full conversation in order.

    SAP accepts image_url in messages_history (verified).
    """
    turn_start = _find_turn_start(messages)

    if turn_start <= 0:
        return []

    # Collect all history messages (before the current turn)
    history_msgs: list[tuple[int, OpenAIMessage]] = []
    for i, message in enumerate(messages):
        if i >= turn_start:
            break
        if message.role == "system":
            continue
        history_msgs.append((i, message))

    # Truncate from the front if too many turns
    max_turns = settings.max_history_turns
    user_indices = [idx for idx, (i, m) in enumerate(history_msgs) if m.role == "user"]
    if len(user_indices) > max_turns:
        cutoff_idx = user_indices[-max_turns]
        history_msgs = history_msgs[cutoff_idx:]

    # Build history entries, preserving image_url
    history: list[dict] = []
    total_chars = 0
    for i, message in history_msgs:
        text = _message_text(message).strip()

        if message.role == "user":
            # Preserve image_url blocks in user messages
            if isinstance(message.content, list):
                content_blocks = []
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            content_blocks.append(part)
                        elif part.get("type") == "text" and isinstance(part.get("text"), str):
                            content_blocks.append({"type": "text", "text": part["text"]})
                if content_blocks:
                    entry = {"role": "user", "content": content_blocks}
                    history.append(entry)
                    # Estimate chars for token budget (image_url data is huge)
                    for b in content_blocks:
                        if b.get("type") == "image_url":
                            total_chars += len(b.get("image_url", {}).get("url", ""))
                        elif b.get("type") == "text":
                            total_chars += len(b.get("text", ""))
                elif text:
                    entry = {"role": "user", "content": text}
                    history.append(entry)
                    total_chars += len(text)
            elif text:
                entry = {"role": "user", "content": text}
                history.append(entry)
                total_chars += len(text)
        elif message.role == "assistant":
            if message.tool_calls:
                entry: dict = {"role": "assistant", "content": text or ""}
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
                history.append(entry)
                total_chars += len(text) + sum(len(tc.function.arguments) for tc in message.tool_calls)
            elif text:
                entry = {"role": "assistant", "content": text}
                history.append(entry)
                total_chars += len(text)
        elif message.role == "tool":
            content = text or _build_tool_result_message(message)
            entry = {"role": "tool", "content": content}
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            history.append(entry)
            total_chars += len(content)

    # Truncate from the front if token budget exceeded.
    # Previously this was done inline with a `break` after each budget check,
    # which caused a critical bug: when a large entry (e.g. image_url 230KB+)
    # pushed the budget over the limit mid-loop, the break dropped ALL
    # subsequent messages — including the assistant with tool_use whose
    # tool_result was in the template, causing SAP 400 "unexpected tool_use_id"
    # errors. Now we collect all entries first, then truncate from the front.
    if total_chars // 4 > settings.max_history_tokens:
        while history and total_chars // 4 > settings.max_history_tokens:
            removed = history.pop(0)
            rc = removed.get("content")
            if isinstance(rc, list):
                for b in rc:
                    if isinstance(b, dict):
                        if b.get("type") == "image_url":
                            total_chars -= len(b.get("image_url", {}).get("url", ""))
                        elif b.get("type") == "text":
                            total_chars -= len(b.get("text", ""))
            elif isinstance(rc, str):
                total_chars -= len(rc)
            for tc in removed.get("tool_calls", []):
                total_chars -= len(tc.get("function", {}).get("arguments", ""))

    # Validate tool_use→tool_result adjacency after truncation.
    # If truncation removed an assistant message containing tool_use,
    # any subsequent tool result referencing that tool_use_id becomes orphaned.
    # SAP's upstream LLM rejects orphaned tool_results with 400.
    history = _repair_tool_adjacency(history)
    _ensure_tool_results_complete(history)

    # Move user(image) messages that break tool result adjacency
    history = _reorder_tool_result_images(history)

    return history


def _build_template_messages(messages: list[OpenAIMessage], tools: list[OpenAITool] | None = None) -> list[dict]:
    """Build template messages for the current turn in native SAP format.

    Contains: system message, and messages from the last user/tool turn onward.
    Uses proper role/content structure, not flat text.
    """
    current: list[dict] = []

    # Find the start of the current turn for template inclusion.
    turn_start = _find_turn_start(messages)

    # If there's a system message, put it in template
    for message in messages:
        if message.role == "system":
            text = _message_text(message).strip()
            if text:
                current.append({"role": "system", "content": text})
            break

    # Add messages from the turn_start onward
    for i, message in enumerate(messages):
        if i < turn_start:
            continue
        if message.role == "system":
            continue

        text = _message_text(message).strip()

        if message.role == "user":
            # Preserve image_url content blocks for multimodal support
            if isinstance(message.content, list):
                # Build content blocks list, preserving image_url
                content_blocks = []
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            content_blocks.append(part)
                        elif part.get("type") == "text" and isinstance(part.get("text"), str):
                            content_blocks.append({"type": "text", "text": part["text"]})
                if content_blocks:
                    current.append({"role": "user", "content": content_blocks})
                elif text:
                    current.append({"role": "user", "content": text})
            else:
                current.append({"role": "user", "content": text})
        elif message.role == "assistant":
            if message.tool_calls:
                entry = {"role": "assistant", "content": text or ""}
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
                current.append(entry)
            elif text:
                current.append({"role": "assistant", "content": text})
        elif message.role == "tool":
            entry = {"role": "tool", "content": text}
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            current.append(entry)

    # Anti-empty-response hint: when tools are present and the template
    # ends with a tool_result, append a user hint to prompt the model.
    # Without this, claude-4.6-opus often returns empty responses after
    # receiving tool results, especially after tool errors (boxd02/boxd03).
    # We use user role (not assistant) so the model treats this as a
    # genuine continuation request rather than continuing its own prior text.
    # Defensive: ensure every assistant(tool_calls) in the template has matching
    # tool results. Some clients (e.g. Shelley agent) send partial tool results
    # when a parallel tool_call hasn't completed yet. SAP rejects these with 400
    # "tool_call_ids did not have response messages". Synthesize missing results.
    _ensure_tool_results_complete(current)

    # Move user(image) messages that break tool result adjacency
    current = _reorder_tool_result_images(current)

    if tools and current and current[-1].get("role") == "tool":
        current.append({"role": "user", "content": "Please continue with the next step based on the tool results above."})

    return current


def _ensure_tool_results_complete(template: list[dict]) -> None:
    """Add synthetic tool results for any orphaned tool_call_ids.

    SAP requires every tool_call in an assistant message to have a matching
    tool result. If a client sends partial results (e.g. only 2 of 3 parallel
    tool calls have results), SAP rejects the request. This function detects
    such orphans and adds placeholder results so the request goes through.
    """
    # Collect all tool_call_ids declared by assistant messages
    declared_ids: set[str] = set()
    for entry in template:
        for tc in entry.get("tool_calls", []):
            tc_id = tc.get("id", "")
            if tc_id:
                declared_ids.add(tc_id)

    # Collect all tool_call_ids that have results
    answered_ids: set[str] = set()
    for entry in template:
        if entry.get("role") == "tool":
            tc_id = entry.get("tool_call_id", "")
            if tc_id:
                answered_ids.add(tc_id)

    # Find orphaned tool_call_ids (declared but not answered)
    orphaned = declared_ids - answered_ids
    if not orphaned:
        return

    # Find the position after the last tool result or assistant(tc) message
    # so we can insert the synthetic results in the right place
    insert_pos = len(template)
    for i in range(len(template) - 1, -1, -1):
        role = template[i].get("role", "")
        if role == "tool":
            insert_pos = i + 1
            break
        elif role == "assistant" and template[i].get("tool_calls"):
            insert_pos = i + 1
            break

    for orphan_id in sorted(orphaned):
        logger.info("synthesizing missing tool result for tool_call_id=%s", orphan_id)
        template.insert(insert_pos, {
            "role": "tool",
            "content": "[tool result pending — no result available]",
            "tool_call_id": orphan_id,
        })
        insert_pos += 1  # keep insertion order correct


def _build_native_tools(tools: list[OpenAITool] | None) -> list[dict] | None:
    """Convert OpenAI tools to SAP native tools format.

    Returns None if no tools, so _build_completion_payload knows
    not to include the tools key.
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            },
        }
        for t in tools
    ]


def _reorder_tool_result_images(entries: list[dict]) -> list[dict]:
    """Move user messages that appear between tool results to after the group.

    Some clients (e.g. Shelley agent) send a user message (often with image_url)
    immediately after a tool result, before the remaining tool results from the
    same assistant(tool_calls). For example:

        assistant(tool_calls: [A, B, C])
        tool(result for A)
        user(image from A)      ← breaks SAP adjacency!
        tool(result for B)
        tool(result for C)

    SAP requires all tool results to be consecutive after assistant(tool_calls).
    This function detects such user messages and moves them to after the last
    tool result in the group, producing:

        assistant(tool_calls: [A, B, C])
        tool(result for A)
        tool(result for B)
        tool(result for C)
        user(image from A)      ← moved after all tool results
    """
    if not entries:
        return entries

    result = list(entries)
    i = 0
    while i < len(result):
        entry = result[i]
        # Find an assistant message with tool_calls
        if entry.get("role") == "assistant" and entry.get("tool_calls"):
            # Collect consecutive tool and user messages that follow
            j = i + 1
            pending_users = []  # user messages to move (indices)
            last_tool_pos = -1  # position of last tool message in this group

            while j < len(result):
                nxt = result[j]
                if nxt.get("role") == "tool":
                    last_tool_pos = j
                    j += 1
                    continue
                elif nxt.get("role") == "user":
                    # user message between tool results — move it later
                    pending_users.append(j)
                    j += 1
                    continue
                else:
                    # Boundary: non-tool, non-user message ends the group
                    break

            # If there are pending user messages and tool results after them,
            # move the user messages to after the last tool result
            if pending_users and last_tool_pos > max(pending_users):
                # Remove user messages from current positions (reverse order to preserve indices)
                moved = []
                for idx in reversed(pending_users):
                    moved.append(result.pop(idx))
                moved.reverse()
                # Insert after the last tool result group
                # Find the insert point by scanning from the assistant(tc)
                insert_at = i + 1
                while insert_at < len(result):
                    if result[insert_at].get("role") in ("tool", "user"):
                        insert_at += 1
                        continue
                    else:
                        break
                for k, msg in enumerate(moved):
                    result.insert(insert_at + k, msg)
                # Don't advance i — re-check this position
                continue

        i += 1

    return result


def _estimate_template_size(template: list[dict]) -> int:
    """Byte-size estimate of a template for payload budget tracking.

    Counts string content + image_url base64 data length + JSON overhead.
    Uses a 1.2x safety factor to account for JSON structure overhead
    (keys, braces, nesting) that pure content-length misses.
    """
    content_len = 0
    for entry in template:
        content = entry.get("content")
        if isinstance(content, str):
            content_len += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        content_len += len(url)  # base64 data URIs are huge
                    elif block.get("type") == "text":
                        content_len += len(block.get("text", ""))
        # tool_calls arguments can be large too
        for tc in entry.get("tool_calls", []):
            content_len += len(tc.get("function", {}).get("arguments", ""))
    # 1.2x safety factor for JSON overhead (keys, braces, nesting)
    return int(content_len * 1.2)


# Conservative payload size limit for SAP multimodal mode.
# SAP has no published hard limit, but community experience suggests
# staying under ~1-2 MB for reliability.
_MULTIMODAL_PAYLOAD_WARN = 1_000_000   # 1 MB — log warning
_MULTIMODAL_PAYLOAD_HARD = 2_000_000   # 2 MB — reject with 400



def _build_image_template_messages(messages: list[OpenAIMessage], tools: list[OpenAITool] | None = None) -> list[dict]:
    """Build template messages containing ALL user messages with image_url blocks.

    Used only for pure multimodal (images, no tools) mode.
    When tools are present, images go in messages_history instead.
    """
    # Find ALL user messages with image_url content
    image_user_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.role == "user" and isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_user_indices.append(i)
                    break

    if not image_user_indices:
        # No image found (shouldn't happen if has_images is True)
        return _build_template_messages(messages, tools)

    template = []
    # System message if present
    for msg in messages:
        if msg.role == "system":
            text = _message_text(msg).strip()
            if text:
                template.append({"role": "system", "content": text})
            break

    # Add ALL image-carrying user messages to template
    for idx in image_user_indices:
        image_msg = messages[idx]
        content_blocks = []
        if isinstance(image_msg.content, list):
            for part in image_msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        content_blocks.append(part)
                    elif part.get("type") == "text" and isinstance(part.get("text"), str):
                        content_blocks.append({"type": "text", "text": part["text"]})
        if content_blocks:
            template.append({"role": "user", "content": content_blocks})

    return template




def _has_image_content(messages: list[OpenAIMessage]) -> bool:
    """Check if any message contains image_url content blocks."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


# Content block types we explicitly support in message content lists.
# Anything else (e.g. "image", "video", "audio") is rejected early
# so users get a clear 400 instead of silent data loss.
_SUPPORTED_CONTENT_TYPES = {"text", "image_url"}


def validate_content_blocks(messages: list[OpenAIMessage]) -> str | None:
    """Validate that all content blocks use supported types.

    Returns an error message string if an unsupported type is found,
    or None if everything is valid.
    """
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    block_type = part.get("type", "")
                    if block_type and block_type not in _SUPPORTED_CONTENT_TYPES:
                        return (f"Unsupported content block type '{block_type}' in {msg.role} message. "
                                f"Supported types: {sorted(_SUPPORTED_CONTENT_TYPES)}. "
                                f"Use base64 image_url format instead.")
    return None


def _to_completion_request(req: OpenAIChatRequest, model_id: str, version: str, deployment_id: str | None = None, resource_group_id: str | None = None) -> CompletionRequest:
    native_tools = _build_native_tools(req.tools)
    has_images = _has_image_content(req.messages)

    # Parse reasoning effort from model name suffix (e.g. 'gpt-5.4:high')
    # The model_id passed in is already resolved (alias → canonical),
    # so we parse effort from the ORIGINAL req.model field.
    _base_model, reasoning_effort = _parse_model_effort(req.model)
    # For OpenAI models: check _supports_reasoning_effort
    # For Claude models: effort is mapped to thinking/output_config in _build_model_params
    is_claude = model_id.startswith("anthropic--claude-")
    if reasoning_effort and not _supports_reasoning_effort(model_id) and not is_claude:
        logger.info("reasoning_effort=%s ignored for %s (not supported)", reasoning_effort, model_id)
        reasoning_effort = None

    # Strategy for image_url content:
    #
    # Previously used hybrid mode: images in template, tools in messages_history.
    # This broke multi-round agent vision because SAP processes template and
    # messages_history separately — the model sees the image as "current input"
    # not as part of the conversation, so it doesn't call read_image for
    # subsequent images sent as text-only paths.
    #
    # New approach: put EVERYTHING in messages_history (text mode), including
    # image_url blocks. SAP accepts image_url in messages_history (verified),
    # and the model sees the complete conversation in order, correctly
    # reasoning about tool use. Stream is preserved.
    #
    # For pure image conversations (no tools, no history), the full multimodal
    # template mode is still used as fallback (no messages_history needed).
    #
    if has_images and req.tools:
        # Has images + tools → text mode with image_url in messages_history
        # This preserves conversation order so the model correctly reasons
        # about when to call tools (like read_image for new images).
        template_messages = _build_template_messages(req.messages, req.tools)
        messages_history = _build_messages_history_with_images(req.messages)
        # Text mode: has_images=False so SAP includes stream + messages_history
        effective_has_images = False
        stream_enabled = False if settings.completion.force_non_stream else req.stream
    elif has_images:
        # Pure image conversation (no tools) → full multimodal template
        # No messages_history needed, image_url goes in template only.
        # SAP multimodal mode: no stream, no messages_history.
        template_messages = _build_image_template_messages(req.messages, req.tools)
        messages_history = []
        effective_has_images = True
        stream_enabled = False  # SAP multimodal doesn't support stream
        if req.stream:
            logger.info("multimodal request with stream=true — downgrading to non-stream (SAP limitation)")
        # Hard payload size limit
        est_size = _estimate_template_size(template_messages)
        if est_size > _MULTIMODAL_PAYLOAD_HARD:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail=f"Multimodal request payload too large (~{est_size // 1000} KB). "
                       f"Reduce conversation length or use URL-based images instead of base64.",
            )
    else:
        # Text only — standard mode
        template_messages = _build_template_messages(req.messages, req.tools)
        messages_history = _build_messages_history(req.messages)
        effective_has_images = False
        stream_enabled = False if settings.completion.force_non_stream else req.stream

    return CompletionRequest(
        prompt_text=_build_current_prompt(req.messages, req.tools),  # fallback
        model_name=model_id,
        model_version=version,
        max_tokens=req.max_tokens or settings.completion.model.max_tokens,
        temperature=req.temperature if req.temperature is not None else settings.completion.model.temperature,
        deployment_id=deployment_id or settings.completion.deployment_id,
        workspace=settings.completion.workspace,
        resource_group_id=resource_group_id or settings.completion.resource_group_id,
        stream_enabled=stream_enabled,
        messages_history=messages_history,
        template_messages=template_messages,
        tools=native_tools,
        has_images=effective_has_images,
        top_p=req.top_p,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        stop=req.stop,
        n=req.n,
        reasoning_effort=reasoning_effort,
    )








def _extract_content_and_usage_from_nonstream_json(obj: dict[str, Any]) -> tuple[str, dict[str, int]]:
    for container_name in ["final_result", "intermediate_results"]:
        container = obj.get(container_name) or {}
        llm = container.get("llm") if isinstance(container, dict) and "llm" in container else container
        if not isinstance(llm, dict):
            continue
        choices = llm.get("choices", [])
        content_parts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                content_parts.append(message["content"])
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
        usage = llm.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if content_parts or usage:
            return "".join(content_parts), {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens)),
            }
    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}



def parse_sap_sse_text(body: str) -> tuple[str, dict[str, int]]:
    stripped = body.strip()
    if stripped and not stripped.startswith("data: "):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return _extract_content_and_usage_from_nonstream_json(obj)

    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for container_name in ["intermediate_results", "final_result"]:
            container = obj.get(container_name) or {}
            llm = container.get("llm") or {}
            for choice in llm.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    chunks.append(content)
            usage = llm.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)
    return "".join(chunks), {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }



def build_openai_response(model: str, content: str, usage: dict[str, int], tool_calls: list[OpenAIToolCall] | None = None) -> OpenAIChatResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    message: dict[str, Any] = {"role": "assistant", "content": content if content else None}
    if tool_calls:
        message["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "message": message, "finish_reason": finish_reason}],
        usage=usage,
    )



def parse_sap_sse_tool_calls(body: str) -> tuple[str, dict[str, int], list[dict]]:
    """Parse SAP SSE body extracting content, usage, and native tool_calls.

    Returns (content, usage, tool_calls_list).
    If no native tool_calls found, returns empty list.
    """
    stripped = body.strip()
    if stripped and not stripped.startswith("data: "):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            content, usage = _extract_content_and_usage_from_nonstream_json(obj)
            # Check for tool_calls in non-stream response
            tool_calls = []
            for container_name in ["intermediate_results", "final_result"]:
                container = obj.get(container_name) or {}
                llm = container.get("llm") or {}
                for choice in llm.get("choices", []):
                    msg = choice.get("message", {})
                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]
                        break
            return content, usage, tool_calls

    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    tool_calls: list[dict] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for container_name in ["intermediate_results", "final_result"]:
            container = obj.get(container_name) or {}
            llm = container.get("llm") or {}
            for choice in llm.get("choices", []):
                delta = choice.get("delta") or {}
                content_chunk = delta.get("content")
                if isinstance(content_chunk, str):
                    chunks.append(content_chunk)
                # Check for native tool_calls in streaming delta
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        # Accumulate tool call parts
                        while len(tool_calls) <= idx:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        current_tc = tool_calls[idx]
                        if tc.get("id"):
                            current_tc["id"] = tc["id"]
                        if tc.get("type"):
                            current_tc["type"] = tc["type"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            current_tc["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            current_tc["function"]["arguments"] += fn["arguments"]
                # Also check message-level tool_calls (non-streaming chunk)
                msg = choice.get("message", {})
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]
            usage = llm.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)
    return "".join(chunks), {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }, tool_calls


def build_openai_response_from_text(model: str, content: str, usage: dict[str, int], has_tools: bool) -> OpenAIChatResponse:
    parsed_tool_calls, remaining = parse_tool_calls(strip_thinking(content)) if has_tools else ([], content)
    tool_calls = [
        OpenAIToolCall(id=item.id, type="function", function=OpenAIToolFunctionCall(name=item.name, arguments=item.arguments))
        for item in parsed_tool_calls
    ]
    return build_openai_response(model, remaining, usage, tool_calls=tool_calls)


def build_openai_response_from_sap(model: str, content: str, usage: dict[str, int], native_tool_calls: list[dict]) -> OpenAIChatResponse:
    """Build OpenAI response using native SAP tool_calls (no text parsing needed).

    If native_tool_calls is non-empty, use them directly.
    Otherwise fall back to text-based parse_tool_calls.
    """
    if native_tool_calls:
        content = strip_thinking(content)
        tool_calls = [
            OpenAIToolCall(
                id=tc.get("id", f"call_{i}"),
                type="function",
                function=OpenAIToolFunctionCall(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", ""),
                ),
            )
            for i, tc in enumerate(native_tool_calls)
        ]
        # If tool_calls present, content is typically empty or just preamble
        remaining = content
        # Strip any intermediate thinking before tool calls
        if tool_calls:
            remaining = ""
        return build_openai_response(model, remaining, usage, tool_calls=tool_calls)
    # Fallback to text-based parsing (legacy prompt injection mode)
    import logging
    logging.getLogger(__name__).warning("No native tool_calls from SAP, falling back to text-based parsing")
    return build_openai_response_from_text(model, content, usage, has_tools=False)



def iter_openai_sse(model: str, sap_result: CompletionHTTPResult, has_tools: bool = False):
    """Stream OpenAI SSE chunks from SAP completion response.

    For true streaming (stream_resp available), reads SAP SSE line-by-line and
    forwards content deltas and native tool_calls in real-time.
    No buffering needed — native tool_calls arrive in SSE chunks.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Send role chunk first
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    if sap_result.stream_resp is not None:
        # True streaming: forward content + native tool_calls in real-time
        yield from _iter_sse_from_stream(model, chunk_id, created, sap_result.stream_resp)
    else:
        # Buffered fallback (non-stream SAP response, or no stream_resp)
        content, usage, native_tool_calls = parse_sap_sse_tool_calls(sap_result.body)
        content = strip_thinking(content)
        if native_tool_calls:
            for i, tc in enumerate(native_tool_calls):
                tool_call = {
                    'index': i,
                    'id': tc.get('id', f'call_{i}'),
                    'type': tc.get('type', 'function'),
                    'function': tc.get('function', {}),
                }
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [tool_call]}, 'finish_reason': None}]})}\n\n"
            finish_reason = 'tool_calls'
        else:
            parsed_tool_calls, remaining = parse_tool_calls(content) if has_tools else ([], content)
            if remaining:
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': remaining}, 'finish_reason': None}]})}\n\n"
            if parsed_tool_calls:
                for item in parsed_tool_calls:
                    tool_call = {'id': item.id, 'type': 'function', 'function': {'name': item.name, 'arguments': item.arguments}}
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [tool_call]}, 'finish_reason': None}]})}\n\n"
                finish_reason = 'tool_calls'
            else:
                finish_reason = 'stop'
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"


def _iter_sse_from_stream(model: str, chunk_id: str, created: int, stream_resp):
    """Read SAP SSE from a live stream response and forward as OpenAI chunks.

    Handles both content deltas and native tool_calls in real-time.
    No buffering needed — everything is forwarded as it arrives.
    """
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    has_tool_calls = False
    try:
        for raw_line in stream_resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Extract content deltas and tool_calls from SAP SSE format
            for container_name in ["intermediate_results", "final_result"]:
                container = obj.get(container_name) or {}
                llm = container.get("llm") if isinstance(container, dict) and "llm" in container else {}
                if not isinstance(llm, dict):
                    continue

                # Always extract usage from any container
                llm_usage = llm.get("usage") or {}
                if llm_usage:
                    usage["prompt_tokens"] = llm_usage.get("prompt_tokens", usage["prompt_tokens"])
                    usage["completion_tokens"] = llm_usage.get("completion_tokens", usage["completion_tokens"])
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

                # SAP sends the same content in both intermediate_results and
                # final_result. Only emit content/tool deltas from intermediate;
                # skip content from final_result to avoid duplication.
                if container_name == "final_result":
                    continue

                for choice in llm.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        chunk_data = {
                            'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created,
                            'model': model,
                            'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    # Forward native tool_calls in streaming delta
                    if delta.get("tool_calls"):
                        has_tool_calls = True
                        tool_calls = delta["tool_calls"]
                        chunk_data = {
                            'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created,
                            'model': model,
                            'choices': [{'index': 0, 'delta': {'tool_calls': tool_calls}, 'finish_reason': None}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
    finally:
        try:
            stream_resp.close()
        except Exception:
            pass

    finish_reason = 'tool_calls' if has_tool_calls else 'stop'
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"



def strip_thinking(text: str) -> str:
    return re.sub(r'(?s)<thinking>.*?</thinking>', '', text).strip()
