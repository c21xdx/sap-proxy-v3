"""Model registry: aliases, resolution, metadata caching, and capability detection.

Extracted from openai_api.py and curl_login.py to consolidate all model-related
logic in one place.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.curl_login import SAPMetadataError, check_model_access_with_password_curl_cffi, get_cached_session_curl_cffi

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────────────

class SAPModelEntry(BaseModel):
    model: str
    version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SAPMetadataResponse(BaseModel):
    globalLLMInfo: dict[str, Any]


class SupportedModel(BaseModel):
    id: str
    version: str
    owned_by: str


class ModelDebugEntry(BaseModel):
    id: str
    version: str
    owned_by: str
    configured: bool
    metadata_match: bool
    access_allowed: bool
    access_status_code: int | None = None
    access_detail: str = ""


# ── Model capability detection ───────────────────────────────────────────────

def _supports_reasoning_effort(model_name: str) -> bool:
    """Check if a model supports the reasoning_effort parameter on SAP."""
    if model_name.startswith('gpt-5'):
        return True
    if model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4'):
        return True
    return False


def _claude_supports_adaptive_thinking(model_name: str) -> bool:
    """Check if a Claude model supports thinking.type=adaptive + output_config.effort.
    Claude 4.6+ supports both enabled and adaptive thinking.
    Claude 4.7+ only supports adaptive (enabled is rejected).
    """
    if not model_name.startswith('anthropic--claude-'):
        return False
    suffix = model_name.split('anthropic--claude-', 1)[1]
    ver_str = suffix.split('-')[0]  # e.g. '4.6' or '4.7'
    try:
        parts = ver_str.split('.')
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (4, 6)
    except (ValueError, IndexError):
        return False


def _claude_deprecates_temperature(model_name: str) -> bool:
    """Check if a Claude model has deprecated the temperature parameter.
    Claude 4.7+ does not accept temperature.
    """
    if not model_name.startswith('anthropic--claude-'):
        return False
    suffix = model_name.split('anthropic--claude-', 1)[1]
    ver_str = suffix.split('-')[0]
    try:
        parts = ver_str.split('.')
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (4, 7)
    except (ValueError, IndexError):
        return False


def _claude_supports_enabled_thinking(model_name: str) -> bool:
    """Check if a Claude model supports thinking.type=enabled + budget_tokens.
    Claude 4.5 and 4.6 support this. Claude 4.7+ does not.
    """
    if not model_name.startswith('anthropic--claude-'):
        return False
    suffix = model_name.split('anthropic--claude-', 1)[1]
    ver_str = suffix.split('-')[0]
    try:
        parts = ver_str.split('.')
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return (4, 5) <= (major, minor) < (4, 7)
    except (ValueError, IndexError):
        return False


# ── Model ID helpers ─────────────────────────────────────────────────────────

def _model_owned_by(model_id: str) -> str:
    if model_id.startswith("gpt-"):
        return "openai"
    return model_id.split("--", 1)[0] if "--" in model_id else "sap"


def _is_supported_model(model_id: str) -> bool:
    """Check if a model is in the allowed list.
    If allowed_models is empty (default), all non-deprecated models are accepted."""
    if not settings.allowed_models:
        return True  # auto-discover: accept everything from SAP metadata
    model = model_id.lower()
    return model in {item.lower() for item in settings.allowed_models}


# ── Model aliases ────────────────────────────────────────────────────────────

# Alias map: common short names → canonical SAP model IDs
# Allows agents/users to use familiar names like "claude-sonnet-4-5" or "claude-4.5-sonnet"
MODEL_ALIASES: dict[str, str] = {
    # Anthropic style: claude-<variant>-<version>
    "claude-haiku-4-5": "anthropic--claude-4.5-haiku",
    "claude-sonnet-4-5": "anthropic--claude-4.5-sonnet",
    "claude-opus-4-5": "anthropic--claude-4.5-opus",
    "claude-opus-4-6": "anthropic--claude-4.6-opus",
    "claude-sonnet-4-6": "anthropic--claude-4.6-sonnet",
    # Claude shorthand without dots
    "claude-4.5-haiku": "anthropic--claude-4.5-haiku",
    "claude-4.5-sonnet": "anthropic--claude-4.5-sonnet",
    "claude-4.5-opus": "anthropic--claude-4.5-opus",
    "claude-4.6-opus": "anthropic--claude-4.6-opus",
    "claude-4.6-sonnet": "anthropic--claude-4.6-sonnet",
    # OpenAI shorthand
    "gpt5.4": "gpt-5.4",
    "gpt-5.4-turbo": "gpt-5.4",
}


def _resolve_alias(requested: str) -> str:
    """If requested model name is a known alias, return the canonical ID."""
    return MODEL_ALIASES.get(requested.lower(), requested)


def _parse_model_effort(model_name: str) -> tuple[str, str | None]:
    """Parse reasoning effort suffix from model name.

    Supports the convention model_name:effort (e.g. 'gpt-5.4:high').
    Valid effort values: low, medium, high, xhigh.
    Returns (model_name_without_effort, effort_or_None).
    """
    if ':' not in model_name:
        return model_name, None
    # Split on the LAST colon to handle model IDs with colons
    # (though SAP IDs don't contain colons)
    base, effort = model_name.rsplit(':', 1)
    effort_low = effort.lower()
    if effort_low in ('low', 'medium', 'high', 'xhigh'):
        return base, effort_low
    # Not a valid effort suffix — treat the whole string as model name
    # (e.g. 'some-model:other' where :other is not an effort level)
    return model_name, None


def _looks_like_model_id(name: str) -> bool:
    """Check if a name looks like a real SAP/OpenAI model ID.

    Rejects completely random strings that would just produce a confusing
    SAP 502 error later.  Accepts known prefix patterns and alias names.
    """
    low = name.lower()
    # Known SAP canonical prefixes: vendor--model
    for prefix in ("anthropic--", "google--", "meta--", "mistral--", "amazon--", "microsoft--"):
        if low.startswith(prefix):
            return True
    # OpenAI GPT family
    if low.startswith("gpt-"):
        return True
    # Known alias names
    if low in MODEL_ALIASES:
        return True
    return False


# ── Model resolution ─────────────────────────────────────────────────────────

def resolve_model(models: list[SupportedModel], requested: str) -> SupportedModel | None:
    # Strip :effort suffix before model lookup
    base, _effort = _parse_model_effort(requested)
    requested_lower = _resolve_alias(base).lower()
    for model in models:
        if model.id.lower() == requested_lower:
            return model
    short = requested_lower.split("--", 1)[-1]
    for model in models:
        if model.id.lower().split("--", 1)[-1] == short:
            return model
    return None


def resolve_model_cached(requested: str, username: str = "", password: str = "", base_url: str | None = None) -> SupportedModel | None:
    """Resolve a model name using cached metadata without requiring credentials.

    When metadata is available and the model is found: returns it with real version.
    When metadata is available but model not found: returns None (unsupported).
    When metadata is unavailable: falls back to alias resolution + pattern check,
    accepting names that look like real model IDs with version "latest".
    base_url: per-credential launchpad URL (multi-user mode).
    """
    # Strip :effort suffix before model lookup (effort is parsed separately)
    base, _effort = _parse_model_effort(requested)
    if username and password:
        try:
            models = _get_cached_models(username, password, base_url=base_url)
            resolved = resolve_model(models, base)
            if resolved:
                return resolved
            # Metadata was available but model not found → genuinely unsupported
            return None
        except Exception:
            pass  # metadata unavailable, fall through to heuristic

    # Fallback when metadata is entirely unavailable:
    # Accept known aliases and names that look like real model IDs.
    # Reject random strings to avoid deferred SAP 502 errors.
    resolved_id = _resolve_alias(base)
    if resolved_id != base.lower():
        # Alias matched → definitely a known model
        return SupportedModel(id=resolved_id, version="latest", owned_by=_model_owned_by(resolved_id))
    if _looks_like_model_id(resolved_id):
        return SupportedModel(id=resolved_id, version="latest", owned_by=_model_owned_by(resolved_id))
    # Completely unknown name — reject early with clear 404
    return None


# ── Metadata extraction ─────────────────────────────────────────────────────

def extract_supported_models(payload: dict[str, Any]) -> list[SupportedModel]:
    parsed = SAPMetadataResponse.model_validate(payload)
    raw_models = parsed.globalLLMInfo.get("models", [])
    result: list[SupportedModel] = []
    for raw_entry in raw_models:
        entry = SAPModelEntry.model_validate(raw_entry)
        deprecated = bool(entry.metadata.get("deprecated", False))
        if deprecated or not _is_supported_model(entry.model):
            continue
        result.append(SupportedModel(id=entry.model, version=entry.version, owned_by=_model_owned_by(entry.model)))
    result.sort(key=lambda item: item.id)
    return result


# ── Metadata fetching & caching ──────────────────────────────────────────────

def _is_html_response_metadata(resp) -> bool:
    """Detect SAP session expiry returning HTML instead of JSON."""
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type and resp.status_code == 200:
        return True
    return False


def _fetch_metadata_payload(username: str, password: str, base_url: str | None = None) -> dict[str, Any]:
    resolved_base = (base_url or settings.sap_base_url).rstrip("/")
    metadata_url = (
        f"{resolved_base}/aic/llm/api/v1/metadataV2"
        f"?workspace={settings.completion.workspace}"
    )
    session, _ = get_cached_session_curl_cffi(username, password, base_url=resolved_base)
    resp = session.get(metadata_url)
    if resp.status_code in {401, 403} or _is_html_response_metadata(resp):
        session, _ = get_cached_session_curl_cffi(username, password, force_refresh=True, base_url=resolved_base)
        resp = session.get(metadata_url)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise SAPMetadataError("metadata response was not an object")
    return data


_metadata_cache_lock = threading.Lock()
_metadata_cache: dict[str, tuple[float, list[SupportedModel]]] = {}
METADATA_CACHE_TTL = 1800  # 30 minutes


def _get_cached_models(username: str, password: str, base_url: str | None = None) -> list[SupportedModel]:
    """Get supported models with TTL cache — no live completion probe."""
    resolved_base = (base_url or settings.sap_base_url).rstrip("/")
    cache_key = f"{username}@{resolved_base}"
    now = time.time()
    with _metadata_cache_lock:
        cached = _metadata_cache.get(cache_key)
        if cached and (now - cached[0]) < METADATA_CACHE_TTL:
            return cached[1]

    payload = _fetch_metadata_payload(username, password, base_url=resolved_base)
    models = extract_supported_models(payload)

    with _metadata_cache_lock:
        _metadata_cache[cache_key] = (now, models)
    return models


def invalidate_metadata_cache() -> None:
    """Force next /v1/models call to re-fetch from SAP."""
    with _metadata_cache_lock:
        _metadata_cache.clear()


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_supported_models(username: str, password: str, base_url: str | None = None) -> list[SupportedModel]:
    """List supported models from cached metadata — no live completion probe.

    Access probing is only done via /debug/models, not on every /v1/models call.
    base_url: per-credential launchpad URL (multi-user mode).
    """
    try:
        return _get_cached_models(username, password, base_url=base_url)
    except SAPMetadataError:
        raise
    except Exception as exc:
        raise SAPMetadataError("failed to fetch supported models") from exc


def inspect_supported_models(username: str, password: str, base_url: str | None = None, deployment_id: str | None = None, resource_group_id: str | None = None) -> list[ModelDebugEntry]:
    try:
        payload = _fetch_metadata_payload(username, password, base_url=base_url)
        parsed = SAPMetadataResponse.model_validate(payload)
        raw_models = parsed.globalLLMInfo.get("models", [])
        entries: list[ModelDebugEntry] = []
        allowed_set = {item.lower() for item in settings.allowed_models}
        for raw_entry in raw_models:
            entry = SAPModelEntry.model_validate(raw_entry)
            deprecated = bool(entry.metadata.get("deprecated", False))
            # Empty allowlist = accept all (consistent with _is_supported_model)
            configured = (not allowed_set) or (entry.model.lower() in allowed_set)
            metadata_match = configured and not deprecated
            if not configured:
                continue
            access = check_model_access_with_password_curl_cffi(
                username, password, entry.model, entry.version,
                base_url=base_url,
                deployment_id=deployment_id,
                resource_group_id=resource_group_id,
            )
            entries.append(
                ModelDebugEntry(
                    id=entry.model,
                    version=entry.version,
                    owned_by=_model_owned_by(entry.model),
                    configured=configured if allowed_set else True,
                    metadata_match=metadata_match,
                    access_allowed=access.allowed,
                    access_status_code=access.status_code,
                    access_detail=access.detail,
                )
            )
        entries.sort(key=lambda item: item.id)
        return entries
    except Exception as exc:
        raise SAPMetadataError("failed to inspect supported models") from exc
