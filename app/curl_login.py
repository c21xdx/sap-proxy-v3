from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests
from pydantic import BaseModel

from app.config import settings


ACCOUNTS_AUTHORIZE = "https://academy.accounts.ondemand.com/oauth2/authorize"
logger = logging.getLogger(__name__)


class SAPProxyError(Exception):
    pass


class SAPLoginError(SAPProxyError):
    pass


class SAPSessionError(SAPProxyError):
    pass


class SAPMetadataError(SAPProxyError):
    pass


class SAPCompletionError(SAPProxyError):
    pass


class CompletionRequest(BaseModel):
    prompt_text: str
    model_name: str
    model_version: str
    max_tokens: int
    temperature: float | None
    deployment_id: str
    workspace: str
    resource_group_id: str
    stream_enabled: bool = True
    messages_history: list[dict] | None = None
    template_messages: list[dict] | None = None  # [{role, content, tool_calls?}, ...]
    tools: list[dict] | None = None  # native tools in SAP format
    has_images: bool = False  # True when request contains image_url blocks
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_settings(cls) -> "CompletionRequest":
        return cls(
            prompt_text=settings.completion.prompt_text,
            model_name=settings.completion.model.name,
            model_version=settings.completion.model.version,
            max_tokens=settings.completion.model.max_tokens,
            temperature=settings.completion.model.temperature,
            deployment_id=settings.completion.deployment_id,
            workspace=settings.completion.workspace,
            resource_group_id=settings.completion.resource_group_id,
            stream_enabled=settings.completion.stream_enabled,
        )


class CurlLoginResult(BaseModel):
    final_url: str
    user_api_status: int
    user_api_content_type: str
    csrf_status: int
    csrf_token_present: bool


class CurlMetadataResult(BaseModel):
    final_url: str
    metadata_status: int
    metadata_content_type: str
    model_count: int


class CurlCompletionResult(BaseModel):
    final_url: str
    completion_status: int
    completion_content_type: str
    csrf_token_present: bool
    response_preview: str


class CompletionHTTPResult(BaseModel):
    final_url: str
    status_code: int
    content_type: str
    csrf_token_present: bool
    body: str
    stream_resp: Any | None = None  # curl_cffi raw response for true streaming


class CachedSession(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    session: Any
    final_url: str
    created_at: float
    username: str
    base_url: str


class ModelAccessResult(BaseModel):
    allowed: bool
    checked_at: float
    status_code: int
    detail: str = ""


class SessionCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached: dict[tuple[str, str], CachedSession] = {}

    def clear(self) -> None:
        with self._lock:
            self._cached.clear()

    def clear_key(self, username: str, base_url: str | None = None) -> None:
        key = self._key(username, base_url or settings.sap_base_url)
        with self._lock:
            self._cached.pop(key, None)

    def get_existing(self, username: str, base_url: str | None = None) -> CachedSession | None:
        resolved_base_url = (base_url or settings.sap_base_url).rstrip("/")
        key = self._key(username, resolved_base_url)
        with self._lock:
            return self._cached.get(key)

    def store(self, cached: CachedSession) -> CachedSession:
        with self._lock:
            self._cached[self._key(cached.username, cached.base_url)] = cached
            return cached

    def get(
        self,
        username: str,
        password: str,
        base_url: str | None = None,
        login_entry_url: str | None = None,
        force_refresh: bool = False,
    ) -> CachedSession:
        resolved_base_url = (base_url or settings.sap_base_url).rstrip("/")
        key = self._key(username, resolved_base_url)
        with self._lock:
            cached = self._cached.get(key)
            if not force_refresh and cached and not self._is_expired(cached):
                return cached
        session, final_url = create_logged_in_session_curl_cffi(
            username,
            password,
            base_url=resolved_base_url,
            login_entry_url=login_entry_url,
        )
        cached = CachedSession(
            session=session,
            final_url=final_url,
            created_at=time.time(),
            username=username,
            base_url=resolved_base_url,
        )
        return self.store(cached)

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "username": cached.username,
                    "base_url": cached.base_url,
                    "age_seconds": max(0, int(time.time() - cached.created_at)),
                    "expires_in_seconds": max(0, settings.session_ttl_seconds - int(time.time() - cached.created_at)),
                    "final_url": cached.final_url,
                }
                for cached in self._cached.values()
            ]

    def has_any(self) -> bool:
        with self._lock:
            return bool(self._cached)

    def _is_expired(self, cached: CachedSession) -> bool:
        return (time.time() - cached.created_at) >= settings.session_ttl_seconds

    def _key(self, username: str, base_url: str) -> tuple[str, str]:
        return username, base_url.rstrip("/")


session_cache = SessionCache()
_model_access_cache: dict[tuple[str, str, str], ModelAccessResult] = {}
_model_access_lock = threading.Lock()
MODEL_ACCESS_TTL_SECONDS = 1800



def _extract_inputs(text: str) -> dict[str, str]:
    soup = BeautifulSoup(text, "html.parser")
    fields: dict[str, str] = {}
    for node in soup.find_all("input"):
        name = node.get("name")
        if not name:
            continue
        fields[name] = node.get("value", "")
    return fields



def _extract_launchpad_bootstrap(text: str) -> tuple[str, str]:
    normalized = " ".join(text.split())
    sig_match = re.search(r'document\.cookie="signature=([^"]+);path=/;"', normalized)
    loc_match = re.search(r'location="([^"]+)"', normalized)
    if not sig_match or not loc_match:
        raise ValueError("failed to parse launchpad bootstrap script")
    return sig_match.group(1), loc_match.group(1)



def _extract_accounts_authorize(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")

    for link in soup.find_all("a", href=True):
        href = html.unescape(link["href"])
        if href.startswith(ACCOUNTS_AUTHORIZE):
            return href

    meta = soup.find("meta", attrs={"name": "redirect"})
    if meta and meta.get("content"):
        return html.unescape(meta["content"])

    href_match = re.search(r'href=["\'](https://academy\.accounts\.ondemand\.com/oauth2/authorize[^"\']+)["\']', text)
    if href_match:
        return html.unescape(href_match.group(1))

    raise ValueError("failed to locate accounts authorize url")



def _launchpad_cookie_domain(base_url: str) -> str:
    domain = urlparse(base_url).hostname
    if not domain:
        raise ValueError(f"invalid launchpad base url: {base_url}")
    return domain



def _build_metadata_url(base_url: str, workspace: str, resource_group_id: str) -> str:
    return (
        f"{base_url}/aic/llm/api/v1/metadataV2"
        f"?workspace={workspace}&resourceGroupId={resource_group_id}"
    )



def _build_completion_url(base_url: str, workspace: str, resource_group_id: str) -> str:
    return (
        f"{base_url}/aic/llm/api/v1/completionV2"
        f"?workspace={workspace}&resourceGroupId={resource_group_id}"
    )



# Parameters that SAP completionV2 actually accepts per model family
# Based on SAP error messages: completionV2 rejects unsupported params
# NOTE: capability functions (_supports_reasoning_effort, etc.) are in
# app.model_registry — imported lazily to avoid circular dependency.

_SAP_SUPPORTED_PARAMS: dict[str, set[str]] = {
    # Claude models (aws-bedrock): max_tokens, temperature (4.5/4.6 only),
    # thinking, output_config (4.6+ adaptive)
    "anthropic": {"max_tokens", "temperature", "thinking", "output_config"},
    # GPT models (azure-openai): max_completion_tokens, reasoning_effort (5.x/o-series)
    "openai": {"max_completion_tokens", "reasoning_effort"},
    # Default: just max_tokens
    "_default": {"max_tokens", "max_completion_tokens", "temperature"},
}


def _filter_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Filter params to only those SAP completionV2 accepts for this model."""
    from app.model_registry import _claude_deprecates_temperature, _supports_reasoning_effort
    if "--" in model_name:
        owner = model_name.split("--", 1)[0]
    elif model_name.startswith(("gpt", "o1", "o3", "o4")):
        owner = "openai"
    else:
        owner = "_default"
    allowed = _SAP_SUPPORTED_PARAMS.get(owner, _SAP_SUPPORTED_PARAMS["_default"])
    filtered = {k: v for k, v in params.items() if k in allowed}
    dropped = set(params.keys()) - allowed
    # Extra: reasoning_effort only for GPT-5.x / o-series
    if "reasoning_effort" in filtered and not _supports_reasoning_effort(model_name):
        del filtered["reasoning_effort"]
        dropped.add("reasoning_effort")
    # Extra: temperature deprecated for Claude 4.7+
    if "temperature" in filtered and _claude_deprecates_temperature(model_name):
        del filtered["temperature"]
        dropped.add("temperature")
    # Extra: thinking/output_config only for Claude models
    if owner != "anthropic":
        for key in ("thinking", "output_config"):
            if key in filtered:
                del filtered[key]
                dropped.add(key)
    if dropped:
        logger.warning("Filtered unsupported params for %s: %s", model_name, sorted(dropped))
    return filtered


def _build_model_params(model_name: str, max_tokens: int, temperature: float | None,
                          top_p: float | None = None,
                          frequency_penalty: float | None = None,
                          presence_penalty: float | None = None,
                          stop: str | list[str] | None = None,
                          n: int | None = None,
                          reasoning_effort: str | None = None) -> dict:
    from app.model_registry import (
        _claude_deprecates_temperature,
        _claude_supports_adaptive_thinking,
        _claude_supports_enabled_thinking,
    )
    params: dict[str, Any] = {}
    # GPT-5.x and o-series use max_completion_tokens (no temperature)
    is_openai_reasoning = model_name.startswith("gpt-5") or model_name.startswith(("o1", "o3", "o4"))
    is_claude = model_name.startswith("anthropic--claude-")
    if is_openai_reasoning:
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
        # Claude 4.7+ deprecates temperature; also, when thinking is enabled
        # for any Claude model, temperature must be 1 or absent
        if temperature is not None and not _claude_deprecates_temperature(model_name) and not is_claude:
            params["temperature"] = temperature
        # For Claude without effort: allow temperature normally (4.5/4.6)
        if temperature is not None and is_claude and reasoning_effort is None and not _claude_deprecates_temperature(model_name):
            params["temperature"] = temperature
    # OpenAI reasoning_effort (gpt-5.x / o-series)
    if reasoning_effort is not None and is_openai_reasoning:
        params["reasoning_effort"] = reasoning_effort
    # Claude thinking + effort (4.6+ adaptive, 4.5 enabled fallback)
    if reasoning_effort is not None and is_claude:
        if _claude_supports_adaptive_thinking(model_name):
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": reasoning_effort}
        elif _claude_supports_enabled_thinking(model_name):
            # Claude 4.5: map effort to budget_tokens approximation
            budget_map = {"low": 1024, "medium": 4096, "high": 16000, "xhigh": 32000}
            budget = budget_map.get(reasoning_effort, 4096)
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Ensure max_tokens > budget_tokens (SAP requirement)
            if params.get("max_tokens", 0) <= budget:
                params["max_tokens"] = budget + 1024
    if top_p is not None:
        params["top_p"] = top_p
    if frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    if stop is not None:
        params["stop"] = stop
    if n is not None:
        params["n"] = n
    # Filter to params SAP completionV2 actually accepts
    return _filter_model_params(model_name, params)



def _build_template_entry(msg: dict) -> dict:
    """Build a single template entry from a template message dict.

    Handles text content, image_url blocks, tool_calls, and tool results.
    Returns entry in SAP completionV2 format.
    """
    entry = {"role": msg["role"]}
    if msg["role"] == "tool":
        entry["content"] = msg.get("content", "")
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = msg["tool_call_id"]
        return entry

    if "tool_calls" in msg:
        entry["content"] = msg.get("content", "")
        entry["tool_calls"] = msg["tool_calls"]
        return entry

    # user/system messages: build content array
    raw_content = msg.get("content", "")

    # If content is already a list of content blocks (from _build_template_messages),
    # pass through image_url blocks and wrap text blocks
    if isinstance(raw_content, list):
        content_blocks = []
        for block in raw_content:
            if isinstance(block, dict):
                btype = block.get("type", "text")
                if btype == "image_url":
                    # Pass through image_url blocks as-is (SAP supports this)
                    content_blocks.append(block)
                elif btype == "text":
                    content_blocks.append({"type": "text", "text": block.get("text", "")})
                # else: skip unknown block types
            elif isinstance(block, str):
                content_blocks.append({"type": "text", "text": block})
        entry["content"] = content_blocks
    else:
        # Plain string content
        entry["content"] = [{"type": "text", "text": str(raw_content)}]

    return entry


def _log_payload_structure(payload: dict, has_images: bool) -> None:
    """Log payload structure for debugging (avoids logging huge base64 data)."""
    try:
        config = payload.get("config", {})
        modules = config.get("modules", {})
        pt = modules.get("prompt_templating", {})
        prompt = pt.get("prompt", {})
        template = prompt.get("template", [])
        template_summary = []
        for entry in template:
            role = entry.get("role", "NO_ROLE")
            content = entry.get("content")
            tc = entry.get("tool_calls")
            if isinstance(content, list):
                parts = []
                for b in content:
                    btype = b.get("type", "?")
                    if btype == "image_url":
                        url = b.get("image_url", {}).get("url", "")
                        parts.append(f"image_url({len(url)} chars)")
                    else:
                        parts.append(btype)
                content_desc = ",".join(parts)
            elif isinstance(content, str):
                content_desc = f'text({len(content)})'
            else:
                content_desc = str(type(content))
            tc_desc = f"+{len(tc)} tool_calls" if tc else ""
            template_summary.append(f"{role}: {content_desc}{tc_desc}")
        has_hist = "messages_history" in payload
        hist_count = len(payload.get("messages_history", [])) if has_hist else 0
        has_stream = "stream" in config
        has_pv = "placeholder_values" in payload
        logger.info(
            "SAP payload: mode=%s template=%d entries[%s] history=%s(%d) stream=%s pv=%s",
            "multimodal" if has_images else "text",
            len(template),
            " | ".join(template_summary) if template_summary else "empty",
            "present" if has_hist else "absent",
            hist_count,
            str(config.get("stream", {}).get("enabled", "?")) if has_stream else "absent",
            "present" if has_pv else "absent",
        )
    except Exception:
        pass  # debug logging, never break the request


def _build_completion_payload(req: CompletionRequest) -> dict:
    """Build SAP completionV2 payload.

    Two modes:
    Two modes:
    - Text mode (default): model at prompt_templating level, supports
      messages_history and stream
    - Multimodal mode (has_images): model at prompt_templating level,
      no messages_history/placeholder_values, image_url blocks in template

    In both modes, model is at prompt_templating level (parallel to prompt),
    NOT inside the prompt object. SAP rejects 'model' inside prompt with:
    "Additional properties are not allowed ('model' was unexpected)".
    """
    # Build template entries
    if req.template_messages:
        template_entries = [_build_template_entry(msg) for msg in req.template_messages]
        prompt_obj = {
            "defaults": {},
            "template": template_entries,
        }
        # Add native tools if present
        if req.tools:
            prompt_obj["tools"] = req.tools
    else:
        # Legacy fallback: flat text in template
        prompt_obj = {
            "defaults": {},
            "template": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": req.prompt_text}],
                }
            ],
        }

    model_spec = {
        "name": req.model_name,
        "params": _build_model_params(req.model_name, req.max_tokens, req.temperature,
                                           top_p=req.top_p,
                                           frequency_penalty=req.frequency_penalty,
                                           presence_penalty=req.presence_penalty,
                                           stop=req.stop,
                                           n=req.n,
                                           reasoning_effort=req.reasoning_effort),
        "version": req.model_version,
    }

    # Model is ALWAYS at prompt_templating level (parallel to prompt).
    # SAP rejects model inside prompt object.
    pt_module = {
        "prompt": prompt_obj,
        "model": model_spec,
    }

    if req.has_images:
        # Multimodal/hybrid mode.
        # Pure multimodal (no messages_history): template has full conversation.
        # Hybrid (has messages_history): template has images, history has tool conversation.
        #
        # SAP docs say multimodal mode doesn't support messages_history,
        # but in practice, sending both works when the template only has
        # image_url content (no tool_calls) and history only has text/tool msgs.
        base = {
            "config": {
                "modules": {
                    "prompt_templating": pt_module,
                },
            },
        }
        if req.messages_history:
            # Hybrid mode: images in template, tool conversation in history
            base["messages_history"] = req.messages_history
            base["placeholder_values"] = {}
            # Stream is downgraded for multimodal, but include for completeness
            base["config"]["stream"] = {"enabled": False}
        return base
    else:
        # Text mode: supports stream, messages_history, placeholder_values
        return {
            "config": {
                "modules": {
                    "prompt_templating": pt_module,
                },
                "stream": {"enabled": req.stream_enabled},
            },
            "placeholder_values": {},
            "messages_history": req.messages_history or [],
        }



def login_with_password_curl_cffi(username: str, password: str, base_url: str | None = None) -> CurlLoginResult:
    session, final_url = create_logged_in_session_curl_cffi(username, password, base_url=base_url)
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    user_api = session.get(f"{launchpad}/aic/api/v1/user")
    csrf = session.head(
        f"{launchpad}/aic/runtime/api/v1/workspaces",
        headers={"X-CSRF-Token": "fetch"},
    )

    return CurlLoginResult(
        final_url=final_url,
        user_api_status=user_api.status_code,
        user_api_content_type=user_api.headers.get("content-type", ""),
        csrf_status=csrf.status_code,
        csrf_token_present=bool(csrf.headers.get("x-csrf-token")),
    )



def create_logged_in_session_curl_cffi(
    username: str,
    password: str,
    base_url: str | None = None,
    login_entry_url: str | None = None,
) -> tuple[requests.Session, str]:
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    entry_url = login_entry_url or f"{launchpad}{settings.sap_login_entry_path}"
    cookie_domain = _launchpad_cookie_domain(launchpad)

    session = requests.Session(impersonate="chrome136")
    session.headers.update({"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})

    try:
        launchpad_index = session.get(entry_url)
        launchpad_index.raise_for_status()
        signature, authorize_url = _extract_launchpad_bootstrap(launchpad_index.text)
        session.cookies.set("fragmentAfterLogin", "", domain=cookie_domain, path="/")
        session.cookies.set("locationAfterLogin", "%2Faic%2Findex.html", domain=cookie_domain, path="/")
        session.cookies.set("signature", signature, domain=cookie_domain, path="/")

        uaa_login = session.get(authorize_url, allow_redirects=True)
        uaa_login.raise_for_status()
        accounts_authorize_url = _extract_accounts_authorize(uaa_login.text)

        accounts_page = session.get(accounts_authorize_url)
        accounts_page.raise_for_status()
        fields = _extract_inputs(accounts_page.text)
        form = {k: v for k, v in fields.items() if k not in {"j_username", "j_password"}}
        form["j_username"] = username
        form["j_password"] = password

        xsrf = None
        for cookie in session.cookies.jar:
            if "XSRF" in cookie.name.upper():
                xsrf = cookie.value
                break
        headers = {"content-type": "application/x-www-form-urlencoded"}
        if xsrf:
            headers["X-XSRF-Token"] = xsrf

        resp = session.post(ACCOUNTS_AUTHORIZE, data=form, headers=headers, allow_redirects=False)
        resp.raise_for_status()

        next_url = resp.headers.get("Location") or resp.headers.get("location")
        last = resp
        for _ in range(10):
            if not next_url:
                break
            if next_url.startswith("/"):
                next_url = urljoin(last.url, next_url)
            last = session.get(next_url, allow_redirects=False)
            next_url = last.headers.get("Location") or last.headers.get("location")

        if next_url:
            if next_url.startswith("/"):
                next_url = urljoin(last.url, next_url)
            last = session.get(next_url, allow_redirects=True)
    except Exception as exc:
        logger.exception("sap login failed", extra={"base_url": launchpad})
        raise SAPLoginError(f"sap login failed for {launchpad}") from exc

    logger.info("sap login succeeded", extra={"base_url": launchpad})
    return session, last.url



def is_session_usable(session: requests.Session, base_url: str | None = None) -> bool:
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    try:
        resp = session.get(f"{launchpad}/aic/api/v1/user")
    except Exception:
        logger.warning("session health check failed with exception", extra={"base_url": launchpad})
        return False
    usable = resp.status_code == 200
    if not usable:
        logger.info("session health check failed", extra={"base_url": launchpad, "status_code": resp.status_code})
    return usable



def get_cached_session_curl_cffi(
    username: str,
    password: str,
    base_url: str | None = None,
    login_entry_url: str | None = None,
    force_refresh: bool = False,
    validate: bool = True,
) -> tuple[requests.Session, str]:
    resolved_base_url = (base_url or settings.sap_base_url).rstrip("/")
    cached = session_cache.get_existing(username, resolved_base_url)
    if not force_refresh and cached and not session_cache._is_expired(cached):
        if not validate or is_session_usable(cached.session, resolved_base_url):
            return cached.session, cached.final_url
        logger.info("dropping unusable cached session", extra={"base_url": resolved_base_url})
        session_cache.clear_key(username, resolved_base_url)

    try:
        cached = session_cache.get(
            username,
            password,
            base_url=resolved_base_url,
            login_entry_url=login_entry_url,
            force_refresh=True if force_refresh else False,
        )
    except SAPLoginError as exc:
        raise SAPSessionError(f"failed to obtain usable SAP session for {resolved_base_url}") from exc
    return cached.session, cached.final_url



def fetch_metadata_with_password_curl_cffi(
    username: str,
    password: str,
    base_url: str | None = None,
) -> CurlMetadataResult:
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    try:
        session, final_url = get_cached_session_curl_cffi(username, password, base_url=launchpad)
        resp = session.get(
            _build_metadata_url(
                launchpad,
                settings.completion.workspace,
                settings.completion.resource_group_id,
            )
        )
        if resp.status_code in {401, 403}:
            logger.info("metadata request got auth failure, refreshing session", extra={"base_url": launchpad, "status_code": resp.status_code})
            session, final_url = get_cached_session_curl_cffi(username, password, base_url=launchpad, force_refresh=True)
            resp = session.get(
                _build_metadata_url(
                    launchpad,
                    settings.completion.workspace,
                    settings.completion.resource_group_id,
                )
            )
    except SAPSessionError:
        raise
    except Exception as exc:
        logger.exception("metadata request failed", extra={"base_url": launchpad})
        raise SAPMetadataError(f"metadata request failed for {launchpad}") from exc

    if resp.status_code != 200:
        logger.warning("metadata request returned non-200", extra={"base_url": launchpad, "status_code": resp.status_code})
        raise SAPMetadataError(f"metadata request failed with status {resp.status_code}")

    model_count = 0
    try:
        data = resp.json()
        model_count = len(data.get("globalLLMInfo", {}).get("models", []))
    except Exception:
        logger.warning("metadata response was not valid json", extra={"base_url": launchpad})
    return CurlMetadataResult(
        final_url=final_url,
        metadata_status=resp.status_code,
        metadata_content_type=resp.headers.get("content-type", ""),
        model_count=model_count,
    )



def _is_html_response(resp) -> bool:
    """Detect SAP's session-expiry response: HTTP 200 with HTML login page."""
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type and resp.status_code == 200:
        body = resp.text[:500].lower()
        if "<html" in body or "<head" in body or "login" in body or "<form" in body:
            return True
    return False


def _is_html_response_status_only(resp) -> bool:
    """Detect SAP's session-expiry by content-type header only (no body read).
    Used for streaming responses where reading the body would consume the stream.
    """
    content_type = resp.headers.get("content-type", "")
    return "text/html" in content_type and resp.status_code == 200


SAP_COMPLETION_TIMEOUT = int(os.getenv("SAP_COMPLETION_TIMEOUT", "300"))  # 5 minutes


def _do_completion_post(session, completion_url: str, payload: dict, headers: dict, stream_enabled: bool):
    """POST to completion URL. For stream requests, use curl_cffi stream mode."""
    timeout = SAP_COMPLETION_TIMEOUT
    if stream_enabled:
        resp = session.post(
            completion_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
    else:
        resp = session.post(
            completion_url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    return resp


def execute_completion_with_password_curl_cffi(
    username: str,
    password: str,
    request: CompletionRequest | None = None,
    base_url: str | None = None,
) -> CompletionHTTPResult:
    req = request or CompletionRequest.from_settings()
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    try:
        session, final_url = get_cached_session_curl_cffi(username, password, base_url=launchpad)

        csrf = session.head(
            f"{launchpad}/aic/runtime/api/v1/workspaces",
            headers={"X-CSRF-Token": "fetch"},
        )
        token = csrf.headers.get("x-csrf-token", "")
        if csrf.status_code in {401, 403} or not token:
            logger.info("csrf fetch failed, refreshing session", extra={"base_url": launchpad, "status_code": csrf.status_code})
            session, final_url = get_cached_session_curl_cffi(username, password, base_url=launchpad, force_refresh=True)
            csrf = session.head(
                f"{launchpad}/aic/runtime/api/v1/workspaces",
                headers={"X-CSRF-Token": "fetch"},
            )
            token = csrf.headers.get("x-csrf-token", "")

        payload = _build_completion_payload(req)
        # DEBUG: log payload structure (not full body — can be huge with base64 images)
        _log_payload_structure(payload, req.has_images)
        headers = {
            "Content-Type": "application/octet-stream",
            "deployment-id": req.deployment_id,
            "is-stream": str(req.stream_enabled).lower(),
            "Referer": f"{launchpad}{settings.sap_login_entry_path}",
            "Origin": launchpad,
            "X-CSRF-Token": token,
        }

        completion_url = _build_completion_url(
            launchpad,
            req.workspace,
            req.resource_group_id,
        )
        resp = _do_completion_post(session, completion_url, payload, headers, req.stream_enabled)

        # Retry on auth failure (401/403) or SAP session expiry (200+HTML)
        if resp.status_code in {401, 403} or (_is_html_response_status_only(resp) and resp.status_code == 200):
            if hasattr(resp, 'close'):
                resp.close()
            reason = "auth failure" if resp.status_code in {401, 403} else "session expired (200+HTML)"
            logger.info(f"completion request got {reason}, refreshing session", extra={"base_url": launchpad, "status_code": resp.status_code})
            session, final_url = get_cached_session_curl_cffi(username, password, base_url=launchpad, force_refresh=True)
            csrf = session.head(
                f"{launchpad}/aic/runtime/api/v1/workspaces",
                headers={"X-CSRF-Token": "fetch"},
            )
            token = csrf.headers.get("x-csrf-token", "")
            headers["X-CSRF-Token"] = token
            resp = _do_completion_post(session, completion_url, payload, headers, req.stream_enabled)
            if _is_html_response_status_only(resp) and resp.status_code == 200:
                if hasattr(resp, 'close'):
                    resp.close()
                raise SAPCompletionError("SAP returned HTML after session refresh — login may be broken")
    except SAPSessionError:
        raise
    except SAPCompletionError:
        raise
    except Exception as exc:
        logger.exception("completion request failed", extra={"base_url": launchpad, "model_name": req.model_name})
        raise SAPCompletionError(f"completion request failed for {launchpad}") from exc

    if resp.status_code != 200:
        error_body = ""
        try:
            error_body = resp.text[:1000]
        except Exception:
            pass
        if hasattr(resp, 'close'):
            resp.close()
        logger.warning(
            "completion request returned non-200",
            extra={"base_url": launchpad, "status_code": resp.status_code, "model_name": req.model_name, "error_body": error_body},
        )
        raise SAPCompletionError(f"completion request failed with status {resp.status_code}: {error_body}")

    # For streaming responses, return the raw response object for real-time forwarding
    if req.stream_enabled:
        return CompletionHTTPResult(
            final_url=final_url,
            status_code=resp.status_code,
            content_type=resp.headers.get("content-type", ""),
            csrf_token_present=bool(token),
            body="",  # body will be read lazily from stream_resp
            stream_resp=resp,
        )

    # Non-streaming: read full body immediately
    body = resp.text
    return CompletionHTTPResult(
        final_url=final_url,
        status_code=resp.status_code,
        content_type=resp.headers.get("content-type", ""),
        csrf_token_present=bool(token),
        body=body,
    )



def check_model_access_with_password_curl_cffi(
    username: str,
    password: str,
    model_name: str,
    model_version: str,
    base_url: str | None = None,
    deployment_id: str | None = None,
    resource_group_id: str | None = None,
    force_refresh: bool = False,
) -> ModelAccessResult:
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    cache_key = (username, launchpad, f"{model_name}:{model_version}")
    with _model_access_lock:
        cached = _model_access_cache.get(cache_key)
        if cached and not force_refresh and (time.time() - cached.checked_at) < MODEL_ACCESS_TTL_SECONDS:
            return cached

    req = CompletionRequest(
        prompt_text="Reply with exactly OK.",
        model_name=model_name,
        model_version=model_version,
        max_tokens=16,
        temperature=None if model_name.startswith(("gpt-5", "o1", "o3", "o4")) else 0.2,
        deployment_id=deployment_id or settings.completion.deployment_id,
        workspace=settings.completion.workspace,
        resource_group_id=resource_group_id or settings.completion.resource_group_id,
        stream_enabled=False,
    )
    try:
        resp = execute_completion_with_password_curl_cffi(username, password, request=req, base_url=launchpad)
        result = ModelAccessResult(
            allowed=True,
            checked_at=time.time(),
            status_code=resp.status_code,
        )
    except SAPCompletionError as exc:
        message = str(exc)
        status_code = 403 if "status 403" in message else 502
        result = ModelAccessResult(
            allowed=False,
            checked_at=time.time(),
            status_code=status_code,
            detail=message,
        )
    with _model_access_lock:
        _model_access_cache[cache_key] = result
    return result



def fetch_completion_with_password_curl_cffi(
    username: str,
    password: str,
    request: CompletionRequest | None = None,
    base_url: str | None = None,
) -> CurlCompletionResult:
    resp = execute_completion_with_password_curl_cffi(
        username,
        password,
        request=request,
        base_url=base_url,
    )
    return CurlCompletionResult(
        final_url=resp.final_url,
        completion_status=resp.status_code,
        completion_content_type=resp.content_type,
        csrf_token_present=resp.csrf_token_present,
        response_preview=resp.body[:500],
    )


# --- Auto-discovery of deployment_id and resource_group_id ---

# Known resource group names to probe (trial instances typically have these)
# doc-grounding first: SAP trial default with grounding support
_PROBE_RESOURCE_GROUPS: list[str] = [
    s.strip()
    for s in os.getenv("SAP_RESOURCE_GROUPS", "doc-grounding,default").split(",")
    if s.strip()
]


# Per-credential deployment cache: (username, base_url) → (deployment_id, resource_group_id)
_deployment_cache_lock = threading.Lock()
_deployment_cache: dict[tuple[str, str], tuple[str, str]] = {}


def cache_deployment(username: str, base_url: str, deployment_id: str, resource_group_id: str) -> None:
    """Store a discovered deployment in the per-credential cache."""
    key = (username, base_url.rstrip("/"))
    with _deployment_cache_lock:
        _deployment_cache[key] = (deployment_id, resource_group_id)


def get_cached_deployment(username: str, base_url: str) -> tuple[str, str] | None:
    """Get cached deployment for a credential, or None if not yet discovered."""
    key = (username, base_url.rstrip("/"))
    with _deployment_cache_lock:
        return _deployment_cache.get(key)


def discover_deployment(username: str, password: str, base_url: str | None = None) -> tuple[str, str]:
    """Auto-discover a usable deployment_id and resource_group_id.

    Probes known resource groups, finds a RUNNING orchestration deployment.
    Returns (deployment_id, resource_group_id).

    Raises SAPSessionError if no suitable deployment found.
    """
    launchpad = (base_url or settings.sap_base_url).rstrip("/")
    session, _ = get_cached_session_curl_cffi(username, password, base_url=launchpad)
    workspace = settings.completion.workspace

    for rg in _PROBE_RESOURCE_GROUPS:
        url = f"{launchpad}/aic/llm/api/v1/deployments?workspace={workspace}&resourceGroupId={rg}"
        try:
            resp = session.get(url)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
        except Exception:
            continue
        resources = data.get("resources", [])
        # Prefer orchestration deployments (serve all models)
        running_orch = [
            d for d in resources
            if d.get("status") == "RUNNING" and d.get("scenarioId") == "orchestration"
        ]
        if running_orch:
            dep = running_orch[0]
            logger.info(
                f"Auto-discovered deployment: id={dep['id']} "
                f"name={dep.get('configurationName', '')} "
                f"resource_group={rg}",
            )
            return dep["id"], rg
        # Fallback: any RUNNING foundation-models deployment
        running_fm = [
            d for d in resources
            if d.get("status") == "RUNNING" and d.get("scenarioId") == "foundation-models"
        ]
        if running_fm:
            dep = running_fm[0]
            logger.info(
                f"Auto-discovered deployment (foundation-models): id={dep['id']} "
                f"name={dep.get('configurationName', '')} "
                f"resource_group={rg}",
            )
            return dep["id"], rg

    raise SAPSessionError(
        f"No RUNNING deployment found in resource groups: {_PROBE_RESOURCE_GROUPS}. "
        f"Set SAP_DEPLOYMENT_ID and SAP_RESOURCE_GROUP_ID manually."
    )
