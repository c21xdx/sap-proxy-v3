from __future__ import annotations

import json
import logging
import time
import uuid
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
for _mod in ('app.curl_login', 'app.openai_api', 'app.main'):
    logging.getLogger(_mod).setLevel(logging.INFO)

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings, SASCredential
from app.curl_login import (
    SAPCompletionError, SAPMetadataError, SAPSessionError,
    cache_deployment, discover_deployment, execute_completion_with_password_curl_cffi,
    get_cached_deployment, session_cache,
)
from app.model_registry import (
    MODEL_ALIASES,
    fetch_supported_models,
    inspect_supported_models,
    resolve_model_cached,
)
from app.openai_api import (
    OpenAIChatRequest,
    OpenAIModel,
    OpenAIModelList,
    build_openai_response_from_sap,
    build_openai_response_from_text,
    iter_openai_sse,
    validate_content_blocks,
    _to_completion_request,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="sap-proxy-v3", version="3.0.0")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.exception(
            "http request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "model": None,
                "elapsed_ms": elapsed_ms,
                "status_code": 500,
            },
        )
        raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "http request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "model": _response_model_hint(request, response),
            "elapsed_ms": elapsed_ms,
            "status_code": response.status_code,
        },
    )
    return response


@app.get("/health")
def health() -> dict:
    snapshot = session_cache.snapshot()
    return {
        "status": "ok",
        "project": "sap-proxy-v3",
        "session_ttl_seconds": settings.session_ttl_seconds,
        "session_cached": session_cache.has_any(),
        "session_cache_entries": len(snapshot),
    }


@app.get("/v1/models", response_model=OpenAIModelList)
def list_models(authorization: str | None = Header(default=None)) -> OpenAIModelList:
    cred = _resolve_credential(authorization)
    try:
        models = fetch_supported_models(cred.sap_user, cred.sap_pass, base_url=cred.base_url)
    except SAPSessionError as exc:
        raise HTTPException(status_code=502, detail=f"sap session error: {exc}") from exc
    except SAPMetadataError as exc:
        raise HTTPException(status_code=502, detail=f"sap metadata error: {exc}") from exc

    canonical_ids = {m.id for m in models}
    alias_entries: list[OpenAIModel] = []
    for alias, canonical in MODEL_ALIASES.items():
        if canonical in canonical_ids and alias not in canonical_ids:
            owner = canonical.split("--", 1)[0] if "--" in canonical else "openai"
            alias_entries.append(OpenAIModel(id=alias, owned_by=owner))

    return OpenAIModelList(
        data=alias_entries + [
            OpenAIModel(id=model.id, owned_by=model.owned_by)
            for model in models
        ]
    )


@app.post("/v1/chat/completions")
def chat_completions(payload: OpenAIChatRequest, authorization: str | None = Header(default=None)):
    cred = _resolve_credential(authorization)
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    content_error = validate_content_blocks(payload.messages)
    if content_error:
        raise HTTPException(status_code=400, detail=content_error)

    deployment_id, resource_group_id = _ensure_deployment(cred)

    resolved = resolve_model_cached(payload.model, cred.sap_user, cred.sap_pass, base_url=cred.base_url)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"unsupported model: {payload.model}")

    completion_request = _to_completion_request(payload, resolved.id, resolved.version, deployment_id=deployment_id, resource_group_id=resource_group_id)
    try:
        sap_response = execute_completion_with_password_curl_cffi(
            cred.sap_user,
            cred.sap_pass,
            request=completion_request,
            base_url=cred.base_url,
        )
    except SAPSessionError as exc:
        raise HTTPException(status_code=502, detail=f"sap session error: {exc}") from exc
    except SAPCompletionError as exc:
        raise HTTPException(status_code=502, detail=f"sap completion error: {exc}") from exc

    if payload.stream:
        stream = StreamingResponse(
            iter_openai_sse(resolved.id, sap_response, has_tools=bool(payload.tools)),
            media_type="text/event-stream",
        )
        stream.headers["X-Model"] = resolved.id
        return stream

    from app.openai_api import parse_sap_sse_tool_calls
    content, usage, native_tool_calls = parse_sap_sse_tool_calls(sap_response.body)
    if native_tool_calls:
        response = build_openai_response_from_sap(resolved.id, content, usage, native_tool_calls)
    else:
        response = build_openai_response_from_text(resolved.id, content, usage, has_tools=bool(payload.tools))
    return response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _response_model_hint(request: Request, response) -> str | None:
    if request.url.path != "/v1/chat/completions":
        return None
    model_header = response.headers.get("X-Model")
    if model_header:
        return model_header
    return None


def _resolve_credential(authorization: str | None) -> SASCredential:
    if settings.is_multi_user:
        if not authorization:
            raise HTTPException(status_code=401, detail="authorization required")
        token = authorization.removeprefix("Bearer ").strip()
        cred = settings.credentials.get(token)
        if not cred:
            raise HTTPException(status_code=401, detail="invalid api key")
        return cred
    if settings.api_key:
        expected = f"Bearer {settings.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid api key")
    return SASCredential(
        api_key=settings.api_key,
        sap_user=settings.sap_user,
        sap_pass=settings.sap_pass,
        base_url=settings.sap_base_url,
    )


def _ensure_deployment(cred: SASCredential) -> tuple[str, str]:
    cached = get_cached_deployment(cred.sap_user, cred.base_url)
    if cached:
        return cached
    if not settings.is_multi_user and settings.completion.deployment_id and settings.completion.resource_group_id:
        cache_deployment(cred.sap_user, cred.base_url, settings.completion.deployment_id, settings.completion.resource_group_id)
        return settings.completion.deployment_id, settings.completion.resource_group_id
    try:
        deployment_id, resource_group_id = discover_deployment(cred.sap_user, cred.sap_pass, base_url=cred.base_url)
        cache_deployment(cred.sap_user, cred.base_url, deployment_id, resource_group_id)
        logger.info(f"Auto-discovered for {cred.sap_user}: deployment_id={deployment_id}, resource_group_id={resource_group_id}")
        return deployment_id, resource_group_id
    except SAPSessionError as exc:
        if not settings.is_multi_user:
            logger.warning(f"Deployment auto-discovery failed: {exc}")
        raise HTTPException(status_code=502, detail=f"deployment auto-discovery failed: {exc}") from exc


def _mask_username(username: str) -> str:
    if not username:
        return ""
    if len(username) <= 2:
        return "*" * len(username)
    return f"{username[:1]}{'*' * (len(username) - 2)}{username[-1:]}"


def _mask_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "[redacted]"
    return f"{parsed.scheme}://{parsed.netloc}/[redacted]"


def _redact_session_entry(entry: dict) -> dict:
    return {
        **entry,
        "username": _mask_username(str(entry.get("username", ""))),
        "base_url": _mask_url(str(entry.get("base_url", ""))),
        "final_url": _mask_url(str(entry.get("final_url", ""))),
    }


@app.get("/debug/session")
def debug_session(authorization: str | None = Header(default=None)) -> dict:
    cred = _resolve_credential(authorization)
    all_entries = session_cache.snapshot()
    if settings.is_multi_user:
        filtered = [
            e for e in all_entries
            if e.get("username") == cred.sap_user
            and e.get("base_url", "").rstrip("/") == (cred.base_url or "").rstrip("/")
        ]
    else:
        filtered = all_entries
    return {
        "session_ttl_seconds": settings.session_ttl_seconds,
        "entries": [_redact_session_entry(entry) for entry in filtered],
    }


@app.get("/debug/models")
def debug_models(authorization: str | None = Header(default=None)) -> dict:
    cred = _resolve_credential(authorization)
    deployment_id, resource_group_id = _ensure_deployment(cred)
    try:
        entries = inspect_supported_models(
            cred.sap_user, cred.sap_pass,
            base_url=cred.base_url,
            deployment_id=deployment_id,
            resource_group_id=resource_group_id,
        )
    except SAPSessionError as exc:
        raise HTTPException(status_code=502, detail=f"sap session error: {exc}") from exc
    except SAPMetadataError as exc:
        raise HTTPException(status_code=502, detail=f"sap metadata error: {exc}") from exc
    return {
        "allowed_models": settings.allowed_models,
        "deployment_id": deployment_id,
        "resource_group_id": resource_group_id,
        "entries": [entry.model_dump() for entry in entries],
    }
