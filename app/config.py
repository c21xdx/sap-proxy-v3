from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urljoin

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


# When ALLOWED_MODELS is empty, all models from SAP metadata are accepted.
# When set, only listed models (comma-separated) are exposed.
DEFAULT_ALLOWED_MODELS: list[str] = []


def _parse_allowed_models() -> list[str]:
    raw = os.getenv("ALLOWED_MODELS", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class SASCredential:
    """A single SAP account credential tied to an API key."""
    api_key: str
    sap_user: str
    sap_pass: str
    base_url: str  # per-credential launchpad URL


def _parse_credentials() -> dict[str, SASCredential]:
    """Parse SAP_CREDENTIALS env var into {api_key: SASCredential} map.

    Format: api_key:sap_user:sap_pass[:base_url],...
    base_url is optional; falls back to SAP_BASE_URL.
    """
    raw = os.getenv("SAP_CREDENTIALS", "").strip()
    if not raw:
        return {}
    result: dict[str, SASCredential] = {}
    default_base = os.getenv("SAP_BASE_URL", "").rstrip("/")
    for entry in raw.split(","):
        parts = [p.strip() for p in entry.split(":", 3)]
        if len(parts) < 3:
            continue
        key, user, pwd = parts[0], parts[1], parts[2]
        base = parts[3].rstrip("/") if len(parts) > 3 and parts[3] else default_base
        if key and user and pwd and base:
            result[key] = SASCredential(api_key=key, sap_user=user, sap_pass=pwd, base_url=base)
    return result


class CompletionModelSettings(BaseModel):
    name: str = os.getenv("SAP_MODEL_NAME", "")
    version: str = os.getenv("SAP_MODEL_VERSION", "latest")
    max_tokens: int = int(os.getenv("SAP_MODEL_MAX_TOKENS", "16384"))
    temperature: float = float(os.getenv("SAP_MODEL_TEMPERATURE", "0.2"))


class CompletionSettings(BaseModel):
    workspace: str = os.getenv("SAP_WORKSPACE", "aicore")
    resource_group_id: str = os.getenv("SAP_RESOURCE_GROUP_ID", "")
    deployment_id: str = os.getenv("SAP_DEPLOYMENT_ID", "")
    stream_enabled: bool = os.getenv("SAP_STREAM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    # When true, always send non-stream requests to SAP but still return
    # SSE format to clients. More reliable for agents (full response
    # before emitting, accurate usage, no thinking leak, empty response
    # detectable). Client sees no difference — same SSE events, just
    # buffered.
    force_non_stream: bool = os.getenv("SAP_FORCE_NON_STREAM", "false").lower() in {"1", "true", "yes", "on"}
    prompt_text: str = os.getenv("SAP_PROMPT_TEXT", "请只回答OK")
    model: CompletionModelSettings = Field(default_factory=CompletionModelSettings)


class Settings(BaseModel):
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8011"))
    allowed_models: list[str] = Field(default_factory=_parse_allowed_models)
    sap_user: str = os.getenv("SAP_USER", "")
    sap_pass: str = os.getenv("SAP_PASS", "")
    api_key: str = os.getenv("API_KEY", "")
    cookie_file: str = os.getenv("COOKIE_FILE", "/tmp/sap-v2-cookies.json")
    sap_base_url: str = os.getenv(
        "SAP_BASE_URL",
        "https://your-sap-launchpad-url",
    ).rstrip("/")
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "20"))
    max_history_tokens: int = int(os.getenv("MAX_HISTORY_TOKENS", "100000"))
    sap_login_entry_path: str = os.getenv("SAP_LOGIN_ENTRY_PATH", "/aic/index.html")
    session_ttl_seconds: int = int(os.getenv("SAP_SESSION_TTL_SECONDS", "1500"))
    completion: CompletionSettings = Field(default_factory=CompletionSettings)
    # Multi-user: parsed from SAP_CREDENTIALS env var
    credentials: dict[str, SASCredential] = Field(default_factory=_parse_credentials)

    @property
    def is_multi_user(self) -> bool:
        """True when SAP_CREDENTIALS is set (multi-user mode)."""
        return bool(self.credentials)

    @property
    def sap_login_entry_url(self) -> str:
        return os.getenv("SAP_LOGIN_ENTRY_URL", urljoin(f"{self.sap_base_url}/", self.sap_login_entry_path.lstrip("/")))

    @property
    def metadata_url(self) -> str:
        return (
            f"{self.sap_base_url}/aic/llm/api/v1/metadataV2"
            f"?workspace={self.completion.workspace}&resourceGroupId={self.completion.resource_group_id}"
        )

    @property
    def completion_url(self) -> str:
        return (
            f"{self.sap_base_url}/aic/llm/api/v1/completionV2"
            f"?workspace={self.completion.workspace}&resourceGroupId={self.completion.resource_group_id}"
        )


settings = Settings()
