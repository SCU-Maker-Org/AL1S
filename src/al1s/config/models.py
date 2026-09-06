"""Strict, immutable AL1S configuration models."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

StrictName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
Snowflake = Annotated[str, Field(pattern=r"^[0-9]{1,20}$")]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppMetaSettings(FrozenModel):
    name: str = "AL1S"
    instance_id: StrictName = "default"
    debug: bool = False


class TelegramSettings(FrozenModel):
    enabled: bool = True
    token_secret: StrictName = "telegram_bot_token"
    proxy_url: str | None = None
    admin_user_ids: tuple[Snowflake, ...] = ()

    @field_validator("proxy_url")
    @classmethod
    def validate_proxy_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https", "socks4", "socks5"} or not parsed.hostname:
            raise ValueError("telegram proxy_url must be an HTTP or SOCKS URL")
        if parsed.fragment:
            raise ValueError("telegram proxy_url must not contain a fragment")
        return value


class DiscordSettings(FrozenModel):
    enabled: bool = False
    token_secret: StrictName = "discord_bot_token"
    admin_user_ids: tuple[Snowflake, ...] = ()
    allowed_guild_ids: tuple[Snowflake, ...] = ()
    allow_direct_messages: bool = True
    sync_commands: bool = True


class TransportSettings(FrozenModel):
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    discord: DiscordSettings = Field(default_factory=DiscordSettings)


class ModelEndpointSettings(FrozenModel):
    base_url: HttpUrl = HttpUrl("https://api.openai.com/v1")
    api_key_secret: StrictName = "openai_api_key"
    model: str = "gpt-4o-mini"
    api_mode: Literal["chat_completions", "responses"] = "chat_completions"
    token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    timeout_seconds: float = Field(60.0, ge=1, le=600)
    max_retries: int = Field(1, ge=0, le=5)

    @model_validator(mode="after")
    def require_https(self) -> ModelEndpointSettings:
        if self.base_url.scheme != "https":
            raise ValueError("model endpoints must use HTTPS")
        if (
            self.base_url.username
            or self.base_url.password
            or self.base_url.query
            or self.base_url.fragment
        ):
            raise ValueError(
                "model endpoints must not contain credentials, query parameters, or fragments"
            )
        return self


class ChatModelSettings(ModelEndpointSettings):
    max_output_tokens: int = Field(2000, ge=64, le=128_000)
    temperature: float = Field(0.7, ge=0, le=2)
    max_concurrency: int = Field(4, ge=1, le=64)
    max_tool_rounds: int = Field(4, ge=0, le=10)
    max_tool_calls: int = Field(12, ge=0, le=50)
    fallback: tuple[ModelEndpointSettings, ...] = ()


class EmbeddingModelSettings(ModelEndpointSettings):
    model: str = "text-embedding-3-small"
    api_mode: Literal["chat_completions", "responses"] = "chat_completions"
    dimensions: int = Field(1536, ge=64, le=8192)
    request_dimensions: bool = True
    batch_size: int = Field(64, ge=1, le=2048)


class ModelsSettings(FrozenModel):
    chat: ChatModelSettings = Field(default_factory=ChatModelSettings)
    embedding: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)


class StorageSettings(FrozenModel):
    database_path: str = Field("al1s.db", min_length=1, max_length=4096)
    busy_timeout_ms: int = Field(5000, ge=100, le=120_000)
    message_retention_days: int = Field(365, ge=1, le=3650)

    @field_validator("database_path")
    @classmethod
    def validate_database_path(cls, value: str) -> str:
        if not value.strip() or "\x00" in value or value.endswith(("/", "\\")):
            raise ValueError("storage.database_path must name a database file")
        return value


class RagSettings(FrozenModel):
    enabled: bool = True
    top_k: int = Field(8, ge=1, le=50)
    dense_candidates: int = Field(30, ge=1, le=500)
    sparse_candidates: int = Field(30, ge=1, le=500)
    chunk_chars: int = Field(1600, ge=200, le=20_000)
    chunk_overlap_chars: int = Field(200, ge=0, le=4000)
    max_chunks_per_document: int = Field(1000, ge=1, le=10_000)
    max_document_bytes: int = Field(8 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_overlap(self) -> RagSettings:
        if self.chunk_overlap_chars >= self.chunk_chars:
            raise ValueError("rag.chunk_overlap_chars must be smaller than rag.chunk_chars")
        if self.chunk_overlap_chars > self.chunk_chars // 2:
            raise ValueError("rag.chunk_overlap_chars must not exceed half of rag.chunk_chars")
        return self


class MemorySettings(FrozenModel):
    enabled: bool = True
    private_auto_learning: bool = True
    group_auto_learning: bool = False
    max_entries_per_scope: int = Field(1000, ge=1, le=100_000)
    prompt_results: int = Field(5, ge=0, le=30)


class MCPServerSettings(FrozenModel):
    name: StrictName
    url: HttpUrl
    enabled: bool = True
    bearer_secret: StrictName | None = None
    allowed_tools: tuple[str, ...] = ()
    trusted_read_only_tools: tuple[str, ...] = ()
    access: Literal["public", "private", "admin", "private_admin"] = "private"
    read_only: bool = True
    timeout_seconds: float = Field(30, ge=1, le=300)
    max_result_bytes: int = Field(256 * 1024, ge=1024, le=4 * 1024 * 1024)
    allow_insecure_http: bool = False
    actor_scoped_workspace: bool = False

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerSettings:
        if self.url.scheme != "https" and not self.allow_insecure_http:
            raise ValueError("MCP URLs must use HTTPS unless allow_insecure_http is true")
        if self.url.username or self.url.password or self.url.query or self.url.fragment:
            raise ValueError(
                "MCP URLs must not contain credentials, query parameters, or fragments"
            )
        if self.enabled and not self.allowed_tools:
            raise ValueError("enabled MCP servers require an explicit allowed_tools list")
        if not set(self.trusted_read_only_tools).issubset(self.allowed_tools):
            raise ValueError("trusted_read_only_tools must be a subset of allowed_tools")
        return self


class MCPSettings(FrozenModel):
    enabled: bool = False
    servers: tuple[MCPServerSettings, ...] = ()

    @model_validator(mode="after")
    def validate_servers(self) -> MCPSettings:
        if self.enabled and not any(server.enabled for server in self.servers):
            raise ValueError("mcp.enabled requires at least one enabled server")
        return self


class MediaSettings(FrozenModel):
    enabled: bool = False
    api_key_secret: StrictName = "dashscope_api_key"
    base_url: HttpUrl = HttpUrl("https://dashscope.aliyuncs.com/api/v1")
    image_model: str = "qwen-image-2.0-pro"
    speech_model: str = "qwen-audio-3.0-tts-flash"
    speech_voice: str = "longanhuan_v3.6"
    max_artifact_bytes: int = Field(20 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_base_url(self) -> MediaSettings:
        parsed = urlsplit(str(self.base_url))
        hostname = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.port not in {None, 443}
            or not (
                hostname == "dashscope.aliyuncs.com" or hostname.endswith(".dashscope.aliyuncs.com")
            )
            or parsed.username
            or parsed.password
            or parsed.path.rstrip("/") != "/api/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("media base_url must be an official DashScope HTTPS endpoint")
        return self


class ReverseImageSettings(FrozenModel):
    enabled: bool = False
    trace_moe_enabled: bool = True
    saucenao_enabled: bool = False
    saucenao_api_key_secret: StrictName = "saucenao_api_key"
    max_image_bytes: int = Field(12 * 1024 * 1024, ge=1024, le=30 * 1024 * 1024)
    max_results: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_engines(self) -> ReverseImageSettings:
        if self.enabled and not (self.trace_moe_enabled or self.saucenao_enabled):
            raise ValueError("reverse_image.enabled requires at least one engine")
        return self


class GroupPolicySettings(FrozenModel):
    enabled: bool = True
    require_mention: bool = True
    allow_reply_trigger: bool = True
    observe_unmentioned_messages: bool = False
    session_scope: Literal["per_user", "shared", "topic"] = "topic"
    wake_words: tuple[str, ...] = ("AL1S", "爱丽丝")
    allowed_chat_ids: tuple[str, ...] = ()
    blocked_chat_ids: tuple[str, ...] = ()
    allowed_thread_ids: tuple[str, ...] = ()
    blocked_thread_ids: tuple[str, ...] = ()
    context_buffer_size: int = Field(30, ge=1, le=500)
    context_buffer_ttl_seconds: int = Field(1800, ge=1, le=86_400)


class RateLimitSettings(FrozenModel):
    per_user_requests: int = Field(10, ge=1, le=10_000)
    per_user_window_seconds: int = Field(60, ge=1, le=86_400)
    per_chat_requests: int = Field(30, ge=1, le=100_000)
    per_chat_window_seconds: int = Field(60, ge=1, le=86_400)


class PolicySettings(FrozenModel):
    group: GroupPolicySettings = Field(default_factory=GroupPolicySettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    max_attachment_bytes: int = Field(20 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    max_input_images: int = Field(4, ge=0, le=20)
    max_output_artifacts: int = Field(10, ge=0, le=10)
    max_output_artifact_bytes: int = Field(50 * 1024 * 1024, ge=1024, le=500 * 1024 * 1024)
    max_history_messages: int = Field(30, ge=1, le=500)
    max_history_chars: int = Field(40_000, ge=1000, le=1_000_000)
    request_timeout_seconds: float = Field(120, ge=5, le=1800)


class LoggingSettings(FrozenModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_output: bool = True


class ProfileSettings(FrozenModel):
    enabled: bool = True
    max_document_bytes: int = Field(32 * 1024, ge=512, le=1024 * 1024)
    max_prompt_chars: int = Field(8000, ge=256, le=100_000)
    reject_secrets: bool = True


class WorkspaceSettings(FrozenModel):
    root: str = "/workspaces"
    max_file_bytes: int = Field(256 * 1024, ge=1024, le=256 * 1024)
    max_results: int = Field(200, ge=1, le=5000)
    command_timeout_seconds: int = Field(25, ge=1, le=300)


class RoleSettings(FrozenModel):
    name: StrictName
    description: str
    system_prompt: str
    greeting: str = "你好，我是 AL1S。"
    farewell: str = "再见。"


def _default_roles() -> tuple[RoleSettings, ...]:
    return (
        RoleSettings(
            name="alice",
            description="可靠、直接的通用 AI 助手",
            system_prompt="你是 AL1S。准确、直接地回答，明确区分事实、推断和不确定信息。",
        ),
    )


class AppSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    config_version: Literal[1] = 1
    app: AppMetaSettings = Field(default_factory=AppMetaSettings)
    transports: TransportSettings = Field(default_factory=TransportSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    media: MediaSettings = Field(default_factory=MediaSettings)
    reverse_image: ReverseImageSettings = Field(default_factory=ReverseImageSettings)
    policies: PolicySettings = Field(default_factory=PolicySettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    default_role: StrictName = "alice"
    roles: tuple[RoleSettings, ...] = Field(default_factory=_default_roles)

    @model_validator(mode="after")
    def validate_application(self) -> AppSettings:
        if not (self.transports.telegram.enabled or self.transports.discord.enabled):
            raise ValueError("at least one transport must be enabled")
        role_names = [role.name for role in self.roles]
        if len(role_names) != len(set(role_names)):
            raise ValueError("role names must be unique")
        if self.default_role not in role_names:
            raise ValueError("default_role must reference a configured role")
        server_names = [server.name for server in self.mcp.servers]
        if len(server_names) != len(set(server_names)):
            raise ValueError("MCP server names must be unique")
        return self
