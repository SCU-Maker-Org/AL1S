"""
配置管理模块
"""

import os
import re
import tomllib
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QWEN_AUDIO_MODEL_VOICES = {
    "qwen-audio-3.0-tts-plus": {"longanlingxin", "longanlufeng"},
    "qwen-audio-3.0-tts-flash": {
        "longanfengyue",
        "longanyuanfei",
        "longanlingxi",
        "longanxiaoxin",
        "longanhuan_v3.6",
        "longjielidou_v3.6",
        "longpaopao_v3.6",
        "longhuohuo_v3.6",
        "longchuanshu_v3.6",
        "loongmary",
        "loongeva_v3.6",
        "loongjohn",
    },
}


class OpenAIConfig(BaseModel):
    """OpenAI配置"""

    api_key: str = Field("", description="OpenAI API密钥")
    base_url: str = Field("https://api.openai.com/v1", description="OpenAI API基础URL")
    model: str = Field("gpt-4o-mini", description="使用的模型名称")
    max_tokens: int = Field(2000, description="最大生成token数")
    temperature: float = Field(0.7, description="生成温度")
    timeout: int = Field(60, description="API超时时间（秒）")


class GroupMemoryConfig(BaseModel):
    """群聊长期记忆配置。"""

    enable_long_term_learning: bool = False
    allow_admin_toggle: bool = True
    namespace_scope: Literal["group", "topic"] = "topic"


class TelegramGroupConfig(BaseModel):
    """Telegram 群聊策略配置。"""

    enabled: bool = True
    require_mention: bool = True
    allow_reply_trigger: bool = True
    observe_unmentioned_messages: bool = True
    ignore_bot_messages: bool = True
    session_scope: Literal["per_user", "shared", "topic"] = "topic"
    allowed_chat_ids: list[int] = Field(default_factory=list)
    blocked_chat_ids: list[int] = Field(default_factory=list)
    allowed_thread_ids: list[int] = Field(default_factory=list)
    ignored_thread_ids: list[int] = Field(default_factory=list)
    wake_words: list[str] = Field(default_factory=lambda: ["爱丽丝", "AL1S"])
    context_buffer_size: int = Field(30, ge=1, le=500)
    context_buffer_ttl: int = Field(1800, ge=1, le=86400)
    memory: GroupMemoryConfig = Field(default_factory=GroupMemoryConfig)

    @field_validator("wake_words")
    @classmethod
    def validate_wake_words(cls, value: list[str]) -> list[str]:
        """去除空值和重复项，避免空字符串匹配所有消息。"""
        result: list[str] = []
        seen: set[str] = set()
        for word in value:
            normalized = word.strip()
            folded = normalized.casefold()
            if normalized and folded not in seen:
                result.append(normalized)
                seen.add(folded)
        return result


class TelegramRateLimitConfig(BaseModel):
    """Telegram 请求限流配置。"""

    enabled: bool = True
    per_user_requests: int = Field(10, ge=1, le=10000)
    per_user_window_seconds: int = Field(60, ge=1, le=86400)
    per_chat_requests: int = Field(30, ge=1, le=100000)
    per_chat_window_seconds: int = Field(60, ge=1, le=86400)


class TelegramConfig(BaseModel):
    """Telegram配置"""

    bot_token: str = Field("", description="Telegram机器人token")
    webhook_url: Optional[str] = Field("", description="Webhook URL（可选）")
    webhook_port: int = Field(8443, description="Webhook端口")
    admin_user_ids: list[int] = Field(default_factory=list)
    media_read_timeout: float = Field(60.0, ge=1.0, le=600.0)
    media_write_timeout: float = Field(120.0, ge=1.0, le=600.0)
    group: TelegramGroupConfig = Field(default_factory=TelegramGroupConfig)
    rate_limit: TelegramRateLimitConfig = Field(default_factory=TelegramRateLimitConfig)


class Ascii2DConfig(BaseModel):
    """Ascii2D配置"""

    base_url: str = Field("https://ascii2d.net", description="Ascii2D基础URL")
    bovw: bool = Field(False, description="是否使用特征搜索")


class RoleConfig(BaseModel):
    """角色配置"""

    name: str = Field(..., description="角色名称")
    english_name: str = Field(..., description="角色英文名")
    description: str = Field(..., description="角色描述")
    personality: str = Field(..., description="角色性格设定")
    greeting: str = Field(..., description="角色问候语")
    farewell: str = Field(..., description="角色告别语")


class MCPServerConfig(BaseModel):
    """MCP服务器配置"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="服务器名称")
    command: str = Field(..., min_length=1, description="启动命令")
    args: list[str] = Field(default_factory=list, description="命令参数")
    env: dict[str, str] = Field(default_factory=dict, description="环境变量")
    enabled: bool = Field(True, description="是否启用")
    connect_timeout: float = Field(60.0, ge=1.0, le=600.0)
    tool_timeout: float = Field(30.0, ge=1.0, le=600.0)
    include_tools: list[str] = Field(default_factory=list)
    exclude_tools: list[str] = Field(default_factory=list)
    tool_prefix: str = Field(default="", pattern=r"^[A-Za-z0-9_-]*$", max_length=32)
    read_only: bool = False
    access: Literal["public", "private", "admin", "private_admin"] = "admin"
    max_result_chars: int = Field(30000, ge=1000, le=200000)

    @model_validator(mode="after")
    def resolve_environment_references(self):
        resolved = {}
        for key, value in self.env.items():
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
            if not match:
                resolved[key] = value
                continue
            environment_name = match.group(1)
            environment_value = os.getenv(environment_name)
            if environment_value is None:
                if self.enabled:
                    raise ValueError(f"MCP环境变量未设置: {environment_name}")
                resolved[key] = value
            else:
                resolved[key] = environment_value
        self.env = resolved
        return self


class MCPConfig(BaseModel):
    """MCP配置"""

    enabled: bool = Field(True, description="是否启用MCP功能")
    servers: list[MCPServerConfig] = Field(
        default_factory=list, description="MCP服务器列表"
    )

    @model_validator(mode="after")
    def validate_unique_server_names(self):
        names = [server.name for server in self.servers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"MCP服务器名称重复: {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def apply_command_environment_overrides(self):
        for server in self.servers:
            normalized_name = re.sub(r"[^A-Za-z0-9]", "_", server.name).upper()
            environment_name = f"AL1S_MCP_{normalized_name}_COMMAND"
            command = os.getenv(environment_name, "").strip()
            if command:
                server.command = command
        return self


class DevWorkspaceConfig(BaseModel):
    """隔离开发工作区与受控 Git 发布配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    root_dir: str = Field("data/dev_workspaces", min_length=1)
    max_workspaces: int = Field(10, ge=1, le=1000)
    max_file_bytes: int = Field(1_000_000, ge=1024, le=100_000_000)
    max_output_chars: int = Field(30_000, ge=1000, le=100_000)
    command_timeout: int = Field(120, ge=1, le=3600)
    git_timeout: int = Field(60, ge=1, le=600)
    git_author_name: str = Field("AL1S", min_length=1, max_length=100)
    git_author_email: str = Field("al1s@localhost", min_length=3, max_length=254)
    branch_prefix: str = Field("al1s/", min_length=1, max_length=100)
    runner_enabled: bool = False
    allowed_github_owners: list[str] = Field(default_factory=list)
    github_token: str = Field(
        default_factory=lambda: os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
        repr=False,
    )

    @field_validator("allowed_github_owners")
    @classmethod
    def normalize_allowed_github_owners(cls, value: list[str]) -> list[str]:
        owners: list[str] = []
        seen: set[str] = set()
        for owner in value:
            normalized = owner.strip()
            if not normalized:
                continue
            if (
                not re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?",
                    normalized,
                )
                or "--" in normalized
            ):
                raise ValueError(f"无效的 GitHub owner: {normalized}")
            folded = normalized.casefold()
            if folded not in seen:
                owners.append(normalized)
                seen.add(folded)
        return owners

    @field_validator("branch_prefix")
    @classmethod
    def validate_branch_prefix(cls, value: str) -> str:
        probe = f"{value}branch"
        if (
            not value.endswith("/")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}", probe)
            or ".." in probe
            or "//" in probe
            or "@{" in probe
            or any(
                part.startswith(".")
                or part.endswith(".")
                or part.casefold().endswith(".lock")
                for part in probe.split("/")
            )
        ):
            raise ValueError(
                "dev_workspace.branch_prefix 必须是以 / 结尾的安全 Git 前缀"
            )
        return value

    @field_validator("git_author_name")
    @classmethod
    def validate_git_author_name(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("Git 作者名称不能包含控制字符")
        return value

    @field_validator("git_author_email")
    @classmethod
    def validate_git_author_email(cls, value: str) -> str:
        if not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+", value):
            raise ValueError("无效的 Git 作者邮箱")
        return value

    @model_validator(mode="after")
    def resolve_github_token(self):
        if not self.github_token:
            self.github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        return self


class RAGConfig(BaseModel):
    """RAG配置"""

    enabled: bool = Field(True, description="是否启用RAG功能")
    vector_store_path: str = Field("data/vector_store", description="向量存储路径")
    embedding_model: str = Field(
        "Qwen/Qwen3-Embedding-0.6B", description="嵌入模型类型"
    )
    max_knowledge_entries: int = Field(10000, description="最大知识条目数")
    similarity_threshold: float = Field(0.3, description="相似度阈值")
    top_k_retrieval: int = Field(5, description="检索返回的最大条目数")
    auto_learning: bool = Field(True, description="是否自动从对话中学习")
    learning_trigger_messages: int = Field(3, description="学习触发的最小消息数")
    importance_threshold: float = Field(0.1, description="知识重要性阈值")
    use_llm_extraction: bool = Field(True, description="是否使用LLM进行高级知识提取")
    technical_collection: str = Field(
        "technical_docs", min_length=1, description="权威技术文档集合名称"
    )
    technical_namespace: str = Field(
        "global:technical", min_length=1, description="权威技术文档命名空间"
    )
    hybrid_search: bool = Field(True, description="是否融合向量检索与 FTS5 检索")
    candidate_k: int = Field(40, ge=5, le=500, description="每路召回候选数量")
    rrf_k: int = Field(60, ge=1, le=1000, description="RRF 融合常数")
    max_context_chars: int = Field(
        12000, ge=1000, le=100000, description="注入模型的 RAG 最大字符数"
    )
    chunk_size: int = Field(1800, ge=256, le=12000)
    chunk_overlap: int = Field(240, ge=0, le=4000)
    max_document_bytes: int = Field(5_000_000, ge=1024, le=100_000_000)

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("rag.chunk_overlap 必须小于 rag.chunk_size")
        return self


class ProfileConfig(BaseModel):
    """用户主动维护的私有画像配置。"""

    enabled: bool = True
    max_prompt_chars: int = Field(12000, ge=500, le=30000)
    max_document_bytes: int = Field(256_000, ge=1024, le=2_000_000)
    private_chat_only: Literal[True] = True
    reject_secrets: bool = True


class MediaConfig(BaseModel):
    """Qwen Media MCP 及 Telegram 媒体派发配置。"""

    enabled: bool = False
    api_key: str = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""), repr=False
    )
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    image_model: str = "qwen-image-2.0-pro"
    speech_model: str = "qwen-audio-3.0-tts-plus"
    speech_voice: str = "longanlingxin"
    output_dir: str = "data/media_outbox"
    max_artifact_bytes: int = Field(20_000_000, ge=1024, le=50_000_000)
    retention_seconds: int = Field(3600, ge=60, le=604800)

    @field_validator("base_url")
    @classmethod
    def validate_dashscope_base_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("无效的 DashScope base_url") from exc
        host = (parsed.hostname or "").lower()
        workspace_host = bool(
            re.fullmatch(
                r"[a-z0-9][a-z0-9-]{1,126}\.(?:cn-beijing|ap-southeast-1)\.maas\.aliyuncs\.com",
                host,
            )
        )
        if (
            parsed.scheme != "https"
            or port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or (
                host not in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}
                and not workspace_host
            )
            or parsed.path.rstrip("/") != "/api/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("media.base_url 必须是官方 DashScope HTTPS /api/v1 端点")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_media_settings(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not self.image_model.startswith("qwen-image"):
            raise ValueError("media.image_model 必须使用 Qwen-Image 模型")
        allowed_voices = QWEN_AUDIO_MODEL_VOICES.get(self.speech_model)
        if allowed_voices is None:
            raise ValueError("media.speech_model 必须使用受支持的 Qwen-Audio-TTS 模型")
        if self.speech_voice not in allowed_voices:
            raise ValueError(
                f"音色 {self.speech_voice} 不适用于模型 {self.speech_model}"
            )
        return self


class AgentConfig(BaseModel):
    """Agent 配置"""

    type: str = Field("unified", description="Agent 类型: unified | langchain")

    # 通用配置
    vector_store: str = Field("faiss", description="向量存储类型: memory | faiss")
    embedding_model: str = Field(
        "Qwen/Qwen3-Embedding-0.6B",
        description="嵌入模型: tfidf | Hugging Face 模型 ID",
    )
    embedding_revision: Optional[str] = Field(
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        description="Hugging Face 模型 revision（提交哈希、标签或分支）",
    )
    embedding_device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    embedding_batch_size: int = Field(8, ge=1, le=128)
    vector_store_path: str = Field("data/vector_store", description="向量存储路径")

    # 学习配置
    auto_learning: bool = Field(True, description="是否启用自动学习")
    learning_threshold: float = Field(0.8, description="学习阈值")
    max_tool_rounds: int = Field(4, ge=1, le=10)
    max_tool_calls: int = Field(12, ge=1, le=50)


class LangChainConfig(BaseModel):
    """LangChain 特定配置"""

    # enabled 属性由 agent.type 自动控制，不再作为配置项
    vector_store: str = Field("faiss", description="向量存储类型: memory | faiss")
    embedding: str = Field(
        "huggingface_bge_m3", description="嵌入模型提供者：openai | huggingface_bge_m3"
    )
    embedding_model_name: str = Field(
        "Qwen/Qwen3-Embedding-0.6B", description="HF嵌入模型名称"
    )
    embedding_device: str = Field("cpu", description="嵌入推理设备：cpu | cuda")
    model_cache_dir: str = Field("data/models", description="模型缓存目录")
    download_timeout: int = Field(300, description="模型下载超时时间（秒）")
    download_retries: int = Field(5, description="下载重试次数")
    retriever_k: int = Field(5, description="检索topK")
    chunk_size: int = Field(1000, description="文本分块大小")
    chunk_overlap: int = Field(200, description="文本分块重叠")

    # 动态属性，由配置验证时设置
    enabled: bool = Field(
        default=False, description="是否启用（由 agent.type 自动控制）"
    )


class AppConfig:
    """应用配置"""

    def __init__(self, **kwargs):
        # 初始化属性
        self.roles = {}
        self.default_role = "天童爱丽丝"
        # 应用元信息
        self.app = None  # 将在下方填充为 AppMetaConfig 实例
        self.debug = False

        # 先加载统一配置文件
        config_data = self._load_unified_config()

        # 从TOML配置中提取各个配置部分
        app_meta_config = config_data.get("app", {}) if config_data else {}
        openai_config = config_data.get("openai", {}) if config_data else {}
        telegram_config = config_data.get("telegram", {}) if config_data else {}
        ascii2d_config = config_data.get("ascii2d", {}) if config_data else {}
        mcp_config = config_data.get("mcp", {}) if config_data else {}
        dev_workspace_config = (
            config_data.get("dev_workspace", {}) if config_data else {}
        )
        rag_config = config_data.get("rag", {}) if config_data else {}
        profile_config = config_data.get("profile", {}) if config_data else {}
        media_config = config_data.get("media", {}) if config_data else {}
        agent_config = config_data.get("agent", {}) if config_data else {}
        lc_config = config_data.get("langchain", {}) if config_data else {}

        # 初始化各个配置对象，使用默认值填充缺失的配置
        # 应用元配置（名称、版本、调试开关）
        try:
            from pydantic import BaseModel, Field  # 重新导入以消除静态检查警告

            class AppMetaConfig(BaseModel):
                name: str = Field("AL1S-Bot", description="应用名称")
                version: str = Field("0.1.0", description="应用版本")
                debug: bool = Field(False, description="是否启用调试日志")

        except Exception:
            # 非常规环境下的兜底（不应触发）
            class AppMetaConfig:  # type: ignore
                def __init__(self, name="AL1S-Bot", version="0.1.0", debug=False):
                    self.name = name
                    self.version = version
                    self.debug = debug

        self.app = AppMetaConfig(**app_meta_config)
        self.debug = bool(getattr(self.app, "debug", False))

        self.openai = OpenAIConfig(**openai_config)
        self.telegram = TelegramConfig(**telegram_config)
        self.ascii2d = Ascii2DConfig(**ascii2d_config)
        self.mcp = MCPConfig(**mcp_config)
        self.dev_workspace = DevWorkspaceConfig(**dev_workspace_config)
        self.rag = RAGConfig(**rag_config)
        self.profile = ProfileConfig(**profile_config)
        self.media = MediaConfig(**media_config)
        self.agent = AgentConfig(**agent_config)
        self.langchain = LangChainConfig(**lc_config)

        # 验证 Agent 配置互斥性
        self._validate_agent_config()

        # 设置角色配置
        if config_data and "default_role" in config_data:
            self.default_role = config_data["default_role"]

        if config_data and "roles" in config_data:
            for role_data in config_data["roles"]:
                role_name = role_data["name"]
                self.roles[role_name] = RoleConfig(**role_data)

        # 验证必需配置
        self._validate_config()

    def _load_unified_config(self):
        """加载统一配置文件"""
        try:
            config_file = Path(__file__).parent.parent / "config.toml"
            if config_file.exists():
                with open(config_file, "rb") as f:
                    config_data = tomllib.load(f)

                print(f"成功加载配置文件: {config_file}")
                return config_data
            else:
                print(f"配置文件不存在: {config_file}")
                print(
                    "请复制 config/config.toml.example 为 config/config.toml 并填写配置信息"
                )
                return {}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("请检查配置文件格式是否正确")
            return {}

    def _validate_config(self):
        """验证必需配置"""
        if not self.openai.api_key:
            print("⚠️  警告: OpenAI API密钥未设置")
        if not self.telegram.bot_token:
            print("⚠️  警告: Telegram Bot Token未设置")
        if not self.roles:
            print("⚠️  警告: 未加载任何角色配置")

    def get_role(self, role_name: str) -> Optional[RoleConfig]:
        """获取指定角色配置"""
        return self.roles.get(role_name)

    def get_default_role(self) -> Optional[RoleConfig]:
        """获取默认角色配置"""
        return self.roles.get(self.default_role)

    def _validate_agent_config(self):
        """验证 Agent 配置"""
        from loguru import logger

        # 根据 agent.type 自动设置相关配置
        if self.agent.type == "langchain":
            # 使用 LangChain Agent 时自动启用 langchain 配置
            self.langchain.enabled = True
            logger.info("使用 LangChain Agent")
        elif self.agent.type == "unified":
            # 使用统一 Agent 时禁用 langchain 配置
            self.langchain.enabled = False
            logger.info("使用统一 Agent")
        else:
            # 未知类型回退到统一 Agent
            logger.warning(f"未知的 Agent 类型: {self.agent.type}，回退到统一 Agent")
            self.agent.type = "unified"
            self.langchain.enabled = False

    def list_roles(self) -> list:
        """列出所有可用角色"""
        return list(self.roles.keys())


# 加载配置
def load_config() -> AppConfig:
    """加载应用配置"""
    return AppConfig(
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
        ),
        telegram=TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            webhook_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
            webhook_port=int(os.getenv("TELEGRAM_WEBHOOK_PORT", "8443")),
        ),
        ascii2d=Ascii2DConfig(
            base_url=os.getenv("ASCII2D_BASE_URL", "https://ascii2d.net"),
            bovw=os.getenv("ASCII2D_BOVW", "false").lower() == "true",
        ),
    )


# 全局配置实例
config = AppConfig()
