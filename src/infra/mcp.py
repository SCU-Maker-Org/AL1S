"""
MCP (Model Context Protocol) 服务模块
"""

import asyncio
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional

from loguru import logger

from ..models import MediaArtifact

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP库未安装，MCP功能将不可用")
    MCP_AVAILABLE = False


MEDIA_CAPTURE_NONCE_ARGUMENT = "al1s_capture_nonce"
MEDIA_CAPTURE_OWNER_ARGUMENT = "al1s_capture_owner"
MAX_MEDIA_TTL_SECONDS = 7 * 24 * 60 * 60
MAX_TOOL_LOG_DESCRIPTION_CHARS = 160


def _tool_log_summary(description: Optional[str]) -> str:
    """将 MCP 工具说明压缩为适合 INFO 日志的单行摘要。"""
    summary = " ".join((description or "").split())
    if len(summary) <= MAX_TOOL_LOG_DESCRIPTION_CHARS:
        return summary
    return summary[: MAX_TOOL_LOG_DESCRIPTION_CHARS - 3].rstrip() + "..."


@dataclass(slots=True)
class _MediaCaptureState:
    nonce: str
    owner_tag: str
    artifacts: List[MediaArtifact] = field(default_factory=list)


class MCPServerConfig:
    """MCP服务器配置"""

    def __init__(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
        connect_timeout: float = 60.0,
        tool_timeout: float = 30.0,
        include_tools: List[str] = None,
        exclude_tools: List[str] = None,
        tool_prefix: str = "",
        read_only: bool = False,
        access: str = "admin",
        max_result_chars: int = 30000,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.connect_timeout = connect_timeout
        self.tool_timeout = tool_timeout
        self.include_tools = include_tools or []
        self.exclude_tools = exclude_tools or []
        self.tool_prefix = tool_prefix
        self.read_only = read_only
        self.access = access
        self.max_result_chars = max_result_chars


class MCPService:
    """MCP服务管理类"""

    def __init__(
        self,
        *,
        media_output_dir: str = "data/media_outbox",
        max_media_bytes: int = 20_000_000,
        max_media_ttl_seconds: int = 3600,
        media_server_name: str = "media",
    ):
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.connected_servers = set()
        self.server_errors: Dict[str, str] = {}
        self.media_output_dir = Path(media_output_dir).resolve()
        self.max_media_bytes = max_media_bytes
        if not 60 <= max_media_ttl_seconds <= MAX_MEDIA_TTL_SECONDS:
            raise ValueError("媒体 TTL 上限必须在 60 秒到 7 天之间")
        self.max_media_ttl_seconds = max_media_ttl_seconds
        self.media_server_name = media_server_name
        self._media_owner_secret = secrets.token_bytes(32)
        self._media_capture: ContextVar[Optional[_MediaCaptureState]] = ContextVar(
            "mcp_media_capture", default=None
        )

    def _media_owner_tag(self, owner: str) -> str:
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("媒体 capture owner 不能为空")
        return hmac.new(
            self._media_owner_secret, owner.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def begin_media_capture(self, owner: str) -> Token:
        """为当前异步 Update 建立独立媒体收件箱。"""
        state = _MediaCaptureState(
            nonce=secrets.token_urlsafe(32),
            owner_tag=self._media_owner_tag(owner),
        )
        return self._media_capture.set(state)

    def finish_media_capture(self, token: Token) -> List[MediaArtifact]:
        state = self._media_capture.get()
        artifacts = list(state.artifacts if state else [])
        self._media_capture.reset(token)
        return artifacts

    @staticmethod
    def _media_payload(value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, str):
            try:
                return MCPService._media_payload(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(value, dict):
            return None
        if isinstance(value.get("al1s_media"), dict):
            return value["al1s_media"]
        required = {"relative_path", "kind", "mime_type", "sha256", "expires_at"}
        if required <= value.keys() and ("byte_size" in value or "size_bytes" in value):
            return value
        for key in ("result", "content", "data"):
            nested = MCPService._media_payload(value.get(key))
            if nested:
                return nested
        return None

    @staticmethod
    def _safe_media_relative_path(value: Any) -> Path:
        relative_path = str(value or "")
        relative = Path(relative_path)
        if (
            not relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or any(part in {"", "."} for part in relative.parts)
        ):
            raise ValueError("媒体路径必须是安全的相对路径")
        return relative

    @contextmanager
    def _secure_open_media_file(
        self, relative: Path, *, delete_after: bool = False
    ) -> Iterator[tuple[BinaryIO, os.stat_result]]:
        """逐级拒绝 symlink，并始终消费已打开的同一 inode。"""
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        file_flags = os.O_RDONLY
        for optional_flag in ("O_NOFOLLOW", "O_CLOEXEC"):
            directory_flags |= getattr(os, optional_flag, 0)
            file_flags |= getattr(os, optional_flag, 0)

        directory_fds: List[int] = []
        file_handle: Optional[BinaryIO] = None
        parent_fd: Optional[int] = None
        filename = relative.parts[-1]
        opened_stat: Optional[os.stat_result] = None
        try:
            root_fd = os.open(self.media_output_dir, directory_flags)
            directory_fds.append(root_fd)
            parent_fd = root_fd
            for part in relative.parts[:-1]:
                parent_fd = os.open(part, directory_flags, dir_fd=parent_fd)
                directory_fds.append(parent_fd)

            file_fd = os.open(filename, file_flags, dir_fd=parent_fd)
            opened_stat = os.fstat(file_fd)
            if not stat.S_ISREG(opened_stat.st_mode):
                os.close(file_fd)
                raise ValueError("媒体 artifact 必须是普通文件")
            file_handle = os.fdopen(file_fd, "rb", closefd=True)
            yield file_handle, opened_stat
        finally:
            if file_handle is not None and not file_handle.closed:
                file_handle.close()
            if delete_after and parent_fd is not None and opened_stat is not None:
                try:
                    current_stat = os.stat(
                        filename, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if (
                        current_stat.st_dev == opened_stat.st_dev
                        and current_stat.st_ino == opened_stat.st_ino
                        and stat.S_ISREG(current_stat.st_mode)
                    ):
                        os.unlink(filename, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("清理媒体 artifact 失败: {}", exc)
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)

    def _verify_open_media_file(
        self,
        file_handle: BinaryIO,
        file_stat: os.stat_result,
        *,
        declared_size: int,
        expected_hash: str,
    ) -> str:
        if (
            file_stat.st_size != declared_size
            or file_stat.st_size > self.max_media_bytes
        ):
            raise ValueError("媒体文件大小不匹配或超过上限")
        digest = hashlib.sha256()
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
        actual_hash = digest.hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or not hmac.compare_digest(
            actual_hash, expected_hash
        ):
            raise ValueError("媒体文件哈希校验失败")
        file_handle.seek(0)
        return actual_hash

    def validate_media_artifact(
        self,
        payload: Dict[str, Any],
        *,
        expected_capture_nonce: Optional[str] = None,
        expected_owner_tag: Optional[str] = None,
    ) -> MediaArtifact:
        """验证由受信 Media MCP 为当前 Telegram Update 生成的 artifact。"""
        kind = str(payload.get("kind", ""))
        if kind == "image":
            kind = "photo"
        if kind not in {"photo", "voice"}:
            raise ValueError("不支持的媒体类型")
        relative = self._safe_media_relative_path(payload.get("relative_path"))
        capture_nonce = str(payload.get("capture_nonce", ""))
        owner_tag = str(payload.get("owner_tag", ""))
        if (
            expected_capture_nonce is None
            or expected_owner_tag is None
            or not hmac.compare_digest(capture_nonce, expected_capture_nonce)
            or not hmac.compare_digest(owner_tag, expected_owner_tag)
        ):
            raise ValueError("媒体 artifact 不属于当前消息")
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", capture_nonce):
            raise ValueError("媒体 capture nonce 格式无效")
        if not re.fullmatch(r"[0-9a-f]{64}", owner_tag):
            raise ValueError("媒体 owner tag 格式无效")
        expected_kind_directory = "image" if kind == "photo" else "voice"
        if len(relative.parts) != 5 or relative.parts[:4] != (
            "requests",
            owner_tag,
            capture_nonce,
            expected_kind_directory,
        ):
            raise ValueError("媒体路径未绑定当前消息")

        try:
            declared_size = int(payload.get("byte_size", payload.get("size_bytes", -1)))
        except (TypeError, ValueError) as exc:
            raise ValueError("媒体文件大小格式无效") from exc
        mime_type = str(payload.get("mime_type", ""))
        allowed_mimes = {
            "photo": {"image/png", "image/jpeg", "image/webp"},
            "voice": {"audio/ogg", "audio/opus", "audio/mpeg", "audio/mp4"},
        }
        if mime_type not in allowed_mimes[kind]:
            raise ValueError("媒体 MIME 类型不受支持")
        expected_hash = str(payload.get("sha256", "")).lower()
        expires_value = payload.get("expires_at", 0)
        try:
            expires_at = float(expires_value)
        except (TypeError, ValueError):
            try:
                parsed_expiry = datetime.fromisoformat(
                    str(expires_value).replace("Z", "+00:00")
                )
                if parsed_expiry.tzinfo is None:
                    raise ValueError("媒体过期时间必须包含时区")
                expires_at = parsed_expiry.astimezone(timezone.utc).timestamp()
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("媒体过期时间格式无效") from exc
        now = time.time()
        if not math.isfinite(expires_at):
            raise ValueError("媒体过期时间格式无效")
        if expires_at <= now:
            raise ValueError("媒体文件已经过期")
        if expires_at - now > self.max_media_ttl_seconds:
            raise ValueError("媒体文件过期时间超过配置上限")

        try:
            with self._secure_open_media_file(relative) as (file_handle, file_stat):
                actual_hash = self._verify_open_media_file(
                    file_handle,
                    file_stat,
                    declared_size=declared_size,
                    expected_hash=expected_hash,
                )
        except OSError as exc:
            raise ValueError("媒体文件无法安全打开") from exc

        artifact_id = str(payload.get("artifact_id", "")) or actual_hash[:16]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", artifact_id):
            raise ValueError("媒体 artifact ID 格式无效")
        return MediaArtifact(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative.as_posix(),
            mime_type=mime_type,
            byte_size=declared_size,
            sha256=actual_hash,
            expires_at=expires_at,
            capture_nonce=capture_nonce,
            owner_tag=owner_tag,
            caption=(
                str(payload.get("caption", ""))
                or ("AI 生成语音" if kind == "voice" else "")
            )[:1024],
        )

    @staticmethod
    def _artifact_payload_from_model(artifact: MediaArtifact) -> Dict[str, Any]:
        return {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "relative_path": artifact.relative_path,
            "mime_type": artifact.mime_type,
            "byte_size": artifact.byte_size,
            "sha256": artifact.sha256,
            "expires_at": artifact.expires_at,
            "capture_nonce": artifact.capture_nonce,
            "owner_tag": artifact.owner_tag,
            "caption": artifact.caption,
        }

    def _verify_artifact_owner(self, artifact: MediaArtifact, owner: str) -> None:
        if not hmac.compare_digest(artifact.owner_tag, self._media_owner_tag(owner)):
            raise ValueError("媒体 artifact 不属于当前调用方")

    def media_artifact_path(self, artifact: MediaArtifact, *, owner: str) -> Path:
        """返回复验后的 canonical path；实际发送必须使用 consume_media_artifact。"""
        self._verify_artifact_owner(artifact, owner)
        validated = self.validate_media_artifact(
            self._artifact_payload_from_model(artifact),
            expected_capture_nonce=artifact.capture_nonce,
            expected_owner_tag=artifact.owner_tag,
        )
        resolved = (self.media_output_dir / validated.relative_path).resolve(
            strict=True
        )
        if not resolved.is_relative_to(self.media_output_dir):
            raise ValueError("媒体文件不在允许的 outbox 中")
        return resolved

    @contextmanager
    def consume_media_artifact(
        self, artifact: MediaArtifact, *, owner: str
    ) -> Iterator[BinaryIO]:
        """复验并打开 artifact，结束发送尝试后安全删除已消费的 inode。"""
        self._verify_artifact_owner(artifact, owner)
        payload = self._artifact_payload_from_model(artifact)
        self.validate_media_artifact(
            payload,
            expected_capture_nonce=artifact.capture_nonce,
            expected_owner_tag=artifact.owner_tag,
        )
        relative = self._safe_media_relative_path(artifact.relative_path)
        try:
            with self._secure_open_media_file(relative, delete_after=True) as (
                file_handle,
                file_stat,
            ):
                self._verify_open_media_file(
                    file_handle,
                    file_stat,
                    declared_size=artifact.byte_size,
                    expected_hash=artifact.sha256,
                )
                yield file_handle
        except OSError as exc:
            raise ValueError("媒体文件无法安全打开") from exc

    def cleanup_media_artifacts(
        self, artifacts: List[MediaArtifact], *, owner: str
    ) -> None:
        """清理当前 Update 未发送或发送失败后遗留的 artifact。"""
        for artifact in artifacts:
            try:
                self._verify_artifact_owner(artifact, owner)
                relative = self._safe_media_relative_path(artifact.relative_path)
                with self._secure_open_media_file(relative, delete_after=True) as (
                    file_handle,
                    file_stat,
                ):
                    self._verify_open_media_file(
                        file_handle,
                        file_stat,
                        declared_size=artifact.byte_size,
                        expected_hash=artifact.sha256,
                    )
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                logger.warning(
                    "无法清理媒体 artifact_id={}: {}", artifact.artifact_id, exc
                )

    def _capture_media_result(
        self, result: Any, text_parts: List[str], *, server_name: str
    ) -> Optional[str]:
        if server_name != self.media_server_name:
            return None
        current = self._media_capture.get()
        if current is None:
            raise RuntimeError("媒体工具只能在 Telegram 消息处理上下文中调用")
        payload = self._media_payload(getattr(result, "structuredContent", None))
        if payload is None:
            for text in text_parts:
                payload = self._media_payload(text)
                if payload is not None:
                    break
        if payload is None:
            return None
        artifact = self.validate_media_artifact(
            payload,
            expected_capture_nonce=current.nonce,
            expected_owner_tag=current.owner_tag,
        )
        current.artifacts.append(artifact)
        return json.dumps(
            {
                "status": "media_ready",
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "byte_size": artifact.byte_size,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _annotation_value(annotations: Any, name: str) -> Any:
        if annotations is None:
            return None
        if isinstance(annotations, dict):
            return annotations.get(name)
        return getattr(annotations, name, None)

    def _include_tool(self, tool: Any, server_config: MCPServerConfig) -> bool:
        name = tool.name
        if server_config.include_tools and not any(
            fnmatchcase(name, pattern) for pattern in server_config.include_tools
        ):
            return False
        if any(fnmatchcase(name, pattern) for pattern in server_config.exclude_tools):
            return False
        if server_config.read_only:
            read_only_hint = self._annotation_value(
                getattr(tool, "annotations", None), "readOnlyHint"
            )
            if read_only_hint is not True:
                return False
        return True

    @staticmethod
    def resolve_caller_access(
        user_id: Optional[int], chat_type: Any, admin_user_ids: List[int]
    ) -> str:
        """将 Telegram 调用方映射为 MCP 访问级别。"""
        normalized_chat_type = getattr(chat_type, "value", chat_type)
        if isinstance(normalized_chat_type, str):
            normalized_chat_type = normalized_chat_type.lower()
        if user_id is not None and user_id in admin_user_ids:
            return "private_admin" if normalized_chat_type == "private" else "admin"
        if normalized_chat_type == "private":
            return "private"
        return "public"

    @staticmethod
    def _access_allowed(required: str, caller_access: str) -> bool:
        ranks = {"public": 0, "private": 1, "admin": 2, "private_admin": 3}
        required_rank = ranks.get(required)
        caller_rank = ranks.get(caller_access)
        return (
            required_rank is not None
            and caller_rank is not None
            and caller_rank >= required_rank
        )

    @classmethod
    def redact_sensitive_data(cls, value: Any) -> Any:
        """在日志和持久化前隐藏常见凭据字段并限制体积。"""
        sensitive = re.compile(
            r"(?:api[_-]?key|token|secret|password|authorization|cookie)", re.I
        )
        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]"
                    if sensitive.search(str(key))
                    else cls.redact_sensitive_data(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls.redact_sensitive_data(item) for item in value[:100]]
        if isinstance(value, tuple):
            return tuple(cls.redact_sensitive_data(item) for item in value[:100])
        if isinstance(value, str) and len(value) > 2000:
            return value[:2000] + "...[TRUNCATED]"
        return value

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)
        if len(sanitized) <= 64:
            return sanitized
        digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:8]
        return f"{sanitized[:55]}_{digest}"

    def _exposed_tool_name(
        self, server_config: MCPServerConfig, native_name: str, existing: set[str]
    ) -> str:
        preferred = self._sanitize_tool_name(
            f"{server_config.tool_prefix}{native_name}"
        )
        if preferred not in existing:
            return preferred

        base = self._sanitize_tool_name(f"{server_config.name}__{native_name}")
        candidate = base
        suffix = 2
        while candidate in existing:
            suffix_text = f"_{suffix}"
            candidate = f"{base[: 64 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        logger.warning(
            f"工具名 {preferred} 冲突，服务器 {server_config.name} 使用别名 {candidate}"
        )
        return candidate

    async def add_server(self, server_config: MCPServerConfig) -> bool:
        """添加MCP服务器"""
        if not MCP_AVAILABLE:
            logger.error("MCP库不可用，无法连接服务器")
            return False

        try:
            logger.info(f"正在连接MCP服务器: {server_config.name}")
            self.servers[server_config.name] = server_config
            self.tools.pop(server_config.name, None)
            self.connected_servers.discard(server_config.name)
            self.server_errors.pop(server_config.name, None)

            # 创建服务器参数
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
            )

            # 使用超时和更简单的连接方式
            try:
                # 临时连接以获取工具信息
                # 首次通过 npx/uvx 启动时可能需要下载依赖。
                async with asyncio.timeout(server_config.connect_timeout):
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            # 初始化会话
                            await session.initialize()

                            # 获取服务器信息（某些版本可能不支持此方法）
                            try:
                                server_info = await session.get_server_info()
                                logger.info(
                                    f"MCP服务器 {server_config.name} 连接成功: {server_info}"
                                )
                            except AttributeError:
                                logger.info(
                                    f"MCP服务器 {server_config.name} 连接成功（服务器信息不可用）"
                                )

                            # 列出可用工具
                            tool_pages = []
                            cursor = None
                            while True:
                                tools_result = await session.list_tools(cursor=cursor)
                                tool_pages.extend(tools_result.tools or [])
                                cursor = getattr(tools_result, "nextCursor", None)
                                if not cursor:
                                    break
                            server_tools = {}
                            existing_names = {
                                tool_name
                                for tools in self.tools.values()
                                for tool_name in tools
                            }
                            if tool_pages:
                                for tool in tool_pages:
                                    if not self._include_tool(tool, server_config):
                                        logger.debug(
                                            f"服务器 {server_config.name} 已过滤工具: {tool.name}"
                                        )
                                        continue
                                    exposed_name = self._exposed_tool_name(
                                        server_config, tool.name, existing_names
                                    )
                                    existing_names.add(exposed_name)
                                    annotations = getattr(tool, "annotations", None)
                                    tool_schema = copy.deepcopy(tool.inputSchema)
                                    if server_config.name == self.media_server_name:
                                        properties = tool_schema.get("properties", {})
                                        properties.pop(
                                            MEDIA_CAPTURE_NONCE_ARGUMENT, None
                                        )
                                        properties.pop(
                                            MEDIA_CAPTURE_OWNER_ARGUMENT, None
                                        )
                                        required = tool_schema.get("required", [])
                                        tool_schema["required"] = [
                                            field
                                            for field in required
                                            if field
                                            not in {
                                                MEDIA_CAPTURE_NONCE_ARGUMENT,
                                                MEDIA_CAPTURE_OWNER_ARGUMENT,
                                            }
                                        ]
                                    server_tools[exposed_name] = {
                                        "description": tool.description,
                                        "schema": tool_schema,
                                        "output_schema": getattr(
                                            tool, "outputSchema", None
                                        ),
                                        "server": server_config.name,
                                        "remote_name": tool.name,
                                        "access": server_config.access,
                                        "read_only": self._annotation_value(
                                            annotations, "readOnlyHint"
                                        ),
                                        "destructive": self._annotation_value(
                                            annotations, "destructiveHint"
                                        ),
                                    }
                                    description_summary = _tool_log_summary(
                                        tool.description
                                    )
                                    if description_summary:
                                        logger.info(
                                            f"发现工具: {exposed_name} - "
                                            f"{description_summary}"
                                        )
                                    else:
                                        logger.info(f"发现工具: {exposed_name}")
                                    logger.debug(
                                        f"MCP工具完整描述 [{exposed_name}]: "
                                        f"{tool.description}"
                                    )
                            if not tool_pages:
                                logger.warning(
                                    f"MCP服务器 {server_config.name} 未提供任何工具"
                                )
                            elif not server_tools:
                                logger.warning(
                                    f"MCP服务器 {server_config.name} 的工具均被策略过滤"
                                )

                            self.tools[server_config.name] = server_tools

                self.connected_servers.add(server_config.name)

                return True

            except asyncio.TimeoutError:
                error = f"连接超时（{server_config.connect_timeout:g}秒）"
                self.server_errors[server_config.name] = error
                logger.error(f"连接MCP服务器 {server_config.name} 超时")
                return False

        except Exception as e:
            self.server_errors[server_config.name] = str(e)
            logger.error(f"连接MCP服务器 {server_config.name} 失败: {e}")
            return False

    async def remove_server(self, server_name: str) -> bool:
        """移除MCP服务器"""
        try:
            if server_name not in self.servers:
                logger.warning(f"MCP服务器 {server_name} 不存在")
                return False

            if server_name in self.sessions:
                session = self.sessions.pop(server_name)
                await session.close()

            self.servers.pop(server_name, None)
            self.tools.pop(server_name, None)
            self.connected_servers.discard(server_name)
            self.server_errors.pop(server_name, None)
            logger.info(f"MCP服务器 {server_name} 已断开连接")
            return True

        except Exception as e:
            logger.error(f"断开MCP服务器 {server_name} 失败: {e}")
            return False

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: str = None,
        caller_access: str = "public",
    ) -> Optional[str]:
        """调用MCP工具"""
        if not MCP_AVAILABLE:
            return "MCP库不可用，无法调用工具"

        try:
            # 查找工具所属服务器
            target_server = None
            if server_name:
                if server_name in self.tools and tool_name in self.tools[server_name]:
                    target_server = server_name
            else:
                # 搜索所有服务器
                for srv_name, tools in self.tools.items():
                    if tool_name in tools:
                        target_server = srv_name
                        break

            if not target_server:
                logger.error(f"工具 {tool_name} 未找到")
                return None

            if target_server not in self.servers:
                logger.error(f"MCP服务器 {target_server} 配置不存在")
                return None

            server_config = self.servers[target_server]
            tool_info = self.tools[target_server][tool_name]
            required_access = tool_info.get("access", server_config.access)
            if not self._access_allowed(required_access, caller_access):
                logger.warning(
                    f"拒绝调用工具 {tool_name}: 需要 {required_access}，当前 {caller_access}"
                )
                return f"工具调用失败: 无权调用 {tool_name}"
            remote_tool_name = tool_info.get("remote_name", tool_name)
            call_arguments = dict(arguments)
            if target_server == self.media_server_name:
                capture = self._media_capture.get()
                if capture is None:
                    logger.warning(
                        "拒绝调用 Media MCP 工具 {}: 当前没有 Telegram capture",
                        tool_name,
                    )
                    return "工具调用失败: 媒体工具只能响应当前 Telegram 消息"
                # 绑定值由受信的调用端覆盖，永不采用模型提供的参数。
                call_arguments[MEDIA_CAPTURE_NONCE_ARGUMENT] = capture.nonce
                call_arguments[MEDIA_CAPTURE_OWNER_ARGUMENT] = capture.owner_tag

            # 创建服务器参数
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
            )

            # 创建临时连接调用工具，添加超时
            try:
                async with asyncio.timeout(server_config.tool_timeout):
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            # 初始化会话
                            await session.initialize()

                            # 调用工具
                            logger.info(
                                f"调用工具: {tool_name} 参数字段: {list(arguments)}"
                            )
                            result = await session.call_tool(
                                remote_tool_name, call_arguments
                            )

                            if result.content:
                                # 处理结果内容
                                content_parts = []
                                for content in result.content:
                                    if hasattr(content, "text"):
                                        content_parts.append(content.text)
                                    elif hasattr(content, "data"):
                                        content_parts.append(str(content.data))
                                    else:
                                        content_parts.append(str(content))

                                media_result = self._capture_media_result(
                                    result,
                                    content_parts,
                                    server_name=target_server,
                                )
                                if media_result is not None:
                                    logger.info(
                                        "工具 {} 生成媒体 artifact，已进入当前消息 outbox",
                                        tool_name,
                                    )
                                    return media_result

                                result_text = "\n".join(content_parts)
                                if len(result_text) > server_config.max_result_chars:
                                    omitted = (
                                        len(result_text)
                                        - server_config.max_result_chars
                                    )
                                    result_text = (
                                        result_text[: server_config.max_result_chars]
                                        + f"\n\n[结果已截断，省略 {omitted} 个字符]"
                                    )
                                if getattr(result, "isError", False):
                                    logger.error(
                                        f"工具 {tool_name} 返回错误: {result_text}"
                                    )
                                    return f"工具调用失败: {result_text}"
                                logger.info(f"工具 {tool_name} 调用成功")
                                return result_text
                            elif getattr(result, "structuredContent", None) is not None:
                                media_result = self._capture_media_result(
                                    result, [], server_name=target_server
                                )
                                if media_result is not None:
                                    return media_result
                                result_text = json.dumps(
                                    result.structuredContent,
                                    ensure_ascii=False,
                                    default=str,
                                )
                                if getattr(result, "isError", False):
                                    return f"工具调用失败: {result_text}"
                                return result_text[: server_config.max_result_chars]
                            elif getattr(result, "isError", False):
                                logger.error(f"工具 {tool_name} 返回错误但没有错误内容")
                                return "工具调用失败: MCP 服务器未提供错误详情"
                            else:
                                logger.warning(f"工具 {tool_name} 返回空结果")
                                return "工具执行完成，但没有返回内容"

            except asyncio.TimeoutError:
                logger.error(f"调用工具 {tool_name} 超时")
                return f"工具调用超时: {tool_name}"

        except Exception as e:
            logger.error(f"调用工具 {tool_name} 失败: {e}")
            return f"工具调用失败: {str(e)}"

    def get_available_tools(
        self, caller_access: str = "public"
    ) -> Dict[str, Dict[str, Any]]:
        """获取所有可用工具"""
        all_tools = {}
        for server_name, tools in self.tools.items():
            for tool_name, tool_info in tools.items():
                if self._access_allowed(
                    tool_info.get("access", "admin"), caller_access
                ):
                    all_tools[tool_name] = tool_info
        return all_tools

    def get_tools_for_llm(self, caller_access: str = "public") -> List[Dict[str, Any]]:
        """获取适用于LLM function calling的工具定义"""
        tools_for_llm = []

        for server_name, tools in self.tools.items():
            for tool_name, tool_info in tools.items():
                if not self._access_allowed(
                    tool_info.get("access", "admin"), caller_access
                ):
                    continue
                # 转换为OpenAI function calling格式
                function_def = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("schema", {}),
                    },
                }
                tools_for_llm.append(function_def)

        return tools_for_llm

    def get_server_status(
        self, caller_access: str = "public"
    ) -> Dict[str, Dict[str, Any]]:
        """获取所有服务器状态"""
        status = {}
        for server_name, server_config in self.servers.items():
            if not self._access_allowed(server_config.access, caller_access):
                continue
            status[server_name] = {
                "name": server_name,
                "command": server_config.command,
                "args": server_config.args,
                "connected": server_name in self.connected_servers,
                "tools_count": len(self.tools.get(server_name, {})),
                "tools": list(self.tools.get(server_name, {}).keys()),
                "error": self.server_errors.get(server_name),
            }
        return status

    async def initialize_default_servers(self, mcp_configs: List[Any]):
        """初始化默认MCP服务器"""
        for config in mcp_configs:
            try:
                # 处理不同类型的配置对象
                if hasattr(config, "name"):
                    # 已经是 MCPServerConfig 对象
                    server_config = MCPServerConfig(
                        name=config.name,
                        command=config.command,
                        **{
                            field: getattr(config, field)
                            for field in (
                                "args",
                                "env",
                                "connect_timeout",
                                "tool_timeout",
                                "include_tools",
                                "exclude_tools",
                                "tool_prefix",
                                "read_only",
                                "access",
                                "max_result_chars",
                            )
                        },
                    )
                else:
                    # 字典类型的配置
                    server_config = MCPServerConfig(
                        name=config.get("name", "unknown"),
                        command=config.get("command", ""),
                        args=config.get("args", []),
                        env=config.get("env", {}),
                        connect_timeout=config.get("connect_timeout", 60.0),
                        tool_timeout=config.get("tool_timeout", 30.0),
                        include_tools=config.get("include_tools", []),
                        exclude_tools=config.get("exclude_tools", []),
                        tool_prefix=config.get("tool_prefix", ""),
                        read_only=config.get("read_only", False),
                        access=config.get("access", "admin"),
                        max_result_chars=config.get("max_result_chars", 30000),
                    )

                # 只初始化启用的服务器
                enabled = (
                    config.enabled
                    if hasattr(config, "enabled")
                    else config.get("enabled", True)
                )
                if not enabled:
                    logger.info(f"跳过禁用的MCP服务器: {server_config.name}")
                    continue

                success = await self.add_server(server_config)
                if success:
                    logger.info(f"MCP服务器 {server_config.name} 初始化成功")
                else:
                    logger.error(f"MCP服务器 {server_config.name} 初始化失败")

            except Exception as e:
                logger.error(f"初始化MCP服务器失败: {e}")

    async def close_all(self):
        """关闭所有MCP连接"""
        # 清理所有数据（不需要关闭会话，因为每次都是临时连接）
        self.sessions.clear()
        self.servers.clear()
        self.tools.clear()
        self.connected_servers.clear()
        self.server_errors.clear()

        logger.info("所有MCP服务器连接已关闭")
