"""
MCP (Model Context Protocol) 服务模块
"""

import asyncio
import hashlib
import json
import re
from fnmatch import fnmatchcase
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP库未安装，MCP功能将不可用")
    MCP_AVAILABLE = False


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

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.connected_servers = set()
        self.server_errors: Dict[str, str] = {}

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
        if user_id is not None and user_id in admin_user_ids:
            return "admin"
        normalized_chat_type = getattr(chat_type, "value", chat_type)
        if normalized_chat_type == "private":
            return "private"
        return "public"

    @staticmethod
    def _access_allowed(required: str, caller_access: str) -> bool:
        ranks = {"public": 0, "private": 1, "admin": 2}
        return ranks.get(caller_access, -1) >= ranks.get(required, 2)

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
                                    server_tools[exposed_name] = {
                                        "description": tool.description,
                                        "schema": tool.inputSchema,
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
                                    logger.info(
                                        f"发现工具: {exposed_name} - {tool.description}"
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
                                remote_tool_name, arguments
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
