"""One command catalog and implementation shared by every transport."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC

from al1s.config import AppSettings
from al1s.domain import (
    ChatKind,
    InboundEvent,
    KnowledgeScope,
    OutboundMessage,
    ResponsePlan,
    RoleDefinition,
    SessionKey,
)
from al1s.ports.mcp import ToolExecutionContext, ToolRegistry
from al1s.ports.repository import Repository
from al1s.ports.search import ImageSearchHit, ReverseImageSearch
from al1s.ports.vector import KnowledgeIndex

from .memory import MemoryService
from .profiles import ProfileError, ProfileService


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    group: str
    admin_only: bool = False
    private_only: bool = False
    private_response: bool = False


COMMANDS = (
    CommandSpec("start", "开始使用 AL1S", "basic"),
    CommandSpec("help", "显示命令帮助", "basic"),
    CommandSpec("ping", "检查机器人状态", "basic"),
    CommandSpec("role", "查看或切换角色", "roles"),
    CommandSpec("roles", "列出可用角色", "roles"),
    CommandSpec("create_role", "创建自定义角色", "roles"),
    CommandSpec("reset", "清空当前会话", "conversation"),
    CommandSpec("stats", "显示当前会话统计", "conversation"),
    CommandSpec("db_stats", "显示数据库统计", "conversation", admin_only=True),
    CommandSpec("my_stats", "显示个人使用统计", "conversation"),
    CommandSpec("rag_stats", "显示检索索引统计", "knowledge"),
    CommandSpec("knowledge", "列出当前作用域记忆", "knowledge"),
    CommandSpec("learn", "明确保存一条长期记忆", "knowledge"),
    CommandSpec("forget", "确认删除一条长期记忆", "knowledge"),
    CommandSpec("rebuild_index", "重建稠密向量索引", "knowledge", admin_only=True),
    CommandSpec("search", "通过图片 URL 反向搜索", "images"),
    CommandSpec("search_engines", "显示反向搜图后端", "images"),
    CommandSpec("test_search", "探测反向搜图服务", "images"),
    CommandSpec("tools", "显示当前可调用工具", "tools"),
    CommandSpec("mcp_status", "显示 MCP 连接状态", "tools"),
    CommandSpec("approve", "批准待执行工具", "tools", private_response=True),
    CommandSpec("deny", "拒绝待执行工具", "tools", private_response=True),
    CommandSpec("profile", "查看画像状态或模板", "profile", private_only=True),
    CommandSpec("profile_set", "替换私有画像", "profile", private_only=True),
    CommandSpec("profile_add", "追加私有画像", "profile", private_only=True),
    CommandSpec("profile_clear", "清除私有画像", "profile", private_only=True),
    CommandSpec("group_status", "查看群聊策略", "group"),
    CommandSpec("group_enable", "启用当前群聊", "group", admin_only=True),
    CommandSpec("group_disable", "停用当前群聊", "group", admin_only=True),
    CommandSpec("group_scope", "设置群聊会话作用域", "group", admin_only=True),
    CommandSpec("group_memory", "设置群聊长期记忆", "group", admin_only=True),
    CommandSpec("group_wake_words", "设置群聊唤醒词", "group", admin_only=True),
)
_COMMAND_MAP = {command.name: command for command in COMMANDS}
_ROLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True, slots=True)
class CommandContext:
    event: InboundEvent
    session: SessionKey
    identity_id: int
    chat_row_id: int

    @property
    def scope(self) -> KnowledgeScope:
        return KnowledgeScope(self.session.knowledge_scope)


class CommandBus:
    def __init__(
        self,
        database: Repository,
        settings: AppSettings,
        profiles: ProfileService,
        memory: MemoryService,
        knowledge: KnowledgeIndex,
        tools: ToolRegistry,
        reverse_image: ReverseImageSearch | None,
    ):
        self.database = database
        self.settings = settings
        self.profiles = profiles
        self.memory = memory
        self.knowledge = knowledge
        self.tools = tools
        self.reverse_image = reverse_image

    async def execute(self, context: CommandContext) -> ResponsePlan:
        invocation = context.event.command
        if invocation is None:
            raise ValueError("command invocation is required")
        if invocation.name == "_action":
            return await self._cmd_action(context, invocation.arguments)
        if invocation.name in {"approve", "deny"}:
            return await self._cmd_action(
                context, f"{invocation.name}:{invocation.arguments.strip()}"
            )
        spec = _COMMAND_MAP.get(invocation.name)
        if spec is None:
            return ResponsePlan.text(f"未知命令：/{invocation.name}")
        if spec.private_only and context.event.address.kind is not ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在私聊中使用。")
        if spec.admin_only and not context.event.permission.is_admin:
            return ResponsePlan.text("该命令需要管理员权限。")
        handler = getattr(self, f"_cmd_{spec.name}")
        return await handler(context, invocation.arguments.strip())

    def preflight(self, event: InboundEvent) -> ResponsePlan | None:
        if event.command is None or event.command.name == "_action":
            return None
        spec = _COMMAND_MAP.get(event.command.name)
        if spec is None:
            return None
        if spec.private_only and event.address.kind is not ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在私聊中使用。")
        if spec.admin_only and not event.permission.is_admin:
            return ResponsePlan.text("该命令需要管理员权限。")
        return None

    async def _cmd_action(self, context: CommandContext, arguments: str) -> ResponsePlan:
        parts = arguments.split(":", 1)
        if (
            len(parts) != 2
            or parts[0] not in {"approve", "deny"}
            or not re.fullmatch(r"[0-9a-f]{32}", parts[1])
        ):
            return ResponsePlan.text("无效的审批操作。", private=True)
        result = await self.tools.resolve_approval(
            parts[1],
            parts[0] == "approve",
            ToolExecutionContext(
                context.session.stable_id,
                context.event.actor.identity.key,
                context.event.permission,
            ),
        )
        return ResponsePlan(
            messages=(
                OutboundMessage(text=result.content, artifacts=result.artifacts, private=True),
            )
        )

    async def _cmd_start(self, context: CommandContext, arguments: str) -> ResponsePlan:
        conversation = await self.database.ensure_conversation(
            context.session, self.settings.default_role
        )
        owner = context.event.actor.identity.key if context.session.owner != "shared" else None
        role = await self.database.get_role(conversation.role_name, owner)
        greeting = role.greeting if role else "你好，我是 AL1S。"
        return ResponsePlan.text(f"{greeting}\n\n使用 /help 查看全部命令。")

    async def _cmd_help(self, context: CommandContext, arguments: str) -> ResponsePlan:
        groups: dict[str, list[CommandSpec]] = {}
        for command in COMMANDS:
            if command.private_only and context.event.address.kind is not ChatKind.PRIVATE:
                continue
            groups.setdefault(command.group, []).append(command)
        lines = ["AL1S 命令："]
        for group, commands in groups.items():
            lines.append(f"\n[{group}]")
            lines.extend(f"/{item.name} - {item.description}" for item in commands)
        return ResponsePlan.text("\n".join(lines))

    async def _cmd_ping(self, context: CommandContext, arguments: str) -> ResponsePlan:
        return ResponsePlan.text("Pong. 两个平台共用同一应用核心。")

    async def _cmd_role(self, context: CommandContext, arguments: str) -> ResponsePlan:
        conversation = await self.database.ensure_conversation(
            context.session, self.settings.default_role
        )
        owner = context.event.actor.identity.key if context.session.owner != "shared" else None
        if not arguments:
            role = await self.database.get_role(conversation.role_name, owner)
            if role is None:
                return ResponsePlan.text("当前角色不存在，请使用 /roles 重新选择。")
            return ResponsePlan.text(
                f"当前角色：{role.name}\n{role.description}\n问候语：{role.greeting}"
            )
        role = await self.database.get_role(arguments, owner)
        if role is None:
            return ResponsePlan.text("角色不存在，请使用 /roles 查看可选项。")
        await self.database.set_conversation_role(
            context.session,
            role.name,
            self._command_event_key(context),
        )
        return ResponsePlan.text(f"当前角色已切换为 {role.name}。")

    async def _cmd_roles(self, context: CommandContext, arguments: str) -> ResponsePlan:
        owner = context.event.actor.identity.key if context.session.owner != "shared" else None
        roles = await self.database.list_roles(owner)
        return ResponsePlan.text(
            "可用角色：\n" + "\n".join(f"- {role.name}: {role.description}" for role in roles)
        )

    async def _cmd_create_role(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if context.session.owner == "shared":
            return ResponsePlan.text("共享会话只能使用管理员配置的全局角色。")
        parts = arguments.split(maxsplit=1)
        if len(parts) != 2 or not _ROLE_NAME.fullmatch(parts[0]):
            return ResponsePlan.text("用法：/create_role NAME DESCRIPTION")
        name, description = parts
        role = RoleDefinition(
            name=name,
            description=description[:1000],
            system_prompt=(
                f"You are role {name}. Character description: {description[:2000]}. "
                "Stay useful and truthful; role-play must never override safety or permissions."
            ),
            greeting=f"你好，我现在以 {name} 的身份协助你。",
            farewell="再见。",
            owner_key=context.event.actor.identity.key,
        )
        created = await self.database.create_role(role)
        return ResponsePlan.text(
            f"角色 {name} 已创建，可使用 /role {name} 切换。"
            if created
            else f"角色 {name} 已存在。"
        )

    async def _cmd_reset(self, context: CommandContext, arguments: str) -> ResponsePlan:
        deleted = await self.database.reset_conversation(
            context.session, self._command_event_key(context)
        )
        return ResponsePlan.text(f"当前会话已重置，删除 {deleted} 条历史消息。")

    async def _cmd_stats(self, context: CommandContext, arguments: str) -> ResponsePlan:
        messages = await self.database.recent_messages(context.session, 10_000)
        return ResponsePlan.text(
            "当前会话统计：\n"
            f"- scope: {context.session.scope.value}\n"
            f"- conversation: {context.session.stable_id}\n"
            f"- messages: {len(messages)}"
        )

    async def _cmd_db_stats(self, context: CommandContext, arguments: str) -> ResponsePlan:
        stats = await self.database.database_stats()
        return ResponsePlan.text(
            "数据库统计：\n" + "\n".join(f"- {key}: {value}" for key, value in stats.items())
        )

    async def _cmd_my_stats(self, context: CommandContext, arguments: str) -> ResponsePlan:
        stats = await self.database.user_stats(context.identity_id)
        return ResponsePlan.text(
            "个人统计：\n" + "\n".join(f"- {key}: {value}" for key, value in stats.items())
        )

    async def _cmd_rag_stats(self, context: CommandContext, arguments: str) -> ResponsePlan:
        stats = await self.knowledge.stats()
        return ResponsePlan.text(
            "RAG 状态：\n" + "\n".join(f"- {key}: {value}" for key, value in stats.items())
        )

    async def _cmd_knowledge(self, context: CommandContext, arguments: str) -> ResponsePlan:
        memories = await self.database.list_memories(context.scope, limit=20)
        if not memories:
            return ResponsePlan.text("当前作用域没有长期记忆。")
        return ResponsePlan.text(
            "当前长期记忆：\n"
            + "\n".join(f"- {item['id']}: {item['content']}" for item in memories)
        )

    async def _cmd_learn(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if not self.settings.memory.enabled:
            return ResponsePlan.text("长期记忆功能未启用。")
        if context.event.address.kind is not ChatKind.PRIVATE:
            stored = await self.database.get_group_settings(context.chat_row_id)
            enabled = bool(stored and stored["memory_enabled"] and stored["consent_at"])
            if not enabled:
                return ResponsePlan.text("群聊长期记忆未启用，请先由管理员执行 /group_memory on。")
        if not arguments:
            return ResponsePlan.text("用法：/learn 要明确保存的内容")
        entry_id = await self.memory.remember(
            context.scope,
            arguments,
            entry_id=self._command_effect_id(context),
            effect_event_key=self._command_event_key(context),
        )
        if entry_id is None:
            return ResponsePlan.text("群聊长期记忆授权已变更，未保存内容。")
        return ResponsePlan.text(f"已保存长期记忆：{entry_id}")

    async def _cmd_forget(self, context: CommandContext, arguments: str) -> ResponsePlan:
        parts = arguments.split()
        if len(parts) == 1:
            return ResponsePlan.text(f"请再次确认：/forget {parts[0]} confirm")
        if len(parts) != 2 or parts[1].casefold() != "confirm":
            return ResponsePlan.text("用法：/forget MEMORY_ID confirm")
        deleted = await self.memory.forget(
            context.scope,
            parts[0],
            effect_event_key=self._command_event_key(context),
        )
        return ResponsePlan.text("记忆已删除。" if deleted else "未找到该作用域中的记忆。")

    async def _cmd_rebuild_index(self, context: CommandContext, arguments: str) -> ResponsePlan:
        rebuilt = await self.knowledge.rebuild_vectors()
        return ResponsePlan.text(f"向量索引已重建，共处理 {rebuilt} 条记录。")

    async def _cmd_search(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if self.reverse_image is None:
            return ResponsePlan.text("未启用反向搜图服务。")
        if not arguments:
            return ResponsePlan.text("用法：/search HTTPS_IMAGE_URL")
        try:
            hits = await self.reverse_image.search_url(arguments)
        except (RuntimeError, ValueError) as exc:
            return ResponsePlan.text(f"反向搜图失败：{exc}")
        return self._format_search_hits(hits)

    async def _cmd_search_engines(self, context: CommandContext, arguments: str) -> ResponsePlan:
        enabled = []
        config = self.settings.reverse_image
        if config.trace_moe_enabled:
            enabled.append("trace.moe")
        if config.saucenao_enabled:
            enabled.append("SauceNAO")
        return ResponsePlan.text("已启用反向搜图后端：" + (", ".join(enabled) or "无"))

    async def _cmd_test_search(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if self.reverse_image is None:
            return ResponsePlan.text("反向搜图服务未启用。")
        status = await self.reverse_image.health()
        return ResponsePlan.text(
            "反向搜图探测：\n" + "\n".join(f"- {key}: {value}" for key, value in status.items())
        )

    async def _cmd_tools(self, context: CommandContext, arguments: str) -> ResponsePlan:
        tools = self.tools.available(context.event.permission)
        if not tools:
            return ResponsePlan.text("当前身份没有可用工具。")
        return ResponsePlan.text(
            "可用工具：\n"
            + "\n".join(
                f"- {tool.definition.name}: {tool.definition.description}" for tool in tools
            )
        )

    async def _cmd_mcp_status(self, context: CommandContext, arguments: str) -> ResponsePlan:
        statuses = self.tools.status()
        if not statuses:
            return ResponsePlan.text("没有配置 MCP Server。")
        return ResponsePlan.text(
            "MCP 状态：\n"
            + "\n".join(
                "- " + ", ".join(f"{key}={value}" for key, value in status.items())
                for status in statuses
            )
        )

    async def _cmd_profile(self, context: CommandContext, arguments: str) -> ResponsePlan:
        content = await self.database.get_profile(context.identity_id)
        if content:
            return ResponsePlan.text(f"当前私有画像：\n\n{content}", private=True)
        return ResponsePlan.text(
            "尚未设置画像。模板如下：\n\n" + self.profiles.template(), private=True
        )

    async def _cmd_profile_set(self, context: CommandContext, arguments: str) -> ResponsePlan:
        arguments = self._profile_input(context, arguments)
        if not arguments:
            return ResponsePlan.text(
                "用法：/profile_set PROFILE_TEXT，或随命令上传 UTF-8 文本文件。",
                private=True,
            )
        try:
            await self.profiles.set(
                context.identity_id,
                arguments,
                event_key=self._command_event_key(context),
            )
        except ProfileError as exc:
            return ResponsePlan.text(str(exc), private=True)
        return ResponsePlan.text("私有画像已替换。", private=True)

    async def _cmd_profile_add(self, context: CommandContext, arguments: str) -> ResponsePlan:
        arguments = self._profile_input(context, arguments)
        if not arguments:
            return ResponsePlan.text(
                "用法：/profile_add PROFILE_TEXT，或随命令上传 UTF-8 文本文件。",
                private=True,
            )
        try:
            await self.profiles.append(
                context.identity_id,
                arguments,
                event_key=self._command_event_key(context),
            )
        except ProfileError as exc:
            return ResponsePlan.text(str(exc), private=True)
        return ResponsePlan.text("内容已追加到私有画像。", private=True)

    async def _cmd_profile_clear(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if arguments.casefold() != "confirm":
            return ResponsePlan.text("请使用 /profile_clear confirm 确认删除。", private=True)
        deleted = await self.database.clear_profile(
            context.identity_id, self._command_event_key(context)
        )
        return ResponsePlan.text(
            "私有画像已清除。" if deleted else "当前没有私有画像。", private=True
        )

    async def _cmd_group_status(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if context.event.address.kind is ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在群聊或服务器频道中使用。")
        stored = await self.database.get_group_settings(context.chat_row_id)
        group = self.settings.policies.group
        if stored:
            values = dict(stored)
            values["memory_enabled"] = bool(
                self.settings.memory.enabled and stored["memory_enabled"] and stored["consent_at"]
            )
        else:
            values = {
                "enabled": group.enabled,
                "session_scope": group.session_scope,
                "memory_enabled": False,
                "wake_words": list(group.wake_words),
            }
        return ResponsePlan.text(
            "群聊策略：\n"
            f"- enabled: {values['enabled']}\n"
            f"- scope: {values['session_scope']}\n"
            f"- memory: {values['memory_enabled']}\n"
            f"- wake words: {', '.join(values['wake_words'])}"
        )

    async def _cmd_group_enable(self, context: CommandContext, arguments: str) -> ResponsePlan:
        return await self._set_group_flag(context, "enabled", True, "当前群聊已启用。")

    async def _cmd_group_disable(self, context: CommandContext, arguments: str) -> ResponsePlan:
        return await self._set_group_flag(context, "enabled", False, "当前群聊已停用。")

    async def _cmd_group_scope(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if context.event.address.kind is ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在群聊或服务器频道中使用。")
        if arguments not in {"per_user", "shared", "topic"}:
            return ResponsePlan.text("用法：/group_scope per_user|shared|topic")
        await self._set_group_settings(context, session_scope=arguments)
        return ResponsePlan.text(f"群聊会话作用域已设置为 {arguments}。")

    async def _cmd_group_memory(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if context.event.address.kind is ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在群聊或服务器频道中使用。")
        if arguments not in {"on", "off"}:
            return ResponsePlan.text("用法：/group_memory on|off")
        enabled = arguments == "on"
        if enabled and not self.settings.memory.enabled:
            return ResponsePlan.text("长期记忆功能未启用。")
        await self._set_group_settings(
            context,
            memory_enabled=enabled,
            consent_at=context.event.timestamp.astimezone(UTC).isoformat() if enabled else None,
        )
        return ResponsePlan.text("群聊长期记忆已启用。" if enabled else "群聊长期记忆已停用。")

    async def _cmd_group_wake_words(self, context: CommandContext, arguments: str) -> ResponsePlan:
        if context.event.address.kind is ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在群聊或服务器频道中使用。")
        words = [] if arguments.casefold() == "clear" else arguments.split()
        if not arguments:
            return ResponsePlan.text("用法：/group_wake_words WORD...，或使用 clear 清空。")
        words = list(dict.fromkeys(word.strip() for word in words if word.strip()))[:20]
        await self._set_group_settings(context, wake_words=words)
        return ResponsePlan.text("群聊唤醒词已更新：" + (", ".join(words) or "无"))

    async def _set_group_flag(
        self, context: CommandContext, key: str, value: bool, message: str
    ) -> ResponsePlan:
        if context.event.address.kind is ChatKind.PRIVATE:
            return ResponsePlan.text("该命令只能在群聊或服务器频道中使用。")
        await self._set_group_settings(context, **{key: value})
        return ResponsePlan.text(message)

    async def _set_group_settings(self, context: CommandContext, **values: object) -> None:
        group = self.settings.policies.group
        defaults = {
            "enabled": group.enabled,
            "session_scope": group.session_scope,
            "memory_enabled": False,
            "wake_words": list(group.wake_words),
            "consent_at": None,
        }
        await self.database.set_group_settings(
            context.chat_row_id,
            defaults,
            event_key=self._command_event_key(context),
            **values,
        )

    @staticmethod
    def _command_event_key(context: CommandContext) -> str:
        return "\x1f".join(context.event.idempotency_key)

    @classmethod
    def _command_effect_id(cls, context: CommandContext) -> str:
        return hashlib.blake2b(cls._command_event_key(context).encode(), digest_size=16).hexdigest()

    @staticmethod
    def _profile_input(context: CommandContext, arguments: str) -> str:
        if arguments:
            return arguments
        for attachment in context.event.attachments:
            if attachment.mime_type in {
                "text/plain",
                "text/markdown",
                "application/octet-stream",
            }:
                try:
                    return attachment.data.decode("utf-8")
                except UnicodeDecodeError:
                    return ""
        return ""

    @staticmethod
    def _format_search_hits(hits: tuple[ImageSearchHit, ...]) -> ResponsePlan:
        if not hits:
            return ResponsePlan.text("没有找到匹配结果。")
        lines = ["反向搜图结果："]
        for index, hit in enumerate(hits, start=1):
            similarity = f" ({hit.similarity:.1%})" if hit.similarity is not None else ""
            lines.append(f"{index}. [{hit.engine}] {hit.title}{similarity}\n{hit.url}")
        return ResponsePlan.text("\n".join(lines))
