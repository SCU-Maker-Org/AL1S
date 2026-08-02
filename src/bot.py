"""
Telegram机器人主类
"""

import asyncio
from typing import Dict, Optional

from loguru import logger
from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agents.langchain_agent_service import LangChainAgentService
from .agents.unified_agent import UnifiedAgentService
from .config import config
from .handlers.chat_handler import ChatHandler
from .handlers.command_handler import CommandHandler as CmdHandler
from .handlers.image_handler import ImageHandler
from .infra.database import DatabaseService
from .infra.mcp import MCPService
from .services.ascii2d_service import Ascii2DService
from .services.conversation_service import ConversationService
from .services.group_chat_service import GroupChatService
from .services.rate_limit_service import RateLimitService
from .utils.database_logger import init_database_logger


class AL1SBot:
    """AL1S-Bot主类"""

    # 命令配置表 - 分组管理
    COMMAND_GROUPS = {
        "basic": {
            "start": "开始使用机器人",
            "help": "显示帮助信息",
        },
        "roles": {
            "role": "设置或查看当前角色",
            "roles": "显示所有可用角色",
            "create_role": "创建自定义角色",
            "reset": "重置当前对话",
        },
        "stats": {
            "stats": "显示对话统计信息",
            "ping": "测试机器人响应",
            "db_stats": "显示数据库统计信息",
            "my_stats": "显示我的使用统计",
            "rag_stats": "显示Agent学习统计",
        },
        "search": {
            "search": "搜索图片URL",
            "search_engines": "显示可用的搜索引擎",
            "test_search": "测试图片搜索服务",
        },
        "tools": {
            "tools": "显示可用的MCP工具",
            "mcp_status": "显示MCP服务器状态",
        },
        "knowledge": {
            "knowledge": "管理知识库",
            "learn": "自动学习说明",
            "forget": "知识清理说明",
            "rebuild_index": "索引重建说明",
        },
        "group": {
            "group_status": "查看当前群聊配置",
            "group_enable": "在当前群启用机器人",
            "group_disable": "在当前群停用机器人",
            "group_scope": "设置当前群会话作用域",
            "group_memory": "设置当前群长期学习",
            "group_wake_words": "设置当前群唤醒词",
        },
    }

    @property
    def COMMANDS(self):
        """获取所有命令的扁平化字典"""
        commands = {}
        for group_commands in self.COMMAND_GROUPS.values():
            commands.update(group_commands)
        return commands

    def __init__(self):
        self.config = config
        self.application: Optional[Application] = None

        # 初始化MCP服务
        self.mcp_service = MCPService() if config.mcp.enabled else None

        # 初始化服务（传入MCP工具处理器）
        self.database_service = DatabaseService()

        # 初始化数据库记录器
        init_database_logger(self.database_service)

        self.ascii2d_service = Ascii2DService()
        self.conversation_service = ConversationService(
            database_service=self.database_service
        )
        self.group_chat_service = GroupChatService(config.telegram.group)
        self.rate_limit_service = RateLimitService(config.telegram.rate_limit)

        # 根据配置选择 Agent 类型（互斥）
        self.unified_agent_service = None
        self.langchain_agent_service = None

        if config.agent.type == "langchain" and config.langchain.enabled:
            # 使用 LangChain Agent
            try:
                self.langchain_agent_service = LangChainAgentService(
                    database_service=self.database_service,
                    mcp_service=self.mcp_service,
                    vector_store_path=config.agent.vector_store_path,
                )
                logger.info("LangChain Agent 服务已创建")
            except Exception as e:
                logger.error(f"创建 LangChain Agent 服务失败: {e}")
                logger.warning("回退到统一 Agent 服务")
                config.agent.type = "unified"

        if config.agent.type == "unified":
            # 使用统一 Agent
            try:
                self.unified_agent_service = UnifiedAgentService(
                    database_service=self.database_service,
                    mcp_service=self.mcp_service,
                    vector_store_path=config.agent.vector_store_path,
                )
                logger.info("统一 Agent 服务已创建")
            except Exception as e:
                logger.error(f"创建统一 Agent 服务失败: {e}")
                raise

        # 初始化处理器（根据 Agent 类型选择）
        self.active_agent_service = (
            self.langchain_agent_service or self.unified_agent_service
        )

        self.chat_handler = ChatHandler(
            self.active_agent_service,
            self.conversation_service,
            self.mcp_service,
            self.database_service,
            self.group_chat_service,
            self.rate_limit_service,
        )
        self.image_handler = ImageHandler(
            self.ascii2d_service,
            self.active_agent_service,
            self.conversation_service,
            self.group_chat_service,
            self.rate_limit_service,
        )

        # 处理器列表
        self.handlers = [self.chat_handler, self.image_handler]

        logger.info("AL1S-Bot 初始化完成")

    def _create_tool_handler(self):
        """创建MCP工具处理器"""

        async def tool_handler(tool_name: str, arguments: dict):
            """处理工具调用"""
            if self.mcp_service:
                return await self.mcp_service.call_tool(tool_name, arguments)
            return None

        return tool_handler

    async def _initialize_mcp_servers(self):
        """初始化MCP服务器"""
        if self.mcp_service and self.config.mcp.enabled:
            # 转换配置格式
            mcp_configs = [
                server_config
                for server_config in self.config.mcp.servers
                if server_config.enabled
            ]

            if mcp_configs:
                logger.info(f"正在初始化 {len(mcp_configs)} 个MCP服务器...")
                await self.mcp_service.initialize_default_servers(mcp_configs)

                # 显示已连接的工具
                tools = self.mcp_service.get_available_tools("admin")
                if tools:
                    logger.info(f"已加载 {len(tools)} 个MCP工具: {list(tools.keys())}")
                else:
                    logger.warning("未找到任何可用的MCP工具")
            else:
                logger.info("没有启用的MCP服务器")

    async def _post_init_callback(self, application):
        """应用初始化后的回调，用于初始化MCP服务器和RAG服务"""
        if self.mcp_service:
            logger.info("正在初始化MCP服务...")
            await self._initialize_mcp_servers()

        # 初始化活动的 Agent 服务
        if self.unified_agent_service:
            logger.info("正在初始化统一 Agent 服务...")
            try:
                await self.unified_agent_service.initialize()
                logger.info("统一 Agent 服务初始化完成")
            except Exception as e:
                logger.error(f"统一 Agent 服务初始化失败: {e}")

        if self.langchain_agent_service:
            logger.info("正在初始化 LangChain Agent 服务...")
            try:
                await self.langchain_agent_service.initialize()
                logger.info("LangChain Agent 服务初始化完成")
            except Exception as e:
                logger.error(f"LangChain Agent 服务初始化失败: {e}")
                self.langchain_agent_service = None

    def start(self):
        """启动机器人（支持优雅关闭的同步方法）"""
        try:
            # 创建应用
            self.application = (
                Application.builder()
                .token(self.config.telegram.bot_token)
                .concurrent_updates(True)
                .build()
            )

            # 设置处理器
            self._setup_handlers()

            # 设置错误处理器
            self.application.add_error_handler(self._error_handler)

            # 设置应用初始化后的回调
            if (
                self.mcp_service
                or self.unified_agent_service
                or self.langchain_agent_service
            ):
                self.application.post_init = self._post_init_callback

            # 启动机器人
            logger.info("启动轮询模式...")

            # 使用标准的run_polling方法，MCP初始化将在post_init中进行
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False,  # 不关闭事件循环
            )

        except KeyboardInterrupt:
            logger.info("收到键盘中断，正在关闭机器人...")
            self._cleanup()
        except Exception as e:
            logger.error(f"启动机器人失败: {e}")
            self._cleanup()
            raise

    def _setup_handlers(self):
        """设置消息处理器"""
        # 命令处理器（传入图片处理器引用）
        self.command_handler = CmdHandler(
            self.conversation_service,
            self.active_agent_service,
            self.image_handler,  # 传入图片处理器引用
            self.mcp_service,  # 传入MCP服务引用
            self.database_service,  # 传入数据库服务引用
            self.group_chat_service,
        )

        # 批量注册命令处理器
        self._register_command_handlers()

        self.application.add_handler(
            ChatMemberHandler(
                self._handle_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER
            )
        )
        self.application.add_handler(
            ChatMemberHandler(self._handle_chat_member, ChatMemberHandler.CHAT_MEMBER)
        )

        # 在业务过滤前记录消息元数据，避免无法区分 Telegram 未投递与策略拒绝。
        self.application.add_handler(
            MessageHandler(filters.ALL, self._log_message_ingress), group=-1
        )

        # 图片处理器
        self.application.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._handle_image)
        )

        # 通用消息处理器
        self.application.add_handler(
            MessageHandler(
                (filters.TEXT | filters.CAPTION) & ~filters.COMMAND,
                self._handle_message,
            )
        )

    def _register_command_handlers(self):
        """批量注册命令处理器"""
        registered_count = 0

        # 分组注册命令
        for group_name, group_commands in self.COMMAND_GROUPS.items():
            # 检查是否应该启用这个命令组
            if not self._should_enable_command_group(group_name):
                logger.info(f"跳过命令组: {group_name}")
                continue

            # 注册该组的所有命令
            for command in group_commands.keys():
                handler_name = f"_handle_{command}_command"
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    guarded = self._guard_command(
                        handler, management_command=group_name == "group"
                    )
                    self.application.add_handler(CommandHandler(command, guarded))
                    registered_count += 1
                else:
                    logger.warning(f"命令处理器不存在: {handler_name}")

        logger.info(f"成功注册 {registered_count} 个命令处理器")

    def _should_enable_command_group(self, group_name: str) -> bool:
        """判断是否应该启用某个命令组"""
        # 基础命令和角色管理始终启用
        if group_name in ["basic", "roles", "stats"]:
            return True

        # MCP工具相关命令
        if group_name == "tools":
            return self.mcp_service is not None

        # 搜索功能
        if group_name == "search":
            return hasattr(self, "ascii2d_service") and self.ascii2d_service is not None

        # 知识管理功能
        if group_name == "knowledge":
            return self.active_agent_service is not None

        return True  # 默认启用

    def _guard_command(self, handler, *, management_command: bool = False):
        """统一执行群权限、命令目标、去重和限流检查。"""

        async def guarded(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if await self.group_chat_service.is_duplicate_update(
                getattr(update, "update_id", None)
            ):
                logger.warning("忽略重复命令 Update update_id={}", update.update_id)
                return
            username = getattr(context.bot, "username", None)
            bot_id = getattr(context.bot, "id", None)
            if not username or not bot_id:
                me = await context.bot.get_me()
                username, bot_id = me.username or "", me.id
            decision = self.group_chat_service.decide(
                update,
                str(username),
                int(bot_id),
                is_command=True,
                management_command=management_command,
            )
            session_key = self.group_chat_service.session_key(update)
            if not decision.allowed:
                logger.bind(
                    chat_id=session_key.chat_id,
                    thread_id=session_key.thread_id,
                    user_id=update.effective_user.id,
                    message_id=update.effective_message.message_id,
                    session_scope=session_key.scope,
                    trigger_type="command",
                ).info(
                    "group_command_decision allowed=false denied_reason={} agent_called=false",
                    decision.denied_reason,
                )
                return
            rate = await self.rate_limit_service.check(
                update.effective_user.id,
                session_key.chat_id,
                session_key.thread_id,
            )
            if not rate.allowed:
                if rate.notify:
                    await self._reply(update, context, "请求过于频繁，请稍后再试。")
                return
            await handler(update, context)

        return guarded

    async def _reply(self, update: Update, context, *args, **kwargs):
        kwargs.setdefault(
            "message_thread_id",
            getattr(update.effective_message, "message_thread_id", None),
        )
        try:
            return await update.effective_message.reply_text(*args, **kwargs)
        except Exception as exc:
            logger.warning("命令引用回复失败，回退为 Topic 普通消息: {}", exc)
            return await context.bot.send_message(
                update.effective_chat.id, *args, **kwargs
            )

    async def _require_group_admin(self, update: Update, context) -> bool:
        if not self.group_chat_service.is_group(update):
            await self._reply(update, context, "该命令只能在群聊中使用。")
            return False
        allowed = await self.group_chat_service.is_admin(
            context,
            update.effective_chat.id,
            update.effective_user.id,
            self.config.telegram.admin_user_ids,
        )
        if not allowed:
            await self._reply(update, context, "权限不足：仅群管理员可执行该命令。")
        return allowed

    def get_enabled_commands(self) -> Dict[str, str]:
        """获取当前启用的命令列表"""
        enabled_commands = {}
        for group_name, group_commands in self.COMMAND_GROUPS.items():
            if self._should_enable_command_group(group_name):
                enabled_commands.update(group_commands)
        return enabled_commands

    def get_command_groups_status(self) -> Dict[str, bool]:
        """获取各命令组的启用状态"""
        return {
            group_name: self._should_enable_command_group(group_name)
            for group_name in self.COMMAND_GROUPS.keys()
        }

    async def _register_commands_async(self):
        """异步注册机器人命令列表"""
        try:
            # 只注册启用的命令组中的命令
            commands = []
            for group_name, group_commands in self.COMMAND_GROUPS.items():
                if self._should_enable_command_group(group_name):
                    for cmd, desc in group_commands.items():
                        commands.append(BotCommand(cmd, desc))

            # 注册命令列表
            await self.application.bot.set_my_commands(commands)
            logger.info(f"异步成功注册 {len(commands)} 个命令到菜单")

        except Exception as e:
            logger.error(f"异步注册命令失败: {e}")
            raise

    async def stop(self):
        """停止机器人"""
        try:
            if self.application:
                logger.info("正在停止机器人应用...")
                await self.application.stop()
                await self.application.shutdown()
                self.application = None
                logger.info("机器人已停止")
        except Exception as e:
            logger.error(f"停止机器人时发生错误: {e}")

        # 清理统一 Agent 服务资源
        if hasattr(self, "unified_agent_service") and self.unified_agent_service:
            try:
                self.unified_agent_service.cleanup()
                logger.debug("已清理统一 Agent 服务资源")
            except Exception as e:
                logger.debug(f"清理统一 Agent 服务资源失败: {e}")

        # 清理 LangChain Agent 资源
        if hasattr(self, "langchain_agent_service") and self.langchain_agent_service:
            try:
                self.langchain_agent_service.cleanup()
                logger.debug("已清理 LangChain Agent 资源")
            except Exception as e:
                logger.debug(f"清理 LangChain Agent 资源失败: {e}")

    # 命令处理方法
    async def _handle_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理start命令"""
        # 在首次使用时注册命令列表
        if not hasattr(self, "_commands_registered"):
            try:
                await self._register_commands_async()
                self._commands_registered = True
                logger.info("首次使用，命令列表注册完成")
            except Exception as e:
                logger.error(f"首次命令注册失败: {e}")

        # 调用原来的start处理
        await self.command_handler._handle_start(update, context)

    async def _handle_help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理help命令 - 显示分组的命令帮助"""
        try:
            help_text = "🤖 <b>AL1S-Bot 帮助</b>\n\n"

            # 命令组显示名称映射
            group_names = {
                "basic": "🏠 基础命令",
                "roles": "🎭 角色管理",
                "stats": "📊 统计信息",
                "search": "🔍 图片搜索",
                "tools": "🛠️ MCP工具",
                "knowledge": "🧠 知识管理",
                "group": "👥 群聊管理",
            }

            # 按组显示启用的命令
            for group_name, group_commands in self.COMMAND_GROUPS.items():
                if self._should_enable_command_group(group_name):
                    group_display_name = group_names.get(group_name, group_name.title())
                    help_text += f"<b>{group_display_name}</b>\n"

                    for cmd, desc in group_commands.items():
                        help_text += f"  /{cmd} - {desc}\n"
                    help_text += "\n"

            help_text += "💡 <i>提示：直接发送消息即可与AI对话</i>"

            await self._reply(update, context, help_text, parse_mode="HTML")

        except Exception as e:
            logger.error(f"处理帮助命令失败: {e}")
            await self._reply(update, context, "❌ 获取帮助信息失败")

    async def _handle_role_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理role命令"""
        args = (
            update.message.text.split(maxsplit=1)[1]
            if len(update.message.text.split()) > 1
            else ""
        )
        await self.command_handler._handle_role(update, context, args)

    async def _handle_roles_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理roles命令"""
        await self.command_handler._handle_roles(update, context)

    async def _handle_create_role_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理create_role命令"""
        args = (
            update.message.text.split(maxsplit=1)[1]
            if len(update.message.text.split()) > 1
            else ""
        )
        await self.command_handler._handle_create_role(update, context, args)

    async def _handle_reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理reset命令"""
        await self.command_handler._handle_reset(update, context)

    async def _handle_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理stats命令"""
        await self.command_handler._handle_stats(update, context)

    async def _handle_ping_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理ping命令"""
        await self.command_handler._handle_ping(update, context)

    async def _handle_search_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理search命令"""
        args = (
            update.message.text.split(maxsplit=1)[1]
            if len(update.message.text.split()) > 1
            else ""
        )
        await self.command_handler._handle_search(update, context, args)

    async def _handle_search_engines_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理search_engines命令"""
        await self.command_handler._handle_search_engines(update, context)

    async def _handle_test_search_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理test_search命令"""
        await self.command_handler._handle_test_search(update, context)

    async def _handle_tools_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理tools命令"""
        await self.command_handler._handle_tools(update, context)

    async def _handle_mcp_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理mcp_status命令"""
        await self.command_handler._handle_mcp_status(update, context)

    async def _handle_db_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理db_stats命令"""
        await self.command_handler._handle_db_stats(update, context)

    async def _handle_my_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理my_stats命令"""
        await self.command_handler._handle_my_stats(update, context)

    async def _handle_rag_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理rag_stats命令"""
        try:
            if not self.active_agent_service:
                await self._reply(update, context, "❌ Agent服务未启用")
                return

            # 尝试获取学习统计信息
            if hasattr(self.active_agent_service, "get_learning_statistics"):
                stats = await self.active_agent_service.get_learning_statistics()
            else:
                stats = {"message": "当前Agent服务不支持统计信息"}

            message = "📊 <b>RAG/学习统计</b>\n\n"

            # 兼容两种返回结构：
            # 1) 新版 LearningService 返回的平铺统计，如 total_learned、total_knowledge_in_db 等
            # 2) 旧版按表名分组的字典结构（包含 vector_index / knowledge_entries 等）
            if isinstance(stats, dict):
                has_dict_values = any(isinstance(v, dict) for v in stats.values())

                if (
                    ("total_learned" in stats)
                    or ("total_knowledge_in_db" in stats)
                    or not has_dict_values
                ):
                    # 新版/平铺结构
                    total_learned = int(stats.get("total_learned", 0) or 0)
                    failed = int(stats.get("failed_extractions", 0) or 0)
                    deduped = int(stats.get("duplicate_filtered", 0) or 0)
                    sessions = int(stats.get("learning_sessions", 0) or 0)
                    total_in_db = int(stats.get("total_knowledge_in_db", 0) or 0)
                    last_time = stats.get("last_learning_time")

                    message += "🧠 <b>学习概览</b>\n"
                    message += f"• 学到知识: {total_learned}\n"
                    message += f"• 失败抽取: {failed}\n"
                    message += f"• 去重过滤: {deduped}\n"
                    message += f"• 学习会话: {sessions}\n"
                    message += f"• 知识库存量: {total_in_db}\n"
                    if last_time:
                        message += f"• 最近学习: {last_time}\n"
                else:
                    # 旧版/分表结构
                    for table_name, info in stats.items():
                        if not isinstance(info, dict):
                            message += f"• {table_name}: {info}\n"
                            continue

                        if table_name == "vector_index":
                            message += "🔍 <b>向量索引</b>\n"
                            message += f"• 向量数量: {info.get('total_vectors', 0)}\n"
                            message += f"• 向量维度: {info.get('dimension', '-')}\n"
                            message += f"• 索引类型: {info.get('index_type', '-')}\n\n"
                        else:
                            display_name = {
                                "knowledge_entries": "知识条目",
                                "embeddings": "向量嵌入",
                                "knowledge_retrievals": "检索记录",
                            }.get(table_name, table_name)

                            message += f"📚 <b>{display_name}</b>\n"
                            message += f"• 总数量: {info.get('total_count', 0)}\n"
                            message += f"• 用户数: {info.get('unique_users', 0)}\n"

                            metric_name = (
                                "平均重要性"
                                if table_name == "knowledge_entries"
                                else (
                                    "平均维度"
                                    if table_name == "embeddings"
                                    else "使用率"
                                )
                            )
                            add_metric = info.get("additional_metric")
                            if isinstance(add_metric, (int, float)):
                                message += f"• {metric_name}: {add_metric:.2f}\n"
                            elif add_metric is not None:
                                message += f"• {metric_name}: {add_metric}\n"

                            last_created = info.get("last_created")
                            if last_created:
                                message += f"• 最后更新: {last_created}\n"
                            message += "\n"

            await self._reply(update, context, message, parse_mode="HTML")

        except Exception as e:
            logger.error(f"获取RAG统计失败: {e}")
            await self._reply(update, context, "❌ 获取RAG统计失败")

    async def _handle_knowledge_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """简化的知识命令处理"""
        try:
            args = context.args

            if not args:
                await self._reply(
                    update,
                    context,
                    "📚 <b>知识管理</b>\n\n"
                    "新版本中，知识通过自动学习功能从对话中提取。\n"
                    "您可以通过与机器人对话来自动积累知识！\n\n"
                    "可用命令:\n"
                    "• <code>/knowledge search 关键词</code> - 搜索知识\n"
                    "• <code>/rag_stats</code> - 查看学习统计",
                    parse_mode="HTML",
                )
                return

            command = args[0].lower()

            if command == "search" and len(args) > 1:
                query = " ".join(args[1:])
                # 使用新的向量搜索服务
                if hasattr(self.active_agent_service, "vector_service"):
                    results = await self.active_agent_service.vector_service.search_knowledge(
                        query,
                        top_k=3,
                        knowledge_namespace=self.group_chat_service.knowledge_namespace(
                            self.group_chat_service.session_key(update)
                        ),
                    )
                    if results:
                        message = f"🔍 <b>搜索结果：{query}</b>\n\n"
                        for i, result in enumerate(results, 1):
                            title = result.get("title", "无标题")
                            content = result.get("content", result.get("summary", ""))[
                                :100
                            ]
                            score = result.get("similarity_score", 0)
                            message += f"{i}. <b>{title}</b>\n{content}...\n相似度: {score:.2f}\n\n"
                        await self._reply(update, context, message, parse_mode="HTML")
                    else:
                        await self._reply(update, context, "🔍 没有找到相关知识")
                else:
                    await self._reply(update, context, "❌ 搜索功能不可用")
            else:
                await self._reply(update, context, "❌ 无效的命令格式")

        except Exception as e:
            logger.error(f"处理知识命令失败: {e}")
            await self._reply(update, context, "❌ 处理命令失败")

    async def _handle_learn_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理learn命令 - 手动触发学习"""
        try:
            await self._reply(
                update,
                context,
                "🧠 <b>自动学习</b>\n\n"
                "新版本中，机器人会自动从对话中学习。\n"
                "无需手动触发学习过程！\n\n"
                "使用 <code>/rag_stats</code> 查看学习进度。",
                parse_mode="HTML",
            )

        except Exception as e:
            logger.error(f"处理学习命令失败: {e}")
            await self._reply(update, context, "❌ 学习失败")

    async def _handle_forget_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理forget命令 - 清理旧知识"""
        try:
            await self._reply(
                update,
                context,
                "🗑️ <b>知识清理</b>\n\n"
                "如需清理知识库，请使用新的优化功能：\n"
                "机器人会自动管理知识生命周期。",
                parse_mode="HTML",
            )

        except Exception as e:
            logger.error(f"处理遗忘命令失败: {e}")
            await self._reply(update, context, "❌ 清理失败")

    async def _handle_rebuild_index_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """处理rebuild_index命令 - 重建向量索引"""
        try:
            await self._reply(
                update,
                context,
                "🔄 <b>重建索引</b>\n\n"
                "新架构中，向量索引会自动维护。\n"
                "如需重新初始化，请重启机器人。",
                parse_mode="HTML",
            )

        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            await self._reply(update, context, "❌ 重建索引失败")

    async def _handle_group_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not await self._require_group_admin(update, context):
            return
        chat_id = update.effective_chat.id
        thread_id = getattr(update.effective_message, "message_thread_id", None) or 0
        group = self.config.telegram.group
        count = await self.group_chat_service.context_count(chat_id, thread_id)
        text = (
            "<b>群聊配置</b>\n\n"
            f"chat_id: <code>{chat_id}</code>\n"
            f"message_thread_id: <code>{thread_id}</code>\n"
            f"群聊功能: {'启用' if self.group_chat_service.is_enabled(chat_id) else '停用'}\n"
            f"会话作用域: <code>{self.group_chat_service.session_scope(chat_id)}</code>\n"
            f"要求触发词/提及: {'是' if group.require_mention else '否'}\n"
            f"记录旁听上下文: {'是' if group.observe_unmentioned_messages else '否'}\n"
            f"上下文缓冲: {count}/{group.context_buffer_size}\n"
            f"长期群知识学习: {'启用' if self.group_chat_service.memory_enabled(chat_id) else '停用'}"
        )
        await self._reply(update, context, text, parse_mode="HTML")

    async def _handle_group_enable_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if await self._require_group_admin(update, context):
            self.group_chat_service.set_enabled(update.effective_chat.id, True)
            await self._reply(update, context, "已在当前群启用机器人。")

    async def _handle_group_disable_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if await self._require_group_admin(update, context):
            self.group_chat_service.set_enabled(update.effective_chat.id, False)
            await self._reply(update, context, "已在当前群停用机器人。")

    async def _handle_group_scope_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not await self._require_group_admin(update, context):
            return
        scope = context.args[0].lower() if context.args else ""
        if not self.group_chat_service.set_scope(update.effective_chat.id, scope):
            await self._reply(
                update,
                context,
                "用法：/group_scope per_user|shared|topic",
            )
            return
        await self._reply(update, context, f"当前群会话作用域已设置为 {scope}。")

    async def _handle_group_memory_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not await self._require_group_admin(update, context):
            return
        value = context.args[0].lower() if context.args else ""
        if value not in {"on", "off"}:
            await self._reply(update, context, "用法：/group_memory on|off")
            return
        if not self.group_chat_service.set_memory_enabled(
            update.effective_chat.id, value == "on"
        ):
            await self._reply(update, context, "配置禁止管理员切换群知识学习。")
            return
        await self._reply(
            update,
            context,
            f"当前群长期学习已{'启用' if value == 'on' else '停用'}。",
        )

    async def _handle_group_wake_words_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        if not await self._require_group_admin(update, context):
            return
        words = [word.strip() for arg in context.args for word in arg.split(",")]
        words = [word for word in words if word]
        if not words:
            current = (
                ", ".join(self.group_chat_service.wake_words(update.effective_chat.id))
                or "（无）"
            )
            await self._reply(update, context, f"当前唤醒词：{current}")
            return
        self.group_chat_service.set_wake_words(update.effective_chat.id, words)
        await self._reply(update, context, "当前群唤醒词已更新。")

    @staticmethod
    def _member_status(member) -> str:
        status = getattr(member, "status", "")
        return str(getattr(status, "value", status)).lower()

    async def _handle_my_chat_member(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        event = update.my_chat_member
        if not event or not update.effective_chat:
            return
        old_status = self._member_status(event.old_chat_member)
        new_status = self._member_status(event.new_chat_member)
        logger.bind(
            chat_id=update.effective_chat.id,
            old_status=old_status,
            new_status=new_status,
        ).info("bot_chat_member_changed")
        joined = old_status in {"left", "kicked"} and new_status in {
            "member",
            "administrator",
        }
        promoted = old_status == "member" and new_status == "administrator"
        removed = new_status in {"left", "kicked"}
        if removed:
            return
        if joined:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=(
                        "已加入群聊。\n\n"
                        "默认仅在 @提及、回复机器人或使用唤醒词时响应。\n"
                        "管理员可使用 /group_status 查看当前配置。"
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "发送入群说明失败 chat_id={} error={}",
                    update.effective_chat.id,
                    exc,
                )
        elif promoted:
            logger.info("机器人已在群 {} 被提升为管理员", update.effective_chat.id)

    async def _handle_chat_member(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        event = update.chat_member
        if not event:
            return
        logger.bind(
            chat_id=update.effective_chat.id if update.effective_chat else None,
            user_id=getattr(event.new_chat_member.user, "id", None),
            old_status=self._member_status(event.old_chat_member),
            new_status=self._member_status(event.new_chat_member),
        ).debug("chat_member_changed")

    # 消息处理方法
    async def _log_message_ingress(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """记录不含正文的 Telegram 消息入口元数据。"""
        message = update.effective_message
        if not message:
            return
        entities = list(getattr(message, "entities", None) or []) + list(
            getattr(message, "caption_entities", None) or []
        )
        entity_types = [
            str(getattr(getattr(entity, "type", None), "value", entity.type))
            for entity in entities
        ]
        sender_chat = getattr(message, "sender_chat", None)
        logger.bind(
            update_id=update.update_id,
            chat_id=getattr(update.effective_chat, "id", None),
            chat_type=str(getattr(update.effective_chat, "type", "")),
            user_id=getattr(update.effective_user, "id", None),
            sender_chat_id=getattr(sender_chat, "id", None),
            message_id=getattr(message, "message_id", None),
        ).info(
            "telegram_message_ingress text={} caption={} entities={}",
            bool(getattr(message, "text", None)),
            bool(getattr(message, "caption", None)),
            entity_types,
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理文本消息"""
        await self.chat_handler.handle(update, context)

    async def _handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理图片消息"""
        await self.image_handler.handle(update, context)

    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """错误处理器"""
        logger.error(f"处理更新时发生错误: {context.error}")

        try:
            if update and update.message:
                await self._reply(
                    update, context, "抱歉，处理您的请求时发生了错误，请稍后再试。"
                )
        except Exception as e:
            logger.error(f"发送错误消息失败: {e}")

    def _cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, "application") and self.application:
                logger.info("正在清理机器人资源...")
                # 这里只记录日志，不调用异步方法
                logger.info("资源清理完成")

            # 清理MCP服务
            if hasattr(self, "mcp_service") and self.mcp_service:
                logger.info("正在清理MCP服务...")
                try:
                    asyncio.run(self.mcp_service.close_all())
                    logger.info("MCP服务清理完成")
                except Exception as mcp_error:
                    logger.error(f"清理MCP服务失败: {mcp_error}")

        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")

    def stop_sync(self):
        """同步停止机器人"""
        try:
            if hasattr(self, "application") and self.application:
                logger.info("正在同步停止机器人...")
                # 对于同步方法，我们只能记录日志
                # 实际的停止操作由信号处理器处理
                logger.info("机器人同步停止完成")
        except Exception as e:
            logger.error(f"同步停止机器人时发生错误: {e}")

    def get_status(self) -> dict:
        """获取机器人状态"""
        return {
            "status": "running" if self.application else "stopped",
            "config": {
                "openai_model": self.config.openai.model,
                "telegram_bot": (
                    self.config.telegram.bot_token[:10] + "..."
                    if self.config.telegram.bot_token
                    else None
                ),
                "webhook_mode": bool(self.config.telegram.webhook_url),
            },
            "conversations": len(self.conversation_service.conversations),
            "users": len(self.conversation_service.users),
            "roles": len(self.conversation_service.roles),
        }
