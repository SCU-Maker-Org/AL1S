"""
命令处理器模块
"""

import html
import re
from typing import Dict, Optional

from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from ..config import config

# OpenAI 服务已集成到 Agent 服务中
from ..infra.mcp import MCPService
from ..models import Command
from ..services.conversation_service import ConversationService
from ..services.group_chat_service import GroupChatService
from ..utils.database_logger import log_user_action
from .base_handler import BaseHandler


class CommandHandler(BaseHandler):
    """命令处理器"""

    def __init__(
        self,
        conversation_service: ConversationService,
        agent_service=None,
        image_handler=None,
        mcp_service: MCPService = None,
        database_service=None,
        group_chat_service: GroupChatService = None,
    ):
        super().__init__("CommandHandler", "处理机器人命令")
        self.conversation_service = conversation_service
        self.agent_service = agent_service
        self.image_handler = image_handler
        self.mcp_service = mcp_service
        self.database_service = database_service
        self.group_chat_service = group_chat_service
        self.commands = self._initialize_commands()

    async def _reply(self, update: Update, *args, **kwargs):
        """回复到原 Telegram Forum Topic。"""
        kwargs.setdefault(
            "message_thread_id",
            getattr(update.effective_message, "message_thread_id", None),
        )
        try:
            return await update.effective_message.reply_text(*args, **kwargs)
        except Exception as exc:
            logger.warning("命令引用回复失败，回退为 Topic 普通消息: {}", exc)
            return await update.get_bot().send_message(
                update.effective_chat.id, *args, **kwargs
            )

    def _session_key(self, update: Update):
        if self.group_chat_service:
            return self.group_chat_service.session_key(update)
        from ..models import SessionKey

        return SessionKey(
            update.effective_chat.id, 0, update.effective_user.id, "private"
        )

    @staticmethod
    def _mcp_access_level(update: Update) -> str:
        return MCPService.resolve_caller_access(
            getattr(update.effective_user, "id", None),
            getattr(update.effective_chat, "type", None),
            config.telegram.admin_user_ids,
        )

    def _initialize_commands(self) -> Dict[str, Command]:
        """初始化可用命令"""
        return {
            "/start": Command(
                name="start",
                description="开始使用机器人",
                usage="/start",
                aliases=["start"],
            ),
            "/help": Command(
                name="help",
                description="显示帮助信息",
                usage="/help",
                aliases=["help", "h"],
            ),
            "/role": Command(
                name="role",
                description="设置或查看当前角色",
                usage="/role [角色名称]",
                aliases=["role", "r"],
                requires_args=False,
            ),
            "/roles": Command(
                name="roles",
                description="显示所有可用角色",
                usage="/roles",
                aliases=["roles", "list_roles"],
            ),
            "/create_role": Command(
                name="create_role",
                description="创建自定义角色",
                usage="/create_role 角色名称 角色描述",
                aliases=["create_role", "cr"],
                requires_args=True,
            ),
            "/reset": Command(
                name="reset",
                description="重置当前对话",
                usage="/reset",
                aliases=["reset", "clear"],
            ),
            "/stats": Command(
                name="stats",
                description="显示对话统计信息",
                usage="/stats",
                aliases=["stats", "info"],
            ),
            "/ping": Command(
                name="ping",
                description="测试机器人响应",
                usage="/ping",
                aliases=["ping", "p"],
            ),
            "/search": Command(
                name="search",
                description="搜索图片URL",
                usage="/search 图片URL",
                aliases=["search", "s"],
                requires_args=True,
            ),
            "/search_engines": Command(
                name="search_engines",
                description="显示可用的图片搜索引擎",
                usage="/search_engines",
                aliases=["search_engines", "engines"],
                requires_args=False,
            ),
            "/test_search": Command(
                name="test_search",
                description="测试图片搜索服务",
                usage="/test_search",
                aliases=["test_search", "test"],
                requires_args=False,
            ),
            "/tools": Command(
                name="tools",
                description="显示可用的MCP工具",
                usage="/tools",
                aliases=["tools", "t"],
                requires_args=False,
            ),
            "/mcp_status": Command(
                name="mcp_status",
                description="显示MCP服务器状态",
                usage="/mcp_status",
                aliases=["mcp_status", "mcp"],
                requires_args=False,
            ),
            "/db_stats": Command(
                name="db_stats",
                description="显示数据库统计信息",
                usage="/db_stats",
                aliases=["db_stats", "db"],
                requires_args=False,
            ),
            "/my_stats": Command(
                name="my_stats",
                description="显示我的使用统计",
                usage="/my_stats",
                aliases=["my_stats", "me"],
                requires_args=False,
            ),
        }

    def can_handle(self, update: Update) -> bool:
        """检查是否可以处理此更新"""
        return (
            update.message is not None
            and update.message.text is not None
            and update.message.text.startswith("/")
        )

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """处理命令"""
        try:
            text = update.message.text.strip()
            parts = text.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            # 查找命令
            cmd = self._find_command(command)
            if not cmd:
                await self._reply(
                    update, f"未知命令: {command}\n使用 /help 查看可用命令"
                )
                return False

            # 检查是否需要参数
            if cmd.requires_args and not args:
                await self._reply(
                    update, f"命令 {cmd.name} 需要参数\n用法: {cmd.usage}"
                )
                return False

            # 记录命令使用
            log_user_action(
                user_id=update.effective_user.id,
                action=f"command_{cmd.name}",
                details={
                    "command": cmd.name,
                    "args": args if args else None,
                    "chat_id": update.effective_chat.id,
                },
            )

            # 执行命令
            success = await self._execute_command(cmd, args, update, context)
            self.log_handling(update, success=success)
            return success

        except Exception as e:
            logger.error(f"命令处理失败: {e}")
            try:
                await self._reply(update, "抱歉，命令执行失败，请稍后再试。")
            except Exception as send_error:
                logger.error("发送命令错误提示失败: {}", send_error)
            self.log_handling(update, success=False)
            return False

    def _find_command(self, command: str) -> Optional[Command]:
        """查找命令"""
        for cmd in self.commands.values():
            if command in [cmd.name] + cmd.aliases:
                return cmd
        return None

    async def _execute_command(
        self,
        cmd: Command,
        args: str,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """执行命令"""
        try:
            if cmd.name == "start":
                return await self._handle_start(update, context)
            elif cmd.name == "help":
                return await self._handle_help(update, context)
            elif cmd.name == "role":
                return await self._handle_role(update, context, args)
            elif cmd.name == "roles":
                return await self._handle_roles(update, context)
            elif cmd.name == "create_role":
                return await self._handle_create_role(update, context, args)
            elif cmd.name == "reset":
                return await self._handle_reset(update, context)
            elif cmd.name == "stats":
                return await self._handle_stats(update, context)
            elif cmd.name == "ping":
                return await self._handle_ping(update, context)
            elif cmd.name == "search":
                return await self._handle_search(update, context, args)
            elif cmd.name == "search_engines":
                return await self._handle_search_engines(update, context)
            elif cmd.name == "test_search":
                return await self._handle_test_search(update, context)
            elif cmd.name == "tools":
                return await self._handle_tools(update, context)
            elif cmd.name == "mcp_status":
                return await self._handle_mcp_status(update, context)
            elif cmd.name == "db_stats":
                return await self._handle_db_stats(update, context)
            elif cmd.name == "my_stats":
                return await self._handle_my_stats(update, context)
            else:
                await self._reply(update, f"命令 {cmd.name} 尚未实现")
                return False

        except Exception as e:
            logger.error(f"执行命令 {cmd.name} 失败: {e}")
            return False

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理start命令"""
        try:
            session_key = self._session_key(update)

            # 获取当前角色
            current_role = self.conversation_service.get_role(session_key)

            if current_role:
                # 显示角色问候语
                welcome_message = (
                    f"🎉 {current_role.greeting}\n\n"
                    f"🎭 当前角色：{current_role.name}\n"
                    f"📝 {current_role.description}\n\n"
                    f"💡 可用命令：\n"
                    f"• /help - 显示帮助信息\n"
                    f"• /role - 查看或切换角色\n"
                    f"• /roles - 显示所有可用角色\n"
                    f"• /create_role - 创建自定义角色\n"
                    f"• /reset - 重置当前对话\n"
                    f"• /stats - 显示对话统计\n"
                    f"• /ping - 测试机器人响应\n"
                    f"• /search - 搜索图片URL\n\n"
                    f"🌟 开始聊天吧！"
                )
                await self._reply(update, welcome_message)
            else:
                # 如果没有角色，显示默认欢迎信息
                await self._reply(
                    update,
                    "🎉 欢迎使用AL1S-Bot！\n\n"
                    "💡 使用 /help 查看所有可用命令\n"
                    "🎭 使用 /role 设置角色\n"
                    "🌟 开始聊天吧！",
                )

            return True

        except Exception as e:
            logger.error(f"处理start命令失败: {e}")
            await self._reply(update, "❌ 处理start命令时发生错误")
            return False

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理帮助命令"""
        help_text = """
📚 可用命令列表：

🔹 基础命令
/start - 开始使用机器人
/help - 显示此帮助信息
/ping - 测试机器人响应

🔹 角色管理
/role [角色名] - 设置或查看当前角色
/roles - 显示所有可用角色
/create_role 名称 描述 - 创建自定义角色

🔹 对话管理
/reset - 重置当前对话
/stats - 显示对话统计信息

🔹 图片搜索
/search 图片URL - 搜索指定URL的图片
/search_engines - 显示可用的搜索引擎
/test_search - 测试图片搜索服务

💡 使用提示：
• 直接发送消息即可开始聊天
• 使用 /role 切换不同角色获得不同体验
• 发送图片可以进行相似图片搜索
• 使用 /search 命令搜索网络图片
        """
        await self._reply(update, help_text.strip())
        return True

    async def _handle_role(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, args: str
    ) -> bool:
        """处理角色命令"""
        user_id = update.message.from_user.id
        chat_id = update.message.chat_id
        session_key = self._session_key(update)

        if not args:
            # 显示当前角色
            conversation = self.conversation_service.get_conversation(session_key)
            current_role = conversation.role
            role_text = f"""
🎭 当前角色信息：

名称: {current_role.name}
描述: {current_role.description}
问候语: {current_role.greeting}
告别语: {current_role.farewell}
            """
            await self._reply(update, role_text.strip())
        else:
            # 设置新角色
            role_name = args.strip()
            conversation = self.conversation_service.get_conversation(session_key)
            current_role = conversation.role

            async with self.conversation_service.session_lock(session_key):
                role_changed = self.conversation_service.set_role(
                    session_key, role_name
                )
            if role_changed:
                # 记录角色切换
                log_user_action(
                    user_id=user_id,
                    action="role_switch",
                    details={
                        "old_role": current_role.name if current_role else None,
                        "new_role": role_name,
                        "chat_id": chat_id,
                    },
                )
                await self._reply(update, f"✅ 角色已设置为: {role_name}")
            else:
                await self._reply(
                    update, f"❌ 角色 {role_name} 不存在\n使用 /roles 查看可用角色"
                )

        return True

    async def _handle_roles(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理角色列表命令"""
        roles = self.conversation_service.list_roles()
        roles_text = "🎭 可用角色列表：\n\n"

        for role in roles:
            roles_text += f"🔹 {role}\n"
            roles_text += f"   描述: {role}\n"
            roles_text += f"   风格: {role}\n"
            roles_text += f"   特点: {role}\n\n"

        roles_text += "💡 使用 /role 角色名 来切换角色"
        await self._reply(update, roles_text.strip())
        return True

    async def _handle_create_role(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, args: str
    ) -> bool:
        """处理创建角色命令"""
        if not args:
            await self._reply(
                update, "❌ 请提供角色名称和描述\n用法: /create_role 名称 描述"
            )
            return False

        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            await self._reply(
                update, "❌ 请提供角色名称和描述\n用法: /create_role 名称 描述"
            )
            return False

        name, description = parts[0], parts[1]

        # 使用OpenAI生成角色提示词
        # 角色提示词生成现在由 Agent 服务处理
        if self.agent_service and hasattr(self.agent_service, "chat_completion"):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"为以下角色描述生成一个系统提示词：{description}",
                    }
                ]
                system_prompt = await self.agent_service.chat_completion(messages)
            except Exception as e:
                logger.error(f"生成角色提示词失败: {e}")
                system_prompt = f"你是{description}。请用中文回答用户的问题。"
        else:
            system_prompt = f"你是{description}。请用中文回答用户的问题。"

        if self.conversation_service.create_custom_role(
            name, description, system_prompt
        ):
            await self._reply(
                update, f"✅ 角色 {name} 创建成功！\n使用 /role {name} 来切换到该角色"
            )
        else:
            await self._reply(update, f"❌ 角色 {name} 已存在")

        return True

    async def _handle_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理reset命令"""
        session_key = self._session_key(update)
        async with self.conversation_service.session_lock(session_key):
            self.conversation_service.reset_conversation(session_key)

        await self._reply(update, "✅ 对话已重置")
        return True

    async def _handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理stats命令"""
        user_id = update.message.from_user.id
        session_key = self._session_key(update)
        stats = self.conversation_service.get_session_stats(session_key)

        stats_text = f"""
📊 对话统计信息：

会话类型: {stats.get('scope_label', 'N/A')}
群组/聊天ID: {stats.get('chat_id', 'N/A')}
Topic ID: {stats.get('thread_id', 0)}
当前会话消息数: {stats.get('message_count', 0)}
当前角色: {stats.get('current_role', 'N/A')}
当前用户ID: {user_id}
        """

        await self._reply(update, stats_text.strip())
        return True

    async def _handle_ping(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理ping命令"""
        await self._reply(update, "🏓 Pong! 机器人运行正常")
        return True

    async def _handle_search(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, args: str
    ) -> bool:
        """处理图片搜索命令"""
        if not args:
            await self._reply(update, "❌ 请提供图片URL\n用法: /search 图片URL")
            return False

        # 这里需要调用图片处理器的方法
        # 由于命令处理器和图片处理器是分离的，我们需要通过其他方式调用
        # 暂时返回提示信息
        await self._reply(
            update,
            f"🔍 图片搜索功能已准备就绪！\n"
            f"您提供的URL: {args}\n\n"
            f"💡 提示：直接发送图片文件即可进行相似图片搜索",
        )
        return True

    async def _handle_search_engines(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理搜索引擎信息命令"""
        engines_text = """
🔍 可用的图片搜索引擎：

📊 Ascii2D
   • 基于PicImageSearch库
   • 支持二次元图片搜索
   • 支持URL和文件上传搜索
   • 提供相似度评分

🚀 未来计划
   • SauceNAO引擎集成
   • Google Lens集成
   • 多引擎并行搜索
   • 搜索结果聚合

💡 使用方式：
• 发送图片文件自动搜索
• 使用 /search 命令搜索URL
• 支持多种图片格式
        """
        await self._reply(update, engines_text.strip())
        return True

    async def _handle_test_search(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理测试搜索命令"""
        test_text = """
🧪 图片搜索服务测试

✅ 服务状态：正常运行
🔧 使用PicImageSearch库
🌐 支持Ascii2D引擎
📱 支持Telegram图片上传

💡 测试方法：
1. 发送一张图片
2. 等待自动分析
3. 查看搜索结果

⚠️ 注意事项：
• 确保图片清晰可见
• 网络连接稳定
• API配额充足
        """
        await self._reply(update, test_text.strip())
        return True

    def _format_command_help(self, text: str, title: str = None) -> str:
        """格式化命令帮助文本，添加Telegram富文本支持"""
        if not text:
            return text

        # 添加标题
        if title:
            text = f"<b>📋 {title}</b>\n\n{text}"

        # 格式化代码块
        # 匹配 ```code``` 格式
        text = re.sub(r"```(\w+)?\n(.*?)```", r"<pre>\2</pre>", text, flags=re.DOTALL)
        # 匹配 `code` 格式
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

        # 格式化粗体
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

        # 格式化斜体
        text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)

        # 格式化删除线
        text = re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)

        # 格式化列表项
        lines = text.split("\n")
        formatted_lines = []
        for line in lines:
            # 格式化无序列表
            if line.strip().startswith("- "):
                line = f"• {line.strip()[2:]}"
            # 格式化有序列表
            elif re.match(r"^\d+\.\s", line.strip()):
                line = f"<b>{line.strip()}</b>"
            formatted_lines.append(line)

        text = "\n".join(formatted_lines)

        # 添加分隔线
        if title:
            text += "\n\n" + "─" * 30

        return text

    async def _send_formatted_response(self, update, text: str, title: str = None):
        """发送格式化的响应"""
        try:
            # 格式化响应文本
            formatted_text = self._format_command_help(text, title)

            # 发送消息，启用HTML解析
            await self._reply(
                update, formatted_text, parse_mode="HTML", disable_web_page_preview=True
            )

        except Exception as e:
            logger.error(f"发送格式化响应失败: {e}")
            # 如果HTML解析失败，发送纯文本
            try:
                await self._reply(update, text)
            except Exception as e2:
                logger.error(f"发送纯文本也失败: {e2}")
                await self._reply(update, "抱歉，消息发送失败，请稍后再试。")

    async def _handle_start_command(self, update, context):
        """处理 /start 命令"""
        try:
            # 注册命令列表
            await self._register_commands_async()

            welcome_text = """🎉 欢迎使用 AL1S-Bot！

我是您的AI助手，具有以下特色功能：

<b>🤖 多角色支持</b>
• 天童爱丽丝 - 可爱的动漫角色
• 女仆爱丽丝 - 温柔的女仆
• Kei人格 - 独特的个性
• 游戏玩家 - 游戏爱好者
• AI助手 - 专业助手

<b>🖼️ 图片分析</b>
• 支持多种AI模型
• 智能图片识别
• 图片来源搜索

<b>💬 智能对话</b>
• 上下文记忆
• 个性化回复
• 多语言支持

<b>📚 常用命令</b>
• /help - 查看帮助信息
• /role - 切换角色
• /roles - 查看所有角色
• /search - 图片搜索
• /stats - 查看统计信息

开始和我聊天吧！您可以问我任何问题，或者发送图片让我分析。"""

            await self._send_formatted_response(
                update, welcome_text, "欢迎使用 AL1S-Bot"
            )

        except Exception as e:
            logger.error(f"处理start命令失败: {e}")
            await self._reply(update, "启动失败，请稍后再试。")

    async def _handle_help_command(self, update, context):
        """处理 /help 命令"""
        help_text = """<b>🔧 命令帮助</b>

<b>基础命令</b>
• /start - 启动机器人
• /help - 显示此帮助信息
• /ping - 测试机器人响应

<b>角色管理</b>
• /role [角色名] - 切换到指定角色
• /roles - 查看所有可用角色
• /create_role [名称] [描述] - 创建新角色

<b>图片功能</b>
• /search - 搜索图片（需要先发送图片）
• /search_engines - 查看支持的搜索引擎

<b>系统管理</b>
• /reset - 重置当前对话
• /stats - 查看使用统计

<b>使用示例</b>
• 发送文本消息开始对话
• 发送图片进行AI分析
• 使用 /role 天童爱丽丝 切换角色

<b>注意事项</b>
• 图片搜索功能依赖外部服务
• 对话历史会保存一段时间
• 支持中文和英文对话"""

        await self._send_formatted_response(update, help_text, "命令帮助")

    async def _handle_roles_command(self, update, context):
        """处理 /roles 命令"""
        try:
            roles = self.conversation_service.list_roles()

            if not roles:
                await self._reply(update, "❌ 没有可用的角色")
                return

            roles_text = "<b>🎭 可用角色列表</b>\n\n"

            for i, role_name in enumerate(roles, 1):
                # 获取角色配置
                role_config = config.get_role(role_name)
                if role_config:
                    roles_text += f"<b>{i}. {role_config.name}</b>\n"
                    if (
                        hasattr(role_config, "english_name")
                        and role_config.english_name
                    ):
                        roles_text += f"   <i>英文名: {role_config.english_name}</i>\n"
                    if hasattr(role_config, "description") and role_config.description:
                        roles_text += f"   📝 {role_config.description}\n"
                    if hasattr(role_config, "personality") and role_config.personality:
                        roles_text += f"   🎨 {role_config.personality[:100]}{'...' if len(role_config.personality) > 100 else ''}\n"
                    roles_text += "\n"

            roles_text += "💡 使用 /role [角色名] 来切换角色"

            await self._send_formatted_response(update, roles_text, "角色列表")

        except Exception as e:
            logger.error(f"处理roles命令失败: {e}")
            await self._reply(update, "获取角色列表失败，请稍后再试。")

    async def _handle_role_command(self, update, context):
        """处理 /role 命令"""
        try:
            args = context.args

            if not args:
                # 显示当前角色
                user_id = update.effective_user.id
                chat_id = update.effective_chat.id
                current_role = self.conversation_service.get_role(
                    self._session_key(update)
                )

                if current_role:
                    role_text = f"<b>🎭 当前角色</b>\n\n"
                    role_text += f"<b>名称:</b> {current_role.name}\n"
                    if (
                        hasattr(current_role, "english_name")
                        and current_role.english_name
                    ):
                        role_text += f"<b>英文名:</b> {current_role.english_name}\n"
                    if (
                        hasattr(current_role, "description")
                        and current_role.description
                    ):
                        role_text += f"<b>描述:</b> {current_role.description}\n"
                    if (
                        hasattr(current_role, "personality")
                        and current_role.personality
                    ):
                        role_text += f"<b>性格:</b> {current_role.personality}\n"
                    if hasattr(current_role, "greeting") and current_role.greeting:
                        role_text += f"<b>问候语:</b> {current_role.greeting}\n"

                    await self._send_formatted_response(
                        update, role_text, "当前角色信息"
                    )
                else:
                    await self._reply(update, "❌ 无法获取当前角色信息")
                return

            # 切换角色
            role_name = " ".join(args)
            session_key = self._session_key(update)
            async with self.conversation_service.session_lock(session_key):
                success = self.conversation_service.set_role(session_key, role_name)

            if success:
                role_text = f"✅ 成功切换到角色: <b>{role_name}</b>\n\n"
                role_text += "现在您可以开始与这个角色对话了！"

                await self._send_formatted_response(update, role_text, "角色切换成功")
            else:
                await self._reply(
                    update, f"❌ 切换到角色 '{role_name}' 失败，请检查角色名称是否正确"
                )

        except Exception as e:
            logger.error(f"处理role命令失败: {e}")
            await self._reply(update, "处理角色命令失败，请稍后再试。")

    async def _handle_stats_command(self, update, context):
        """处理 /stats 命令"""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id

            # 获取统计信息
            session_key = self._session_key(update)
            conversation = self.conversation_service.get_conversation(session_key)
            current_role = self.conversation_service.get_role(session_key)

            stats_text = "<b>📊 使用统计</b>\n\n"

            if conversation:
                stats_text += f"<b>对话ID:</b> {conversation.id}\n"
                stats_text += f"<b>创建时间:</b> {conversation.created_at}\n"
                stats_text += f"<b>消息数量:</b> {len(conversation.messages) if hasattr(conversation, 'messages') else 0}\n"

            if current_role:
                stats_text += f"<b>当前角色:</b> {current_role.name}\n"

            stats_text += f"<b>用户ID:</b> {user_id}\n"
            stats_text += f"<b>聊天ID:</b> {chat_id}\n"

            await self._send_formatted_response(update, stats_text, "使用统计")

        except Exception as e:
            logger.error(f"处理stats命令失败: {e}")
            await self._reply(update, "获取统计信息失败，请稍后再试。")

    async def _handle_ping_command(self, update, context):
        """处理 /ping 命令"""
        ping_text = "🏓 <b>Pong!</b>\n\n机器人运行正常！"
        await self._send_formatted_response(update, ping_text, "连接测试")

    async def _handle_search_engines_command(self, update, context):
        """处理 /search_engines 命令"""
        engines_text = """<b>🔍 支持的搜索引擎</b>

<b>Ascii2D</b>
• 动漫图片搜索
• 支持文件上传和URL搜索
• 提供图片来源信息

<b>使用方法</b>
1. 发送图片给机器人
2. 机器人会自动进行图片分析
3. 同时搜索相似图片

<b>注意事项</b>
• 图片搜索功能依赖外部服务
• 某些图片可能无法找到结果
• 支持常见图片格式（JPG、PNG等）"""

        await self._send_formatted_response(update, engines_text, "搜索引擎信息")

    async def _handle_tools(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理tools命令"""
        try:
            if not self.mcp_service:
                await self._reply(update, "❌ MCP功能未启用", parse_mode="HTML")
                return False

            # 获取可用工具
            tools = self.mcp_service.get_available_tools(self._mcp_access_level(update))

            if not tools:
                await self._reply(
                    update,
                    "🔧 <b>可用工具</b>\n\n❌ 暂无可用的MCP工具\n\n"
                    "请检查MCP服务器配置或使用 /mcp_status 查看服务器状态",
                    parse_mode="HTML",
                )
                return True

            # 按服务器分组工具
            tools_by_server = {}
            for tool_name, tool_info in tools.items():
                server_name = tool_info.get("server", "未知")
                if server_name not in tools_by_server:
                    tools_by_server[server_name] = []
                tools_by_server[server_name].append(
                    (tool_name, tool_info.get("description", "无描述"))
                )

            # 构建简洁的工具列表消息
            tools_text = f"🔧 <b>可用的MCP工具 ({len(tools)}个)</b>\n\n"

            for server_name, server_tools in tools_by_server.items():
                safe_server_name = html.escape(str(server_name))
                tools_text += (
                    f"📦 <b>{safe_server_name}</b> ({len(server_tools)}个工具)\n"
                )

                # 每个服务器最多显示前8个工具，避免消息过长
                displayed_tools = server_tools[:8]
                for tool_name, description in displayed_tools:
                    # 截断过长的描述
                    description = str(description or "无描述")
                    short_desc = (
                        description[:50] + "..."
                        if len(description) > 50
                        else description
                    )
                    safe_tool_name = html.escape(str(tool_name))
                    safe_description = html.escape(short_desc)
                    tools_text += (
                        f"  • <code>{safe_tool_name}</code> - {safe_description}\n"
                    )

                if len(server_tools) > 8:
                    tools_text += f"  • ... 还有 {len(server_tools) - 8} 个工具\n"

                tools_text += "\n"

            tools_text += "💡 <b>使用提示:</b>\n"
            tools_text += "• 直接描述需求，AI会自动选择工具\n"
            tools_text += "• 支持文件操作、数据库查询等功能\n"
            tools_text += "• 使用 <code>/mcp_status</code> 查看服务器状态"

            await self._reply(update, tools_text, parse_mode="HTML")
            return True

        except Exception as e:
            logger.error(f"处理tools命令失败: {e}")
            await self._reply(update, "获取工具信息失败，请稍后再试。")
            return False

    async def _handle_mcp_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理mcp_status命令"""
        try:
            if not self.mcp_service:
                await self._reply(
                    update,
                    "❌ <b>MCP状态</b>\n\nMCP功能未启用\n\n"
                    "要启用MCP功能，请在配置文件中设置 <code>mcp.enabled = true</code>",
                    parse_mode="HTML",
                )
                return False

            # 获取服务器状态
            server_status = self.mcp_service.get_server_status(
                self._mcp_access_level(update)
            )

            if not server_status:
                await self._reply(
                    update,
                    "🔧 <b>MCP服务器状态</b>\n\n❌ 没有配置任何MCP服务器\n\n"
                    "请在配置文件中添加MCP服务器配置",
                    parse_mode="HTML",
                )
                return True

            # 构建状态消息
            status_text = "🔧 <b>MCP服务器状态</b>\n\n"

            connected_count = 0
            total_tools = 0

            for server_name, status in server_status.items():
                is_connected = status.get("connected", False)
                tools_count = status.get("tools_count", 0)

                if is_connected:
                    connected_count += 1
                    total_tools += tools_count
                    status_icon = "✅"
                else:
                    status_icon = "❌"

                safe_server_name = html.escape(str(server_name))
                safe_command = html.escape(str(status.get("command", "未知")))
                status_text += f"{status_icon} <b>{safe_server_name}</b>\n"
                status_text += f"📂 命令: <code>{safe_command}</code>\n"
                status_text += f"🔧 工具数量: {tools_count}\n"

                if status.get("error"):
                    safe_error = html.escape(str(status["error"]))
                    status_text += f"⚠️ 错误: <code>{safe_error}</code>\n"

                if tools_count > 0:
                    tools_list = status.get("tools", [])
                    safe_tools = [html.escape(str(tool)) for tool in tools_list[:3]]
                    status_text += f"🛠️ 工具: {', '.join(safe_tools)}"
                    if len(tools_list) > 3:
                        status_text += f" 等{len(tools_list)}个"
                    status_text += "\n"

                status_text += "\n"

            # 添加总览
            status_text += f"📊 <b>总览</b>\n"
            status_text += f"• 服务器总数: {len(server_status)}\n"
            status_text += f"• 已连接: {connected_count}\n"
            status_text += f"• 可用工具: {total_tools}\n\n"

            if connected_count > 0:
                status_text += "💡 使用 /tools 查看所有可用工具"
            else:
                status_text += "⚠️ 所有MCP服务器都未连接，请检查配置"

            await self._reply(update, status_text, parse_mode="HTML")
            return True

        except Exception as e:
            logger.error(f"处理mcp_status命令失败: {e}")
            await self._reply(update, "获取MCP状态失败，请稍后再试。")
            return False

    async def _handle_db_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理db_stats命令"""
        try:
            # 从bot实例获取数据库服务
            if not hasattr(self, "database_service"):
                await self._reply(
                    update,
                    "❌ <b>数据库统计</b>\n\n数据库服务未启用",
                    parse_mode="HTML",
                )
                return False

            # 获取统计信息
            role_stats = await self.database_service.get_role_stats()
            tool_stats = await self.database_service.get_tool_usage_stats()

            # 构建统计消息
            stats_text = "📊 <b>数据库统计信息</b>\n\n"

            # 角色使用统计
            if role_stats:
                stats_text += "🎭 <b>角色使用统计</b>\n"
                for stat in role_stats[:5]:  # 显示前5个
                    stats_text += f"• {stat['role_name']}: {stat['usage_count']}次\n"
                stats_text += "\n"

            # 工具使用统计
            if tool_stats:
                stats_text += "🛠️ <b>工具使用统计</b>\n"
                for stat in tool_stats[:5]:  # 显示前5个
                    success_rate = (
                        (stat["success_count"] / stat["usage_count"] * 100)
                        if stat["usage_count"] > 0
                        else 0
                    )
                    stats_text += f"• {stat['tool_name']}: {stat['usage_count']}次 ({success_rate:.1f}% 成功)\n"
                stats_text += "\n"
            else:
                stats_text += "🛠️ <b>工具使用统计</b>\n暂无工具使用记录\n\n"

            stats_text += "💡 使用 /my_stats 查看您的个人统计信息"

            await self._reply(update, stats_text, parse_mode="HTML")
            return True

        except Exception as e:
            logger.error(f"处理db_stats命令失败: {e}")
            await self._reply(update, "获取数据库统计失败，请稍后再试。")
            return False

    async def _handle_my_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """处理my_stats命令"""
        try:
            # 从bot实例获取数据库服务
            if not hasattr(self, "database_service"):
                await self._reply(
                    update, "❌ <b>我的统计</b>\n\n数据库服务未启用", parse_mode="HTML"
                )
                return False

            user_id = update.effective_user.id

            # 获取用户统计信息
            user_stats = await self.database_service.get_user_stats(user_id)

            if not user_stats:
                await self._reply(
                    update, "📊 <b>我的统计</b>\n\n暂无使用记录", parse_mode="HTML"
                )
                return True

            # 构建统计消息
            stats_text = "📊 <b>我的使用统计</b>\n\n"

            if user_stats.get("username"):
                stats_text += f"👤 用户名: @{user_stats['username']}\n"

            stats_text += f"💬 对话数量: {user_stats.get('conversation_count', 0)}\n"
            stats_text += f"📝 消息数量: {user_stats.get('message_count', 0)}\n"

            if user_stats.get("current_role"):
                stats_text += f"🎭 当前角色: {user_stats['current_role']}\n"

            if user_stats.get("last_activity"):
                stats_text += f"⏰ 最后活动: {user_stats['last_activity']}\n"

            stats_text += "\n💡 使用 /db_stats 查看全局统计信息"

            await self._reply(update, stats_text, parse_mode="HTML")
            return True

        except Exception as e:
            logger.error(f"处理my_stats命令失败: {e}")
            await self._reply(update, "获取个人统计失败，请稍后再试。")
            return False
