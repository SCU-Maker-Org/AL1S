"""
图片处理器模块
"""

import time
from typing import List

from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from ..agents.unified_agent import EnhancedUnifiedAgentService as UnifiedAgentService
from ..models import ImageSearchResult, Message
from ..services.ascii2d_service import Ascii2DService
from ..services.conversation_service import ConversationService
from ..services.group_chat_service import GroupChatService
from ..services.rate_limit_service import RateLimitService
from .base_handler import BaseHandler


class ImageHandler(BaseHandler):
    """图片处理器"""

    def __init__(
        self,
        ascii2d_service: Ascii2DService,
        unified_agent_service: UnifiedAgentService,
        conversation_service: ConversationService,
        group_chat_service: GroupChatService = None,
        rate_limit_service: RateLimitService = None,
    ):
        super().__init__("ImageHandler", "处理图片消息和图片搜索")
        self.ascii2d_service = ascii2d_service
        self.unified_agent_service = unified_agent_service
        self.conversation_service = conversation_service
        self.group_chat_service = group_chat_service
        self.rate_limit_service = rate_limit_service

    async def _reply(self, update: Update, *args, **kwargs):
        kwargs.setdefault(
            "message_thread_id",
            getattr(update.effective_message, "message_thread_id", None),
        )
        try:
            return await update.effective_message.reply_text(*args, **kwargs)
        except Exception as exc:
            logger.warning("图片引用回复失败，回退为 Topic 普通消息: {}", exc)
            return await update.get_bot().send_message(
                update.effective_chat.id, *args, **kwargs
            )

    async def _replace_placeholder(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        message_id: int,
        text: str,
    ) -> None:
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=message_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as html_error:
            logger.warning("图片回复 HTML 编辑失败，回退纯文本: {}", html_error)
            import re

            plain_text = re.sub(r"<[^>]+>", "", text)
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=message_id,
                    text=plain_text,
                )
            except Exception as edit_error:
                logger.warning(
                    "图片占位消息编辑失败，回退 Topic 新消息: {}", edit_error
                )
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=plain_text,
                    message_thread_id=getattr(
                        update.effective_message, "message_thread_id", None
                    ),
                )

    def can_handle(self, update: Update) -> bool:
        """检查是否可以处理此更新"""
        return update.message is not None and (
            update.message.photo or update.message.document
        )

    def _format_image_analysis(
        self,
        analysis_text: str,
        search_results: List[ImageSearchResult] = None,
        role: str = "user",
    ) -> str:
        """格式化图片分析结果，添加Telegram富文本支持"""
        if not analysis_text:
            return "无法分析图片内容"

        # 构建响应文本
        response_text = "<b>🖼️ 图片分析结果</b>\n\n"
        response_text += f"<i>{analysis_text}</i>\n\n"

        # 添加图片搜索结果
        if search_results and len(search_results) > 0:
            response_text += "<b>🔍 相似图片搜索结果</b>\n\n"

            for i, result in enumerate(search_results[:5], 1):  # 只显示前5个结果
                response_text += f"<b>{i}. {result.title or '无标题'}</b>\n"

                if result.url:
                    response_text += f"   🔗 <a href='{result.url}'>查看图片</a>\n"

                if result.source and result.source != "Ascii2D":
                    response_text += f"   📍 来源: {result.source}\n"

                if result.metadata and result.metadata.get("author"):
                    response_text += f"   👤 作者: {result.metadata['author']}\n"

                response_text += "\n"

            if len(search_results) > 5:
                response_text += f"<i>... 还有 {len(search_results) - 5} 个结果</i>\n"
        else:
            response_text += "<i>⚠️ 未找到相似的图片搜索结果</i>\n"

        # 添加分隔线
        response_text += "\n" + "─" * 30

        return response_text

    def _get_image_placeholder_message(self, role) -> str:
        """根据角色生成个性化的图片分析占位信息"""
        if not role or not hasattr(role, "name"):
            return "🖼️ 正在分析图片中..."

        # 根据角色名称生成个性化占位信息
        role_name = role.name.lower()

        if "爱丽丝" in role_name or "alice" in role_name:
            placeholders = [
                "🌸 爱丽丝正在仔细观察图片...",
                "🎀 让我看看这是什么图片呢...",
                "✨ 正在用心分析图片内容...",
                "💭 这张图片很有趣，分析中...",
            ]
        elif "女仆" in role_name:
            placeholders = [
                "🎀 女仆正在为您分析图片...",
                "🌹 恭敬地检查图片内容中...",
                "💝 用心观察图片，请稍候...",
                "✨ 为您仔细分析这张图片...",
            ]
        elif "kei" in role_name or "Kei" in role_name:
            placeholders = [
                "⚡ Kei正在扫描图片数据...",
                "🔥 分析图片特征中...",
                "🎯 识别图片内容，请稍候...",
                "💪 全力分析图片信息...",
            ]
        elif "游戏" in role_name or "玩家" in role_name:
            placeholders = [
                "🎮 正在识别游戏截图...",
                "🕹️ 分析图片攻略信息...",
                "🎲 检测图片游戏元素...",
                "🏆 搜索图片相关信息...",
            ]
        elif "助手" in role_name or "AI" in role_name:
            placeholders = [
                "🤖 AI正在处理图片数据...",
                "💻 分析图片内容和特征...",
                "🔍 识别图片中的元素...",
                "⚙️ 图像识别系统运行中...",
            ]
        else:
            # 默认占位信息
            placeholders = [
                f"🖼️ {role.name}正在分析图片...",
                f"👀 {role.name}仔细观察中...",
                f"🔍 {role.name}识别图片内容...",
            ]

        # 随机选择一个占位信息（简单轮换）
        import time

        index = int(time.time()) % len(placeholders)
        return placeholders[index]

    async def _send_formatted_response(self, update, text: str):
        """发送格式化的响应"""
        try:
            # 发送消息，启用HTML解析
            await self._reply(
                update, text, parse_mode="HTML", disable_web_page_preview=True
            )

        except Exception as e:
            logger.error(f"发送格式化响应失败: {e}")
            # 如果HTML解析失败，发送纯文本
            try:
                # 移除HTML标签
                import re

                clean_text = re.sub(r"<[^>]+>", "", text)
                await self._reply(update, clean_text)
            except Exception as e2:
                logger.error(f"发送纯文本也失败: {e2}")
                await self._reply(update, "抱歉，消息发送失败，请稍后再试。")

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """处理图片消息"""
        try:
            if (
                self.group_chat_service
                and await self.group_chat_service.is_duplicate_update(
                    getattr(update, "update_id", None)
                )
            ):
                return True
            bot_username = getattr(context.bot, "username", None)
            bot_id = getattr(context.bot, "id", None)
            if not bot_username or not bot_id:
                me = await context.bot.get_me()
                bot_username, bot_id = me.username or "", me.id
            if self.group_chat_service:
                decision = self.group_chat_service.decide(
                    update, str(bot_username), int(bot_id)
                )
                if not decision.allowed:
                    if decision.denied_reason == "not_triggered":
                        await self.group_chat_service.observe(update)
                    logger.bind(
                        chat_id=update.effective_chat.id,
                        thread_id=getattr(
                            update.effective_message, "message_thread_id", None
                        )
                        or 0,
                        user_id=update.effective_user.id,
                        message_id=update.effective_message.message_id,
                        trigger_type=getattr(
                            decision.trigger_type, "value", decision.trigger_type
                        ),
                    ).info(
                        "group_image_decision allowed=false denied_reason={} agent_called=false",
                        decision.denied_reason,
                    )
                    return True
                session_key = self.group_chat_service.session_key(update)
            else:
                from ..models import SessionKey

                session_key = SessionKey(
                    update.effective_chat.id, 0, update.effective_user.id, "private"
                )

            if self.rate_limit_service:
                rate = await self.rate_limit_service.check(
                    update.effective_user.id,
                    session_key.chat_id,
                    session_key.thread_id,
                )
                if not rate.allowed:
                    if rate.notify:
                        await self._reply(update, "请求过于频繁，请稍后再试。")
                    return True

            # 提取消息信息
            message_info = self.extract_message_info(update)
            if not message_info:
                logger.error("无法提取消息信息")
                return False

            user_id = message_info["user_id"]
            message = message_info["message"]

            conversation = self.conversation_service.get_conversation(session_key)

            # 获取当前角色
            role = conversation.role
            if not role:
                # 如果没有设置角色，使用默认角色
                role = self.conversation_service.get_role(session_key)
                if not role:
                    # 如果还是没有角色，创建一个默认角色
                    default_role_name = "AI助手"
                    self.conversation_service.set_role(session_key, default_role_name)
                    role = self.conversation_service.get_role(session_key)

            # 先发送个性化占位信息
            placeholder_text = self._get_image_placeholder_message(role)
            placeholder_message = await self._reply(update, placeholder_text)

            try:
                async with self.conversation_service.session_lock(session_key):
                    self.conversation_service.add_message(session_key, message)
                    photo_file = await self._get_photo_file(update)
                    if photo_file is None:
                        raise ValueError("无法读取图片文件")
                    image_data = await photo_file.download_as_bytearray()

                    image_bytes = bytes(image_data)

                    analysis_result = await self.unified_agent_service.analyze_image(
                        image_bytes
                    )

                    search_results = await self.ascii2d_service.search_by_image_file(
                        image_bytes
                    )

                    formatted_response = self._format_image_analysis(
                        analysis_result, search_results, role
                    )

                    await self._replace_placeholder(
                        update,
                        context,
                        placeholder_message.message_id,
                        formatted_response,
                    )

                    bot_message = Message(
                        role="assistant",
                        content=f"图片分析：{analysis_result}\n\n图片来源搜索：{len(search_results)} 个结果",
                        timestamp=time.time(),
                    )
                    self.conversation_service.add_message(session_key, bot_message)

                logger.info(f"成功处理用户 {user_id} 的图片")
                return True

            except Exception as e:
                # 如果图片处理失败，编辑占位信息为错误消息
                error_message = "❌ 处理图片时出现错误，请稍后重试。"
                await self._replace_placeholder(
                    update,
                    context,
                    placeholder_message.message_id,
                    error_message,
                )
                logger.error(f"图片处理失败: {e}")
                return False

        except Exception as e:
            logger.error(f"处理图片消息失败: {e}")
            # 尝试发送错误消息
            try:
                await self._reply(update, "❌ 处理图片时出现错误，请稍后重试。")
            except Exception as send_error:
                logger.error(f"发送错误消息失败: {send_error}")
            return False

    async def _get_photo_file(self, update: Update):
        """获取图片文件"""
        try:
            if update.message.photo:
                # 获取最大尺寸的图片
                photo = max(update.message.photo, key=lambda p: p.file_size)
                return await update.get_bot().get_file(photo.file_id)
            elif (
                update.message.document
                and update.message.document.mime_type.startswith("image/")
            ):
                return await update.get_bot().get_file(update.message.document.file_id)
            else:
                return None
        except Exception as e:
            logger.error(f"获取图片文件失败: {e}")
            return None

    async def _analyze_image(self, photo_file) -> str:
        """分析图片内容"""
        try:
            # 获取图片URL
            file_url = photo_file.file_path
            if not file_url.startswith("http"):
                file_url = f"https://api.telegram.org/file/bot{photo_file.bot.token}/{file_url}"

            # 使用OpenAI分析图片
            analysis = await self.unified_agent_service.analyze_image(
                file_url, "请简要描述这张图片的内容和风格"
            )

            return analysis

        except Exception as e:
            logger.error(f"图片分析失败: {e}")
            return "无法分析图片内容"

    async def _search_similar_images(self, photo_file) -> List[ImageSearchResult]:
        """搜索相似图片"""
        try:
            # 下载图片数据
            image_data = await photo_file.download_as_bytearray()

            # 将bytearray转换为bytes
            image_bytes = bytes(image_data)

            # 使用Ascii2D搜索
            results = await self.ascii2d_service.search_by_image_file(image_bytes)

            return results

        except Exception as e:
            logger.error(f"图片搜索失败: {e}")
            return []

    def _build_image_reply(
        self, analysis: str, search_results: List[ImageSearchResult]
    ) -> str:
        """构建图片回复消息"""
        reply_parts = []

        # 添加图片分析结果
        reply_parts.append("🔍 图片分析结果：")
        reply_parts.append(analysis)
        reply_parts.append("")

        # 添加搜索结果
        if search_results:
            reply_parts.append("🖼️ 相似图片搜索结果：")

            for i, result in enumerate(search_results[:5], 1):  # 只显示前5个结果
                source_info = f"来源: {result.source}"
                if result.similarity:
                    source_info += f" (相似度: {result.similarity:.1f}%)"

                reply_parts.append(f"{i}. {source_info}")
                if result.title:
                    reply_parts.append(f"   标题: {result.title}")
                reply_parts.append(f"   链接: {result.url}")
                reply_parts.append("")

            if len(search_results) > 5:
                reply_parts.append(f"... 还有 {len(search_results) - 5} 个结果")
        else:
            reply_parts.append("❌ 未找到相似图片")

        return "\n".join(reply_parts)

    async def handle_image_search_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, image_url: str
    ) -> bool:
        """处理图片搜索命令"""
        try:
            # 验证URL
            if not self.ascii2d_service.validate_image_url(image_url):
                await self._reply(update, "❌ 无效的图片URL")
                return False

            # 发送"正在搜索"状态
            await context.bot.send_chat_action(
                chat_id=update.message.chat_id,
                action="typing",
                message_thread_id=getattr(update.message, "message_thread_id", None),
            )

            # 搜索相似图片（不再需要异步上下文管理器）
            search_results = await self.ascii2d_service.search_by_image_url(image_url)

            # 构建回复
            if search_results:
                reply_text = self._build_image_reply(
                    "通过URL搜索的图片", search_results
                )
            else:
                reply_text = "❌ 未找到相似图片"

            await self._reply(update, reply_text)
            return True

        except Exception as e:
            logger.error(f"URL图片搜索失败: {e}")
            await self._reply(update, "抱歉，图片搜索失败，请稍后再试。")
            return False

    async def handle_multiple_engine_search(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        image_data: bytes,
        engines: List[str] = None,
    ) -> bool:
        """处理多引擎图片搜索"""
        try:
            # 发送"正在搜索"状态
            await context.bot.send_chat_action(
                chat_id=update.message.chat_id,
                action="typing",
                message_thread_id=getattr(update.message, "message_thread_id", None),
            )

            # 使用多个搜索引擎搜索
            all_results = await self.ascii2d_service.search_multiple_engines(
                image_data, engines
            )

            # 构建多引擎搜索结果回复
            reply_text = self._build_multi_engine_reply(all_results)

            await self._reply(update, reply_text)
            return True

        except Exception as e:
            logger.error(f"多引擎图片搜索失败: {e}")
            await self._reply(update, "抱歉，多引擎搜索失败，请稍后再试。")
            return False

    def _build_multi_engine_reply(self, all_results: dict) -> str:
        """构建多引擎搜索结果回复"""
        reply_parts = []
        reply_parts.append("🔍 多引擎图片搜索结果：")
        reply_parts.append("")

        total_results = 0

        for engine_name, results in all_results.items():
            if results:
                reply_parts.append(f"📊 {engine_name.upper()} 引擎：")
                reply_parts.append(f"   找到 {len(results)} 个结果")

                # 显示前3个结果
                for i, result in enumerate(results[:3], 1):
                    source_info = f"来源: {result.source}"
                    if result.similarity:
                        source_info += f" (相似度: {result.similarity:.1f}%)"

                    reply_parts.append(f"   {i}. {source_info}")
                    if result.title:
                        reply_parts.append(f"      标题: {result.title}")
                    reply_parts.append(f"      链接: {result.url}")

                if len(results) > 3:
                    reply_parts.append(f"   ... 还有 {len(results) - 3} 个结果")

                reply_parts.append("")
                total_results += len(results)
            else:
                reply_parts.append(f"❌ {engine_name.upper()} 引擎：未找到结果")
                reply_parts.append("")

        reply_parts.append(f"📈 总计找到 {total_results} 个结果")

        return "\n".join(reply_parts)
