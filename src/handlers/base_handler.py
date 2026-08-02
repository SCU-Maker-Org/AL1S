"""
基础处理器模块
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from ..models import Message, User


class BaseHandler(ABC):
    """基础处理器抽象类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logger.bind(handler=name)

    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """
        处理更新

        Args:
            update: Telegram更新对象
            context: 上下文对象

        Returns:
            bool: 是否成功处理
        """
        pass

    def can_handle(self, update: Update) -> bool:
        """
        检查是否可以处理此更新

        Args:
            update: Telegram更新对象

        Returns:
            bool: 是否可以处理
        """
        return True

    def extract_user_info(self, update: Update) -> Optional[User]:
        """提取用户信息"""
        try:
            if update.effective_user:
                return User(
                    id=update.effective_user.id,
                    username=update.effective_user.username,
                    first_name=update.effective_user.first_name,
                    last_name=update.effective_user.last_name,
                    is_bot=update.effective_user.is_bot,
                    language_code=update.effective_user.language_code,
                )
        except Exception as e:
            self.logger.error(f"提取用户信息失败: {e}")

        return None

    def extract_message_info(self, update: Update) -> Optional[Dict]:
        """提取消息信息，返回包含用户ID、聊天ID和消息内容的字典"""
        try:
            if update.message:
                # 提取消息文本，确保不为空
                message_text = update.message.text or update.message.caption or ""
                if not message_text.strip() and (
                    update.message.photo or update.message.document
                ):
                    message_text = "[图片]"
                if not message_text.strip():
                    self.logger.warning(
                        f"收到空消息: user_id={update.effective_user.id if update.effective_user else 'unknown'}"
                    )
                    return None

                return {
                    "user_id": (
                        update.effective_user.id if update.effective_user else None
                    ),
                    "chat_id": (
                        update.effective_chat.id if update.effective_chat else None
                    ),
                    "message": Message(
                        role="user",
                        content=message_text.strip(),
                        timestamp=(
                            update.message.date.timestamp()
                            if update.message.date
                            else time.time()
                        ),
                    ),
                }
        except Exception as e:
            self.logger.error(f"提取消息信息失败: {e}")

        return None

    def log_handling(self, update: Update, success: bool = True):
        """记录处理日志"""
        user_id = update.effective_user.id if update.effective_user else "unknown"
        chat_id = update.effective_chat.id if update.effective_chat else "unknown"

        if success:
            self.logger.info(f"成功处理来自用户 {user_id} 在聊天 {chat_id} 的请求")
        else:
            self.logger.warning(f"处理来自用户 {user_id} 在聊天 {chat_id} 的请求失败")

    def get_handler_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }
