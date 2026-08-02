"""
自主学习服务
- 从对话中自动提取知识
- 评估知识重要性
- 管理知识生命周期
- 支持增量学习
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..config import config
from ..models import KnowledgeEntry


class KnowledgeExtractor:
    """知识提取器"""

    def __init__(self, llm_service):
        """
        初始化知识提取器
        Args:
            llm_service: 支持 chat_completion 接口的 LLM 服务
        """
        self.llm_service = llm_service

    async def extract_from_conversation(
        self,
        user_message: str,
        bot_response: str,
        conversation_context: List[Dict[str, str]] = None,
    ) -> List[KnowledgeEntry]:
        """从对话中提取知识"""
        try:
            # 构建提取提示词
            extraction_prompt = self._build_extraction_prompt(
                user_message, bot_response, conversation_context
            )

            # 调用 LLM 提取知识
            extraction_result = await self.llm_service.chat_completion(
                [
                    {
                        "role": "system",
                        "content": "你是一个知识提取专家，专门从对话中提取有价值的信息。",
                    },
                    {"role": "user", "content": extraction_prompt},
                ]
            )

            if not extraction_result or extraction_result.strip() == "无":
                return []

            # 解析提取结果
            knowledge_entries = self._parse_extraction_result(extraction_result)

            logger.debug(f"从对话中提取了 {len(knowledge_entries)} 个知识点")
            return knowledge_entries

        except Exception as e:
            logger.error(f"从对话中提取知识失败: {e}")
            return []

    def _parse_extraction_result(self, extraction_result: str) -> List[KnowledgeEntry]:
        """解析知识提取结果"""
        try:
            logger.debug(f"开始解析提取结果: {extraction_result[:100]}...")

            # 清理结果，移除可能的 markdown 代码块标记
            cleaned_result = extraction_result.strip()
            logger.debug(f"清理前: {repr(cleaned_result[:100])}")

            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:]  # 移除 ```json
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]  # 移除 ```
            cleaned_result = cleaned_result.strip()

            logger.debug(f"清理后: {repr(cleaned_result[:100])}")

            # 如果结果是 "无" 或空，返回空列表
            if not cleaned_result or cleaned_result.lower() in ["无", "none", "null"]:
                logger.debug("结果为空或'无'，返回空列表")
                return []

            # 尝试解析 JSON
            try:
                knowledge_data = json.loads(cleaned_result)
                logger.debug(f"JSON 解析成功，数据类型: {type(knowledge_data)}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败，尝试修复: {e}")
                # 尝试修复常见的 JSON 格式问题
                cleaned_result = self._fix_json_format(cleaned_result)
                knowledge_data = json.loads(cleaned_result)
                logger.debug(f"修复后 JSON 解析成功")

            # 转换为 KnowledgeEntry 对象
            knowledge_entries = []

            if isinstance(knowledge_data, list):
                logger.debug(f"处理列表数据，包含 {len(knowledge_data)} 个条目")
                for i, item in enumerate(knowledge_data):
                    logger.debug(f"处理第 {i+1} 个条目: {type(item)}")
                    if isinstance(item, dict):
                        entry = self._dict_to_knowledge_entry(item)
                        if entry:
                            knowledge_entries.append(entry)
                            logger.debug(f"成功转换条目: {entry.title}")
                        else:
                            logger.warning(f"第 {i+1} 个条目转换失败")
            elif isinstance(knowledge_data, dict):
                logger.debug("处理单个字典数据")
                # 单个知识条目
                entry = self._dict_to_knowledge_entry(knowledge_data)
                if entry:
                    knowledge_entries.append(entry)
                    logger.debug(f"成功转换单个条目: {entry.title}")

            logger.debug(f"最终转换了 {len(knowledge_entries)} 个知识条目")
            return knowledge_entries

        except Exception as e:
            logger.error(f"解析知识提取结果失败: {e}")
            logger.debug(f"原始结果: {extraction_result}")
            import traceback

            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return []

    def _fix_json_format(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        # 移除多余的逗号
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
        # 确保字符串被正确引用
        json_str = re.sub(r"(\w+):", r'"\1":', json_str)
        return json_str

    def _dict_to_knowledge_entry(self, data: dict) -> Optional[KnowledgeEntry]:
        """将字典转换为 KnowledgeEntry 对象"""
        try:
            entry = KnowledgeEntry(
                title=data.get("title", ""),
                content=data.get("content", ""),
                summary=data.get("summary", ""),
                keywords=data.get("keywords", ""),
                category=data.get("category", "general"),
                importance_score=float(data.get("importance_score", 5.0)),
            )
            return entry
        except Exception as e:
            logger.error(f"转换知识条目失败: {e}")
            logger.debug(f"数据: {data}")
            return None

    def _build_extraction_prompt(
        self, user_message: str, bot_response: str, context: List[Dict[str, str]] = None
    ) -> str:
        """构建知识提取提示词"""
        prompt = f"""
请从以下对话中提取有价值的知识点。重点关注：
1. 个人信息（姓名、生日、喜好、经历等）
2. 技术知识（概念、方法、工具等）
3. 事实信息（日期、地点、数据等）
4. 重要的对话内容和决定

用户问题: {user_message}
助手回答: {bot_response}
"""

        if context:
            prompt += "\n对话上下文:\n"
            for msg in context[-3:]:  # 最近3条消息作为上下文
                role = "用户" if msg["role"] == "user" else "助手"
                prompt += f"{role}: {msg['content']}\n"

        prompt += """
请提取关键信息，每个知识点使用以下JSON格式：
{
    "title": "简短标题",
    "content": "详细内容",
    "summary": "简要摘要",
    "keywords": "关键词1,关键词2,关键词3",
    "category": "分类（如：技术、生活、学习等）",
    "importance_score": 数字（1-10，表示重要程度）
}

如果有多个知识点，请用JSON数组格式返回。
如果没有有价值的知识，请只返回 "无"。
"""
        return prompt

    def _parse_extraction_result(self, result: str) -> List[KnowledgeEntry]:
        """解析知识提取结果"""
        try:
            # 尝试解析JSON
            if result.strip().startswith("["):
                # JSON数组格式
                data = json.loads(result)
                if isinstance(data, list):
                    return [
                        self._create_knowledge_entry(item)
                        for item in data
                        if isinstance(item, dict)
                    ]
            elif result.strip().startswith("{"):
                # 单个JSON对象
                data = json.loads(result)
                if isinstance(data, dict):
                    return [self._create_knowledge_entry(data)]

            # 如果JSON解析失败，尝试文本解析
            return self._parse_text_result(result)

        except json.JSONDecodeError:
            # JSON解析失败，尝试文本解析
            return self._parse_text_result(result)

    def _parse_text_result(self, result: str) -> List[KnowledgeEntry]:
        """解析文本格式的提取结果"""
        try:
            knowledge_entries = []
            lines = result.strip().split("\n")

            current_entry = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 匹配字段
                if line.startswith("标题:") or line.startswith("title:"):
                    current_entry["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("内容:") or line.startswith("content:"):
                    current_entry["content"] = line.split(":", 1)[1].strip()
                elif line.startswith("摘要:") or line.startswith("summary:"):
                    current_entry["summary"] = line.split(":", 1)[1].strip()
                elif line.startswith("关键词:") or line.startswith("keywords:"):
                    current_entry["keywords"] = line.split(":", 1)[1].strip()
                elif line.startswith("分类:") or line.startswith("category:"):
                    current_entry["category"] = line.split(":", 1)[1].strip()
                elif line.startswith("重要性:") or line.startswith("importance:"):
                    try:
                        score_text = line.split(":", 1)[1].strip()
                        current_entry["importance_score"] = float(
                            re.findall(r"\d+\.?\d*", score_text)[0]
                        )
                    except (IndexError, ValueError):
                        current_entry["importance_score"] = 5.0

                # 如果遇到分隔符或新的标题，保存当前条目
                if line.startswith("---") or (
                    line.startswith("标题:") and current_entry
                ):
                    if len(current_entry) >= 2:  # 至少有标题和内容
                        knowledge_entries.append(
                            self._create_knowledge_entry(current_entry)
                        )
                    current_entry = {}
                    if line.startswith("标题:"):
                        current_entry["title"] = line.split(":", 1)[1].strip()

            # 保存最后一个条目
            if len(current_entry) >= 2:
                knowledge_entries.append(self._create_knowledge_entry(current_entry))

            return knowledge_entries

        except Exception as e:
            logger.error(f"解析文本提取结果失败: {e}")
            return []

    def _create_knowledge_entry(self, data: Dict[str, Any]) -> KnowledgeEntry:
        """创建知识条目对象"""
        return KnowledgeEntry(
            title=data.get("title", "未知标题"),
            content=data.get("content", ""),
            summary=data.get("summary", data.get("content", "")[:200]),
            keywords=data.get("keywords", ""),
            category=data.get("category", "general"),
            importance_score=float(data.get("importance_score", 5.0)),
            created_at=datetime.now(),
        )


class LearningService:
    """自主学习服务"""

    def __init__(self, database_service, vector_service, llm_service):
        self.database_service = database_service
        self.vector_service = vector_service
        self.llm_service = llm_service

        # 初始化知识提取器
        self.knowledge_extractor = KnowledgeExtractor(llm_service)

        # 学习统计
        self.learning_stats = {
            "total_learned": 0,
            "last_learning_time": None,
            "learning_sessions": 0,
        }

    async def learn_from_conversation(
        self,
        user_message: str,
        bot_response: str,
        conversation_id: int,
        user_id: int,
        conversation_context: List[Dict[str, str]] = None,
        knowledge_namespace: str = "",
    ) -> int:
        """从对话中学习知识"""
        try:
            if not self._should_learn(user_message, bot_response):
                return 0

            # 提取知识
            knowledge_entries = (
                await self.knowledge_extractor.extract_from_conversation(
                    user_message, bot_response, conversation_context
                )
            )

            if not knowledge_entries:
                return 0

            # 保存知识
            saved_count = 0
            for entry in knowledge_entries:
                entry.user_id = user_id
                entry.conversation_id = conversation_id
                entry.knowledge_namespace = knowledge_namespace

                # 检查重复性
                if await self._is_duplicate_knowledge(entry):
                    logger.debug(f"跳过重复知识: {entry.title}")
                    continue

                # 保存到数据库
                if await self._save_knowledge_entry(entry):
                    # 添加到向量存储
                    await self.vector_service.add_knowledge(entry)
                    saved_count += 1
                    logger.debug(f"学习了新知识: {entry.title}")

            # 更新学习统计
            if saved_count > 0:
                self.learning_stats["total_learned"] += saved_count
                self.learning_stats["last_learning_time"] = datetime.now()
                self.learning_stats["learning_sessions"] += 1

                logger.info(f"从对话中学习了 {saved_count} 个新知识点")

            return saved_count

        except Exception as e:
            logger.error(f"从对话中学习失败: {e}")
            return 0

    def _should_learn(self, user_message: str, bot_response: str) -> bool:
        """判断是否应该从这次对话中学习"""
        # 检查配置 - 优先使用 agent 配置，回退到 rag 配置
        auto_learning = False
        if hasattr(config, "agent") and config.agent.auto_learning:
            auto_learning = True
        elif hasattr(config, "rag") and config.rag.auto_learning:
            auto_learning = True

        if not auto_learning:
            return False

        # 检查消息长度
        if len(user_message) < 10 or len(bot_response) < 20:
            return False

        # 检查是否包含有价值的内容
        valuable_patterns = [
            r"如何",
            r"什么是",
            r"为什么",
            r"怎么样",
            r"解释",
            r"说明",
            r"方法",
            r"步骤",
            r"技巧",
            r"原理",
            r"定义",
            r"概念",
            # 个人信息相关
            r"生日",
            r"喜欢",
            r"爱好",
            r"姓名",
            r"年龄",
            r"职业",
            r"家庭",
            r"我是",
            r"我的",
            r"我叫",
            r"我在",
            r"我会",
            r"我想",
            # 记忆和确认相关
            r"记住",
            r"记录",
            r"保存",
            r"知道了",
            r"明白了",
            r"好的",
        ]

        combined_text = user_message + " " + bot_response
        for pattern in valuable_patterns:
            if re.search(pattern, combined_text):
                return True

        # 检查回答质量（长度和结构）
        if len(bot_response) > 50 and (
            "。" in bot_response or "：" in bot_response or "！" in bot_response
        ):
            return True

        return False

    async def _is_duplicate_knowledge(self, entry: KnowledgeEntry) -> bool:
        """检查是否为重复知识"""
        try:
            # 搜索相似知识
            similar_knowledge = await self.vector_service.search_knowledge(
                entry.title + " " + entry.content,
                top_k=3,
                threshold=0.8,  # 高相似度阈值
                knowledge_namespace=entry.knowledge_namespace or None,
            )

            for similar in similar_knowledge:
                # 检查标题相似度
                if (
                    self._calculate_text_similarity(
                        entry.title, similar.get("title", "")
                    )
                    > 0.8
                ):
                    return True

                # 检查内容相似度
                if (
                    self._calculate_text_similarity(
                        entry.content, similar.get("content", "")
                    )
                    > 0.9
                ):
                    return True

            return False

        except Exception as e:
            logger.error(f"检查重复知识失败: {e}")
            return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单的Jaccard相似度）"""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def _save_knowledge_entry(self, entry: KnowledgeEntry) -> bool:
        """保存知识条目到数据库"""
        try:
            if not self.database_service:
                return False

            # 调用数据库服务保存知识条目
            entry_id = await asyncio.to_thread(
                self.database_service.save_knowledge_entry,
                entry.user_id,
                entry.conversation_id,
                entry.title,
                entry.content,
                entry.summary,
                entry.keywords,
                entry.category,
                entry.importance_score,
                entry.source_message_id,
                entry.knowledge_namespace,
            )

            if entry_id:
                entry.id = entry_id
                return True

            return False

        except Exception as e:
            logger.error(f"保存知识条目失败: {e}")
            return False

    async def optimize_knowledge_base(self) -> Dict[str, int]:
        """优化知识库（清理重复、过期知识等）"""
        try:
            stats = {
                "removed_duplicates": 0,
                "removed_outdated": 0,
                "updated_scores": 0,
            }

            if not self.database_service:
                return stats

            # 获取所有知识条目
            all_entries = await asyncio.to_thread(
                self.database_service.get_all_knowledge_entries
            )

            if not all_entries:
                return stats

            # 查找并删除重复项
            seen_titles = set()
            for entry in all_entries:
                title = entry.get("title", "").lower().strip()
                if title in seen_titles:
                    # 删除重复项
                    await asyncio.to_thread(
                        self.database_service.delete_knowledge_entry, entry.get("id")
                    )
                    stats["removed_duplicates"] += 1
                else:
                    seen_titles.add(title)

            # 删除过期知识（可选，根据具体需求实现）
            cutoff_date = datetime.now() - timedelta(days=365)  # 一年前
            for entry in all_entries:
                created_at = entry.get("created_at")
                if created_at and isinstance(created_at, str):
                    try:
                        created_date = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                        if (
                            created_date < cutoff_date
                            and entry.get("importance_score", 0) < 3
                        ):
                            await asyncio.to_thread(
                                self.database_service.delete_knowledge_entry,
                                entry.get("id"),
                            )
                            stats["removed_outdated"] += 1
                    except ValueError:
                        pass

            logger.info(f"知识库优化完成: {stats}")
            return stats

        except Exception as e:
            logger.error(f"优化知识库失败: {e}")
            return {"removed_duplicates": 0, "removed_outdated": 0, "updated_scores": 0}

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        try:
            # 从数据库获取统计信息
            if self.database_service:
                total_knowledge = await asyncio.to_thread(
                    self.database_service.get_knowledge_count
                )
            else:
                total_knowledge = 0

            stats = self.learning_stats.copy()
            stats["total_knowledge_in_db"] = total_knowledge

            return stats

        except Exception as e:
            logger.error(f"获取学习统计失败: {e}")
            return self.learning_stats.copy()

    async def suggest_learning_opportunities(
        self, recent_conversations: List[Dict[str, Any]]
    ) -> List[str]:
        """建议学习机会"""
        try:
            suggestions = []

            # 分析最近的对话，找出可能的学习点
            for conv in recent_conversations[-10:]:  # 最近10次对话
                user_msg = conv.get("user_message", "")
                bot_msg = conv.get("bot_message", "")

                # 检查是否包含问题但没有学习
                if any(
                    keyword in user_msg
                    for keyword in ["如何", "什么是", "为什么", "怎么"]
                ):
                    if len(bot_msg) > 100:  # 有详细回答
                        suggestions.append(f"可以从对话中学习: {user_msg[:50]}...")

            return suggestions[:5]  # 最多返回5个建议

        except Exception as e:
            logger.error(f"生成学习建议失败: {e}")
            return []

    def cleanup(self):
        """清理资源"""
        try:
            # 保存学习统计
            if self.learning_stats["total_learned"] > 0:
                logger.info(
                    f"学习会话结束，总共学习了 {self.learning_stats['total_learned']} 个知识点"
                )

            # 清理引用
            self.knowledge_extractor = None

        except Exception as e:
            logger.debug(f"学习服务清理失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass
