"""
增强的统一 Agent 服务 - 集成任务编排与执行能力
- 基于 unified_agent_service 的增强版本
- 集成任务编排器，支持复杂多步骤任务
- 提供智能任务构建与执行接口
- 支持工作流模板与自定义任务链
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from .task_builder import (
    SmartTaskBuilder,
    Task,
    TaskChain,
    TaskStatus,
    TaskTemplates,
    TaskType,
)
from .task_orchestrator import TaskExecutionContext, TaskOrchestrator, TaskResult
from .unified_agent_service import UnifiedAgentService


class EnhancedUnifiedAgentService(UnifiedAgentService):
    """增强的统一 Agent 服务 - 集成任务编排能力"""

    def __init__(
        self,
        database_service=None,
        mcp_service=None,
        vector_store_path: str = "data/vector_store",
        max_concurrent_tasks: int = 5,
    ):
        # 调用父类构造函数
        super().__init__(
            database_service=database_service,
            mcp_service=mcp_service,
            vector_store_path=vector_store_path,
        )

        # 任务编排器
        self.task_orchestrator = TaskOrchestrator(
            agent_service=self,
            vector_service=self.vector_service,
            mcp_service=self.mcp_service,
            database_service=database_service,
        )

        # 智能任务构建器
        self.task_builder = SmartTaskBuilder(self.task_orchestrator)

        # 任务模板
        self.task_templates = TaskTemplates()

        # 配置
        self.max_concurrent_tasks = max_concurrent_tasks

        # 状态
        self._task_orchestrator_initialized = False

    async def initialize(self) -> bool:
        """初始化增强的统一服务"""
        try:
            # 先初始化父类
            parent_initialized = await super().initialize()
            if not parent_initialized:
                return False

            # 初始化任务编排器
            orchestrator_initialized = await self.task_orchestrator.initialize()
            if not orchestrator_initialized:
                return False

            # 启动任务编排器
            await self.task_orchestrator.start_orchestrator()

            self._task_orchestrator_initialized = True
            logger.info("增强统一 Agent 服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"增强统一 Agent 服务初始化失败: {e}")
            return False

    async def process_with_task_orchestration(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
        enable_smart_analysis: bool = True,
        enable_knowledge_search: bool = True,
        enable_web_search: bool = True,
        enable_learning: bool = True,
        conversation_id: int = None,
        user_id: int = None,
    ) -> Optional[str]:
        """使用任务编排处理复杂查询"""
        try:
            if not self._task_orchestrator_initialized:
                logger.warning("任务编排器未初始化，使用基础模式")
                return await self.chat_completion(messages, tools)

            # 提取用户查询
            user_query = self._extract_user_query(messages)
            if not user_query:
                return "抱歉，无法提取用户查询。"

            # 提取系统上下文
            system_context = self._extract_system_context(messages)

            # 提取对话历史
            conversation_history = self._extract_conversation_history(messages)

            logger.info(f"使用任务编排处理查询: {user_query[:50]}...")

            if enable_smart_analysis:
                # 使用智能任务构建器
                task_chain = await self.task_builder.build_intelligent_response_task(
                    user_query=user_query,
                    conversation_history=conversation_history,
                    system_context=system_context,
                    enable_web_search=enable_web_search,
                    enable_knowledge_search=enable_knowledge_search,
                    enable_learning=enable_learning,
                )
            else:
                # 使用基础任务链
                task_chain = await self._build_basic_response_task_chain(
                    user_query=user_query,
                    conversation_history=conversation_history,
                    system_context=system_context,
                    enable_knowledge_search=enable_knowledge_search,
                    enable_web_search=enable_web_search,
                    enable_learning=enable_learning,
                )

            # 创建执行上下文
            execution_context = await self._create_execution_context(
                conversation_id=conversation_id, user_id=user_id
            )

            # 执行任务链
            results = await task_chain.execute(
                orchestrator=self.task_orchestrator, execution_context=execution_context
            )

            # 提取最终结果
            final_result = self._extract_final_result(results)

            if final_result:
                logger.info("任务编排执行成功")
                return final_result
            else:
                logger.warning("任务编排未产生结果，回退到基础模式")
                return await self.chat_completion(messages, tools)

        except Exception as e:
            logger.error(f"任务编排处理失败: {e}")
            # 回退到基础模式
            return await self.chat_completion(messages, tools)

    async def execute_research_workflow(
        self,
        query: str,
        search_web: bool = True,
        search_knowledge: bool = True,
        conversation_id: int = None,
        user_id: int = None,
    ) -> Optional[str]:
        """执行研究工作流"""
        try:
            if not self._task_orchestrator_initialized:
                logger.warning("任务编排器未初始化")
                return None

            logger.info(f"执行研究工作流: {query}")

            # 使用研究工作流模板
            task_chain = self.task_templates.create_research_workflow(
                query=query, search_web=search_web, search_knowledge=search_knowledge
            )

            # 创建执行上下文
            execution_context = await self._create_execution_context(
                conversation_id=conversation_id, user_id=user_id
            )

            # 执行任务
            results = await task_chain.execute(
                orchestrator=self.task_orchestrator, execution_context=execution_context
            )

            # 提取结果
            final_result = self._extract_final_result(results)

            if final_result:
                logger.info("研究工作流执行成功")
                return final_result
            else:
                logger.warning("研究工作流未产生结果")
                return None

        except Exception as e:
            logger.error(f"研究工作流执行失败: {e}")
            return None

    async def execute_content_analysis_workflow(
        self,
        content: str,
        analysis_type: str = "comprehensive",
        conversation_id: int = None,
        user_id: int = None,
    ) -> Optional[str]:
        """执行内容分析工作流"""
        try:
            if not self._task_orchestrator_initialized:
                logger.warning("任务编排器未初始化")
                return None

            logger.info(f"执行内容分析工作流: {analysis_type}")

            # 使用内容分析模板
            task_chain = self.task_templates.create_content_analysis_workflow(
                content=content, analysis_type=analysis_type
            )

            # 创建执行上下文
            execution_context = await self._create_execution_context(
                conversation_id=conversation_id, user_id=user_id
            )

            # 执行任务
            results = await task_chain.execute(
                orchestrator=self.task_orchestrator, execution_context=execution_context
            )

            # 提取结果
            final_result = self._extract_final_result(results)

            if final_result:
                logger.info("内容分析工作流执行成功")
                return final_result
            else:
                logger.warning("内容分析工作流未产生结果")
                return None

        except Exception as e:
            logger.error(f"内容分析工作流执行失败: {e}")
            return None

    async def execute_custom_task_chain(
        self, task_chain: TaskChain, conversation_id: int = None, user_id: int = None
    ) -> List[Any]:
        """执行自定义任务链"""
        try:
            if not self._task_orchestrator_initialized:
                logger.warning("任务编排器未初始化")
                return []

            logger.info(f"执行自定义任务链: {len(task_chain.tasks)} 个任务")

            # 创建执行上下文
            execution_context = await self._create_execution_context(
                conversation_id=conversation_id, user_id=user_id
            )

            # 执行任务链
            results = await task_chain.execute(
                orchestrator=self.task_orchestrator, execution_context=execution_context
            )

            logger.info(f"自定义任务链执行完成: {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"自定义任务链执行失败: {e}")
            return []

    async def create_and_execute_tasks(
        self,
        tasks_config: List[Dict[str, Any]],
        conversation_id: int = None,
        user_id: int = None,
    ) -> List[TaskResult]:
        """创建并执行任务"""
        try:
            if not self._task_orchestrator_initialized:
                logger.warning("任务编排器未初始化")
                return []

            logger.info(f"创建并执行任务: {len(tasks_config)} 个任务")

            # 创建任务
            tasks = []
            for task_config in tasks_config:
                task = Task(**task_config)
                tasks.append(task)

            # 创建执行上下文
            execution_context = await self._create_execution_context(
                conversation_id=conversation_id, user_id=user_id
            )

            # 提交任务
            task_ids = await self.task_orchestrator.submit_tasks(
                tasks, execution_context
            )

            # 等待所有任务完成
            results = []
            for task_id in task_ids:
                while True:
                    status = await self.task_orchestrator.get_task_status(task_id)
                    if status in [
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    ]:
                        result = await self.task_orchestrator.get_task_result(task_id)
                        results.append(result)
                        break

                    await asyncio.sleep(0.5)

            logger.info(f"任务执行完成: {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"任务创建与执行失败: {e}")
            return []

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if not self._task_orchestrator_initialized:
            return None

        return await self.task_orchestrator.get_task_status(task_id)

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        if not self._task_orchestrator_initialized:
            return None

        return await self.task_orchestrator.get_task_result(task_id)

    async def get_all_task_statuses(self) -> Dict[str, TaskStatus]:
        """获取所有任务状态"""
        if not self._task_orchestrator_initialized:
            return {}

        return await self.task_orchestrator.get_all_task_statuses()

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if not self._task_orchestrator_initialized:
            return False

        return await self.task_orchestrator.cancel_task(task_id)

    def _extract_user_query(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """从消息中提取用户查询"""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        return None

    def _extract_system_context(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """从消息中提取系统上下文"""
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                return msg["content"]
        return None

    def _extract_conversation_history(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """提取对话历史"""
        history = []
        for msg in messages:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                history.append(msg)
        return history

    async def _build_basic_response_task_chain(
        self,
        user_query: str,
        conversation_history: List[Dict[str, str]] = None,
        system_context: str = None,
        enable_knowledge_search: bool = True,
        enable_web_search: bool = True,
        enable_learning: bool = True,
    ) -> TaskChain:
        """构建基础响应任务链"""
        chain = TaskChain()

        # 步骤1: 知识搜索
        if enable_knowledge_search:
            chain.add_knowledge_search(
                name="knowledge_search", query=user_query, description="搜索相关知识"
            )

        # 步骤2: 判断是否需要网页搜索
        if enable_web_search:
            web_keywords = [
                "最新",
                "新闻",
                "实时",
                "当前",
                "今天",
                "现在",
                "网页",
                "网站",
            ]
            if any(keyword in user_query for keyword in web_keywords):
                # 可以在这里添加网页搜索逻辑
                pass

        # 步骤3: 生成响应
        messages = []

        if system_context:
            messages.append({"role": "system", "content": system_context})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_query})

        chain.add_llm_task(
            name="generate_response",
            messages=messages,
            description="生成最终响应",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None,
        )

        # 步骤4: 学习对话
        if enable_learning:
            chain.add_task(
                task_type=TaskType.LEARNING,
                name="learn_from_conversation",
                description="从对话中学习",
                parameters={
                    "user_message": user_query,
                    "bot_response": "${generate_response.result}",
                },
                dependencies={chain.tasks[-1].task_id} if chain.tasks else None,
            )

        return chain

    async def _create_execution_context(
        self, conversation_id: int = None, user_id: int = None
    ) -> TaskExecutionContext:
        """创建任务执行上下文"""
        return TaskExecutionContext(
            execution_id=f"enhanced_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_service=self,
            vector_service=self.vector_service,
            mcp_service=self.mcp_service,
            database_service=self.database_service,
            conversation_id=conversation_id or self._conversation_id_context.get(),
            user_id=user_id,
            global_context={},
            task_results={},
        )

    def _extract_final_result(self, results: List[Any]) -> Optional[str]:
        """从任务结果中提取最终结果"""
        if not results:
            return None

        # 查找最后一个成功完成的任务结果
        for result in reversed(results):
            if hasattr(result, "status") and result.status == TaskStatus.COMPLETED:
                if hasattr(result, "result") and result.result:
                    return str(result.result)

        return None

    async def cleanup(self):
        """清理资源"""
        try:
            # 清理任务编排器
            if self.task_orchestrator:
                await self.task_orchestrator.cleanup()

            # 调用父类清理
            await super().cleanup()

            logger.info("增强统一 Agent 服务资源清理完成")

        except Exception as e:
            logger.error(f"增强统一 Agent 服务清理失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            asyncio.create_task(self.cleanup())
        except Exception:
            pass
