"""
Agent 模块 - 统一 Agent 服务与任务编排

提供完整的 Agent 服务能力，包括：
- 统一 Agent 服务 (UnifiedAgentService)
- LangChain Agent 服务 (LangChainAgentService)  
- 增强统一 Agent 服务 (EnhancedUnifiedAgentService)
- 任务编排与执行能力
- 智能任务构建与工作流模板
"""

from .unified_agent import UnifiedAgentService, EnhancedUnifiedAgentService
from .langchain_agent_service import LangChainAgentService
from ..agents.unified_agent.task_orchestrator import (
    TaskOrchestrator,
    TaskExecutor,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
    TaskExecutionContext
)
from ..agents.unified_agent.task_builder import (
    TaskChain,
    TaskTemplates,
    SmartTaskBuilder
)

__all__ = [
    # 基础 Agent 服务
    "UnifiedAgentService",
    "LangChainAgentService",
    "EnhancedUnifiedAgentService",
    
    # 任务编排核心
    "TaskOrchestrator",
    "TaskExecutor", 
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    "TaskExecutionContext",
    
    # 任务构建器
    "TaskChain",
    "TaskTemplates", 
    "SmartTaskBuilder"
]

# 版本信息
__version__ = "0.1.0"
__description__ = "统一 Agent 服务与任务编排系统"
