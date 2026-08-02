from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.langchain_agent_service import LangChainAgentService


class FakeVectorService:
    def __init__(self):
        self.calls = []

    async def search_knowledge(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            {
                "record_type": "document_chunk",
                "title": "PostgreSQL MVCC",
                "content": "Snapshots determine row visibility.",
                "source_uri": "https://www.postgresql.org/docs/current/mvcc.html",
            }
        ]


class FakeMCPService:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments, caller_access="public"):
        self.calls.append((name, arguments, caller_access))
        return '{"status":"media_ready"}'


class FakeToolLLM:
    def __init__(self):
        self.bound_tools = None
        self.messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.messages.append(list(messages))
        if len(self.messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "generate_image",
                        "args": {"prompt": "MVCC snapshot diagram"},
                        "id": "call_media",
                        "type": "tool_call",
                    }
                ],
            )
        assert isinstance(messages[-1], ToolMessage)
        return AIMessage(content="图片已经生成。")


@pytest.mark.asyncio
async def test_scoped_langchain_rag_keeps_mcp_tools_reachable(tmp_path):
    mcp = FakeMCPService()
    service = LangChainAgentService(
        database_service=None,
        mcp_service=mcp,
        vector_store_path=str(tmp_path),
    )
    service._initialized = True
    service.vector_service = FakeVectorService()
    service._llm = FakeToolLLM()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
            },
        }
    ]

    answer = await service.chat_completion(
        [
            {"role": "system", "content": "Answer with citations."},
            {"role": "user", "content": "画一张 MVCC 快照图"},
        ],
        tools=tools,
        tool_access="admin",
        knowledge_namespaces=["global:technical", "private:42"],
    )

    assert answer == "图片已经生成。"
    assert service.vector_service.calls[0][1]["knowledge_namespaces"] == [
        "global:technical",
        "private:42",
    ]
    assert service._llm.bound_tools == tools
    assert mcp.calls == [
        (
            "generate_image",
            {"prompt": "MVCC snapshot diagram"},
            "admin",
        )
    ]
