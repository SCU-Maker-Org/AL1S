from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.unified_agent.unified_agent_service import UnifiedAgentService


def _agent():
    agent = UnifiedAgentService.__new__(UnifiedAgentService)
    agent.vector_service = SimpleNamespace()
    return agent


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.mark.asyncio
async def test_tool_results_follow_the_assistant_tool_call_message():
    agent = _agent()
    agent.tool_handler = AsyncMock(side_effect=["first result", "second result"])

    completion_create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="final response"))]
        )
    )
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))
    )

    assistant_message = SimpleNamespace(
        content=None,
        tool_calls=[
            _tool_call("call_1", "first_tool", '{"query":"one"}'),
            _tool_call("call_2", "second_tool", '{"query":"two"}'),
        ],
    )
    messages = [{"role": "user", "content": "search"}]

    result = await agent._handle_tool_calls(assistant_message, messages)

    assert result == "final response"
    sent_messages = completion_create.await_args.kwargs["messages"]
    assert [message["role"] for message in sent_messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [call["id"] for call in sent_messages[1]["tool_calls"]] == [
        "call_1",
        "call_2",
    ]
    assert sent_messages[2]["tool_call_id"] == "call_1"
    assert sent_messages[3]["tool_call_id"] == "call_2"


@pytest.mark.asyncio
async def test_tool_calls_can_continue_across_multiple_rounds():
    agent = _agent()
    agent.tool_handler = AsyncMock(side_effect=["library id", "documentation"])
    second_tool_call = SimpleNamespace(
        content=None,
        tool_calls=[
            _tool_call(
                "call_docs",
                "ctx7_query-docs",
                '{"libraryId":"/pydantic/pydantic","query":"validation"}',
            )
        ],
    )
    final_message = SimpleNamespace(content="final answer", tool_calls=[])
    completion_create = AsyncMock(
        side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(message=second_tool_call)]),
            SimpleNamespace(choices=[SimpleNamespace(message=final_message)]),
        ]
    )
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))
    )
    first_tool_call = SimpleNamespace(
        content=None,
        tool_calls=[
            _tool_call(
                "call_resolve",
                "ctx7_resolve-library-id",
                '{"libraryName":"pydantic","query":"validation"}',
            )
        ],
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ctx7_resolve-library-id",
                "parameters": {"type": "object"},
            },
        }
    ]

    result = await agent._handle_tool_calls(
        first_tool_call,
        [{"role": "user", "content": "look up pydantic validation"}],
        tools,
        caller_access="private",
    )

    assert result == "final answer"
    assert completion_create.await_count == 2
    assert all(
        call.kwargs["tools"] == tools for call in completion_create.await_args_list
    )
    assert agent.tool_handler.await_args_list[0].args[2] == "private"
    assert agent.tool_handler.await_args_list[1].args[2] == "private"
    final_messages = completion_create.await_args_list[1].kwargs["messages"]
    assert [message["role"] for message in final_messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


@pytest.mark.asyncio
async def test_mcp_tool_uses_record_tool_call():
    agent = _agent()
    agent.mcp_service = SimpleNamespace(call_tool=AsyncMock(return_value="tool result"))
    agent.database_service = SimpleNamespace(
        record_tool_call=AsyncMock(return_value=True)
    )
    agent._conversation_id_context = ContextVar("test_conversation_id", default=None)
    agent._conversation_id_context.set(42)

    result = await agent._handle_mcp_tool("test_tool", {"value": 1})

    assert result == "tool result"
    call = agent.database_service.record_tool_call.await_args.kwargs
    assert call["conversation_id"] == 42
    assert call["tool_name"] == "test_tool"
    assert call["arguments"] == {"value": 1}
    assert call["result"] == "tool result"
    assert call["success"] is True
    assert call["error_message"] is None
    assert call["execution_time"] >= 0


@pytest.mark.asyncio
async def test_database_logging_failure_does_not_replace_tool_result():
    agent = _agent()
    agent.mcp_service = SimpleNamespace(call_tool=AsyncMock(return_value="tool result"))
    agent.database_service = SimpleNamespace(
        record_tool_call=AsyncMock(side_effect=RuntimeError("database unavailable"))
    )
    agent._conversation_id_context = ContextVar("test_conversation_id", default=None)
    agent._conversation_id_context.set(42)

    result = await agent._handle_mcp_tool("test_tool", {})

    assert result == "tool result"
