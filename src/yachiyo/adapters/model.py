"""Request-scoped OpenAI-compatible chat and tool calls."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any

from openai import APIError, AsyncOpenAI

from yachiyo.config import AppSettings, SecretStore
from yachiyo.config.models import ModelEndpointSettings
from yachiyo.ports.model import (
    ModelImage,
    ModelMessage,
    ModelRequest,
    ModelResult,
    ModelUsage,
    ToolCall,
)


class ModelProviderError(RuntimeError):
    pass


@dataclass(slots=True)
class _ConfiguredClient:
    settings: ModelEndpointSettings
    client: AsyncOpenAI
    endpoint_id: str = ""


class OpenAICompatibleModel:
    def __init__(self, settings: AppSettings, secrets: SecretStore):
        self.settings = settings
        self.secrets = secrets
        self._chat_clients: list[_ConfiguredClient] = []
        self._chat_slots = asyncio.Semaphore(settings.models.chat.max_concurrency)

    def validate_credentials(self) -> None:
        for endpoint in (self.settings.models.chat, *self.settings.models.chat.fallback):
            self.secrets.read(endpoint.api_key_secret, required=True)

    def _create_client(self, endpoint: ModelEndpointSettings) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.secrets.read(endpoint.api_key_secret, required=True),
            base_url=str(endpoint.base_url).rstrip("/"),
            timeout=endpoint.timeout_seconds,
            max_retries=endpoint.max_retries,
        )

    def _get_chat_clients(self) -> list[_ConfiguredClient]:
        if not self._chat_clients:
            endpoints = (self.settings.models.chat, *self.settings.models.chat.fallback)
            self._chat_clients = [
                _ConfiguredClient(
                    endpoint,
                    self._create_client(endpoint),
                    self._endpoint_id(endpoint, index),
                )
                for index, endpoint in enumerate(endpoints)
            ]
        return self._chat_clients

    async def complete(self, request: ModelRequest) -> ModelResult:
        errors: list[str] = []
        clients = self._get_chat_clients()
        if request.continuation_endpoint:
            clients = [
                item for item in clients if item.endpoint_id == request.continuation_endpoint
            ]
            if not clients:
                raise ModelProviderError("the response continuation endpoint is unavailable")
        for configured in clients:
            try:
                async with self._chat_slots:
                    if configured.settings.api_mode == "responses":
                        return await self._responses(configured, request)
                    return await self._chat_completions(configured, request)
            except (APIError, OSError, TimeoutError) as exc:
                errors.append(f"{configured.settings.model}: {type(exc).__name__}")
        raise ModelProviderError("all configured chat endpoints failed: " + ", ".join(errors))

    @staticmethod
    def _endpoint_id(settings: ModelEndpointSettings, index: int) -> str:
        return (
            f"{index}|{settings.api_mode}|{settings.api_key_secret}|"
            f"{str(settings.base_url).rstrip('/')}|{settings.model}"
        )

    async def _chat_completions(
        self, configured: _ConfiguredClient, request: ModelRequest
    ) -> ModelResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.instructions}]
        messages.extend(self._chat_message(message) for message in request.messages)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]
        arguments: dict[str, Any] = {
            "model": configured.settings.model,
            "messages": messages,
            "temperature": request.temperature,
            configured.settings.token_parameter: request.max_output_tokens,
        }
        if tools:
            arguments["tools"] = tools
        response = await configured.client.chat.completions.create(**arguments)
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for call in message.tool_calls or ():
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_invalid_json": call.function.arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = response.usage
        return ModelResult(
            text=message.content or "",
            tool_calls=tuple(calls),
            usage=ModelUsage(
                input_tokens=int(usage.prompt_tokens) if usage else 0,
                output_tokens=int(usage.completion_tokens) if usage else 0,
            ),
            provider_request_id=getattr(response, "_request_id", None),
            model=response.model,
        )

    async def _responses(self, configured: _ConfiguredClient, request: ModelRequest) -> ModelResult:
        input_items: list[dict[str, Any]] = []
        source_messages = request.messages
        if request.previous_response_id:
            trailing_tools = []
            for message in reversed(request.messages):
                if message.role != "tool":
                    break
                trailing_tools.append(message)
            source_messages = tuple(reversed(trailing_tools))
        for message in source_messages:
            if message.tool_calls:
                if message.content:
                    input_items.append({"role": "assistant", "content": message.content})
                input_items.extend(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    }
                    for call in message.tool_calls
                )
            elif message.role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
            else:
                input_items.append(self._response_message(message))
        tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            }
            for tool in request.tools
        ]
        arguments = {
            "model": configured.settings.model,
            "instructions": request.instructions,
            "input": input_items,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        if tools:
            arguments["tools"] = tools
        if request.previous_response_id:
            arguments["previous_response_id"] = request.previous_response_id
        response = await configured.client.responses.create(**arguments)
        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_arguments = getattr(item, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"_invalid_json": raw_arguments}
            calls.append(
                ToolCall(
                    id=getattr(item, "call_id", getattr(item, "id", "")),
                    name=getattr(item, "name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                )
            )
        usage = response.usage
        return ModelResult(
            text=response.output_text,
            tool_calls=tuple(calls),
            usage=ModelUsage(
                input_tokens=int(usage.input_tokens) if usage else 0,
                output_tokens=int(usage.output_tokens) if usage else 0,
            ),
            provider_request_id=getattr(response, "_request_id", None),
            model=response.model,
            continuation_id=response.id,
            continuation_endpoint=configured.endpoint_id,
        )

    @staticmethod
    def _image_uri(image: ModelImage) -> str:
        encoded = base64.b64encode(image.data).decode("ascii")
        return f"data:{image.mime_type};base64,{encoded}"

    def _chat_message(self, message: ModelMessage) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message.role}
        if message.images:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {"type": "image_url", "image_url": {"url": self._image_uri(image)}}
                for image in message.images
            )
            result["content"] = content
        else:
            result["content"] = message.content
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.name:
            result["name"] = message.name
        return result

    def _response_message(self, message: ModelMessage) -> dict[str, Any]:
        if not message.images:
            return {"role": message.role, "content": message.content}
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "input_text", "text": message.content})
        content.extend(
            {
                "type": "input_image",
                "image_url": self._image_uri(image),
                "detail": "auto",
            }
            for image in message.images
        )
        return {"role": message.role, "content": content}

    async def close(self) -> None:
        for configured in self._chat_clients:
            await configured.client.close()
        self._chat_clients.clear()
