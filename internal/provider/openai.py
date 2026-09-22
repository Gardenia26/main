"""OpenAI-compatible provider implementation.

DeepSeek exposes an OpenAI-compatible Chat Completions API.  The provider
therefore contains one message/tool translation layer that can be reused for
both OpenAI-compatible services and DeepSeek.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # Keep offline schema tests usable without the SDK.
    OpenAI = None  # type: ignore[assignment,misc]

from internal.provider.interface import LLMProvider
from internal.schema.message import (
    Message,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    ToolCall,
    ToolDefinition,
)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a dictionary response."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _content_to_text(content: Any) -> str:
    """Normalize the text content returned by different SDK versions."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            text = _get_field(part, "text")
            if text is not None:
                text_parts.append(str(text))
        if text_parts:
            return "".join(text_parts)
    return str(content)


class OpenAIProvider(LLMProvider):
    """Adapt the internal message protocol to OpenAI Chat Completions."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    @staticmethod
    def _message_to_openai(message: Message) -> dict[str, Any]:
        """Translate one internal message into an OpenAI-compatible object."""

        if message.tool_call_id:
            # Internally tool results are attached to a user message.  The
            # OpenAI-compatible protocol represents them with role=tool.
            return {
                "role": ROLE_TOOL,
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }

        if message.role == ROLE_SYSTEM:
            return {"role": ROLE_SYSTEM, "content": message.content}

        if message.role == ROLE_USER:
            return {"role": ROLE_USER, "content": message.content}

        if message.role == ROLE_ASSISTANT:
            payload: dict[str, Any] = {
                "role": ROLE_ASSISTANT,
                # ``None`` is the standard representation when an assistant
                # response consists only of tool calls.
                "content": message.content or None,
            }
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload

        if message.role == ROLE_TOOL:
            raise ValueError("tool 消息必须提供 tool_call_id")

        raise ValueError(f"不支持的消息角色: {message.role!r}")

    @staticmethod
    def _tool_to_openai(tool: ToolDefinition) -> dict[str, Any]:
        """Translate a framework tool definition to function-calling schema."""

        parameters = tool.input_schema or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

    @staticmethod
    def _response_to_message(response: Any) -> Message:
        """Translate an SDK response object into the internal Message type."""

        choices = _get_field(response, "choices", [])
        if not choices:
            raise RuntimeError("模型响应中没有 choices")

        choice = choices[0]
        choice_message = _get_field(choice, "message")
        if choice_message is None:
            # This fallback also makes simple fake responses convenient in
            # tests, where content/tool_calls may be placed on the choice.
            choice_message = choice

        content = _content_to_text(_get_field(choice_message, "content", ""))
        raw_tool_calls = _get_field(choice_message, "tool_calls", []) or []
        tool_calls: list[ToolCall] = []

        for raw_call in raw_tool_calls:
            function = _get_field(raw_call, "function", {}) or {}
            arguments = _get_field(function, "arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                ToolCall(
                    id=str(_get_field(raw_call, "id", "") or ""),
                    name=str(_get_field(function, "name", "") or ""),
                    arguments=arguments,
                )
            )

        return Message(
            role=ROLE_ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        """Send the conversation to the model and normalize its response."""

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                self._message_to_openai(message) for message in messages
            ],
        }

        if available_tools:
            request["tools"] = [
                self._tool_to_openai(tool) for tool in available_tools
            ]

        response = self.client.chat.completions.create(**request)
        return self._response_to_message(response)


def new_deepseek_provider(model: str) -> OpenAIProvider:
    """Construct an :class:`OpenAIProvider` backed by DeepSeek."""

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("请先配置 DEEPSEEK_API_KEY 环境变量")

    if OpenAI is None:
        raise RuntimeError("未安装 openai SDK，请先执行: pip install openai")

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )
    return OpenAIProvider(client, model)
