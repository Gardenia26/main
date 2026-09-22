"""Common message types used by the agent engine and providers."""

from __future__ import annotations

from dataclasses import dataclass, field


ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    ``arguments`` is kept as the original JSON string so the registry can
    decide how to validate and decode each tool's input.
    """

    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class Message:
    """A provider-neutral conversation message."""

    role: str = ""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""


@dataclass
class ToolResult:
    """The result of executing one tool call."""

    tool_call_id: str = ""
    output: str = ""
    is_error: bool = False


@dataclass
class ToolDefinition:
    """A tool exposed to the model using a JSON Schema input description."""

    name: str = ""
    description: str = ""
    input_schema: dict | None = None
