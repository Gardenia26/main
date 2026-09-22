"""Tool registry abstraction used by the agent engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

from internal.schema.message import ToolCall, ToolDefinition, ToolResult


class Registry(ABC):
    """Interface for discovering and executing agent tools."""

    @abstractmethod
    def get_available_tools(self) -> list[ToolDefinition]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, call: ToolCall) -> ToolResult:
        raise NotImplementedError
