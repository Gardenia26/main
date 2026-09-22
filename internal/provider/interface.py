"""Provider abstraction used by the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

from internal.schema.message import Message, ToolDefinition


class LLMProvider(ABC):
    """The model-facing interface consumed by :class:`AgentEngine`."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        """Generate one assistant message from the current conversation."""

        raise NotImplementedError
