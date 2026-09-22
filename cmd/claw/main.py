"""Experiment entry point for DeepSeek Provider integration."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internal.engine.loop import AgentEngine
from internal.provider.openai import new_deepseek_provider
from internal.schema.message import ToolCall, ToolDefinition, ToolResult
from internal.tools.registry import Registry

logging.basicConfig(level=logging.INFO, format="%(message)s")
WEATHER_OUTPUT = "API 返回：今天是晴天，气温25度。"


class MockRegistry(Registry):
    """Provide a deterministic weather tool for the classroom experiment."""

    def get_available_tools(self) -> list[ToolDefinition]:
        return [ToolDefinition(
            name="get_weather",
            description="获取指定城市的当前天气情况",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        )]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name != "get_weather":
            return ToolResult(tool_call_id=call.id, output=f"未知工具：{call.name}", is_error=True)
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            return ToolResult(tool_call_id=call.id, output=f"工具参数不是有效 JSON：{exc}", is_error=True)
        city = arguments.get("city") if isinstance(arguments, dict) else None
        if not isinstance(city, str) or not city.strip():
            return ToolResult(tool_call_id=call.id, output="工具参数缺少必填字段 city", is_error=True)
        return ToolResult(tool_call_id=call.id, output=WEATHER_OUTPUT, is_error=False)


MODEL = "deepseek-chat"
PROMPT = "我想去北京跑步，帮我查查天气适合吗？"
ENABLE_THINKING = False


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("请先配置 DEEPSEEK_API_KEY 环境变量")
    provider = new_deepseek_provider(MODEL)
    engine = AgentEngine(provider, MockRegistry(), str(PROJECT_ROOT), ENABLE_THINKING)
    engine.run(PROMPT)


if __name__ == "__main__":
    main()