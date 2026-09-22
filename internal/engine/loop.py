"""Main agent loop with an optional planning/thinking phase."""

from __future__ import annotations

import logging

from internal.provider.interface import LLMProvider
from internal.schema.message import Message, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER, ToolResult
from internal.tools.registry import Registry

log = logging.getLogger(__name__)


class AgentEngine:
    """Coordinate model calls, tool execution, and conversation history."""

    MAX_TURNS = 16

    def __init__(self, provider: LLMProvider, registry: Registry, work_dir: str, enable_thinking: bool):
        self.provider = provider
        self.registry = registry
        self.work_dir = work_dir
        self.enable_thinking = enable_thinking

    def _initial_context(self, user_prompt: str) -> list[Message]:
        return [
            Message(role=ROLE_SYSTEM, content=(
                "You are py-tiny-claw, an expert coding assistant. "
                "You have full access to tools in the workspace.\n"
                f"Current workspace: {self.work_dir}"
            )),
            Message(role=ROLE_USER, content=user_prompt),
        ]

    def _run_thinking_phase(self, context: list[Message]) -> None:
        log.info("[Engine][Phase 1] 开始慢思考，暂不挂载工具...")
        plan = self.provider.generate(context, available_tools=None)
        if plan.tool_calls:
            log.warning("[Engine][Phase 1] 忽略规划阶段返回的工具调用")
            plan = Message(role=ROLE_ASSISTANT, content=plan.content)
        context.append(plan)
        if plan.content:
            log.info("  [思考计划]: %s", plan.content)

    def run(self, user_prompt: str) -> str:
        if not user_prompt.strip():
            raise ValueError("user_prompt 不能为空")

        log.info("[Engine] 引擎启动，工作区：%s", self.work_dir)
        log.info("[Engine] 慢思考模式 (Thinking Phase): %s", self.enable_thinking)
        context = self._initial_context(user_prompt)
        if self.enable_thinking:
            self._run_thinking_phase(context)

        for turn_count in range(1, self.MAX_TURNS + 1):
            log.info("========== [Turn %d]开始 ==========", turn_count)
            available_tools = self.registry.get_available_tools()
            log.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")
            response = self.provider.generate(context, available_tools)
            context.append(response)
            if response.content:
                log.info("  [对外回复]: %s", response.content)
            if not response.tool_calls:
                log.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                return response.content or ""

            log.info("  [Engine] 模型请求调用 %d 个工具...", len(response.tool_calls))
            for call in response.tool_calls:
                log.info("  -> 执行工具: %s, 参数: %s", call.name, call.arguments)
                try:
                    result = self.registry.execute(call)
                except Exception as exc:
                    log.exception("[Engine] 工具执行失败: %s", call.name)
                    result = ToolResult(tool_call_id=call.id, output=f"工具执行失败：{exc}", is_error=True)
                context.append(Message(role=ROLE_USER, content=result.output, tool_call_id=result.tool_call_id or call.id))
                if result.is_error:
                    log.info("  -> 工具执行失败: %s", result.output)
                else:
                    log.info("  -> 工具执行成功 (返回 %d 字节)", len(result.output.encode("utf-8")))

        raise RuntimeError(f"Agent 在 {self.MAX_TURNS} 轮内没有生成最终回答")