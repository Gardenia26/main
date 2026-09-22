"""Run experiment test 1 and test 2 with identical control variables."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the test title and logging output in the same stream and order.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)

from internal.engine.loop import AgentEngine
from internal.provider.openai import new_deepseek_provider
from cmd.claw.main import MODEL, PROMPT, MockRegistry


def run_test(provider, registry, enable_thinking: bool) -> None:
    """Run one experiment while changing only EnableThinking."""

    print(
        f"\n====测试{'1' if enable_thinking else '2'} "
        f"EnableThinking = {enable_thinking}====",
        flush=True,
    )
    engine = AgentEngine(
        provider=provider,
        registry=registry,
        work_dir=str(PROJECT_ROOT),
        enable_thinking=enable_thinking,
    )
    engine.run(PROMPT)


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("请先配置 DEEPSEEK_API_KEY 环境变量")

    # 两次测试共用同一个 Provider、模型、Prompt 和 MockRegistry。
    provider = new_deepseek_provider(MODEL)
    registry = MockRegistry()

    run_test(provider, registry, enable_thinking=True)
    run_test(provider, registry, enable_thinking=False)


if __name__ == "__main__":
    main()
