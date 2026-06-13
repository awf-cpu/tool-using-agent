"""CLI entry point: ``python -m agent.run "What is 12*7 and who wrote Hamlet?"``."""

from __future__ import annotations

import argparse

from .core import Agent
from .planner import get_planner
from .tools import build_default_tools


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent.run",
        description="Run a minimal tool-using agent on a task.",
    )
    parser.add_argument("task", help="the task for the agent to solve")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--sandbox", default=None, help="dir for the read_file tool")
    parser.add_argument(
        "--planner", default=None, help="planner: mock|openai (default: env or mock)"
    )
    args = parser.parse_args(argv)

    agent = Agent(
        tools=build_default_tools(sandbox_dir=args.sandbox),
        planner=get_planner(args.planner),
        max_steps=args.max_steps,
    )
    result = agent.run(args.task)

    print(f"Task: {result.task}")
    print("\nSteps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step.tool}({step.tool_input!r}) -> {step.observation}")
    print(f"\nAnswer: {result.answer}")
    print(f"(stopped: {result.stopped_reason})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
