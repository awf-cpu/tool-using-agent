"""The agent reason->act->observe loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .planner import Planner, PlannerAction, get_planner
from .tools import Tool, ToolError, build_default_tools


@dataclass
class Step:
    """A single executed step in the loop."""

    tool: str
    tool_input: str
    observation: str


@dataclass
class AgentResult:
    """Final outcome of an agent run."""

    task: str
    answer: str
    steps: List[Step] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # or "max_steps"


class Agent:
    """Runs a planner against a set of tools until done or out of steps."""

    def __init__(
        self,
        tools: Sequence[Tool] | None = None,
        planner: Planner | None = None,
        max_steps: int = 6,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or build_default_tools())}
        self.planner = planner or get_planner()
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResult:
        """Execute the loop for ``task`` and return the result."""
        history: List[tuple[str, str, str]] = []
        steps: List[Step] = []

        for _ in range(self.max_steps):
            action = self.planner.plan(task, list(self.tools.values()), history)
            if action.final:
                return AgentResult(task=task, answer=action.answer, steps=steps)

            observation = self._execute(action)
            steps.append(Step(action.tool, action.tool_input, observation))
            history.append((action.tool, action.tool_input, observation))

        # max steps reached: ask the planner for a best-effort final answer
        final = self.planner.plan(
            task, list(self.tools.values()), history
        )
        answer = final.answer if final.final else _fallback_answer(history)
        return AgentResult(
            task=task, answer=answer, steps=steps, stopped_reason="max_steps"
        )

    def _execute(self, action: PlannerAction) -> str:
        """Validate and run the selected tool, capturing safe errors."""
        tool = self.tools.get(action.tool)
        if tool is None:
            return f"error: unknown tool {action.tool!r}"
        try:
            return tool(action.tool_input)
        except ToolError as exc:
            return f"error: {exc}"


def _fallback_answer(history: List[tuple[str, str, str]]) -> str:
    if not history:
        return "No answer produced before reaching the step limit."
    return "Partial result(s): " + "; ".join(f"{t} = {o}" for t, _, o in history)
