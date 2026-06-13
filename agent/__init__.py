"""agent: a minimal, framework-free tool-using agent loop.

An :class:`~agent.core.Agent` runs a reason -> act -> observe loop: a planner
proposes a tool call, the tool executes, the observation is fed back, and the
loop repeats until the planner emits a final answer or a step limit is hit.

The planner is abstracted behind :class:`~agent.planner.Planner`. A deterministic
rule-based :class:`~agent.planner.MockPlanner` ships by default (no API key), and
an optional OpenAI function-calling planner activates behind ``OPENAI_API_KEY``.
"""

from .core import Agent, AgentResult, Step
from .tools import Tool, ToolError, build_default_tools
from .planner import Planner, MockPlanner, get_planner, PlannerAction

__all__ = [
    "Agent",
    "AgentResult",
    "Step",
    "Tool",
    "ToolError",
    "build_default_tools",
    "Planner",
    "MockPlanner",
    "get_planner",
    "PlannerAction",
]

__version__ = "0.1.0"
