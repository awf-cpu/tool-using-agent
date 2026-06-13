"""Tests for the end-to-end agent loop, including the max-steps guard."""

import os

from agent.core import Agent
from agent.planner import MockPlanner, Planner, PlannerAction
from agent.tools import Tool, build_default_tools

SANDBOX = os.path.join(os.path.dirname(__file__), "..", "sandbox")


def test_agent_solves_multistep_task():
    agent = Agent(tools=build_default_tools(sandbox_dir=SANDBOX), planner=MockPlanner())
    result = agent.run("What is 12*7 and who wrote Hamlet?")

    tools_used = [s.tool for s in result.steps]
    assert "calculator" in tools_used
    assert "search" in tools_used
    assert "84" in result.answer
    assert "Shakespeare" in result.answer
    assert result.stopped_reason == "final_answer"


def test_agent_reads_file_when_referenced():
    agent = Agent(tools=build_default_tools(sandbox_dir=SANDBOX), planner=MockPlanner())
    result = agent.run("Read notes.txt and summarise it")
    assert any(s.tool == "read_file" for s in result.steps)
    assert "release is scheduled" in result.answer


def test_max_steps_guard():
    """A planner that never finalises must be stopped by the step limit."""

    class NeverDonePlanner(Planner):
        def plan(self, task, tools, history):
            return PlannerAction(tool="calculator", tool_input="1 + 1")

    agent = Agent(planner=NeverDonePlanner(), max_steps=3)
    result = agent.run("loop forever")
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3


def test_unknown_tool_is_handled_gracefully():
    class BadPlanner(Planner):
        def __init__(self):
            self.calls = 0

        def plan(self, task, tools, history):
            self.calls += 1
            if self.calls == 1:
                return PlannerAction(tool="nonexistent", tool_input="x")
            return PlannerAction(final=True, answer="done")

    agent = Agent(planner=BadPlanner())
    result = agent.run("trigger unknown tool")
    assert "unknown tool" in result.steps[0].observation
    assert result.answer == "done"
