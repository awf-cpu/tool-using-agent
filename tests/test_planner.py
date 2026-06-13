"""Tests for the mock planner's action selection and validation."""

from agent.planner import MockPlanner, PlannerAction
from agent.tools import build_default_tools


def _tools():
    return build_default_tools()


def test_planner_selects_calculator_for_arithmetic():
    planner = MockPlanner()
    action = planner.plan("What is 12 * 7?", _tools(), history=[])
    assert isinstance(action, PlannerAction)
    assert action.tool == "calculator"
    assert action.tool_input == "12 * 7"


def test_planner_selects_search_for_factual_question():
    planner = MockPlanner()
    action = planner.plan("Who wrote Hamlet?", _tools(), history=[])
    assert action.tool == "search"
    assert "hamlet" in action.tool_input.lower()


def test_planner_finalises_when_nothing_left():
    planner = MockPlanner()
    history = [("calculator", "12 * 7", "84")]
    action = planner.plan("What is 12 * 7?", _tools(), history=history)
    assert action.final is True
    assert "84" in action.answer


def test_planner_does_not_repeat_a_tool():
    planner = MockPlanner()
    history = [("search", "hamlet", "Shakespeare wrote Hamlet.")]
    action = planner.plan("Who wrote Hamlet?", _tools(), history=history)
    assert action.final is True
