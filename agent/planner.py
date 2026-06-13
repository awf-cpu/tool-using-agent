"""Planners decide the next action given the task and history.

A :class:`Planner` returns a :class:`PlannerAction` that is either a tool call
(``tool`` + ``tool_input``) or a final answer (``final=True``). The default
:class:`MockPlanner` is a deterministic, keyword-driven planner that needs no API
key; an optional OpenAI function-calling planner sits behind ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from .tools import Tool


@dataclass
class PlannerAction:
    """One decision from a planner.

    Either a tool call or a final answer. When ``final`` is True, ``answer``
    holds the response and ``tool``/``tool_input`` are ignored.
    """

    final: bool = False
    answer: str = ""
    tool: str = ""
    tool_input: str = ""


class Planner(ABC):
    """Chooses the next :class:`PlannerAction`."""

    @abstractmethod
    def plan(
        self,
        task: str,
        tools: Sequence[Tool],
        history: List[tuple[str, str, str]],
    ) -> PlannerAction:
        """Decide the next action.

        Args:
            task: the user's original task.
            tools: tools available to call.
            history: list of ``(tool, tool_input, observation)`` already executed.
        """
        raise NotImplementedError


_CALC_RE = re.compile(r"[-+]?\d[\d\s]*(?:[-+*/%]|\*\*|//)[\s\d().+\-*/%]*\d")


class MockPlanner(Planner):
    """Deterministic, rule-based planner for offline use.

    Strategy: handle each sub-goal in the task exactly once, then answer.
    * If the task contains an arithmetic expression not yet computed -> calculator.
    * If the task asks "who/what/where" about a topic not yet searched -> search.
    * If it references a file not yet read -> read_file.
    * Otherwise -> compose a final answer from observations.
    """

    def plan(self, task, tools, history):
        tool_names = {t.name for t in tools}
        done_tools = {tool for tool, _, _ in history}

        # 1. arithmetic
        if "calculator" in tool_names and "calculator" not in done_tools:
            match = _CALC_RE.search(task)
            if match:
                return PlannerAction(tool="calculator", tool_input=match.group().strip())

        # 2. factual lookup
        if "search" in tool_names and "search" not in done_tools:
            topic = _extract_search_topic(task)
            if topic:
                return PlannerAction(tool="search", tool_input=topic)

        # 3. file read
        if "read_file" in tool_names and "read_file" not in done_tools:
            fname = _extract_filename(task)
            if fname:
                return PlannerAction(tool="read_file", tool_input=fname)

        # 4. done -> synthesise an answer from what we observed
        return PlannerAction(final=True, answer=_compose_answer(task, history))


class OpenAIPlanner(Planner):
    """Function-calling planner backed by OpenAI (optional, needs OPENAI_API_KEY)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # lazy import keeps it optional

        self._client = OpenAI()
        self.model = model

    def plan(self, task, tools, history):  # pragma: no cover - needs network
        import json

        functions = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            }
            for t in tools
        ]
        transcript = "\n".join(f"{t}({i}) -> {o}" for t, i, o in history)
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the task by calling tools one at a time. When you have "
                    "enough information, reply with the final answer and no tool call."
                ),
            },
            {"role": "user", "content": f"Task: {task}\n\nHistory:\n{transcript}"},
        ]
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, tools=functions, temperature=0.0
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = json.loads(call.function.arguments or "{}")
            return PlannerAction(tool=call.function.name, tool_input=str(args.get("input", "")))
        return PlannerAction(final=True, answer=msg.content or "")


def get_planner(name: str | None = None) -> Planner:
    """Return a planner (``"mock"`` or ``"openai"``).

    Reads ``AGENT_PLANNER`` when ``name`` is omitted, defaulting to ``"mock"``.
    """
    name = (name or os.getenv("AGENT_PLANNER") or "mock").lower()
    if name == "mock":
        return MockPlanner()
    if name == "openai":
        return OpenAIPlanner()
    raise ValueError(f"unknown planner: {name!r}")


# --------------------------------------------------------------------------- #
# Heuristics used by the mock planner
# --------------------------------------------------------------------------- #
_FILE_RE = re.compile(r"([\w./-]+\.(?:txt|md|json|csv|log))", re.IGNORECASE)
_QUESTION_WORDS = ("who", "what", "where", "which", "name")
_STOPWORDS = {"is", "are", "the", "a", "an", "of", "and", "to", "in", "by"}


def _extract_filename(task: str) -> str:
    match = _FILE_RE.search(task)
    return match.group(1) if match else ""


def _extract_search_topic(task: str) -> str:
    """Pull a likely lookup topic from a 'who/what ...' clause."""
    lowered = task.lower()
    if not any(w in lowered for w in _QUESTION_WORDS):
        return ""
    # grab the clause after the last filler verb, strip punctuation
    for word in ("wrote ", "painted ", "created ", "discovered ", "made "):
        idx = lowered.rfind(word)
        if idx != -1:
            return _clean_topic(task[idx + len(word):])
    # fallback: text after the question word
    for w in _QUESTION_WORDS:
        idx = lowered.rfind(w)
        if idx != -1:
            return _clean_topic(task[idx + len(w):])
    return ""


def _clean_topic(text: str) -> str:
    """Normalise a candidate topic; reject ones with no real lookup content.

    Keeps the planner from treating a leftover arithmetic fragment (e.g.
    "12 * 7" in "What is 12 * 7?") as a knowledge-base lookup. After stripping
    punctuation and leading stopwords, the topic must contain a non-stopword
    alphabetic token to count as a valid search.
    """
    topic = re.sub(r"[?.!]", "", text).strip()
    tokens = topic.split()
    while tokens and tokens[0].lower() in _STOPWORDS:
        tokens.pop(0)
    topic = " ".join(tokens)
    meaningful = any(
        tok.isalpha() and tok.lower() not in _STOPWORDS for tok in topic.split()
    )
    return topic if meaningful else ""


def _compose_answer(task: str, history: List[tuple[str, str, str]]) -> str:
    if not history:
        return "I could not determine any actions to take for this task."
    parts = [f"{tool}({inp!r}) = {obs}" for tool, inp, obs in history]
    return "Result(s): " + "; ".join(parts)
