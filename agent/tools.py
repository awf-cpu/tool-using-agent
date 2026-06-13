"""Tools the agent can call.

Each tool is a small, validated, side-effect-aware callable. Three are provided:

* ``calculator`` — evaluates arithmetic expressions safely (no ``eval``).
* ``search`` — looks up an offline in-memory knowledge base.
* ``read_file`` — reads a file, sandboxed to a single base directory.
"""

from __future__ import annotations

import ast
import operator
import os
from dataclasses import dataclass
from typing import Callable, Dict, List


class ToolError(Exception):
    """Raised when a tool is called with invalid arguments or fails safely."""


@dataclass
class Tool:
    """A named, documented callable the agent can invoke.

    Attributes:
        name: unique identifier the planner uses to select the tool.
        description: one-line summary shown to the planner.
        func: callable taking a single string argument and returning a string.
    """

    name: str
    description: str
    func: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.func(arg)


# --------------------------------------------------------------------------- #
# Safe calculator
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> str:
    """Evaluate an arithmetic expression without using ``eval``.

    Supports + - * / // % ** and parentheses over numeric literals. Anything
    else (names, function calls, attribute access) raises :class:`ToolError`.
    """
    expression = expression.strip()
    if not expression:
        raise ToolError("empty expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ToolError(f"invalid expression: {expression!r}") from exc
    result = _eval_node(tree.body)
    # render ints without trailing ".0"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ToolError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("unsupported expression element")


# --------------------------------------------------------------------------- #
# Offline knowledge-base search
# --------------------------------------------------------------------------- #
DEFAULT_KB: Dict[str, str] = {
    "hamlet": "Hamlet was written by William Shakespeare around 1600.",
    "mona lisa": "The Mona Lisa was painted by Leonardo da Vinci.",
    "gravity": "The law of universal gravitation was formulated by Isaac Newton.",
    "relativity": "The theory of relativity was developed by Albert Einstein.",
    "python": "Python is a programming language created by Guido van Rossum.",
}


def make_search(knowledge_base: Dict[str, str] | None = None) -> Callable[[str], str]:
    """Build a search function over an in-memory knowledge base (offline)."""
    kb = knowledge_base if knowledge_base is not None else DEFAULT_KB

    def search(query: str) -> str:
        q = query.strip().lower()
        if not q:
            raise ToolError("empty query")
        # exact key, then substring match on keys, then value match
        if q in kb:
            return kb[q]
        for key, value in kb.items():
            if key in q or q in key:
                return value
        for key, value in kb.items():
            if any(word in value.lower() for word in q.split()):
                return value
        return f"No knowledge-base entry found for {query!r}."

    return search


# --------------------------------------------------------------------------- #
# Sandboxed file reader
# --------------------------------------------------------------------------- #
def make_read_file(base_dir: str) -> Callable[[str], str]:
    """Build a ``read_file`` tool restricted to ``base_dir`` (prevents traversal)."""
    base = os.path.realpath(base_dir)

    def read_file(rel_path: str) -> str:
        rel_path = rel_path.strip()
        if not rel_path:
            raise ToolError("empty path")
        target = os.path.realpath(os.path.join(base, rel_path))
        if target != base and not target.startswith(base + os.sep):
            raise ToolError("path escapes the sandbox directory")
        if not os.path.isfile(target):
            raise ToolError(f"file not found in sandbox: {rel_path!r}")
        with open(target, "r", encoding="utf-8") as fh:
            return fh.read()

    return read_file


# --------------------------------------------------------------------------- #
# Default tool set
# --------------------------------------------------------------------------- #
def build_default_tools(
    sandbox_dir: str | None = None,
    knowledge_base: Dict[str, str] | None = None,
) -> List[Tool]:
    """Return the standard three tools, optionally customised."""
    sandbox_dir = sandbox_dir or os.path.join(os.path.dirname(__file__), "..", "sandbox")
    return [
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression, e.g. '12 * 7'.",
            func=safe_calculate,
        ),
        Tool(
            name="search",
            description="Look up a fact in the offline knowledge base.",
            func=make_search(knowledge_base),
        ),
        Tool(
            name="read_file",
            description="Read a text file by relative path inside the sandbox.",
            func=make_read_file(sandbox_dir),
        ),
    ]
