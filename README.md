# tool-using-agent

A minimal, **framework-free tool-using agent** in pure Python. Given a task, the
agent runs a **reason → act → observe** loop: a planner picks a tool, the tool
runs, the observation is fed back, and the loop repeats until the planner emits a
final answer or hits a step limit. It runs fully **offline** with a deterministic
rule-based planner, and can switch to **OpenAI function-calling** behind an env var.

## Why it matters

"Agent" frameworks hide the part that actually matters: the control loop, tool
schemas, argument validation, observation feedback, and the guardrails that stop
runaway loops. This repo strips it to the essentials so the mechanics are
visible and testable — the same shape that maps directly onto provider
function-calling APIs.

## Architecture (in 4 lines)

1. **Tools** (`tools.py`) — a safe calculator (AST-evaluated, no `eval`), an offline knowledge-base `search`, and a path-sandboxed `read_file`.
2. **Planner** (`planner.py`) — `Planner` ABC returning a `PlannerAction` (tool call or final answer); deterministic `MockPlanner` by default, optional OpenAI function-calling planner.
3. **Agent** (`core.py`) — drives the loop, validates tool calls, captures errors, and enforces a `max_steps` guard.
4. Swapping the mock planner for OpenAI requires only `AGENT_PLANNER=openai` + `OPENAI_API_KEY`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install openai   # optional, only for the real planner
```

## Usage

```bash
python -m agent.run "What is 12*7 and who wrote Hamlet?"
```

Output (offline mock planner):

```
Task: What is 12*7 and who wrote Hamlet?

Steps:
  1. calculator('12*7') -> 84
  2. search('hamlet') -> Hamlet was written by William Shakespeare around 1600.

Answer: Result(s): calculator('12*7') = 84; search('hamlet') = Hamlet was written by William Shakespeare around 1600.
(stopped: final_answer)
```

Use the real planner:

```bash
export OPENAI_API_KEY=sk-...
python -m agent.run "Read notes.txt and tell me when the release is" --planner openai
```

## How it works

- The **mock planner** is keyword/regex-driven and intentionally boring: it
  detects an arithmetic expression, a who/what factual question, or a referenced
  filename, dispatches the matching tool exactly once each, then composes a final
  answer from the observations. This makes multi-step behaviour deterministic and
  unit-testable while exercising the identical loop the real planner uses.
- The **agent loop** records every `(tool, input, observation)` step, validates
  that the requested tool exists, converts `ToolError`s into safe observations
  (so a bad tool call never crashes the run), and guarantees termination via
  `max_steps`.
- **Safety**: the calculator parses with `ast` and only allows numeric literals
  and arithmetic operators; `read_file` resolves real paths and refuses anything
  outside its sandbox directory (blocking `../` traversal and absolute paths).

## Tests

```bash
pytest -q
```

Covers the calculator (incl. rejection of code injection), search hits/misses,
sandbox traversal blocking, planner action selection, a full multi-step solve,
unknown-tool handling, and the max-steps guard.

## License

MIT © 2026 awf-cpu
