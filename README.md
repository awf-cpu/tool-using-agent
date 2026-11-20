# tool-using-agent

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/awf-cpu/tool-using-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/awf-cpu/tool-using-agent/actions/workflows/ci.yml)

A minimal, **framework-free tool-using agent** in pure Python. Given a task, the agent runs a **reason, act, observe** loop: a planner picks a tool, the tool runs, the observation is fed back, and the loop repeats until the planner returns a final answer or hits a step limit. It runs fully **offline** with a deterministic rule-based planner and can switch to **OpenAI function-calling** with an environment variable. Zero runtime dependencies.

## The business problem

Most "AI agent" projects fail in production not because of the model but because of the control loop around it: tools that accept bad arguments, errors that crash the run, loops that never stop, file access that escapes its folder. This repo shows those guardrails in a few hundred readable, tested lines, the same shape that maps onto provider function-calling APIs.

## Try it in one line

```bash
python -m agent.run "What is 12*7 and who wrote Hamlet?"
```

Output (offline mock planner):

```
Task: What is 12*7 and who wrote Hamlet?

Steps:
  1. calculator('12*7') -> 84
  2. search('Hamlet') -> Hamlet was written by William Shakespeare around 1600.

Answer: Result(s): calculator('12*7') = 84; search('Hamlet') = Hamlet was written by William Shakespeare around 1600.
(stopped: final_answer)
```

Real planner (needs `pip install openai`):

```bash
export OPENAI_API_KEY=...
python -m agent.run "Read notes.txt and tell me when the release is" --planner openai
```

## Architecture

1. **Tools** (`tools.py`): a safe calculator (AST-evaluated, no `eval`), an offline knowledge-base `search`, and a path-sandboxed `read_file`.
2. **Planner** (`planner.py`): a `Planner` interface returning a tool call or a final answer. Deterministic `MockPlanner` by default, optional OpenAI function-calling planner.
3. **Agent** (`core.py`): drives the loop, validates tool calls, converts tool errors into safe observations, and enforces a `max_steps` guard.

**Guardrails built in:** no `eval` (the calculator parses with `ast` and allows only numbers and arithmetic), `read_file` refuses anything outside its sandbox (`../` and absolute paths blocked), unknown tools do not crash the run, and termination is guaranteed.

## Tests

```bash
pip install pytest && pytest -q
```

14 tests: calculator (including code-injection rejection), search hits and misses, sandbox traversal blocking, planner action selection, a full multi-step solve, unknown-tool handling and the max-steps guard. CI runs them on Python 3.11 and 3.12 with no API keys.

## How this becomes a solution for your company

In real projects the tools are your systems (CRM, case-management, calendar, billing) and the guardrails are your rules: what the agent may do alone, what needs a human, what is logged for audit. If you want an agent that acts inside your operation with those limits made explicit, see **Work with me** below.

## Work with me

I am André Feix, founder of **Pindorama Tecnologia** (Florianópolis, Brazil). We build custom AI systems for companies and law firms: WhatsApp agents with compliance guardrails (LGPD, bar-association rules), RAG over documents, document and deadline automation, LLM routing with cost control, and local AI with data kept in Brazil.

- LinkedIn: https://www.linkedin.com/in/andrefeix
- Email: andrefeix777@gmail.com
- More: https://github.com/awf-cpu

## License

MIT © 2026 awf-cpu
