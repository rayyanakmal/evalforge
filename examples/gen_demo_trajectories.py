#!/usr/bin/env python3
"""Generate demo trajectories for verdictlab v2 (Phase 5).

Runs a tiny ReAct-style agent (search + calculator tools) on the
hk-islands-facts suite via the DeepSeek API, twice:

  - Run A (clean):     normal system prompt, efficient tool use
  - Run B (degraded):  loop-prone prompt that encourages repeating the
                       same search before answering

Both runs use the SAME suite, SAME tools, SAME model — the only difference
is the system prompt (isolation of variables). Outputs per-test trajectory
files that `verdictlab import-trajectory` / `verdictlab compare --trajectory`
can grade.

Requires DEEPSEEK_API_KEY (env or ~/.hermes/.env). Deps: httpx + verdictlab.

Usage:
    python examples/gen_demo_trajectories.py [--model deepseek-chat]
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
from pathlib import Path

import httpx

from verdictlab.models import TokenCount
from verdictlab.trajectory.capture import StepRecorder

# Documented deepseek-chat pricing (USD per 1M tokens) — demo cost is an
# estimate from real token counts, marked as such in reports.
INPUT_RATE_PER_M = 0.27
OUTPUT_RATE_PER_M = 1.10

BASE_URL = "https://api.deepseek.com/v1"
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Knowledge base (deterministic "search" tool — no network)
# ---------------------------------------------------------------------------

FACTS = {
    "hong kong island": {"population": 1_270_000, "area": 78},
    "lantau": {"population": 200_000, "area": 147},
    "cheung chau": {"population": 9_000, "area": 2.5},
    "lamma": {"population": 6_000, "area": 13.9},
}


def tool_search(query: str) -> str:
    """Canned search: match the query against known HK island facts.

    Returns every fact whose island name appears in the query; if the query
    asks for a list/ranking (largest, list, all, islands), returns the full
    table so the agent can rank by area.
    """
    q = query.lower()
    if any(kw in q for kw in ("largest", "list", "all islands", "by area", "three")):
        return json.dumps(
            [{"name": n, "population": f["population"], "area": f["area"],
              "unit": "sq km"} for n, f in FACTS.items()],
            default=str,
        )
    matches = []
    for name, facts in FACTS.items():
        if name in q:
            matches.append(
                {"name": name, "population": facts["population"],
                 "area": facts["area"], "unit": "sq km"}
            )
    if matches:
        return json.dumps(matches if len(matches) > 1 else matches[0])
    return json.dumps({"error": "no results for: " + query})


# ---------------------------------------------------------------------------
# Safe calculator (deterministic, no eval)
# ---------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError("unsupported expression")


def tool_calculator(expr: str) -> str:
    """Safe arithmetic evaluation via AST (no eval)."""
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
        return json.dumps({"result": result})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"invalid expression: {exc}"})


TOOLS = {"search": tool_search, "calculator": tool_calculator}

# ---------------------------------------------------------------------------
# LLM call (DeepSeek chat completions, OpenAI-compatible)
# ---------------------------------------------------------------------------

SYSTEM_CLEAN = (
    "You are a helpful agent with two tools: search(query) and calculator(expr).\n"
    "Use them only when needed, then answer.\n"
    "Format each step as exactly one line:\n"
    '  TOOL: search {"query": "..."}\n'
    '  TOOL: calculator {"expr": "..."}\n'
    '  ANSWER: <your final answer as a number>\n'
    "Never output multiple lines per step. Stop as soon as you have the answer."
)

SYSTEM_LOOPY = (
    "You are a helpful agent with two tools: search(query) and calculator(expr).\n"
    "IMPORTANT: to be safe, ALWAYS call search twice with the same query before "
    "moving on, and call calculator twice with the same expression to double-check.\n"
    "Format each step as exactly one line:\n"
    '  TOOL: search {"query": "..."}\n'
    '  TOOL: calculator {"expr": "..."}\n'
    '  ANSWER: <your final answer as a number>\n'
    "Never output multiple lines per step."
)


def call_llm(client: httpx.Client, model: str, messages: list[dict]) -> tuple[str, int, int]:
    """One chat completion. Returns (text, input_tokens, output_tokens)."""
    resp = client.post(
        f"{BASE_URL}/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 256},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def run_one(client: httpx.Client, model: str, question: str, system: str) -> dict:
    """Run the agent on one question, recording the trajectory."""
    rec = StepRecorder()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    final_answer = None

    for _ in range(8):  # max steps
        text, in_tok, out_tok = call_llm(client, model, messages)
        cost = (in_tok / 1e6 * INPUT_RATE_PER_M) + (out_tok / 1e6 * OUTPUT_RATE_PER_M)
        rec.emit(
            tool="llm", args={"model": model},
            result=text, tokens_in=in_tok, tokens_out=out_tok,
            cost_usd=cost,
        )
        messages.append({"role": "assistant", "content": text})

        m_tool = re.search(r"TOOL:\s*(\w+)\s*(\{.*\})", text, re.DOTALL)
        if m_tool:
            name, raw_args = m_tool.group(1), m_tool.group(2)
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"raw": raw_args}
            fn = TOOLS.get(name)
            if fn is None:
                result = json.dumps({"error": f"unknown tool {name}"})
            else:
                result = fn(*args.values())
            rec.emit(tool=name, args=args, result=result)
            messages.append({"role": "user", "content": result})
            continue

        m_ans = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
        if m_ans:
            final_answer = m_ans.group(1).strip()
            break
        # Neither tool nor answer — push a nudge and continue
        messages.append({"role": "user", "content": "Please respond with TOOL: or ANSWER: only."})

    rec.finish(final_answer)
    return rec.trajectory().model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        env_file = Path.home() / ".hermes/.env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found (env or ~/.hermes/.env)")

    suite = json.loads((OUT_DIR / "suite_trajectory.json").read_text())
    questions = {t["id"]: t["prompt"] for t in suite["tests"]}

    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(headers=headers) as client:
        run_a: dict[str, dict] = {}
        run_b: dict[str, dict] = {}
        for tid, prompt in questions.items():
            print(f"  {tid} ...", flush=True)
            run_a[tid] = run_one(client, args.model, prompt, SYSTEM_CLEAN)
            run_b[tid] = run_one(client, args.model, prompt, SYSTEM_LOOPY)

    (OUT_DIR / "trajectories_run_a.json").write_text(
        json.dumps(run_a, indent=2), encoding="utf-8")
    (OUT_DIR / "trajectories_run_b.json").write_text(
        json.dumps(run_b, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'trajectories_run_a.json'} and "
          f"{OUT_DIR / 'trajectories_run_b.json'}")


if __name__ == "__main__":
    main()
