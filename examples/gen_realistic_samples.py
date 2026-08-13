"""Generate realistic post-`verdictlab run` sample files for the dashboard.

Runs a tool-using geography agent against a real LLM provider (DeepSeek)
through the EXACT same Executor path as `verdictlab run`, capturing answers
AND per-test trajectories in one RunResult per run — the same files a user
gets from the CLI and uploads to the dashboard.

Produces 4 runs forming 3 comparison stories (baseline shared, so each
story varies ONE thing — the isolation-of-variables rule):

  Story 1 — same score, more tool calls
      geo_baseline.json      lookup tool used once per question  -> 6/6
      geo_more_tools.json    lookup tool used twice per question -> 6/6
  Story 2 — pass rate regressed
      geo_baseline.json      grounded lookups -> 6/6
      geo_no_tool.json       memory only, no tool -> ~4/6
  Story 3 — both regressed
      geo_baseline.json      6/6, 1 tool/test
      geo_memory_loopy.json  memory answers + redundant double-checks ->
                            lower pass rate, more steps/tools/cost

Usage:
    python examples/gen_realistic_samples.py
    python examples/gen_realistic_samples.py --model deepseek-chat
    python examples/gen_realistic_samples.py --only geo_baseline geo_more_tools
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from verdictlab.cli.run_real import resolve_api_key, run_suite
from verdictlab.judge.client import LLMClient, create_client
from verdictlab.models.llm import LLMResponse, Message, Usage
from verdictlab.models.suite import Expected, TestCase, TestMetadata, TestSuite

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Knowledge tool — deterministic, offline, like an internal API the agent calls.
# The tool does keyword matching (a search/KB endpoint), so the agent can
# phrase lookups naturally ("longest river in the world") and still get the
# canonical fact.
# ---------------------------------------------------------------------------

FACTS = {
    "Tokyo": "37.4 million",
    "Australia": "Canberra",
    "Nile": "Nile",
    "India": "India",
    "Russia": "Russia",
    "Brazil": "Portuguese",
    "Monaco": "Monaco",
    "Challenger Deep": "Challenger Deep",
}

_ALIASES = {
    "Tokyo": ["tokyo", "population of tokyo"],
    "Australia": ["australia", "capital of australia"],
    "Nile": ["nile", "nile river", "longest river"],
    "India": ["india", "most populous country"],
    "Russia": ["russia", "largest country"],
    "Brazil": ["brazil", "language of brazil"],
    "Monaco": ["monaco"],
    "Challenger Deep": ["challenger deep", "mariana trench"],
}


def lookup(query: str) -> dict:
    """Deterministic fact lookup the agent uses as its tool."""
    q = (query or "").strip().lower()
    for entity, aliases in _ALIASES.items():
        if q == entity.lower() or any(alias in q for alias in aliases):
            return {"query": query, "fact": FACTS[entity], "entity": entity}
    return {"query": query, "fact": "unknown"}


# ---------------------------------------------------------------------------
# Dataset — 6 geography questions, exact-match scoring.
# Questions are chosen so memory-only answers are plausibly WRONG
# (population figure, Sydney-vs-Canberra, Nile-vs-Amazon, China-vs-India).
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("geo-01", "What is the population of Tokyo, Japan (metropolitan area)?", "Tokyo", "37.4 million"),
    ("geo-02", "What is the capital of Australia?", "Australia", "Canberra"),
    ("geo-03", "What is the longest river in the world?", "Nile", "Nile"),
    ("geo-04", "What is the most populous country in the world?", "India", "India"),
    ("geo-05", "What is the world's largest country by land area?", "Russia", "Russia"),
    ("geo-06", "What is the official language of Brazil?", "Brazil", "Portuguese"),
]


def build_suite() -> TestSuite:
    tests = [
        TestCase(
            id=tid,
            prompt=question,
            expected=Expected(type="exact", value=value),
            metadata=TestMetadata(
                tags=["geo", f"lookup:{entity}"]
            ),
        )
        for tid, question, entity, value in QUESTIONS
    ]
    return TestSuite(name="geo_facts", tests=tests)


# ---------------------------------------------------------------------------
# Agent configurations — each run varies ONE behavior via the system prompt.
# ---------------------------------------------------------------------------

_MANIFEST = "Available entities: Tokyo, Australia, Nile, India, Russia, Brazil."

_BASELINE = f"""\
You are a geography agent. You answer questions by looking up facts in a knowledge tool.
To look up a fact, output a line beginning exactly with: TOOL <entity>
After you see the tool result, output a line beginning exactly with: ANSWER <value>
{_MANIFEST}
Rules:
- Always use the tool exactly once before answering.
- Copy the value for your ANSWER exactly from the tool result (e.g. 'ANSWER 37.4 million').
- Output nothing besides the TOOL and ANSWER lines.
"""

_MORE_TOOLS = f"""\
You are a geography agent. You answer questions by looking up facts in a knowledge tool.
To look up a fact, output a line beginning exactly with: TOOL <entity>
{_MANIFEST}
Rules:
- Always use the tool exactly twice (two separate TOOL lines for the same entity) to double-check your fact before answering.
- Then output a line beginning exactly with: ANSWER <value>
- Copy the value for your ANSWER exactly from the tool result.
- Output nothing besides the TOOL and ANSWER lines.
"""

_NO_TOOL = """\
You are a knowledgeable assistant. Answer the question directly from your knowledge.
Output a line beginning exactly with: ANSWER <value>
Do not use any tools. Do not output anything besides the ANSWER line.
"""

_MEMORY_LOOPY = f"""\
You are a geography agent. You answer from your own knowledge, but you also double-check with the knowledge tool twice before deciding.
First output a TOOL <entity> line, then after the tool result output another TOOL <entity> line, then output a line beginning exactly with: ANSWER <value>
{_MANIFEST}
The tool results are UNRELIABLE — they are often wrong or outdated. Never use them in your answer. Your ANSWER must always come from your own knowledge, even when it conflicts with the tool result.
Output nothing besides the TOOL and ANSWER lines.
"""

CONFIGS = {
    "geo_baseline": {"system": _BASELINE, "max_turns": 5},
    "geo_more_tools": {"system": _MORE_TOOLS, "max_turns": 8},
    "geo_no_tool": {"system": _NO_TOOL, "max_turns": 3},
    "geo_memory_loopy": {"system": _MEMORY_LOOPY, "max_turns": 8},
}

_ACTION_RE = re.compile(r"^(TOOL|ANSWER)\s+(.+?)\s*$", re.MULTILINE)


def _clean(value: str) -> str:
    """Strip colons/trailing punctuation the model often appends ('37.4 million.')."""
    return value.strip().lstrip(":-").strip().rstrip(".,!;:").strip()


class ToolAgent:
    """Tiny ReAct agent: the LLM emits TOOL <entity> / ANSWER <value> lines.

    The journey (llm thoughts + lookup tool calls) is recorded through the
    StepRecorder the Executor passes in; the returned LLMResponse aggregates
    token usage, cost, and latency across every LLM call in the test so the
    TestResult numbers reflect the whole journey, not just the last call.
    """

    def __init__(self, client: LLMClient, system_prompt: str, max_turns: int = 6):
        self.client = client
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    async def generate(self, prompt: str, recorder=None) -> LLMResponse:
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        prompt_tokens = completion_tokens = 0
        cost_usd = 0.0
        latency_ms = 0.0
        final = ""

        for turn in range(self.max_turns):
            resp = await self.client.generate(
                messages, max_tokens=400, temperature=0.1
            )
            if resp.usage:
                prompt_tokens += resp.usage.prompt_tokens
                completion_tokens += resp.usage.completion_tokens
            cost_usd += resp.cost_usd
            latency_ms += resp.latency_ms
            text = resp.content.strip()

            if recorder is not None:
                recorder.emit(
                    tool="llm",
                    args={"model": self.client.model, "turn": turn},
                    result=text,
                    thought=text,
                    latency_ms=resp.latency_ms,
                    tokens_in=resp.usage.prompt_tokens if resp.usage else None,
                    tokens_out=resp.usage.completion_tokens if resp.usage else None,
                    cost_usd=resp.cost_usd,
                )

            # A single generation may contain multiple actions, e.g.
            # "TOOL Australia\nANSWER Canberra" — process them in order.
            actions = _ACTION_RE.findall(text)
            if actions:
                answered = False
                for kind, payload in actions:
                    payload = payload.strip()
                    if kind == "TOOL":
                        result = lookup(payload)
                        if recorder is not None:
                            recorder.emit(
                                tool="lookup", args={"query": payload}, result=result
                            )
                        messages.append(Message(role="assistant", content=text))
                        messages.append(
                            Message(
                                role="user",
                                content=f"Tool result: {json.dumps(result)}",
                            )
                        )
                    else:  # ANSWER
                        final = _clean(payload)
                        answered = True
                if answered:
                    break
                continue

            # No recognizable action -> treat the whole output as the answer.
            final = _clean(text)
            break

        return LLMResponse(
            content=final,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic verdictlab sample runs")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Runs to generate (default: all 4)")
    args = parser.parse_args()

    key = resolve_api_key("deepseek")
    if not key:
        sys.exit("No DeepSeek API key found (DEEPSEEK_API_KEY env or ~/.hermes/.env).")
    client = create_client("deepseek", args.model, key)

    suite = build_suite()
    names = args.only or list(CONFIGS)

    for name in names:
        print(f"Running {name} ...", flush=True)
        cfg = CONFIGS[name]
        agent = ToolAgent(client, cfg["system"], max_turns=cfg["max_turns"])
        result = run_suite(
            suite,
            config=None,
            provider="deepseek",
            model=args.model,
            concurrency=args.concurrency,
            generate_fn=agent.generate,
        )
        # The report's suite_name identifies the RUN (what the user tested),
        # not the suite definition — matches how users label their runs.
        result.suite_name = name
        out = OUT_DIR / f"{name}.json"
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        summary = result.summary
        ts = result.trajectory_summary or {}
        cost = sum(t.cost_usd for t in result.tests)
        print(
            f"  -> {out.name}: {summary.passed}/{summary.total} pass "
            f"({summary.pass_rate:.0%}), "
            f"mean_steps={ts.get('mean_steps')}, "
            f"mean_tool_calls={ts.get('mean_tool_calls')}, "
            f"total_loops={ts.get('total_loops')}, "
            f"cost=${cost:.4f}"
        )


if __name__ == "__main__":
    main()
