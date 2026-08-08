#!/usr/bin/env python3
"""Generate evalforge sample results: v1 (23/24 pass) vs v2 (20/24 pass).

Story: model v2 regressed on 3 tool-selection cases that v1 passed.
Only t05, t13, t20 flip from pass to fail (the regression trio).
All other cases stay identical so the diff is easy to read.
"""
import json
from pathlib import Path

OUT = Path("/home/rayyan/projects/evalforge/examples")

BASE = [
    # (id, response, expected, dim, reasoning, latency, tokens, cost)
    ("t01", "The tool call is routed to the calculator with arguments a=12, b=7.", "calculator with a=12, b=7", "tool_selection", "Correct tool and args.", 1204, (412, 38), 0.0008),
    ("t02", "No tool needed. Direct answer provided from knowledge.", "direct answer", "tool_selection", "Correctly avoided unnecessary tool.", 890, (380, 45), 0.0007),
    ("t03", "Calling search with query=current_weather_hk.", "search with query=current_weather_hk", "tool_selection", "Search is the correct tool.", 1502, (421, 33), 0.0008),
    ("t04", "The document says the project deadline is May 30.", "deadline May 30", "faithfulness", "Supported by context.", 1310, (500, 42), 0.0009),
    ("t05", "Routed to code_runner with language=python, code=def f(): return 42.", "code_runner", "tool_selection", "Code execution requires code_runner.", 2410, (445, 51), 0.0010),
    ("t06", "I will use the calendar tool to check availability.", "calendar check", "tool_selection", "Calendar is appropriate.", 950, (398, 37), 0.0007),
    ("t07", "The weather in HK is 28C with light rain.", "28C light rain", "faithfulness", "Matches retrieved weather data.", 1620, (510, 40), 0.0009),
    ("t08", "The email should go to the sales team per the policy doc.", "sales team", "answer_correctness", "Matches policy.", 1180, (466, 39), 0.0008),
    ("t09", "Yes, the refund policy allows 30 days.", "30 days refund", "answer_correctness", "Accurate.", 1005, (455, 36), 0.0008),
    ("t10", "Routing to database tool to fetch user 123 profile.", "database fetch", "tool_selection", "DB query correct.", 1740, (432, 41), 0.0008),
    ("t11", "The shipment is scheduled for Friday.", "Friday", "answer_correctness", "Correct.", 870, (390, 32), 0.0007),
    ("t12", "No, the API key must not be logged.", "do not log", "answer_correctness", "Matches security policy.", 1120, (444, 38), 0.0008),
    ("t13", "Calling search for population of Indonesia.", "search", "tool_selection", "Search correct.", 1330, (405, 30), 0.0007),
    ("t14", "The exchange rate is 7.85 HKD per USD.", "7.85", "faithfulness", "Matches source.", 1440, (478, 35), 0.0009),
    ("t15", "Use the notify tool to alert the user.", "notify", "tool_selection", "Notify is correct.", 890, (412, 29), 0.0007),
    ("t16", "The meeting moved to 3 PM.", "3 PM", "answer_correctness", "Correct.", 910, (388, 31), 0.0007),
    ("t17", "I cannot find this in the documents, so I will not guess.", "decline gracefully", "answer_correctness", "Honest decline is correct.", 990, (430, 34), 0.0008),
    ("t18", "Routing to calculator: 156 * 4 = 624.", "624", "tool_selection", "Correct.", 1210, (421, 39), 0.0008),
    ("t19", "The policy allows 2 concurrent sessions.", "2 sessions", "answer_correctness", "Matches.", 880, (399, 33), 0.0007),
    ("t20", "Use the translate tool to convert to Cantonese.", "translate", "tool_selection", "Translate correct.", 1010, (415, 30), 0.0008),
    ("t21", "The flight departs at 14:30.", "14:30", "faithfulness", "Matches flight data.", 1090, (420, 28), 0.0008),
    ("t22", "The vault is unlocked via the security tool.", "security tool", "tool_selection", "Correct.", 1130, (411, 34), 0.0008),
    ("t23", "The battery life is 18 hours per the spec sheet.", "18 hours", "answer_correctness", "Accurate.", 970, (408, 32), 0.0008),
    ("t24", "Confirmed: the store closes at 22:00.", "22:00", "faithfulness", "Matches.", 850, (392, 29), 0.0007),
]

# The regression trio: v1 passes these, v2 fails them
REGRESSION = {"t05", "t13", "t20"}


def make_tests(version: int) -> list[dict]:
    tests = []
    for tid, response, expected, dim, reasoning, latency, (tin, tout), cost in BASE:
        status = "pass"
        score = {"overall": 1.0, "dimensions": [{"name": dim, "score": 5, "reasoning": reasoning}], "method": "rubric"}
        error = None
        if version == 2 and tid in REGRESSION:
            status = "fail"
            score = {
                "overall": 0.2,
                "dimensions": [
                    {"name": dim, "score": 1, "reasoning": f"v2 misrouted this case. Expected {expected}, got a different tool."}
                ],
                "method": "rubric",
            }
            response = "I will use the general assistant to answer this instead of a specialized tool."
            error = "tool_selection mismatch"
        tests.append({
            "id": tid,
            "status": status,
            "response": response,
            "expected_value": expected,
            "score": score,
            "tokens": {"input": tin, "output": tout, "total": tin + tout},
            "latency_ms": latency,
            "cost_usd": cost,
            "error": error,
        })
    return tests


def make_summary(tests: list[dict]) -> dict:
    passed = sum(1 for t in tests if t["status"] == "pass")
    failed = sum(1 for t in tests if t["status"] == "fail")
    errored = sum(1 for t in tests if t["status"] == "error")
    total = len(tests)
    latencies = sorted(t["latency_ms"] for t in tests)
    cost = sum(t["cost_usd"] for t in tests)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "pass_rate": round(passed / total, 3),
        "total_cost_usd": round(cost, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "latency_p50": latencies[len(latencies) // 2],
        "latency_p95": latencies[int(len(latencies) * 0.95) - 1],
        "latency_p99": latencies[-1],
    }


def write(name: str, version: int):
    tests = make_tests(version)
    result = {
        "suite_name": f"agent_tool_use_v{version}",
        "timestamp": f"2026-08-0{version}T09:30:00Z",
        "duration_ms": 48210,
        "tests": tests,
        "summary": make_summary(tests),
    }
    path = OUT / name
    path.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(f"{name}: {s['passed']}/{s['total']} pass ({s['pass_rate']:.1%}), "
          f"failed={s['failed']}, errored={s['errored']}, cost=${s['total_cost_usd']}, "
          f"p95={s['latency_p95']}ms")


write("sample_results_v1.json", 1)
write("sample_results_v2.json", 2)
print("DONE")
