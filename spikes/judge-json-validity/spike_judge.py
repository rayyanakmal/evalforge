#!/usr/bin/env python3
"""
Spike: Validate DeepSeek v4 flash reliability for structured JSON scoring.

Steps:
  1. Define 10 varied "candidate responses" (simulating target LLM outputs).
  2. For each, call DeepSeek v4 flash as a judge, asking for accuracy/completeness/tone scores as JSON.
  3. Record JSON parse success, score ranges, cost.
  4. Output verdict.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

# DeepSeek v4 flash pricing (approximate):
PRICING = {
    "input_per_mtok": 0.07,
    "output_per_mtok": 0.28,
}

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluation judge. "
    "Return ONLY valid JSON with this exact structure (no markdown, no extra text):\n"
    '{"accuracy": <1-5>, "completeness": <1-5>, "tone": <1-5>, '
    '"overall": <1-5>, "reasoning": "<brief justification>"}'
)

TEST_CASES = [
    ("What is the capital of France?",
     "Paris is the capital and largest city of France, located on the Seine River in northern France."),
    ("What is the capital of France?",
     "Lyon is the capital of France. It's known for its gastronomy."),
    ("Explain how HTTP caching works.",
     "HTTP caching stores copies of responses to reduce server load. The Cache-Control header controls caching behavior."),
    ("Explain how HTTP caching works.",
     "HTTP caching is a mechanism where responses from a server are stored by the client or intermediary proxies to serve future requests faster. Key headers include Cache-Control (max-age, no-cache, no-store, must-revalidate), ETag (validation tokens), and Last-Modified dates. When a cached resource is still fresh (within max-age), the cache serves it directly. When stale, the client can validate with If-None-Match (ETag) or If-Modified-Since headers, and the server returns 304 Not Modified if unchanged."),
    ("Could you help me understand quantum computing?",
     "That's a dumb question. Quantum computing is obviously about using qubits. Read a book."),
    ("What's the weather like today?",
     "Oh magnificent seeker of atmospheric knowledge! How blessed am I to bask in the radiance of your weather-related inquiry! The celestial bodies have aligned to bring you... partly cloudy skies at 72°F."),
    ("Write a Python function to sort a list.",
     "def sort_list(lst): return sorted(lst)"),
    ("Who won the 2028 Super Bowl?",
     "The 2028 Super Bowl was won by the London Monarchs, who defeated the Tokyo Samurai 38-35 in a thrilling overtime game."),
    ("How do I implement binary search in Python?",
     "Here's a clean binary search implementation:\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
    ("What's the JSON format for a user profile?",
     'A user profile JSON looks like: {"name": "John", "age": 30, "email": "john@example.com"}. But you should also include an id field and handle nested objects for addresses.'),
]


def call_deepseek(messages, max_tokens=700, temperature=0.1):
    """Call DeepSeek API and return parsed response."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            latency = time.time() - start
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage", {})
            cost_usd = (
                usage.get("prompt_tokens", 0) / 1_000_000 * PRICING["input_per_mtok"]
                + usage.get("completion_tokens", 0) / 1_000_000 * PRICING["output_per_mtok"]
            )
            content = data["choices"][0]["message"].get("content", "")
            reasoning = data["choices"][0]["message"].get("reasoning_content", "")
            return {
                "content": content,
                "reasoning": reasoning,
                "usage": usage,
                "cost_usd": cost_usd,
                "latency_s": latency,
                "error": None,
            }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return {
            "content": None,
            "reasoning": "",
            "usage": {},
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": f"HTTP {e.code}: {body[:200]}",
        }
    except Exception as e:
        return {
            "content": None,
            "reasoning": "",
            "usage": {},
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": str(e),
        }


def try_parse_json(text):
    """Try to extract and parse JSON from response text."""
    if not text or not text.strip():
        return None, "empty response"
    text = text.strip()

    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip()), None
            except json.JSONDecodeError:
                pass
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip()), None
            except json.JSONDecodeError:
                pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1]), None
        except json.JSONDecodeError:
            pass

    return None, "no valid JSON found in: " + text[:120]


def validate_schema(obj):
    """Validate the parsed JSON has correct score fields."""
    if not isinstance(obj, dict):
        return False, "not a dict"
    required = ["accuracy", "completeness", "tone"]
    for key in required:
        if key not in obj:
            return False, f"missing field: {key}"
        val = obj[key]
        if not isinstance(val, (int, float)):
            return False, f"field {key} is not numeric: {type(val).__name__}"
        if val < 1 or val > 5:
            return False, f"field {key} out of range (1-5): {val}"
    if "overall" in obj:
        val = obj["overall"]
        if not isinstance(val, (int, float)):
            return False, f"overall not numeric: {type(val).__name__}"
        if val < 1 or val > 5:
            return False, f"overall out of range (1-5): {val}"
    return True, None


def main():
    if not API_KEY:
        print("FATAL: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("SPIKE: DeepSeek v4 Flash JSON Judge Reliability")
    print("=" * 70)
    print(f"Model:     {MODEL}")
    print(f"Cases:     {len(TEST_CASES)}")
    print(f"Max tokens: 700 (to allow room for reasoning + content)")
    print(f"Pricing:   ${PRICING['input_per_mtok']}/M in, ${PRICING['output_per_mtok']}/M out")
    print()

    results = []
    total_cost = 0.0
    total_latency = 0.0
    parse_ok = 0
    schema_ok = 0
    total_tok = 0
    total_reasoning_tok = 0

    for i, (question, response) in enumerate(TEST_CASES, 1):
        print(f"--- Test {i}/{len(TEST_CASES)} ---")
        print(f"  Q: {question[:70]}")
        print(f"  R: {response[:70].replace(chr(10), ' ')}")

        result = call_deepseek([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nResponse to evaluate:\n{response}"},
        ])

        cost = result["cost_usd"]
        lat = result["latency_s"]
        total_cost += cost
        total_latency += lat
        total_tok += result["usage"].get("total_tokens", 0)
        token_details = result["usage"].get("completion_tokens_details", {})
        total_reasoning_tok += token_details.get("reasoning_tokens", 0)

        parsed, parse_err = try_parse_json(result["content"])
        if parsed:
            parse_ok += 1
            valid, schema_err = validate_schema(parsed)
            if valid:
                schema_ok += 1
                print(f"  ✓ acc={parsed['accuracy']} comp={parsed['completeness']} tone={parsed['tone']} overall={parsed.get('overall','?')}")
            else:
                print(f"  ✗ Schema: {schema_err}")
                print(f"    Raw: {result['content'][:120]}")
        else:
            print(f"  ✗ Parse: {parse_err}")

        usage = result["usage"]
        rt = token_details.get("reasoning_tokens", 0)
        print(f"  Cost=${cost:.5f}  Lat={lat:.2f}s  Tokens={usage.get('total_tokens',0)}  (reasoning={rt})")
        print()

        results.append({
            "test": i,
            "parsed": parsed,
            "parse_ok": parsed is not None,
            "schema_valid": valid if parsed else False,
            "cost": cost,
            "latency": lat,
            "tokens": result["usage"],
            "reasoning_tokens": rt,
            "raw": result["content"],
        })

    print("=" * 70)
    print("SUMMARY STATS")
    print("=" * 70)
    parse_rate = parse_ok / len(TEST_CASES) * 100
    schema_rate = schema_ok / len(TEST_CASES) * 100

    print(f"  JSON Parse Success:      {parse_ok}/{len(TEST_CASES)} ({parse_rate:.0f}%)")
    print(f"  Schema Compliance:       {schema_ok}/{len(TEST_CASES)} ({schema_rate:.0f}%)")
    print(f"  Total Cost:              ${total_cost:.5f}")
    print(f"  Avg Cost per Eval:       ${total_cost / len(TEST_CASES):.5f}")
    print(f"  Total Tokens:            {total_tok}")
    print(f"  Total Reasoning Tokens:  {total_reasoning_tok}")
    print(f"  Reasoning % of output:   {total_reasoning_tok / max(total_tok - (total_tok - total_reasoning_tok), 1) * 100:.0f}%")
    print(f"  Avg Latency:             {total_latency / len(TEST_CASES):.2f}s")

    valid_scores = [r["parsed"] for r in results if r.get("schema_valid")]
    if valid_scores:
        print()
        print("Score Distributions:")
        for dim in ["accuracy", "completeness", "tone", "overall"]:
            vals = [s.get(dim) for s in valid_scores if s.get(dim) is not None]
            if vals:
                print(f"  {dim:15s}: mean={sum(vals)/len(vals):.2f}  range={min(vals)}-{max(vals)}  values={sorted(vals)}")

    failed = [r for r in results if not r["parse_ok"]]
    if failed:
        print()
        print(f"PARSE FAILURES ({len(failed)}):")
        for r in failed:
            print(f"  Test {r['test']}: raw='{r['raw'][:100] if r['raw'] else 'empty'}'")

    failed_schema = [r for r in results if r["parse_ok"] and not r["schema_valid"]]
    if failed_schema:
        print()
        print(f"SCHEMA FAILURES ({len(failed_schema)}):")
        for r in failed_schema:
            print(f"  Test {r['test']}: {r['parsed']}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Primary metric: did we get valid JSON with correct schema?
    if schema_rate >= 90:
        verdict = "VALIDATED"
        detail = "DeepSeek v4 flash reliably returns valid JSON for rubric scoring."
    elif schema_rate >= 70:
        verdict = "PARTIAL"
        detail = "Mostly reliable, but retry/fallback logic is advisable."
    elif schema_rate >= 40:
        verdict = "PARTIAL"
        detail = "Inconsistent. Retry with stricter prompting is essential."
    else:
        verdict = "INVALIDATED"
        detail = "DeepSeek v4 flash is NOT reliable for structured JSON scoring."

    print(f"  Verdict: {verdict}")
    print(f"  Detail:  {detail}")
    print(f"  Cost:    ${total_cost:.5f}")
    print()


if __name__ == "__main__":
    main()
