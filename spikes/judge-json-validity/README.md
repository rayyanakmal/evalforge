# Spike: DeepSeek v4 Flash JSON Judge Reliability

**Date:** 2026-07-06
**Model:** `deepseek-v4-flash`
**Total Cost:** $0.00109

---

## Verdict: VALIDATED ✅

DeepSeek v4 flash **reliably returns valid structured JSON** for LLM-as-Judge rubric scoring — **100% success rate** across 10 diverse test cases.

---

## What Worked

| Metric | Value |
|--------|-------|
| JSON Parse Success | **100%** (10/10) |
| Schema Compliance | **100%** (10/10) — all responses had `accuracy`, `completeness`, `tone`, `overall`, `reasoning` |
| Avg Cost per Eval | **$0.00011** (~482 total tokens avg) |
| Avg Latency | **0.17s** |
| Output Token Overhead | ~60% used for reasoning, ~40% for actual JSON content |

- The judge correctly penalized **wrong answers** (acc=1), **rude tone** (tone=1), **hallucinations** (acc=1), and **incomplete responses** (comp=1).
- The judge correctly rewarded **accurate, comprehensive, well-toned responses** (4-5s).
- Scoring range was well-distributed across 1–5 for all dimensions.
- The judge **always included reasoning** alongside scores.

## What Didn't Work

- **With `max_tokens=300`**: Only 40% success rate. The model's reasoning consumes most output tokens, leaving the JSON truncated.
- **With `max_tokens=500`**: 90% success rate. One truncation edge case for longer reasoning.
- **With `max_tokens=700`**: 100% success rate. **Recommend this as the minimum.**

| max_tokens | Success Rate | Failure Mode |
|------------|-------------|--------------|
| 300        | 40%         | Truncated mid-JSON (ran out of tokens during reasoning) |
| 500        | 90%         | One truncation on long reasoning case |
| 700        | 100%        | No failures |

## Surprises

1. **`deepseek-v4-flash` is a reasoning model** — it emits `reasoning_content` in every response, separate from `content`. This consumes ~60% of the output token budget. The `content` field holds the final JSON answer.
2. **Extremely cheap** — ~$0.0001/eval. At 10,000 evals/month, that's ~$1.10 total.
3. **No markdown wrapping** — unlike some models, DeepSeek returned raw JSON without `` ```json `` fences, which simplified parsing.
4. **Test 10 (JSON-in-response)** — the response being evaluated contained a JSON example itself (`{"name": "John", ...}`), but the judge still returned clean scoring JSON without confusion.

## Recommendations for Real Build

1. **Set `max_tokens ≥ 700`** for the judge LLM call. The model needs headroom for reasoning + JSON output.
2. **Parse `content`, not `reasoning_content`** — the final answer is always in `content`.
3. **Implement fallback retry** for robustness: if JSON parse fails → retry once with `"IMPORTANT: Return ONLY valid JSON"` prepended to the prompt. (Not needed in our tests, but wise for production.)
4. **No need for a separate JSON extraction step** — the model reliably returns raw JSON without markdown fences.
5. **Cost tracking is accurate** — token counts (prompt, completion, reasoning) are all provided in the API response.
6. **Consider caching** for identical judge prompts (37% of input tokens were cached in some calls).

## Score Consistency Notes

The judge showed coherent scoring:
| Dimension | Mean | Range | Notes |
|-----------|------|-------|-------|
| accuracy  | 3.7  | 1–5   | Correctly 1 for wrong answers, 5 for correct |
| completeness | 2.7 | 1–5  | Harsh on short answers (comp=1–2), generous on detailed (comp=4–5) |
| tone      | 3.5  | 1–5   | Correctly 1 for rude, 2 for sycophantic, 4–5 for professional |
| overall   | 2.9  | 1–5   | Reasonable aggregate |

The judge was **appropriately strict** on tone and accuracy, which is good for an eval framework.

---

## Appendix: Raw Tabular Results

| # | Question (abbrev) | acc | comp | tone | overall | Cost ($) | Lat (s) |
|---|-------------------|-----|------|------|---------|----------|---------|
| 1 | Capital of France (correct) | 5 | 5 | 5 | 5 | 0.00005 | 0.18 |
| 2 | Capital of France (wrong) | 1 | 1 | 3 | 1 | 0.00011 | 0.17 |
| 3 | HTTP caching (short) | 5 | 2 | 4 | 3 | 0.00012 | 0.16 |
| 4 | HTTP caching (detailed) | 5 | 4 | 5 | 4 | 0.00008 | 0.16 |
| 5 | Quantum computing (rude) | 2 | 1 | 1 | 1 | 0.00016 | 0.18 |
| 6 | Weather (sycophantic) | 4 | 4 | 2 | 3 | 0.00011 | 0.20 |
| 7 | Sort function (minimal) | 5 | 4 | 4 | 4 | 0.00014 | 0.17 |
| 8 | 2028 Super Bowl (hallucinated) | 1 | 1 | 3 | 1 | 0.00008 | 0.17 |
| 9 | Binary search (code) | 5 | 2 | 4 | 3 | 0.00011 | 0.18 |
| 10 | User profile JSON | 4 | 3 | 4 | 4 | 0.00012 | 0.17 |

---

*Test script: `spikes/judge-json-validity/spike_judge.py`*
