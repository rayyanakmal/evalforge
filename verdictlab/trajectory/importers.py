"""Trajectory importers — the A fallback (D3).

Sealed-box agents with no hooks export their trace; verdictlab grades it.
Accepted formats:
1. Our own Trajectory JSON ({"steps": [...], "final_answer": ...})
2. OTel-style span JSON ({"spans": [{"name": "tool.search", ...}]})
3. Minimal JSONL (one {"tool", "args", "result"} line per step)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from verdictlab.models import TokenCount, Trajectory, TrajectoryStep


def load_trajectory_json(path: str) -> Trajectory:
    """Load a trajectory file in any supported format into a Trajectory."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"trajectory file not found: {path}")

    if p.suffix == ".jsonl":
        return _from_jsonl(p)

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"trajectory file must contain a JSON object, got {type(raw).__name__}")

    if "steps" in raw:
        steps = [_coerce_step(s, i) for i, s in enumerate(raw["steps"])]
        return Trajectory(steps=steps, final_answer=raw.get("final_answer"))
    if "spans" in raw:
        return _from_otel_spans(raw["spans"], raw.get("final_answer"))
    raise ValueError(
        f"unrecognized trajectory format in {path}: expected 'steps' or 'spans' "
        f"keys, got {sorted(raw.keys())}"
    )


def load_trajectories_file(path: str, required_ids: Optional[list[str]] = None) -> dict[str, Trajectory]:
    """Load a per-test-id map of trajectories (run-level import).

    Format: {"test_id": {"steps": [...], "final_answer": ...}, ...}
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"trajectories file not found: {path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"trajectories file must be a JSON object keyed by test id, got {type(raw).__name__}")

    missing = [tid for tid in (required_ids or []) if tid not in raw]
    if missing:
        raise ValueError(f"trajectories file missing required test ids: {missing}")

    result: dict[str, Trajectory] = {}
    for tid, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"trajectory for test '{tid}' must be an object, got {type(payload).__name__}")
        steps = [_coerce_step(s, i) for i, s in enumerate(payload.get("steps", []))]
        result[tid] = Trajectory(steps=steps, final_answer=payload.get("final_answer"))
    return result


def _from_jsonl(p: Path) -> Trajectory:
    steps = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {lineno + 1} of {p}: {exc}") from exc
        steps.append(_coerce_step(obj, len(steps)))
    return Trajectory(steps=steps)


def _from_otel_spans(spans: list[dict], final_answer: Optional[str]) -> Trajectory:
    steps = []
    for i, span in enumerate(spans):
        name = span.get("name", "")
        tool = name.split(".")[-1] if name else "unknown"
        attrs = span.get("attributes") or {}
        error = span.get("error") or (None if span.get("status", "ok") == "ok" else "error")
        steps.append(TrajectoryStep(
            index=i,
            tool=tool,
            args=attrs.get("args") or {},
            result=span.get("result"),
            latency_ms=float(span.get("duration_ms") or span.get("latency_ms") or 0.0),
            error=error,
        ))
    return Trajectory(steps=steps, final_answer=final_answer)


def _coerce_step(obj: dict, index: int) -> TrajectoryStep:
    if not isinstance(obj, dict):
        raise ValueError(f"step at index {index} must be an object, got {type(obj).__name__}")
    if "index" in obj and obj.get("index") != index:
        # keep the original index; Trajectory validator will catch ordering
        index = obj["index"]
    tokens = None
    tok = obj.get("tokens")
    if isinstance(tok, dict):
        tokens = TokenCount(
            input=tok.get("input", 0),
            output=tok.get("output", 0),
            total=tok.get("total") or (tok.get("input", 0) + tok.get("output", 0)),
        )
    return TrajectoryStep(
        index=index,
        tool=str(obj.get("tool", "unknown")),
        args=obj.get("args") or {},
        result=obj.get("result"),
        thought=obj.get("thought"),
        latency_ms=float(obj.get("latency_ms") or obj.get("duration_ms") or 0.0),
        tokens=tokens,
        cost_usd=float(obj.get("cost_usd") or 0.0),
        error=obj.get("error"),
    )
