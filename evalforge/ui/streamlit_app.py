"""evalforge dashboard — Streamlit UI for exploring evaluation results.

Features:
- Load 1-2 result JSON files (sample regression story by default)
- Summary metric cards: pass rate, cases, avg latency, total cost
- Per-case results table with status highlighting
- Side-by-side matrix view when 2 runs are loaded (v1 vs v2)
- Cost/latency percentiles + token usage (p50/p95/p99)
- RAGAS-style metric vocabulary in the scorer selector
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evalforge.models.result import RunResult
from evalforge.ui.metrics import RAGAS_METRICS, metric_names
from evalforge.ui.trajectory_display import (
    per_tool_df,
    regression_df,
    steps_df,
    trajectory_metrics_df,
    verdict_badge_html,
)

st.set_page_config(page_title="evalforge — Eval Dashboard", page_icon="⚡", layout="wide")

# Fonts (Inter body / Space Grotesk display / JetBrains Mono data) + design CSS
CSS = Path(__file__).resolve().parent / "style.css"
if CSS.exists():
    st.markdown(
        """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">""",
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_run(path: str) -> RunResult:
    """Load and validate a RunResult JSON."""
    data = json.loads(Path(path).read_text())
    return RunResult.model_validate(data)


SAMPLE_PAIRS: dict[str, tuple[str, str]] = {
    "Sample: same score, more tools": ("geo_baseline.json", "geo_more_tools.json"),
    "Sample: pass rate regressed": ("geo_baseline.json", "geo_no_tool.json"),
    "Sample: both regressed": ("geo_baseline.json", "geo_memory_loopy.json"),
}


def load_sample_pair(filenames: tuple[str, str]) -> dict[str, RunResult]:
    """Load a built-in comparison pair as real RunResult files.

    Uses the exact same load_run path as uploads, so the sample view is
    identical to what a user sees after uploading two `evalforge run`
    outputs (answers + trajectories, one file per run).
    """
    runs: dict[str, RunResult] = {}
    for f in filenames:
        p = EXAMPLES / f
        if not p.exists():
            st.warning(
                f"Sample file {f} not found. Run examples/gen_realistic_samples.py first."
            )
            continue
        r = load_run(str(p))
        runs[r.suite_name] = r
    return runs


def load_uploaded_trajectory(uploaded) -> RunResult | None:
    """Load an uploaded trajectory file (single or per-test map) as a run."""
    from evalforge.trajectory.importers import (
        load_trajectories_file,
        load_trajectory_json,
    )
    from evalforge.trajectory.metrics import summarize_trajectories
    from evalforge.models import TestResult, build_summary_from_tests

    try:
        raw = json.loads(uploaded.getvalue())
        is_map = (
            isinstance(raw, dict)
            and bool(raw)
            and all(isinstance(v, dict) and "steps" in v for v in raw.values())
        )
        if is_map:
            trajs = load_trajectories_file(
                _write_temp(uploaded, raw))
            tests = [
                TestResult(id=tid, status="pass", response=t.final_answer, trajectory=t)
                for tid, t in trajs.items()
            ]
        else:
            traj = load_trajectory_json(_write_temp(uploaded, raw))
            tests = [
                TestResult(id="imported", status="pass",
                           response=traj.final_answer, trajectory=traj)
            ]
        report = RunResult(
            suite_name=uploaded.name.replace(".json", ""),
            timestamp="upload",
            duration_ms=0.0,
            tests=tests,
            summary=build_summary_from_tests(tests),
        )
        report.trajectory_summary = summarize_trajectories(tests)
        return report
    except Exception as e:
        st.error(f"Could not parse trajectory {uploaded.name}: {e}")
        return None


def _write_temp(uploaded, raw) -> str:
    """Materialize an uploaded trajectory to a temp file for the importer."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(raw, f)
        return f.name


def load_uploaded(uploaded) -> RunResult | None:
    """Load an uploaded RunResult JSON. Returns None (silently) when the file
    isn't a RunResult — the caller falls back to the trajectory loader."""
    try:
        data = json.loads(uploaded.getvalue())
        return RunResult.model_validate(data)
    except Exception:
        # Not a RunResult — likely a trajectory file; caller handles it.
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def status_color(status: str) -> str:
    s = status or ""
    if "pass" in s:
        return "#16A34A"
    if "fail" in s:
        return "#DC2626"
    if "error" in s:
        return "#D97706"
    return "#64748B"


STATUS_ICON = {"pass": "✅", "fail": "❌", "error": "⚠️"}


def status_text(status: str) -> str:
    if not status:
        return "—"
    return f"{STATUS_ICON.get(status, '•')} {status}"


def render_metrics(run: RunResult, key_prefix: str):
    s = run.summary
    c1, c2, c3, c4 = st.columns(4)
    delta = None
    c1.metric("Pass rate", f"{s.pass_rate:.0%}", delta)
    c2.metric("Cases", f"{s.total}", f"{s.failed} failed")
    c3.metric("Avg latency", f"{s.avg_latency_ms:.0f}ms")
    c4.metric("Total cost", f"${s.total_cost_usd:.3f}")


def render_percentiles(run: RunResult, key_prefix: str):
    s = run.summary
    with st.expander("Latency & token detail"):
        cols = st.columns(4)
        cols[0].write(f"**p50** {s.latency_p50:.0f}ms" if s.latency_p50 else "**p50** n/a")
        cols[1].write(f"**p95** {s.latency_p95:.0f}ms" if s.latency_p95 else "**p95** n/a")
        cols[2].write(f"**p99** {s.latency_p99:.0f}ms" if s.latency_p99 else "**p99** n/a")
        total_tokens = sum((t.tokens.total or 0) for t in run.tests if t.tokens)
        cols[3].write(f"**tokens** {total_tokens:,}")
        if run.tests:
            st.markdown(
                f"Total input tokens: {sum((t.tokens.input or 0) for t in run.tests if t.tokens):,} · "
                f"output: {sum((t.tokens.output or 0) for t in run.tests if t.tokens):,}"
            )


def render_trajectory_report(run: RunResult):
    """Trajectory report card: process metrics, per-tool rollup, step timeline."""
    if not run.trajectory_summary:
        return

    st.subheader("Process metrics (trajectory)")
    mdf = trajectory_metrics_df(run.trajectory_summary)
    cols = st.columns(4)
    for i in range(min(4, len(mdf))):
        with cols[i]:
            st.metric(str(mdf["metric"].iloc[i]), str(mdf["value"].iloc[i]))

    st.markdown("**Per-tool rollup**")
    tdf = per_tool_df(run.trajectory_summary)
    if not tdf.empty:
        st.dataframe(tdf, use_container_width=True, hide_index=True)

    with st.expander("Step timeline — pick a test to inspect the journey"):
        traj_tests = [t for t in run.tests if t.trajectory is not None]
        if traj_tests:
            options = {t.id: t for t in traj_tests}
            selected = st.selectbox(
                "Test", list(options.keys()), key=f"timeline_{run.suite_name}"
            )
            sdf = steps_df(options[selected].trajectory)
            st.dataframe(sdf, use_container_width=True, hide_index=True)
        else:
            st.markdown("No trajectory data on individual tests.")


def run_to_df(run: RunResult) -> pd.DataFrame:
    rows = []
    for t in run.tests:
        rows.append({
            "case": t.id,
            "status": status_text(t.status),
            "score": round(t.score.overall, 2) if t.score else None,
            "latency_ms": t.latency_ms,
            "cost_usd": t.cost_usd,
            "method": t.score.method if t.score else "",
            "error": t.error or "",
        })
    return pd.DataFrame(rows)


def style_status(val: str) -> str:
    color = status_color(val)
    return f"color: {color}; font-weight: 600;"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="hero">
  <span class="eyebrow">Eval-driven agent testing</span>
  <h1 class="hero-title">The report card for your AI agents</h1>
  <p class="hero-sub">Score model outputs against test suites — pass rates, cost,
  latency, and regressions. Load one run, or compare two to catch what broke.</p>
</div>""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("Controls")
    source = st.radio(
        "Data source",
        [*SAMPLE_PAIRS.keys(), "Upload"],
        index=0,
    )

    runs: dict[str, RunResult] = {}
    if source in SAMPLE_PAIRS:
        runs = load_sample_pair(SAMPLE_PAIRS[source])
        if not runs:
            st.error(
                "Sample data not found. Run examples/gen_realistic_samples.py first."
            )
    else:
        up = st.file_uploader(
            "Upload result JSON or trajectory JSON",
            type=["json"],
            accept_multiple_files=True,
        )
        if up:
            for f in up:
                r = load_uploaded(f)
                if r is None:
                    r = load_uploaded_trajectory(f)
                if r:
                    runs[r.suite_name] = r

    metric = st.selectbox("Metric vocabulary", metric_names(), index=0)
    st.caption(RAGAS_METRICS[metric]["description"])

    st.divider()
    st.caption("evalforge — eval-driven agent testing · v0.2.0")

if not runs:
    st.warning("No runs loaded. Use the sample data or upload a result JSON.")
    st.stop()

names = list(runs.keys())

# --- Header story ---
if len(runs) >= 2:
    v1, v2 = names[0], names[1]
    r1, r2 = runs[v1], runs[v2]
    delta = r2.summary.pass_rate - r1.summary.pass_rate
    direction = "regressed" if delta < 0 else "improved"
    tone = "danger" if delta < 0 else "success"
    st.markdown(
        f"""<div class="callout {tone}">
  <span class="callout-label">{v1} → {v2}</span>
  <span>Pass rate <strong>{r1.summary.pass_rate:.0%} → {r2.summary.pass_rate:.0%}</strong> —
  {direction} by {abs(delta):.0%}. See the matrix below for which cases moved.</span>
</div>""",
        unsafe_allow_html=True,
    )

# --- Single run view ---
if len(runs) == 1:
    name = names[0]
    run = runs[name]
    st.subheader(name)
    render_metrics(run, "single")
    render_percentiles(run, "single")
    render_trajectory_report(run)

    st.subheader("Per-case results")
    df = run_to_df(run)
    st.dataframe(
        df.style.map(style_status, subset=["status"]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Failed cases — error analysis"):
        fails = [t for t in run.tests if t.status != "pass"]
        if not fails:
            st.markdown("No failures. Clean run.")
        for t in fails:
            st.markdown(
                f"**{t.id}** ({t.status}) — {t.error or 'no error message'}"
            )
            if t.score and t.score.dimensions:
                for d in t.score.dimensions:
                    st.markdown(f"- {d.name}: {d.score}/5 — {d.reasoning or 'no reasoning'}")

# --- Matrix view (2+ runs) ---
else:
    st.subheader("Side-by-side comparison")
    col_names = st.columns(len(names))
    col_rates = st.columns(len(names))
    for i, name in enumerate(names):
        run = runs[name]
        s = run.summary
        col_rates[i].metric(
            f"{name} pass rate",
            f"{s.pass_rate:.0%}",
            f"{s.failed} failed · {s.total} cases",
        )

    # Per-case matrix
    all_ids = sorted({t.id for run in runs.values() for t in run.tests})
    matrix_rows = []
    for tid in all_ids:
        row = {"case": tid}
        for name in names:
            run = runs[name]
            match = next((t for t in run.tests if t.id == tid), None)
            row[name] = status_text(match.status) if match else "—"
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)

    st.markdown("**Per-case status across runs** — a pass→fail flip is a regression.")
    st.dataframe(
        matrix_df.style.map(style_status, subset=names),
        use_container_width=True,
        hide_index=True,
    )

    # Regression list
    st.markdown("**Regressions / fixes**")
    for tid in all_ids:
        statuses = []
        for name in names:
            run = runs[name]
            match = next((t for t in run.tests if t.id == tid), None)
            statuses.append(match.status if match else None)
        if len(statuses) >= 2 and statuses[0] == "pass" and statuses[-1] == "fail":
            st.markdown(f"- 🔴 **{tid}** regressed: passed in {names[0]}, failed in {names[-1]}")
        elif len(statuses) >= 2 and statuses[0] == "fail" and statuses[-1] == "pass":
            st.markdown(f"- 🟢 **{tid}** fixed: failed in {names[0]}, passed in {names[-1]}")

    # Trajectory regression (process quality) — first two runs with data
    traj_names = [n for n in names if runs[n].trajectory_summary]
    if len(traj_names) >= 2:
        from evalforge.trajectory.regression import compare_trajectories

        rep = compare_trajectories(runs[traj_names[0]], runs[traj_names[1]])
        st.markdown("---")
        st.subheader("Trajectory regression (process quality)")
        st.markdown(
            f"Comparing **{traj_names[0]}** → **{traj_names[1]}**: "
            f"same answers may hide worse journeys — more repeated calls, "
            f"more steps, more cost."
        )
        verdict = rep.get("verdict")
        if verdict:
            st.markdown(f"Verdict: {verdict_badge_html(verdict)}", unsafe_allow_html=True)
        rdf = regression_df(rep)
        if not rdf.empty:
            st.dataframe(rdf, use_container_width=True, hide_index=True)
        # Per-test step deltas
        ptd = rep.get("per_test") or {}
        if ptd:
            rows = []
            for tid, d in ptd.items():
                rows.append({
                    "test": tid,
                    "steps_delta": d.get("steps_delta", ""),
                    "repeats_delta": d.get("repeated_calls_delta", ""),
                    "cost_delta": d.get("cost_delta", ""),
                    "note": "appeared" if d.get("appeared") else ("disappeared" if d.get("disappeared") else ""),
                })
            st.markdown("**Per-test journey deltas**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-run details in tabs
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        with tab:
            run = runs[name]
            render_metrics(run, f"tab_{name}")
            render_percentiles(run, f"tab_{name}")
            render_trajectory_report(run)
            df = run_to_df(run)
            st.dataframe(
                df.style.map(style_status, subset=["status"]),
                use_container_width=True,
                hide_index=True,
            )

st.markdown(
    """<div class="footer">
  <span>evalforge v0.2.0 · MIT</span>
  <span><a href="https://github.com/rayyanakmal/evalforge">GitHub</a> · Streamlit</span>
</div>""",
    unsafe_allow_html=True,
)
