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

st.set_page_config(page_title="evalforge — Eval Dashboard", page_icon="⚡", layout="wide")

CSS = Path(__file__).resolve().parent / "style.css"
if CSS.exists():
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


def load_default_runs() -> dict[str, RunResult]:
    """Load the built-in v1/v2 regression sample."""
    runs = {}
    for f in ("sample_results_v1.json", "sample_results_v2.json"):
        p = EXAMPLES / f
        if p.exists():
            runs[f.replace("sample_results_", "").replace(".json", "")] = load_run(str(p))
    return runs


def load_uploaded(uploaded) -> RunResult | None:
    """Load an uploaded JSON file as a run."""
    try:
        data = json.loads(uploaded.getvalue())
        return RunResult.model_validate(data)
    except Exception as e:
        st.error(f"Could not parse {uploaded.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def status_color(status: str) -> str:
    return {"pass": "#16A34A", "fail": "#DC2626", "error": "#D97706"}.get(status, "#64748B")


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
        total_tokens = sum((t.tokens.total or 0) for t in run.tests)
        cols[3].write(f"**tokens** {total_tokens:,}")
        if run.tests:
            st.markdown(
                f"Total input tokens: {sum((t.tokens.input or 0) for t in run.tests):,} · "
                f"output: {sum((t.tokens.output or 0) for t in run.tests):,}"
            )


def run_to_df(run: RunResult) -> pd.DataFrame:
    rows = []
    for t in run.tests:
        rows.append({
            "case": t.id,
            "status": t.status,
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
st.title("evalforge — LLM Evaluation Dashboard")
st.markdown(
    "Scores model outputs against test suites. Load a run to see pass rates, "
    "cost, latency, and error analysis — or compare **two runs** to catch regressions."
)

# --- Sidebar ---
with st.sidebar:
    st.title("Controls")
    source = st.radio("Data source", ["Sample (v1 vs v2)", "Upload"], index=0)

    runs: dict[str, RunResult] = {}
    if source == "Sample (v1 vs v2)":
        runs = load_default_runs()
        if not runs:
            st.error("Sample data not found. Run examples/gen_samples.py first.")
    else:
        up = st.file_uploader("Upload result JSON", type=["json"], accept_multiple_files=True)
        if up:
            for f in up:
                r = load_uploaded(f)
                if r:
                    runs[r.suite_name] = r

    metric = st.selectbox("Metric vocabulary", metric_names(), index=0)
    st.caption(RAGAS_METRICS[metric]["description"])

    st.divider()
    st.caption("evalforge — eval-driven agent testing")

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
    st.info(
        f"**{v1} → {v2}:** pass rate {r1.summary.pass_rate:.0%} → "
        f"{r2.summary.pass_rate:.0%} ({direction} by {abs(delta):.0%}). "
        f"See the matrix below for which cases moved."
    )

# --- Single run view ---
if len(runs) == 1:
    name = names[0]
    run = runs[name]
    st.subheader(name)
    render_metrics(run, "single")
    render_percentiles(run, "single")

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
            if match is None:
                row[name] = "—"
            else:
                row[name] = match.status
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

    # Per-run details in tabs
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        with tab:
            run = runs[name]
            render_metrics(run, f"tab_{name}")
            render_percentiles(run, f"tab_{name}")
            df = run_to_df(run)
            st.dataframe(
                df.style.map(style_status, subset=["status"]),
                use_container_width=True,
                hide_index=True,
            )
