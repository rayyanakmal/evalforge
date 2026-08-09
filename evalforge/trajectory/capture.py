"""Capture layer: StepRecorder + record() context manager (D4 ladder rung 3).

The recorder is the "notebook" the evaluator writes into while driving an
agent. Users with a custom agent loop add ONE line per tool call site —
``emit(tool, args, result)`` — or use the timed ``call`` context manager.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from evalforge.models import TokenCount, Trajectory, TrajectoryStep


class StepRecorder:
    """Collects TrajectorySteps in order and builds a Trajectory at the end.

    Callable: ``recorder(tool, args, ...)`` is sugar for ``recorder.emit(...)``
    so ``with record() as emit: emit(tool, args)`` reads naturally.
    """

    def __init__(self) -> None:
        self._steps: list[TrajectoryStep] = []
        self._final_answer: Optional[str] = None

    def __call__(self, tool: str, args: Optional[dict] = None, **kwargs: Any) -> TrajectoryStep:
        return self.emit(tool, args, **kwargs)

    def emit(
        self,
        tool: str,
        args: Optional[dict] = None,
        result: Any = None,
        thought: Optional[str] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd: float = 0.0,
    ) -> TrajectoryStep:
        """Record one step. Auto-assigns the next index."""
        tokens = None
        if tokens_in is not None or tokens_out is not None:
            tokens = TokenCount(
                input=tokens_in or 0,
                output=tokens_out or 0,
                total=(tokens_in or 0) + (tokens_out or 0),
            )
        step = TrajectoryStep(
            index=len(self._steps),
            tool=tool,
            args=args or {},
            result=result,
            thought=thought,
            latency_ms=latency_ms,
            tokens=tokens,
            cost_usd=cost_usd,
            error=error,
        )
        self._steps.append(step)
        return step

    @contextmanager
    def call(self, tool: str, args: Optional[dict] = None) -> Iterator[TrajectoryStep]:
        """Timed call context: auto-records latency and exceptions.

        Usage::

            with recorder.call("search", {"q": "hk"}) as step:
                step.result = do_search(**step.args)
        """
        start = time.monotonic()
        step = TrajectoryStep(index=len(self._steps), tool=tool, args=args or {})
        self._steps.append(step)
        try:
            yield step
        except Exception as exc:  # noqa: BLE001 — we record, then re-raise
            step.error = f"{type(exc).__name__}: {exc}"
            step.latency_ms = (time.monotonic() - start) * 1000.0
            raise
        else:
            step.latency_ms = (time.monotonic() - start) * 1000.0

    def finish(self, final_answer: Optional[str]) -> None:
        """Record the agent's final answer (marks convergence)."""
        self._final_answer = final_answer

    def trajectory(self) -> Trajectory:
        """Build the immutable-at-rest Trajectory from collected steps."""
        return Trajectory(steps=list(self._steps), final_answer=self._final_answer)


@contextmanager
def record(recorder: Optional[StepRecorder] = None) -> Iterator[StepRecorder]:
    """Context manager exposing the emit function for custom agent loops.

    Usage::

        with evalforge.trajectory.record() as emit:
            emit("search", {"q": "hk"}, result="islands")
            with emit.call("calc", {"expr": "1+1"}) as step:
                step.result = "2"
        traj = emit.trajectory()

    Pass an existing ``StepRecorder`` to share it with other adapters.
    """
    rec = recorder or StepRecorder()
    yield rec
