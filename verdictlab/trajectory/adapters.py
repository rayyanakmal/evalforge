"""Capture adapters — the friction ladder (D4).

Rung 2: drop-in client shims — swap the import, every completion auto-emits
an ``llm`` step. Tested against fake clients (no real SDK on the Pi).
Rung 1: framework callback registration — thin integration points for
LangChain/LangGraph native traces. Tested against fake callback handlers;
real-framework validation happens on the Mac if ever needed.
"""

from __future__ import annotations

from typing import Any, Optional

from .capture import StepRecorder

# Rung 2: drop-in client shims ----------------------------------------------


class _ClientShim:
    """Wraps an SDK client; the overridden method emits llm steps.

    Attribute passthrough keeps the rest of the client's API intact.
    """

    def __init__(self, inner: Any, recorder: StepRecorder, method_name: str) -> None:
        self._inner = inner
        self._recorder = recorder
        self._method = method_name
        # Bind the instrumented method name to the shim itself so __getattr__
        # passthrough never shadows it (the shim is callable).
        setattr(self, method_name, self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, **request: Any) -> Any:
        response = getattr(self._inner, self._method)(**request)
        return self._emit_llm(request, response)

    def _emit_llm(self, request: dict, response: Any) -> Any:
        content = self._extract_content(response)
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None)
        tokens_out = getattr(usage, "completion_tokens", None)
        self._recorder.emit(
            tool="llm",
            args={"model": request.get("model"), "messages": request.get("messages")},
            result=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        return response

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            return getattr(choices[0].message, "content", "")
        content = getattr(response, "content", None)  # Anthropic shape
        if isinstance(content, list):
            return "".join(getattr(block, "text", "") for block in content)
        return content or ""


def wrap_openai(client: Any, recorder: Optional[StepRecorder] = None):
    """Wrap an OpenAI-style client; chat_completions_create emits llm steps.

    The shim exposes ``chat_completions_create(model=..., messages=...)``
    directly (plus any other attribute passthrough). A real OpenAI client
    exposes ``chat.completions.create`` — pass a partial or adapt the call::

        client = wrap_openai(OpenAI(api_key=...), recorder=rec)
        client.chat_completions_create(model=..., messages=...)
    """
    if recorder is None:
        raise ValueError("wrap_openai requires a recorder=StepRecorder()")
    return _ClientShim(client, recorder, "chat_completions_create")


def wrap_anthropic(client: Any, recorder: Optional[StepRecorder] = None):
    """Wrap an Anthropic-style client; messages_create emits llm steps."""
    if recorder is None:
        raise ValueError("wrap_anthropic requires a recorder=StepRecorder()")
    return _ClientShim(client, recorder, "messages_create")


# Rung 1: framework callback registration -------------------------------------


def register_langchain_callback(recorder: Optional[StepRecorder] = None):
    """Integration point for LangChain native traces (D4 rung 1).

    Real LangChain is NOT installed on the Pi; this returns a documented
    callback-handler stub that forwards tool events to the recorder. Wire it
    into ``CallbackHandler`` on the Mac for real framework use.
    """
    if recorder is None:
        raise ValueError("register_langchain_callback requires a recorder=StepRecorder()")
    return _FrameworkCallbackStub(recorder, "langchain")


def register_langgraph_callback(recorder: Optional[StepRecorder] = None):
    """Integration point for LangGraph native traces (D4 rung 1)."""
    if recorder is None:
        raise ValueError("register_langgraph_callback requires a recorder=StepRecorder()")
    return _FrameworkCallbackStub(recorder, "langgraph")


class _FrameworkCallbackStub:
    """Forwards tool events to a recorder; mirrors real callback interfaces.

    The method names follow LangChain's BaseCallbackHandler shape
    (on_tool_start / on_tool_end / on_tool_error) so a real adapter only
    needs to translate framework events into these calls.
    """

    def __init__(self, recorder: StepRecorder, framework: str) -> None:
        self._recorder = recorder
        self.framework = framework

    def on_tool_start(self, tool: str, args: Optional[dict] = None, **_: Any) -> None:
        self._recorder.emit(tool=tool, args=args or {})

    def on_tool_end(self, result: Any, **_: Any) -> None:
        # result lands on the most recent open step
        steps = self._recorder.trajectory().steps
        if steps:
            steps[-1].result = result

    def on_tool_error(self, error: Exception, **_: Any) -> None:
        steps = self._recorder.trajectory().steps
        if steps:
            steps[-1].error = f"{type(error).__name__}: {error}"
