"""Strands-based chat agent for the Bedrock chatbot Streamlit app.

Replaces the hand-rolled converse_stream loop (bedrock_claude_/bedrock_streemer/
_invoke_bedrock_with_retries) for the plain-chat path. Conversation history is
owned by the Strands session manager (local/S3/DynamoDB, see agent/sessions.py);
the model-visible window is bounded by SlidingWindowConversationManager.

Model routing is registry-driven (model_id.json): each display name maps to a spec
whose "engine" selects the Strands provider -
  runtime          -> BedrockModel (Converse API)
  mantle-responses -> OpenAIResponsesModel against the Bedrock Mantle OpenAI endpoint
  mantle-chat      -> OpenAIModel (Chat Completions) against Bedrock Mantle
and whose "reasoning" field selects the reasoning-parameter dialect
("adaptive" = Claude adaptive thinking, "config" = runtime reasoning_config,
"effort" = OpenAI-style effort).
"""
import json
import os
import re
import threading
from typing import Callable, Optional

from botocore.config import Config

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:  # allow use outside a Streamlit runtime (tests, notebooks)
    add_script_run_ctx = get_script_run_ctx = None

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel
from strands.models.model import CacheConfig

from .sessions import build_session_manager

# read_timeout is the max silent gap between bytes on the stream. Streaming responses
# emit events continuously (reasoning deltas included), so a long silence means the
# stream has stalled server-side - fail in minutes, not the urllib3-default forever.
# NOTE: mid-stream read timeouts are NOT retried by botocore (retries cover the
# initial call only); they surface as ReadTimeoutError and are handled in the UI.
BOTO_CONFIG = Config(read_timeout=300, retries={"max_attempts": 10, "mode": "adaptive"})

_ROOT = os.path.dirname(os.path.dirname(__file__))

# Model registry: display name -> spec (id, engine, profile, reasoning, vision, tools).
with open(os.path.join(_ROOT, "model_id.json"), encoding="utf-8") as _f:
    MODEL_REGISTRY = json.load(_f)

try:
    with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as _f:
        _APP_CONFIG = json.load(_f)
except (OSError, json.JSONDecodeError):
    _APP_CONFIG = {}

# Reasoning knobs (config.json overrides; effort tier selects the token budgets).
REASONING_EFFORT = _APP_CONFIG.get("reasoning-effort", "medium")
REASONING_MAX_TOKENS = {"low": 15000, "medium": 25000, "high": 35000,
                        **_APP_CONFIG.get("reasoning-max-tokens", {})}
# thinking budget per effort tier, for "budget"-dialect models (haiku-4.5)
REASONING_BUDGET_TOKENS = {"low": 5000, "medium": 10000, "high": 20000,
                           **_APP_CONFIG.get("reasoning-budget-tokens", {})}


def _reasoning_max_tokens() -> int:
    return REASONING_MAX_TOKENS.get(REASONING_EFFORT, REASONING_MAX_TOKENS["medium"])


def model_spec(name: str) -> dict:
    """Spec for a display name; raises a clear error for unknown models."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; expected one of {list(MODEL_REGISTRY)}") from None


# Human-readable, per-tool activity lines shown while a tool runs. Each entry maps a
# tool name to (spinner label, the input field to surface once it streams in).
_TOOL_ACTIVITY = {
    "web_fetch": ("🌐 Fetching", "url"),
    "tavily_search": ("🔍 Searching the web", "query"),
    "data_analysis": ("🔬 Analyzing data", None),
    "worker_agents": ("🧵 Delegating to worker agents", None),
}


def _fmt_activity(text: str) -> str:
    """Italicize an activity message line-by-line (markdown emphasis cannot span
    newlines), so multi-line progress blocks - e.g. one line per research worker -
    render cleanly in the same placeholder as single-line spinners."""
    return "  \n".join(f"_{ln}_" if ln.strip() else "" for ln in text.split("\n"))


def _partial_json_field(partial: str, field: str) -> str:
    """Best-effort extraction of one string field from an incomplete JSON fragment.

    Tool input streams in as partial JSON (e.g. '{"url": "https://exa'), so a full
    json.loads() usually fails mid-stream. Try a full parse first, then fall back to
    a regex that grabs the value-so-far."""
    try:
        return json.loads(partial).get(field, "") or ""
    except (json.JSONDecodeError, AttributeError):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)', partial)
        return m.group(1) if m else ""


class StreamlitCallbackHandler:
    """Streams model text, reasoning, and tool activity into a Streamlit placeholder.

    Mirrors the old bedrock_streemer rendering: reasoning is shown under a
    '**MODEL REASONING**' header while it streams, then the answer text replaces it.
    While a tool runs, a live activity line (e.g. '🌐 Fetching https://...') is shown
    in the same placeholder and is overwritten as soon as answer text starts arriving.
    """

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
        self.thinking = ""
        self._activity = ""       # current tool-activity line (transient, not persisted)
        self._tool_active = False  # a tool call is in flight (activity line wins render)
        self._tool_use_id = None   # id of the in-flight tool call, to detect a new one
        self._closed = False       # set after the turn; blocks late worker-thread writes
        # Strands streams from a worker thread; Streamlit UI writes need the
        # script-run context, so capture it here (main thread) and attach later.
        self._ctx = get_script_run_ctx() if get_script_run_ctx else None

    def close(self):
        """Mark the turn finished. Any straggler event from a worker thread after this
        is dropped instead of writing to the placeholder - a delta delivered after the
        script run ends desyncs the frontend's running indicator (spinner never stops)."""
        self._closed = True

    def _render(self):
        """Redraw the placeholder from current state.

        While a tool runs, the activity line wins so the bubble reads e.g.
        '🌐 Fetching https://...'. Otherwise the answer text wins over reasoning; the
        answer only ever reflects text produced AFTER the last tool call, so a model's
        pre-tool preamble ('let me search...') never merges into the final bubble.
        """
        if self._tool_active and self._activity:
            self.placeholder.markdown(_fmt_activity(self._activity), unsafe_allow_html=True)
        elif self.text:
            self.placeholder.markdown(self.text.replace("$", "\\$"), unsafe_allow_html=True)
        elif self.thinking:
            self.placeholder.markdown(
                "**MODEL REASONING**\n\n" + self.thinking.replace("$", "\\$"),
                unsafe_allow_html=True,
            )
        elif self._activity:
            self.placeholder.markdown(_fmt_activity(self._activity), unsafe_allow_html=True)

    def status(self, message: str):
        """Push a transient progress line from inside a tool (e.g. the data-analysis
        sub-agent) into the same placeholder. Runs on the worker thread, so re-attach
        the script-run context like __call__ does."""
        if self._closed:
            return
        if self._ctx is not None and get_script_run_ctx() is None:
            add_script_run_ctx(threading.current_thread(), self._ctx)
        self._tool_active = True
        self._activity = message
        self._render()

    def __call__(self, **kwargs):
        if self._closed:
            return
        if self._ctx is not None and get_script_run_ctx() is None:
            add_script_run_ctx(threading.current_thread(), self._ctx)
        reasoning = kwargs.get("reasoningText")
        data = kwargs.get("data", "")
        current_tool_use = kwargs.get("current_tool_use")

        if current_tool_use:
            tool_use_id = current_tool_use.get("toolUseId")
            # A new tool call is starting: drop any preamble text/reasoning emitted
            # before it so only the post-tool answer surfaces in the final bubble.
            if tool_use_id != self._tool_use_id:
                self._tool_use_id = tool_use_id
                self.text = ""
                self.thinking = ""
            self._tool_active = True
            name = current_tool_use.get("name", "")
            label, field = _TOOL_ACTIVITY.get(name, (f"🔧 Using {name}", None))
            value = _partial_json_field(current_tool_use.get("input", ""), field) if field else ""
            self._activity = f"{label} {value}…" if value else f"{label}…"
        if reasoning:
            self.thinking += reasoning
        if data:
            self.text += data
        # model output (answer text or post-tool reasoning) arriving without a tool
        # event means the round-trip is over: retire the spinner so it stops winning.
        if (reasoning or data) and not current_tool_use:
            self._tool_active = False
        if reasoning or data or current_tool_use:
            self._render()


class SubAgentWorkingsPanel:
    """Live '🧬 Agent workings' panel for a SINGLE sub-agent (e.g. data_analysis).

    Doubles as the sub-agent's Strands callback handler: streams its reasoning/text
    into an st.status container materialized lazily inside a placeholder slot (created
    on the main thread, under the answer spinner), expanded while the tool runs and
    collapsed on completion. Deliberately not used for parallel workers - N concurrent
    writers would interleave into noise; workers report line-per-worker progress via
    the activity spinner instead.
    """

    _TAIL_CHARS = 8000  # cap the re-rendered transcript so per-token redraws stay cheap

    def __init__(self, slot, label: str = "🔬 Agent workings"):
        self.slot = slot          # st.empty() created on the main thread, in the bubble
        self.label = label
        self._status = None
        self._body = None
        self._buffer = ""
        self._tool_use_id = None
        self._ctx = get_script_run_ctx() if get_script_run_ctx else None

    def _attach_ctx(self):
        if self._ctx is not None and get_script_run_ctx() is None:
            add_script_run_ctx(threading.current_thread(), self._ctx)

    def begin(self):
        """Open (or reopen) the panel for a sub-agent run."""
        self._attach_ctx()
        self._buffer = ""
        self._tool_use_id = None
        self._status = self.slot.status(f"{self.label}…", expanded=True)
        self._body = self._status.empty()

    def __call__(self, **kwargs):
        if self._status is None:
            return
        self._attach_ctx()
        reasoning = kwargs.get("reasoningText")
        data = kwargs.get("data", "")
        current_tool_use = kwargs.get("current_tool_use")
        if current_tool_use:
            tool_use_id = current_tool_use.get("toolUseId")
            if tool_use_id != self._tool_use_id:  # transcript marker per code execution
                self._tool_use_id = tool_use_id
                self._buffer += "\n\n⚙️ *writing & executing code…*\n\n"
        if reasoning:
            self._buffer += reasoning
        if data:
            self._buffer += data
        if reasoning or data or current_tool_use:
            tail = self._buffer[-self._TAIL_CHARS:]
            if len(self._buffer) > self._TAIL_CHARS:
                tail = "…" + tail
            self._body.markdown(tail.replace("$", "\\$"), unsafe_allow_html=True)

    def complete(self, error: bool = False):
        if self._status is None:
            return
        self._attach_ctx()
        self._status.update(label=self.label, expanded=False,
                            state="error" if error else "complete")


def _build_runtime_model(spec: dict, region: str, reasoning: bool, max_tokens: int) -> BedrockModel:
    """BedrockModel (Converse API) for engine 'runtime'."""
    profile = spec.get("profile") or ""
    model_id = f"{profile}.{spec['id']}" if profile else spec["id"]
    config = {
        "model_id": model_id,
        "region_name": region,
        "boto_client_config": BOTO_CONFIG,
        # auto prompt caching: detects model support and injects cachePoints across the
        # system prompt and message history (replaces the deprecated cache_prompt flag)
        "cache_config": CacheConfig(strategy="auto"),
    }
    if reasoning:
        # reasoning dialect per registry: current Claude models take adaptive
        # thinking + effort; haiku-4.5 predates adaptive and needs a fixed thinking
        # budget; other runtime models (deepseek/kimi) take reasoning_config.
        # Effort tier and token budgets come from config.json (reasoning-effort,
        # reasoning-max-tokens, reasoning-budget-tokens).
        dialect = spec.get("reasoning")
        if dialect == "adaptive":
            thinking_fields = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": REASONING_EFFORT},
            }
        elif dialect == "budget":
            budget = REASONING_BUDGET_TOKENS.get(REASONING_EFFORT,
                                                 REASONING_BUDGET_TOKENS["medium"])
            thinking_fields = {"thinking": {"type": "enabled", "budget_tokens": budget}}
        else:
            thinking_fields = {"reasoning_config": REASONING_EFFORT}
        # reasoning models: temperature must stay unset, generous output budget
        config.update({
            "max_tokens": _reasoning_max_tokens(),
            "additional_request_fields": thinking_fields,
        })
    else:
        config.update({"max_tokens": max_tokens})
    return BedrockModel(**config)


def _build_mantle_model(spec: dict, region: str, reasoning: bool, max_tokens: int):
    """OpenAI-compatible providers for Bedrock Mantle engines.

    mantle-responses -> OpenAIResponsesModel (gpt-* and gemma-4 models)
    mantle-chat      -> OpenAIModel / Chat Completions (glm-5)
    bedrock_mantle_config mints a fresh SigV4 bearer token per request (no manual
    provide_token plumbing, no token-expiry failures on long sessions).
    """
    if spec["engine"] == "mantle-responses":
        from aws_bedrock_token_generator import provide_token
        from strands.models.openai_responses import OpenAIResponsesModel
        # explicit /openai/v1 base URL: ALL responses-API models live there, but
        # strands' bedrock_mantle_config path heuristic only routes openai.gpt-5.*
        # ids to it - gemma would land on /v1, where the responses stream stalls
        # after 2 events (validated live). Token minted per model construction,
        # which is per turn here, so expiry is a non-issue.
        return OpenAIResponsesModel(
            client_args={
                "api_key": provide_token(),
                "base_url": f"https://bedrock-mantle.{region}.api.aws/openai/v1",
            },
            model_id=spec["id"],
            params={
                "max_output_tokens": _reasoning_max_tokens() if reasoning else max_tokens,
                "reasoning": {"effort": REASONING_EFFORT if reasoning else "none"},
                "store": False,  # no server-side conversation storage (sessions own history)
            },
            stateful=False,
        )
    from strands.models.openai import OpenAIModel
    params = {
        "max_tokens": _reasoning_max_tokens() if reasoning else max_tokens,
        "store": False,
    }
    if reasoning:
        params["reasoning_effort"] = REASONING_EFFORT
    return OpenAIModel(bedrock_mantle_config={"region": region}, model_id=spec["id"],
                       params=params)


def build_model(model_name: str, region: str, reasoning: bool = False,
                max_tokens: int = 4000):
    """Strands model provider for a registry display name, routed by engine."""
    spec = model_spec(model_name)
    if spec["engine"] == "runtime":
        return _build_runtime_model(spec, region, reasoning, max_tokens)
    if spec["engine"] in ("mantle-responses", "mantle-chat"):
        return _build_mantle_model(spec, region, reasoning, max_tokens)
    raise ValueError(f"Unknown engine {spec['engine']!r} for model {model_name!r}")


def build_bedrock_model(model_id: str, region: str, reasoning: bool = False,
                        max_tokens: int = 4000) -> BedrockModel:
    """Back-compat shim for callers holding a raw Bedrock model id.

    Wraps the id in a minimal runtime spec; reasoning dialect inferred: Claude ids
    use adaptive thinking, anything else the generic reasoning_config.
    """
    spec = {"id": model_id, "engine": "runtime", "profile": "",
            "reasoning": "adaptive" if "claude" in model_id else "config"}
    return _build_runtime_model(spec, region, reasoning, max_tokens)


def resolve_model(name_or_id: str, region: str, reasoning: bool = False,
                  max_tokens: int = 4000):
    """Model provider from either a registry display name ("gpt-sol") or a raw
    Bedrock model id ("us.anthropic.claude-..."). The single entry point for the
    orchestrator and all sub-agents."""
    if name_or_id in MODEL_REGISTRY:
        return build_model(name_or_id, region, reasoning=reasoning, max_tokens=max_tokens)
    return build_bedrock_model(name_or_id, region, reasoning=reasoning, max_tokens=max_tokens)


def build_chat_agent(*, model_id: str, region: str, session_id: str,
                     session_storage: str, system_prompt: str,
                     history_window: int = 10, reasoning: bool = False,
                     max_tokens: int = 4000, user_id: str = "", bucket: str = "",
                     dynamodb_table: str = "", agentcore_memory_id: str = "",
                     tools: Optional[list] = None,
                     callback_handler: Optional[Callable] = None) -> Agent:
    """Create a chat Agent bound to a persisted session.

    The session manager restores prior messages for session_id on construction,
    so a new Agent per Streamlit rerun resumes the conversation transparently.

    tools: optional list of Strands @tool callables to enable for this turn
        (e.g. web search/fetch, selected in the sidebar).
    """
    session_manager = build_session_manager(
        session_storage, session_id,
        user_id=user_id, bucket=bucket, region=region, dynamodb_table=dynamodb_table,
        agentcore_memory_id=agentcore_memory_id,
    )
    # model_id is a registry display name ("gpt-sol") when present there, else a raw
    # Bedrock model id (back-compat for direct callers/tests).
    model = resolve_model(model_id, region, reasoning=reasoning, max_tokens=max_tokens)
    return Agent(
        model=model,
        agent_id="chat",
        system_prompt=system_prompt,
        session_manager=session_manager,
        conversation_manager=SlidingWindowConversationManager(window_size=history_window),
        callback_handler=callback_handler,
        tools=tools or [],
    )
