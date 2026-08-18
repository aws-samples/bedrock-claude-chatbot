"""Worker-agent swarm (agent-as-tool) for the Strands chat orchestrator.

Lets the main agent fan self-contained subtasks - typically parallel web research -
out to ephemeral worker agents, keeping its own context free of intermediate tool
noise (each worker's searches/fetches stay inside the worker; only its finished,
cited answer returns).

Topology:
    orchestrator (persisted chat agent)
      └─ tool: worker_agents(tasks: list[str])          [this module]
            ├─ worker 1 (own model, web tools, single-shot)
            ├─ worker 2 ...                              } ThreadPoolExecutor
            └─ worker N (bounded by _MAX_WORKERS)

Workers are stateless and never persisted: follow-ups re-delegate with fresh,
self-contained task briefs (the orchestrator owns continuity). Progress is reported
line-per-worker through the activity spinner - deliberately NOT a live text panel,
since N concurrent writers would interleave into noise.
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from strands import Agent, tool

from .chat_agent import resolve_model
from .web_tools import WEB_TOOLS

# Matches URLs in the workers' markdown answers - both bare and [text](url) links.
# Parens are allowed mid-URL (wiki-style paths); _trim_url balances them afterwards,
# so a [text](url) delimiter is dropped but a real trailing "...(x)" is kept.
_URL_RE = re.compile(r"https?://[^\s\]>\"'<]+")


def _trim_url(url: str) -> str:
    """Strip trailing punctuation and any unbalanced closing parens off a matched URL."""
    url = url.rstrip(".,;:!?")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:!?")
    return url


def _extract_citations(answers) -> List[str]:
    """Collect the unique source URLs cited across the workers' answers (in order).

    Extracted here - rather than asking each worker for structured output - so one
    regex covers every worker regardless of how it formats its citations. The list is
    returned under a top-level "citations" key, which the display reconstruction
    (sessions._urls_from_result_text) renders under the Sources attachment expander;
    any future tool can opt into the same rendering by returning that key.
    """
    seen = []
    for text in answers:
        for match in _URL_RE.findall(text):
            url = _trim_url(match)
            if url and url not in seen:
                seen.append(url)
    return seen

# Hard cap on concurrent workers per call: swarm cost/rate-limit pressure scales
# linearly with fan-out, so an over-eager orchestrator cannot spawn 20 workers.
# Default; overridable via config.json "worker-agent-max-workers" (see
# make_worker_agents_tool max_workers param).
_MAX_WORKERS = 6

_WORKER_TIMEOUT_SECONDS = 480  # a hung worker must not stall the whole turn

_WORKER_SYSTEM_PROMPT = """You are a diligent research worker agent executing ONE \
self-contained task delegated by an orchestrator. You have no other context - the task \
brief is everything you know, so follow it exactly.

TOOLS:
- tavily_search: fast web search returning snippets and source URLs. Use it for \
breadth - overviews, finding authoritative sources, recent news/updates.
- web_fetch: fetch one URL and extract its full text. Use it for depth when snippets \
are insufficient - complete articles, documentation pages, detailed data.

METHOD:
- If the task is answerable from your own knowledge with high confidence and does not \
need current information, answer directly without tools.
- Otherwise: search first for sources, then fetch the most promising pages for detail. \
Cross-reference multiple sources on anything contentious or fast-moving.
- Stay strictly on-task; do not expand scope beyond the brief.

ANSWER:
- Return a clear, well-structured markdown answer with IN-LINE CITATIONS (source URLs) \
for every factual claim from the web - uncited research is not credible and will be \
discarded. Make it self-contained: the reader will not see your searches or sources \
list unless you include them."""


def _run_worker(task: str, model_id: str, region: str, reasoning: bool) -> str:
    """One ephemeral, single-shot worker. Raises on failure; the pool catches."""
    agent = Agent(
        model=resolve_model(model_id, region, reasoning=reasoning, max_tokens=8000),
        agent_id="worker",
        system_prompt=_WORKER_SYSTEM_PROMPT,
        tools=list(WEB_TOOLS),
        callback_handler=None,  # workers never stream to the UI
    )
    return str(agent(task))


def make_worker_agents_tool(*, model_id: str, region: str, reasoning: bool = False,
                            max_workers: int = _MAX_WORKERS,
                            status_fn: Optional[Callable[[str], None]] = None):
    """Build the worker_agents tool the orchestrator calls.

    model_id: Bedrock model for every worker (from config `worker-agent-model`) -
        deliberately fixed, not randomized, for reproducible behavior and cost.
    max_workers: per-call fan-out cap (config `worker-agent-max-workers`); the tool
        docstring the orchestrator reads is formatted with this value, so the model
        is told the real limit.
    status_fn: pushes a multi-line progress block (one line per worker) into the
        chat bubble's activity spinner as workers finish.
    """
    max_workers = max(1, int(max_workers))

    def _progress(states: dict):
        if status_fn:
            lines = ["🧵 Worker agents"]
            for label, done in states.items():
                lines.append(f"{'✅' if done else '⏳'} {label}")
            status_fn("\n".join(lines))

    def worker_agents(tasks: List[str]) -> dict:
        tasks = [t for t in (tasks or []) if isinstance(t, str) and t.strip()]
        if not tasks:
            return {"error": "No tasks provided."}
        dropped = tasks[max_workers:]
        tasks = tasks[:max_workers]

        labels = {t: (t[:60] + "…" if len(t) > 60 else t) for t in tasks}
        states = {labels[t]: False for t in tasks}
        _progress(states)

        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_run_worker, t, model_id, region, reasoning): t
                       for t in tasks}
            try:
                # overall deadline lives on as_completed: one hung worker must not
                # stall the turn forever (result() is instant once a future yields)
                for future in as_completed(futures, timeout=_WORKER_TIMEOUT_SECONDS):
                    task = futures[future]
                    try:
                        results[task] = future.result()
                    except Exception as e:
                        results[task] = f"(worker failed: {e})"
                    states[labels[task]] = True
                    _progress(states)
            except TimeoutError:
                for future, task in futures.items():
                    if task not in results:
                        future.cancel()
                        results[task] = f"(worker timed out after {_WORKER_TIMEOUT_SECONDS}s)"
                        states[labels[task]] = True
                _progress(states)

        if dropped:
            results["_note"] = (f"{len(dropped)} task(s) beyond the {max_workers}-worker "
                                "limit were not run: " + "; ".join(d[:80] for d in dropped)
                                + ". If their results matter for the answer, call this "
                                "tool again with the remaining tasks.")
        results["citations"] = _extract_citations(results.values())
        return results

    # Docstring is assigned dynamically so the fan-out cap the orchestrator reads
    # matches the configured max_workers; @tool is applied AFTER so the generated
    # tool spec picks the formatted docstring up.
    worker_agents.__doc__ = f"""Delegate self-contained subtasks to parallel worker agents (with web search
        and page-fetch tools) and collect their finished, cited answers.

        Use this to offload work that would clutter the conversation - especially
        research that parallelizes well: comparing multiple options, gathering facts on
        several entities, investigating independent questions at once. Also useful for
        a single deep-dive you want done off to the side. Prefer answering directly for
        trivial lookups.

        Each task brief must be FULLY SELF-CONTAINED and directive: the worker sees
        nothing but its brief (no conversation history), so spell out the complete
        task, all relevant context, and the expected output format.

        Args:
            tasks: One task brief per worker, each comprehensive and self-contained.
                Maximum {max_workers} per call - consolidate or prioritize if you have more.

        Returns:
            A dict mapping each task brief to that worker's markdown answer (with
            source citations), or an error note if the worker failed.
        """
    return tool(worker_agents)
