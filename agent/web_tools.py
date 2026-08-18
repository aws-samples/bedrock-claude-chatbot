"""Web research tools for the Strands chat agent.

Two on-demand tools (enabled via the sidebar "Tools" dropdown):
  - tavily_search: web/news/finance search via the Tavily API
  - web_fetch:     fetch a single URL and extract its main text (JS-free)

The fetch tool is deliberately lightweight (requests + trafilatura) - no headless
browser - so it runs in SageMaker Studio without a Chrome/chromedriver stack. It
returns server-rendered text and will not execute client-side JavaScript; for
JS-heavy pages, prefer tavily_search (which returns cleaned page content).
"""
import json
import os
from typing import Literal, Optional

import requests
import trafilatura
from strands import tool

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.json")

# Secrets come from the environment (see .env.example). A repo-root .env is loaded
# here as a convenience so it works for any entrypoint (app, tests, notebooks);
# load_dotenv never overrides variables already exported in the shell.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except ImportError:  # optional dependency; exported env vars still work without it
    pass

# Cap the text handed back to the model so a large page cannot blow the context
# window (or, under DynamoDB session storage, the persisted message size).
_MAX_FETCH_CHARS = 50_000
_FETCH_TIMEOUT = 15
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _tavily_api_key() -> Optional[str]:
    """TAVILY_API_KEY from the environment (populated from .env if present).
    A config.json `tavily-api-key` is honored as a legacy fallback but should not
    be used - config files get committed; .env is gitignored."""
    key = os.environ.get("TAVILY_API_KEY")
    if key:
        return key
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f).get("tavily-api-key") or None
    except (OSError, json.JSONDecodeError):
        return None


@tool
def tavily_search(
    query: str,
    topic: Literal["general", "news", "finance"] = "general",
    search_depth: Literal["basic", "advanced"] = "basic",
    max_results: int = 5,
    time_range: Optional[Literal["day", "week", "month", "year"]] = None,
    days: int = 7,
    include_raw_content: bool = False,
) -> dict | str:
    """Search the web for current information using the Tavily API.

    Use this when a question needs up-to-date facts, news, prices, or anything
    beyond the model's training cutoff, or to find source URLs to read with web_fetch.

    Args:
        query: The search query.
        topic: Search category. 'news' for real-time updates, 'finance' for markets,
            'general' otherwise.
        search_depth: 'advanced' retrieves more relevant sources; 'basic' is faster.
        max_results: Number of results to return (1-20).
        time_range: Restrict results to the last day/week/month/year.
        days: For topic='news', how many days back to include (>= 1).
        include_raw_content: If true, include the cleaned page text of each result
            (useful to avoid a separate web_fetch call).

    Returns:
        A list of result dicts (title, url, content, ...), or an error string.
    """
    api_key = _tavily_api_key()
    if not api_key:
        return ("Tavily API key not configured. Set the TAVILY_API_KEY environment "
                "variable or add 'tavily-api-key' to config.json.")
    if not 1 <= max_results <= 20:
        return "max_results must be between 1 and 20"
    if days < 1:
        return "days must be 1 or greater"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        params = {
            "query": query,
            "topic": topic,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
        }
        if time_range:
            params["time_range"] = time_range
        if topic == "news":
            params["days"] = days
        response = client.search(**params)
        return response.get("results", response)
    except Exception as e:  # network / auth / quota errors surface to the model
        return f"Tavily search failed: {e}"


@tool
def web_fetch(url: str) -> str:
    """Fetch a single web page and return its main text content.

    Downloads the page over HTTP and extracts the primary article/text, stripping
    navigation, ads and boilerplate. Does NOT run JavaScript, so content rendered
    client-side (some SPAs) may be missing - use tavily_search for those.

    Args:
        url: The absolute URL to fetch (must start with http:// or https://).

    Returns:
        The extracted page text (truncated if very long), or an error string.
    """
    if not url.lower().startswith(("http://", "https://")):
        return f"Invalid URL (must start with http:// or https://): {url}"
    try:
        resp = requests.get(url, headers=_FETCH_HEADERS, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {e}"

    text = trafilatura.extract(
        resp.text, url=url, include_comments=False, include_tables=True,
    )
    if not text:
        # fall back to raw text if extraction found no main content
        text = resp.text
    text = text.strip()
    if len(text) > _MAX_FETCH_CHARS:
        text = text[:_MAX_FETCH_CHARS] + "\n\n[... content truncated ...]"
    return f"URL: {url}\n\n{text}"


# Tools exposed by name so the sidebar/agent wiring can select them on demand.
WEB_TOOLS = [tavily_search, web_fetch]
