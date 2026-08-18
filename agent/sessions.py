"""Session storage backends for the Strands chat agent.

Three interchangeable backends, selected via the "session-storage" key in config.json:
  - "local":    FileSessionManager under .strands_sessions/
  - "s3":       S3SessionManager under s3://<Bucket_Name>/<prefix>
  - "dynamodb": RepositorySessionManager backed by DynamoDBSessionRepository

Each backend independently holds the entire conversation history (Strands session,
agent state and messages).
"""
import json
import os
import re
import time
from decimal import Decimal
from typing import Any, Optional

import boto3
from boto3.dynamodb.conditions import Key

from strands.session.file_session_manager import FileSessionManager
from strands.session.s3_session_manager import S3SessionManager
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.content import Message
from strands.types.session import Session, SessionAgent, SessionMessage

LOCAL_SESSION_DIR = ".strands_sessions"
S3_SESSION_PREFIX = "strands-sessions"
# oversized DynamoDB messages (large docs / base64 images) spill to S3 under this prefix
DDB_OVERFLOW_PREFIX = f"{S3_SESSION_PREFIX}/ddb-overflow"
# leave headroom under DynamoDB's 400KB per-item hard limit
MAX_DDB_ITEM_BYTES = 350_000

CHAT_AGENT_ID = "chat"

# AgentCore Memory session ids must match [a-zA-Z0-9][a-zA-Z0-9-_]* (no dots), but the
# app's session ids are time.time() strings ("1786134138.387"). "." <-> "_" is bijective
# for those (digits + one dot, never an underscore).
def _acm_session_id(session_id: str) -> str:
    return session_id.replace(".", "_")


def _app_session_id(acm_session_id: str) -> str:
    return acm_session_id.replace("_", ".")


# LTM namespaces on the memory resource (created with these exact templates); facts and
# preferences are cross-session per actor and injected as context on retrieval. Session
# summaries exist on the resource too but are not auto-retrieved (the live window
# already covers the current session; other sessions' summaries rarely match queries).
ACM_RETRIEVAL_NAMESPACES = ("/facts/{actorId}", "/preferences/{actorId}")

# Documents attached to a user turn are injected into the message text wrapped in this
# sentinel so the clean question can be recovered for display (see strip_attached_documents).
ATTACHED_DOCS_OPEN = "<attached_documents>"
ATTACHED_DOCS_CLOSE = "</attached_documents>"
_ATTACHED_DOCS_RE = re.compile(
    re.escape(ATTACHED_DOCS_OPEN) + r".*?" + re.escape(ATTACHED_DOCS_CLOSE) + r"\s*",
    re.DOTALL,
)


def _to_dynamo(obj: Any) -> Any:
    """Recursively convert floats to Decimal (DynamoDB rejects float)."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo(v) for v in obj]
    return obj


def _from_dynamo(obj: Any) -> Any:
    """Recursively convert Decimal back to int/float."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(v) for v in obj]
    return obj


class DynamoDBSessionRepository(SessionRepository):
    """Stores Strands sessions in a single DynamoDB table.

    Item layout (composite key, reuses the existing chat table's key schema):
      UserId    (partition key) : "<user_id>"
      SessionId (sort key)      : "SESSION#<session_id>"                  -> session + agents metadata
      SessionId (sort key)      : "MSG#<session_id>#<agent_id>#<000123>"  -> one message per item

    One item per message keeps writes small (400KB DynamoDB item cap applies per
    message, not per conversation) and lets messages be listed with a Query.

    A single message can still exceed the 400KB item cap (large injected documents
    or base64 image bytes). Such messages are transparently offloaded to S3: the
    DynamoDB item stores only a pointer ({"s3_ref": <uri>}) and the full message
    JSON lives at s3://<bucket>/<DDB_OVERFLOW_PREFIX>/... . Offload is invisible to
    callers - read_message/list_messages dereference pointers automatically.
    """

    def __init__(self, table_name: str, user_id: str, region_name: Optional[str] = None,
                 bucket: str = ""):
        self.table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)
        self.user_id = user_id
        self.bucket = bucket
        self._s3 = boto3.client("s3", region_name=region_name) if bucket else None

    # ---- key helpers -------------------------------------------------------
    def _session_key(self, session_id: str) -> dict:
        return {"UserId": self.user_id, "SessionId": f"SESSION#{session_id}"}

    def _message_sk(self, session_id: str, agent_id: str, message_id: int) -> str:
        return f"MSG#{session_id}#{agent_id}#{message_id:09d}"

    def _get_session_item(self, session_id: str) -> Optional[dict]:
        resp = self.table.get_item(Key=self._session_key(session_id))
        return resp.get("Item")

    def _put_session_item(self, item: dict) -> None:
        self.table.put_item(Item=_to_dynamo(item))

    # ---- session -----------------------------------------------------------
    def create_session(self, session: Session, **kwargs: Any) -> Session:
        item = {
            **self._session_key(session.session_id),
            "session": session.to_dict(),
            "agents": {},
            "multi_agents": {},
            "time": str(time.time()),
        }
        self._put_session_item(item)
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        item = self._get_session_item(session_id)
        if item is None:
            return None
        return Session.from_dict(_from_dynamo(item["session"]))

    def list_session_ids(self) -> list[str]:
        """Return the ids of all sessions for this user (SESSION# items only)."""
        session_ids, args = [], {
            "KeyConditionExpression": Key("UserId").eq(self.user_id) & Key("SessionId").begins_with("SESSION#")
        }
        while True:
            resp = self.table.query(**args)
            session_ids.extend(item["SessionId"].split("SESSION#", 1)[1] for item in resp["Items"])
            if "LastEvaluatedKey" not in resp:
                break
            args["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return session_ids

    # ---- agent -------------------------------------------------------------
    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        item = self._get_session_item(session_id)
        item.setdefault("agents", {})[session_agent.agent_id] = session_agent.to_dict()
        self._put_session_item(item)

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        item = self._get_session_item(session_id)
        if item is None:
            return None
        agent = item.get("agents", {}).get(agent_id)
        return SessionAgent.from_dict(_from_dynamo(agent)) if agent else None

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        self.create_agent(session_id, session_agent)

    # ---- S3 offload for oversized messages ---------------------------------
    def _overflow_key(self, session_id: str, agent_id: str, message_id: int) -> str:
        return f"{DDB_OVERFLOW_PREFIX}/{self.user_id}/{session_id}/{agent_id}/message_{message_id}.json"

    def _message_item(self, session_id: str, agent_id: str, session_message: SessionMessage) -> dict:
        """Build the DynamoDB item for a message, spilling to S3 if it would exceed the item cap."""
        message_dict = session_message.to_dict()
        item = {
            "UserId": self.user_id,
            "SessionId": self._message_sk(session_id, agent_id, session_message.message_id),
            "message": message_dict,
            "time": str(time.time()),
        }
        # only the message payload is unbounded; measure it as stored (JSON) plus slack for keys
        if len(json.dumps(message_dict, ensure_ascii=False).encode("utf-8")) <= MAX_DDB_ITEM_BYTES:
            return _to_dynamo(item)
        if self._s3 is None:
            raise ValueError(
                "session-storage 'dynamodb' needs Bucket_Name in config.json to offload messages "
                "larger than the 400KB DynamoDB item limit"
            )
        key = self._overflow_key(session_id, agent_id, session_message.message_id)
        self._s3.put_object(
            Bucket=self.bucket, Key=key,
            Body=json.dumps(message_dict, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        item["message"] = {"s3_ref": f"s3://{self.bucket}/{key}"}
        return _to_dynamo(item)

    def _load_message(self, stored: dict) -> SessionMessage:
        """Reverse of _message_item: dereference an S3 pointer if present."""
        message = _from_dynamo(stored)
        if isinstance(message, dict) and "s3_ref" in message:
            _, _, rest = message["s3_ref"].partition("s3://")
            bucket, _, key = rest.partition("/")
            body = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            message = json.loads(body.decode("utf-8"))
        return SessionMessage.from_dict(message)

    # ---- messages ----------------------------------------------------------
    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        self.table.put_item(Item=self._message_item(session_id, agent_id, session_message))

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        resp = self.table.get_item(
            Key={"UserId": self.user_id, "SessionId": self._message_sk(session_id, agent_id, message_id)}
        )
        item = resp.get("Item")
        return self._load_message(item["message"]) if item else None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        self.create_message(session_id, agent_id, session_message)

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> list[SessionMessage]:
        prefix = f"MSG#{session_id}#{agent_id}#"
        messages, args = [], {
            "KeyConditionExpression": Key("UserId").eq(self.user_id) & Key("SessionId").begins_with(prefix)
        }
        while True:
            resp = self.table.query(**args)
            messages.extend(resp["Items"])
            if "LastEvaluatedKey" not in resp:
                break
            args["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        # sort key zero-padding makes lexicographic order == numeric order
        messages.sort(key=lambda m: m["SessionId"])
        messages = messages[int(offset):]
        if limit is not None:
            messages = messages[:int(limit)]
        return [self._load_message(m["message"]) for m in messages]

    # ---- multi-agent (not used by the chat path; stored on the session item) ----
    def create_multi_agent(self, session_id: str, multi_agent: Any, **kwargs: Any) -> None:
        item = self._get_session_item(session_id)
        item.setdefault("multi_agents", {})[multi_agent.id] = multi_agent.serialize_state()
        self._put_session_item(item)

    def read_multi_agent(self, session_id: str, multi_agent_id: str, **kwargs: Any) -> Optional[dict]:
        item = self._get_session_item(session_id)
        if item is None:
            return None
        return item.get("multi_agents", {}).get(multi_agent_id)

    def create_multiagent(self, *args, **kwargs):  # older strands naming, delegate
        return self.create_multi_agent(*args, **kwargs)

    def update_multi_agent(self, session_id: str, multi_agent: Any, **kwargs: Any) -> None:
        self.create_multi_agent(session_id, multi_agent)


def _build_agentcore_manager(session_id: str, *, memory_id: str, user_id: str,
                             region: Optional[str], with_retrieval: bool = True):
    """AgentCoreMemorySessionManager for one (actor, session). Conversation events are
    the session store; the resource's LTM strategies extract facts/preferences per
    actor in the background and get injected as <user_context> on each user turn."""
    from bedrock_agentcore.memory.integrations.strands.config import (
        AgentCoreMemoryConfig, RetrievalConfig)
    from bedrock_agentcore.memory.integrations.strands.session_manager import (
        AgentCoreMemorySessionManager)
    retrieval = ({ns: RetrievalConfig(top_k=5, relevance_score=0.3)
                  for ns in ACM_RETRIEVAL_NAMESPACES} if with_retrieval else None)
    config = AgentCoreMemoryConfig(
        memory_id=memory_id,
        session_id=_acm_session_id(session_id),
        actor_id=user_id or "default",
        retrieval_config=retrieval,
        # batch_size MUST stay 1: buffered create_message returns no eventId, which
        # would leave _latest_agent_message.message_id = None and break turn_meta
        # keying. Immediate writes are naturally spaced by model/tool latency, well
        # under the 5 CreateEvent/s per-actor-session limit.
        batch_size=1,
    )
    return AgentCoreMemorySessionManager(agentcore_memory_config=config, region_name=region)


def build_session_manager(storage: str, session_id: str, *, user_id: str = "",
                          bucket: str = "", region: Optional[str] = None,
                          dynamodb_table: str = "", agentcore_memory_id: str = ""):
    """Return a Strands session manager for the configured backend.

    storage: "local" | "s3" | "dynamodb" | "agentcore"
    """
    storage = (storage or "local").lower()
    if storage == "local":
        return FileSessionManager(session_id=session_id, storage_dir=LOCAL_SESSION_DIR)
    if storage == "s3":
        if not bucket:
            raise ValueError("session-storage 's3' requires Bucket_Name in config.json")
        return S3SessionManager(session_id=session_id, bucket=bucket,
                                prefix=f"{S3_SESSION_PREFIX}/{user_id}" if user_id else S3_SESSION_PREFIX,
                                region_name=region)
    if storage == "dynamodb":
        if not dynamodb_table:
            raise ValueError("session-storage 'dynamodb' requires DynamodbTable in config.json")
        repo = DynamoDBSessionRepository(table_name=dynamodb_table, user_id=user_id,
                                         region_name=region, bucket=bucket)
        return RepositorySessionManager(session_id=session_id, session_repository=repo)
    if storage == "agentcore":
        if not agentcore_memory_id:
            raise ValueError("session-storage 'agentcore' requires 'agentcore-memory-id' in config.json")
        return _build_agentcore_manager(session_id, memory_id=agentcore_memory_id,
                                        user_id=user_id, region=region)
    raise ValueError(f"Unknown session-storage: {storage!r} (expected local | s3 | dynamodb | agentcore)")


# ============================================================================
# Document-injection sentinel helpers
# ============================================================================
def wrap_attached_documents(result_string: str) -> str:
    """Wrap the per-file <name>...</name> blocks produced by process_files() in the
    <attached_documents> sentinel so the injected text can be split back out of the
    persisted user message for display (see strip_attached_documents)."""
    if not result_string:
        return ""
    return f"{ATTACHED_DOCS_OPEN}\n{result_string}{ATTACHED_DOCS_CLOSE}\n"


def strip_attached_documents(text: str) -> str:
    """Remove any <attached_documents>...</attached_documents> block, returning the
    clean user question that was typed in the chat box."""
    return _ATTACHED_DOCS_RE.sub("", text).strip()


# Marker the document_generator tool appends (deterministically, outside the model) to
# its relayed answer, so generated-artifact NAMES persist word-for-word in the
# orchestrator's window: [generated artifacts: report.docx, deck.pptx]
_GEN_ARTIFACTS_RE = re.compile(r"\[generated artifacts: ([^\]]*)\]")


def generated_uris_in_window(messages: list, turn_meta: dict) -> list:
    """S3 uris of generated documents whose [generated artifacts: ...] marker is still
    visible in the model's conversation window.

    Counterpart of dataset_uris_in_window for docgen outputs: marker names in the
    window's user messages (tool results are user-role) are matched against basenames
    recorded in turn_meta[*].doc_output — turn_meta stays the sole uri source, so a
    fabricated marker name yields nothing. Entries are read in ascending message-id
    order so a re-generated filename resolves to its LATEST uri (which also matches
    S3, where same prefix + basename overwrites).
    """
    # chronological key order. int keys are FileSession/S3/DynamoDB sequential message
    # ids; agentcore keys are eventIds ("<zero-padded-millis>#<hash>") whose string
    # sort IS time order. A session never mixes backends, so the two groups never
    # actually compete.
    def _key_order(k):
        k = str(k)
        return (0, int(k), "") if k.isdigit() else (1, 0, k)

    uri_by_name = {}
    for key in sorted((turn_meta or {}), key=_key_order):
        for uri in (turn_meta[key].get("doc_output") or []):
            uri_by_name[os.path.basename(uri)] = uri  # latest occurrence wins
    if not uri_by_name:
        return []
    visible = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content", []):
            texts = [block["text"]] if block.get("text") else []
            for c in (block.get("toolResult") or {}).get("content", []):
                if c.get("text"):
                    texts.append(c["text"])
            for text in texts:
                for m in _GEN_ARTIFACTS_RE.finditer(text):
                    visible.update(n.strip() for n in m.group(1).split(",") if n.strip())
    return [uri for name, uri in uri_by_name.items() if name in visible]


def dataset_uris_in_window(messages: list, turn_meta: dict) -> list:
    """S3 uris of previously attached documents whose injected content is still
    visible in the model's conversation window.

    `messages` is agent.messages right after build_chat_agent(): the session manager
    restores exactly the window the sliding-window trim left visible (verified live),
    so scanning it - rather than re-deriving a cutoff - keeps "visible to us" equal to
    "visible to the agent" by construction.

    Scans user messages for <attached_documents> blocks and matches their per-file
    </name> tags word-for-word against the basenames recorded in turn_meta[*].documents.
    turn_meta stays the source of truth for the uris (tags are just names); a tag with
    no recorded uri - or a recorded uri whose content has scrolled past the cutoff -
    yields nothing. This deliberately mirrors what the orchestrator can still ground:
    datasets it can no longer see are not offered to sub-agents (the user re-attaches).
    """
    uri_by_name = {}
    for meta in (turn_meta or {}).values():
        for uri in meta.get("documents", []):
            uri_by_name.setdefault(os.path.basename(uri), uri)
    if not uri_by_name:
        return []
    visible = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content", []):
            text = block.get("text") or ""
            for m in _ATTACHED_DOCS_RE.finditer(text):
                visible.update(re.findall(r"</([^<>\n]+)>", m.group(0)))
    return [uri for name, uri in uri_by_name.items() if name in visible]


# ============================================================================
# Display-log reconstruction (read-only; never creates a session)
# ============================================================================
# These helpers read persisted Strands sessions WITHOUT constructing a session
# manager, because instantiating a manager for a missing session id creates an
# empty session as a side effect. The chatbot's sidebar shows a "New Chat" id
# before the first turn is ever sent, so a read must be side-effect-free.

def _urls_from_result_text(text: str) -> list[str]:
    """Pull source URLs out of a toolResult's text payload.

    Two shapes are understood:
    - tavily_search: a JSON list of result dicts (each with a "url") - the search sources.
    - any tool returning a JSON object with a top-level "citations" list of URLs
      (e.g. worker_agents) - the generic opt-in for Sources rendering.
    web_fetch returns a plain "URL: ...\\n\\n<text>" string, whose fetched URL is taken
    from the toolUse input instead (see _extract_turn), so a non-JSON payload here
    simply yields nothing.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, dict) and isinstance(data.get("citations"), list):
        return [u for u in data["citations"] if isinstance(u, str)]
    items = data if isinstance(data, list) else [data]
    return [item["url"] for item in items
            if isinstance(item, dict) and isinstance(item.get("url"), str)]


def _extract_turn(message: Message) -> dict:
    """Pull the display-relevant fields out of one Strands message's content blocks.

    Tool-calling turns interleave extra messages: an assistant message may carry
    only a toolUse block, and the following user message carries the toolResult.
    These are internal to the turn and not shown as chat bubbles, so we also report
    which block types are present and any source URLs referenced (pages fetched via
    web_fetch and result links from tavily_search) so they can be surfaced under the
    final answer's attachments expander.
    """
    text_parts, thinking_parts, source_urls = [], [], []
    has_tool_use = has_tool_result = False
    for block in message.get("content", []):
        if "text" in block:
            text_parts.append(block["text"])
        elif "reasoningContent" in block:
            reasoning_text = block["reasoningContent"].get("reasoningText", {})
            if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                thinking_parts.append(reasoning_text["text"])
        elif "toolUse" in block:
            has_tool_use = True
            tool_input = block["toolUse"].get("input", {})
            if isinstance(tool_input, dict) and isinstance(tool_input.get("url"), str):
                source_urls.append(tool_input["url"])  # web_fetch target
        elif "toolResult" in block:
            has_tool_result = True
            for content in block["toolResult"].get("content", []):
                if isinstance(content, dict) and content.get("text"):
                    source_urls.extend(_urls_from_result_text(content["text"]))
    return {
        "text": "".join(text_parts),
        "thinking": "".join(thinking_parts),
        "source_urls": source_urls,
        "has_tool_use": has_tool_use,
        "has_tool_result": has_tool_result,
    }


def _dedupe(seq: list) -> list:
    """Order-preserving de-duplication, dropping falsy entries."""
    seen, out = set(), []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def messages_to_display(session_messages: list, turn_meta: dict) -> list[dict]:
    """Turn a session's messages (+ per-turn metadata from agent.state) into the
    formatted_data list the Streamlit UI renders.

    turn_meta maps the assistant message_id (as a str) to extra info that cannot be
    reconstructed from the conversation alone: model id, s3 doc/image uris, cost.

    Tool round-trips (assistant toolUse -> user toolResult -> assistant text) are not
    shown as their own bubbles: only the final assistant answer is rendered, any
    reasoning the intermediate messages carried is carried forward, and the source
    URLs (fetched pages / search-result links) are collected into the attachments
    expander beneath that answer.
    """
    formatted = []
    pending_thinking = ""   # reasoning emitted on intermediate toolUse messages
    pending_urls = []       # source URLs referenced before the final answer
    for sm in session_messages:
        role = sm.message.get("role")
        parsed = _extract_turn(sm.message)

        # user turn carrying a toolResult is internal to a tool round-trip: harvest
        # its source URLs (tavily results) but do not render a bubble
        if role == "user" and parsed["has_tool_result"]:
            pending_urls.extend(parsed["source_urls"])
            continue
        # assistant turn that invokes a tool is internal to the round-trip, even when
        # it also carries preamble text (real Bedrock puts "let me search..." in the
        # same message as the toolUse). Buffer its reasoning/fetched URLs, drop the
        # preamble, and skip the bubble - only the post-tool answer is rendered.
        if role == "assistant" and parsed["has_tool_use"]:
            pending_thinking += parsed["thinking"]
            pending_urls.extend(parsed["source_urls"])
            continue

        if role == "user":
            formatted.append({
                "role": "user",
                "content": strip_attached_documents(parsed["text"]),
                "thinking": "",
            })
        elif role == "assistant":
            meta = turn_meta.get(str(sm.message_id), {})
            docs = meta.get("documents", [])
            images = meta.get("images", [])
            sources = _dedupe(pending_urls + parsed["source_urls"])
            attachment_parts = [os.path.basename(x) for x in (images + docs)]
            if sources:
                attachment_parts.append(
                    "**Sources**\n" + "\n".join(f"- [{u}]({u})" for u in sources)
                )
            attachment = "\n\n".join(attachment_parts)
            formatted.append({
                "role": "assistant",
                "content": parsed["text"],
                "thinking": pending_thinking + parsed["thinking"],
                "attachment": attachment,
                "code": "",
                "code-result": "",
                # sub-agent artifacts (s3 uris) surfaced by the UI; persisted in
                # turn_meta, not reconstructable from the messages themselves.
                # image_output/plotly render inline; doc_output renders as presigned
                # download links in the "artifacts" expander.
                "image_output": meta.get("image_output", []),
                "plotly": meta.get("plotly", []),
                "doc_output": meta.get("doc_output", []),
            })
            pending_thinking, pending_urls = "", []
    return formatted


def _iter_raw_messages_local(session_id: str, agent_id: str = CHAT_AGENT_ID):
    """Read persisted local message files without touching FileSessionManager."""
    agent_dir = os.path.join(LOCAL_SESSION_DIR, f"session_{session_id}", "agents",
                             f"agent_{agent_id}")
    messages_dir = os.path.join(agent_dir, "messages")
    agent_file = os.path.join(agent_dir, "agent.json")
    if not os.path.isdir(messages_dir):
        return [], {}
    files = []
    for name in os.listdir(messages_dir):
        if name.startswith("message_") and name.endswith(".json"):
            files.append((int(name[len("message_"):-5]), name))
    messages = []
    for _, name in sorted(files):
        with open(os.path.join(messages_dir, name), encoding="utf-8") as f:
            messages.append(SessionMessage.from_dict(json.load(f)))
    turn_meta = {}
    if os.path.exists(agent_file):
        with open(agent_file, encoding="utf-8") as f:
            turn_meta = json.load(f).get("state", {}).get("turn_meta", {})
    return messages, turn_meta


def _iter_raw_messages_s3(session_id: str, *, bucket: str, user_id: str,
                          region: Optional[str], agent_id: str = CHAT_AGENT_ID):
    """Read persisted S3 message objects without touching S3SessionManager."""
    s3 = boto3.client("s3", region_name=region)
    prefix = f"{S3_SESSION_PREFIX}/{user_id}" if user_id else S3_SESSION_PREFIX
    agent_prefix = f"{prefix}/session_{session_id}/agents/agent_{agent_id}/"
    messages = []
    paginator = s3.get_paginator("list_objects_v2")
    indexed = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{agent_prefix}messages/"):
        for obj in page.get("Contents", []):
            name = os.path.basename(obj["Key"])
            if name.startswith("message_") and name.endswith(".json"):
                indexed.append((int(name[len("message_"):-5]), obj["Key"]))
    for _, key in sorted(indexed):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        messages.append(SessionMessage.from_dict(json.loads(body.decode("utf-8"))))
    turn_meta = {}
    try:
        body = s3.get_object(Bucket=bucket, Key=f"{agent_prefix}agent.json")["Body"].read()
        turn_meta = json.loads(body.decode("utf-8")).get("state", {}).get("turn_meta", {})
    except Exception:
        pass
    return messages, turn_meta


def _iter_raw_messages_agentcore(session_id: str, *, memory_id: str, user_id: str,
                                 region: Optional[str]):
    """Read one session's messages + turn_meta straight from AgentCore Memory events,
    WITHOUT constructing a session manager (which would create a SESSION state event
    for ids that don't exist yet, materializing empty sessions on every rerun).

    The integration stores message payloads with message_id=0 and uses the AgentCore
    eventId as the live message id (see append_message), so turn_meta is keyed by
    eventId - each restored message is re-stamped with its event's id here so the
    messages_to_display lookup matches.
    """
    from bedrock_agentcore.memory.client import MemoryClient
    from bedrock_agentcore.memory.integrations.strands.bedrock_converter import (
        AgentCoreMemoryConverter)
    from bedrock_agentcore.memory.models.filters import (
        EventMetadataFilter, LeftExpression, MetadataValue, OperatorType, RightExpression)

    client = MemoryClient(region_name=region)
    actor = user_id or "default"
    acm_sid = _acm_session_id(session_id)
    try:
        events = client.list_events(memory_id=memory_id, actor_id=actor,
                                    session_id=acm_sid, max_results=10000)
    except Exception:
        return [], {}

    messages = []
    for event in reversed(events):  # list_events returns newest-first
        for msg in AgentCoreMemoryConverter.events_to_messages([event]):
            msg.message_id = event.get("eventId")
            messages.append(msg)

    turn_meta = {}
    try:
        agent_events = client.list_events(
            memory_id=memory_id, actor_id=actor, session_id=acm_sid, max_results=1,
            event_metadata=[
                EventMetadataFilter.build_expression(
                    left_operand=LeftExpression.build("stateType"),
                    operator=OperatorType.EQUALS_TO,
                    right_operand=RightExpression.build("AGENT")),
                EventMetadataFilter.build_expression(
                    left_operand=LeftExpression.build("agentId"),
                    operator=OperatorType.EQUALS_TO,
                    right_operand=RightExpression.build(CHAT_AGENT_ID)),
            ])
        if agent_events:
            agent_data = json.loads(agent_events[0]["payload"][0]["blob"])
            turn_meta = (agent_data.get("state") or {}).get("turn_meta", {})
    except Exception:
        pass
    return messages, turn_meta


def read_display_history(storage: str, session_id: str, *, user_id: str = "",
                         bucket: str = "", region: Optional[str] = None,
                         dynamodb_table: str = "", agentcore_memory_id: str = "") -> list[dict]:
    """Reconstruct the Streamlit display log for one session from its persisted
    Strands session, without creating a session. Returns [] for unknown sessions."""
    storage = (storage or "local").lower()
    if not session_id:
        return []
    if storage == "local":
        messages, turn_meta = _iter_raw_messages_local(session_id)
    elif storage == "s3":
        if not bucket:
            raise ValueError("session-storage 's3' requires Bucket_Name in config.json")
        messages, turn_meta = _iter_raw_messages_s3(
            session_id, bucket=bucket, user_id=user_id, region=region)
    elif storage == "dynamodb":
        if not dynamodb_table:
            raise ValueError("session-storage 'dynamodb' requires DynamodbTable in config.json")
        repo = DynamoDBSessionRepository(table_name=dynamodb_table, user_id=user_id,
                                         region_name=region, bucket=bucket)
        if repo.read_session(session_id) is None:
            return []
        messages = repo.list_messages(session_id, CHAT_AGENT_ID)
        agent = repo.read_agent(session_id, CHAT_AGENT_ID)
        turn_meta = (agent.state.get("turn_meta", {}) if agent else {})
    elif storage == "agentcore":
        if not agentcore_memory_id:
            raise ValueError("session-storage 'agentcore' requires 'agentcore-memory-id' in config.json")
        messages, turn_meta = _iter_raw_messages_agentcore(
            session_id, memory_id=agentcore_memory_id, user_id=user_id, region=region)
    else:
        raise ValueError(f"Unknown session-storage: {storage!r}")
    return messages_to_display(messages, turn_meta)


def list_user_sessions(storage: str, *, user_id: str = "", bucket: str = "",
                       region: Optional[str] = None, dynamodb_table: str = "",
                       agentcore_memory_id: str = "") -> dict[str, str]:
    """Map session_id -> first user message, for the sidebar session picker."""
    storage = (storage or "local").lower()
    session_ids: list[str] = []
    if storage == "local":
        base = LOCAL_SESSION_DIR
        if os.path.isdir(base):
            session_ids = [name[len("session_"):] for name in os.listdir(base)
                           if name.startswith("session_")]
    elif storage == "s3":
        if not bucket:
            return {}
        s3 = boto3.client("s3", region_name=region)
        prefix = f"{S3_SESSION_PREFIX}/{user_id}/" if user_id else f"{S3_SESSION_PREFIX}/"
        paginator = s3.get_paginator("list_objects_v2")
        seen = set()
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                leaf = cp["Prefix"].rstrip("/").split("/")[-1]
                if leaf.startswith("session_"):
                    seen.add(leaf[len("session_"):])
        session_ids = list(seen)
    elif storage == "dynamodb":
        if not dynamodb_table:
            return {}
        repo = DynamoDBSessionRepository(table_name=dynamodb_table, user_id=user_id,
                                         region_name=region, bucket=bucket)
        session_ids = repo.list_session_ids()
    elif storage == "agentcore":
        if not agentcore_memory_id:
            return {}
        from bedrock_agentcore.memory.client import MemoryClient
        client = MemoryClient(region_name=region)
        try:
            paginator = client.gmdp_client.get_paginator("list_sessions")
            for page in paginator.paginate(memoryId=agentcore_memory_id,
                                           actorId=user_id or "default"):
                for summary in page.get("sessionSummaries", []):
                    session_ids.append(_app_session_id(summary["sessionId"]))
        except Exception:
            return {}
    else:
        raise ValueError(f"Unknown session-storage: {storage!r}")

    result = {}
    for sid in session_ids:
        history = read_display_history(storage, sid, user_id=user_id, bucket=bucket,
                                       region=region, dynamodb_table=dynamodb_table,
                                       agentcore_memory_id=agentcore_memory_id)
        for entry in history:
            if entry["role"] == "user":
                result[sid] = entry["content"]
                break
    return result
