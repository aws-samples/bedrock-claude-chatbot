"""Data-analysis sub-agent (agent-as-tool) for the Strands chat orchestrator.

Replaces the legacy structured function-calling path (utils/function_calling_utils
+ Docker Lambda / Athena) with an agentic sub-agent that writes and runs code in an
Amazon Bedrock AgentCore **Code Interpreter** sandbox.

Topology:
    orchestrator (persisted chat agent)
      └─ tool: data_analysis(request, dataset_names?)   [this module]
            └─ ephemeral sub-agent (own model, vision)
                  └─ tool: execute_code(code)           [one of two backends]
                        ├─ "python":  AgentCore Code Interpreter (CodeSessionManager)
                        └─ "pyspark": Athena for Apache Spark, PySpark engine v3
                                      (AthenaSparkSessionManager)

Design notes (validated against a real interpreter):
  - The sandbox is a stateful IPython kernel: variables and files persist across
    execute_code calls within one session (see CodeSessionManager).
  - The sandbox has S3 egress + an IAM role, but NO s3fs and no general internet, so
    datasets are read with plain boto3 + io.BytesIO (not pd.read_csv("s3://...")).
  - Generated files stay on the sandbox FS; the agent prints [[ARTIFACT:...]] sentinels
    and we read the bytes back out with download_file (relative paths only).
  - Athena (validated likewise): inline code via StartCalculationExecution works ONLY on
    "PySpark engine version 3" workgroups (Spark 3.5 rejects the calculation APIs). The
    session is a stateful kernel too, but there is no download API — artifacts leave the
    sandbox only via S3, so a bootstrap calculation defines a save_artifact() helper that
    uploads the file and prints the same sentinel. No pip binary, S3 read/write via the
    workgroup execution role. Dated runtime: Python 3.9 / pandas 1.4 / plotly 5.9.
  - Charts are fed back to the (vision) sub-agent as PNG image content blocks so it can
    validate them, and their S3 uris are collected in a per-turn sink for the UI. Plotly
    is preferred (interactive in the UI); the sandbox has no kaleido, so .plotly json is
    rendered to PNG app-side (kaleido lives in the app env) for the vision feedback.
"""
import io
import os
import re
import time
from typing import Callable, Optional

import boto3

from strands import Agent, tool

from .chat_agent import resolve_model

# Sentinel a sub-agent prints after saving each output file, so we can extract artifact
# paths deterministically from stdout: [[ARTIFACT:image:chart.png]] /
# [[ARTIFACT:plotly:chart.plotly]] / [[ARTIFACT:document:report.docx]]
_ARTIFACT_RE = re.compile(r"\[\[ARTIFACT:(image|plotly|document):(.+?)\]\]")
_IMAGE_EXTS = {".png", ".jpeg", ".jpg", ".gif", ".webp"}

# System prompt = environment truths + guardrails only (no analysis recipe; the agent
# figures out the analysis natively). Dataset uris are appended per invocation.
_SYSTEM_PROMPT = """You are a data-analysis agent. You answer questions about structured \
datasets by writing and running Python code in a sandboxed code interpreter, then \
explaining the results.

EXECUTION ENVIRONMENT (read carefully):
- You have one tool, execute_code(code), running a STATEFUL IPython kernel. Variables, \
imports and files you create PERSIST across calls, so build up your analysis \
incrementally and reuse prior results instead of recomputing.
- Standard data-science libraries are pre-installed: pandas, numpy, matplotlib, plotly, \
scipy, scikit-learn, openpyxl, boto3, etc. You CANNOT pip install anything (no internet).
- There is NO s3fs, so DO NOT use pd.read_csv("s3://..."). Read datasets from S3 with \
boto3 into a buffer, e.g.:
    import boto3, io, pandas as pd
    obj = boto3.client("s3", region_name="{region}").get_object(Bucket=BKT, Key=KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
  Always pass region_name="{region}" to the S3 client (other endpoints are unreachable).
- Start by exploring the data (df.head(), df.info(), df.describe()) before analyzing.

CHARTS (so the user and you can see them):
- PREFER plotly for charts — the user sees plotly figures as interactive charts. Save each \
plotly figure as JSON to a RELATIVE filename and print its sentinel right after:
    import plotly.io as pio
    pio.write_json(fig, "revenue.plotly")
    print("[[ARTIFACT:plotly:revenue.plotly]]")
- Do NOT call fig.write_image() / fig.to_image() — the kaleido engine is NOT installed \
here. Saving the .plotly json is enough; it is rendered for the user automatically.
- Fall back to matplotlib ONLY when plotly is a poor fit for the visual, or if plotly \
keeps erroring after a couple of attempts. Save it as a PNG:
    fig.savefig("revenue.png")
    print("[[ARTIFACT:image:revenue.png]]")
- Always use plain RELATIVE filenames; absolute paths like /tmp/x.png are REJECTED.
- After execution you will be shown a PNG render of every figure you saved (plotly \
figures are converted to PNG for you); verify they look correct and reference them in \
your answer.

ANSWER:
- When done, write a clear, well-formatted markdown answer (use tables/lists where useful) \
based strictly on the computed results. Do not fabricate numbers. This answer is returned \
verbatim to the user, so make it self-contained.
- If SEVERAL datasets could plausibly match the request and it does not disambiguate, do \
NOT guess: instead of analyzing, return a short question listing the candidate files so \
the user can pick. When only one dataset fits (or one is listed), just proceed."""

# Variant for the Athena Spark backend ("pyspark" engine). Same skeleton, different
# environment truths: pre-bound `spark`, dated libs, save_artifact() instead of manual
# sentinels (no download API — files leave the sandbox only via the S3 upload helper).
_SPARK_SYSTEM_PROMPT = """You are a data-analysis agent. You answer questions about \
structured datasets by writing and running Python/PySpark code in a sandboxed Spark \
environment, then explaining the results.

EXECUTION ENVIRONMENT (read carefully):
- You have one tool, execute_code(code), running a STATEFUL kernel on Amazon Athena for \
Apache Spark. Variables, temp views and files you create PERSIST across calls, so build \
up your analysis incrementally and reuse prior results instead of recomputing.
- A live SparkSession is already bound to the variable `spark` — do not create your own.
- Load datasets with Spark (native S3 support), then aggregate/filter in Spark and only \
bring SMALL results into pandas:
    df = spark.read.option("header", True).option("inferSchema", True).csv("s3://bucket/key.csv")
    df.createOrReplaceTempView("data")
    small = spark.sql("select ... group by ...").toPandas()
  Do NOT use pd.read_csv("s3://...") (no s3fs installed); for non-CSV/parquet edge cases \
read via boto3 into io.BytesIO.
- DATED library versions: Python 3.9, pandas 1.4, numpy 1.23, plotly 5.9, matplotlib 3.5, \
scikit-learn 1.1, Spark 3.2. Avoid pandas 2.x-only APIs. You CANNOT install packages.
- Only printed output comes back — print() any result you need to see.
- Start by exploring the data (schema, sample rows, counts) before analyzing.

CHARTS (so the user and you can see them):
- PREFER plotly for charts — the user sees plotly figures as interactive charts. Save each \
figure to a file under /tmp, then call save_artifact(path) — a helper already defined in \
your kernel that publishes the file and prints its capture sentinel:
    import plotly.io as pio
    pio.write_json(fig, "/tmp/revenue.plotly")
    save_artifact("/tmp/revenue.plotly")
- Do NOT call fig.write_image() / fig.to_image() — the kaleido engine is NOT installed. \
Saving the .plotly json is enough; it is rendered for the user automatically.
- Fall back to matplotlib ONLY when plotly is a poor fit for the visual, or if plotly \
keeps erroring after a couple of attempts:
    fig.savefig("/tmp/revenue.png")
    save_artifact("/tmp/revenue.png")
- After execution you will be shown a PNG render of every figure you saved (plotly \
figures are converted to PNG for you); verify they look correct and reference them in \
your answer.

ANSWER:
- When done, write a clear, well-formatted markdown answer (use tables/lists where useful) \
based strictly on the computed results. Do not fabricate numbers. This answer is returned \
verbatim to the user, so make it self-contained."""

# Bootstrap calculation run once per Athena session: defines the save_artifact() helper
# the prompt references. Written brace-free so .format() only touches the placeholders.
_SPARK_BOOTSTRAP = '''\
import boto3 as _boto3, os as _os
_ART_BUCKET = "{bucket}"
_ART_PREFIX = "{prefix}"
def save_artifact(path):
    name = _os.path.basename(path)
    kind = "plotly" if name.endswith(".plotly") else "image"
    _boto3.client("s3").upload_file(path, _ART_BUCKET, _ART_PREFIX + "/" + name)
    print("[[ARTIFACT:" + kind + ":" + name + "]]")
print("spark session ready")
'''


class CodeSessionManager:
    """A warm AgentCore Code Interpreter session with lazy start and auto-restart.

    One instance is cached per chat session (in st.session_state) so dataframes and
    imports survive across turns. If the session has expired/terminated, run() starts a
    fresh one transparently (kernel state is lost, but datasets reload cheaply from S3).
    """

    engine = "python"

    def __init__(self, region: str, identifier: str, timeout_seconds: int = 3600):
        self.region = region
        self.identifier = identifier
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _start(self):
        # Imported lazily so the rest of the app runs even if bedrock-agentcore is absent.
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
        client = CodeInterpreter(self.region)
        client.start(identifier=self.identifier, session_timeout_seconds=self.timeout_seconds)
        self._client = client

    def run(self, code: str) -> dict:
        """Execute code, restarting the session once if it has gone away.

        Returns {stdout, stderr, exitCode, isError}.
        """
        if self._client is None:
            self._start()
        try:
            resp = self._client.execute_code(code, language="python", clear_context=False)
        except Exception:
            # session likely expired/terminated - start a fresh one and retry once
            self._start()
            resp = self._client.execute_code(code, language="python", clear_context=False)
        result = list(resp["stream"])[0]["result"]
        sc = result.get("structuredContent", {})
        return {
            "stdout": sc.get("stdout", ""),
            "stderr": sc.get("stderr", ""),
            "exitCode": sc.get("exitCode", 0),
            "isError": bool(result.get("isError", False)),
        }

    def download(self, path: str):
        """Return the bytes/str of a file from the sandbox (relative path)."""
        return self._client.download_file(path)

    def fetch_artifact(self, path: str, *, bucket: str, prefix: str, region: str):
        """Pull an artifact out of the sandbox FS, publish it to S3, and return
        (bytes, s3_uri)."""
        data = self.download(path)
        uri = _upload_artifact(data, path, bucket=bucket, prefix=prefix, region=region)
        if isinstance(data, str):
            data = data.encode("utf-8")
        return data, uri

    def stop(self):
        if self._client is not None:
            try:
                self._client.stop()
            finally:
                self._client = None


class AthenaSparkSessionManager:
    """A warm Athena-for-Apache-Spark session ("pyspark" engine) with lazy start and
    auto-restart, mirroring CodeSessionManager's interface.

    Requires a workgroup on **PySpark engine version 3** — the newer "Apache Spark 3.5"
    engine rejects StartCalculationExecution (inline code) entirely. The session is a
    stateful kernel with a pre-bound `spark` SparkSession; a bootstrap calculation
    defines save_artifact() (S3 upload + sentinel print), since Athena has no API to
    download files out of the sandbox.
    """

    engine = "pyspark"

    def __init__(self, region: str, workgroup: str, *, bucket: str, prefix: str,
                 idle_timeout_minutes: int = 20):
        self.region = region
        self.workgroup = workgroup
        self.bucket = bucket
        self.prefix = prefix
        self.idle_timeout_minutes = idle_timeout_minutes
        self._athena = boto3.client("athena", region_name=region)
        self._session_id = None

    def _start(self):
        resp = self._athena.start_session(
            Description="chatbot data-analysis session",
            WorkGroup=self.workgroup,
            EngineConfiguration={
                "CoordinatorDpuSize": 1,
                "MaxConcurrentDpus": 20,
                "DefaultExecutorDpuSize": 1,
            },
            SessionIdleTimeoutInMinutes=self.idle_timeout_minutes,
        )
        session_id = resp["SessionId"]
        deadline = time.time() + 300
        while True:
            state = self._athena.get_session_status(SessionId=session_id)["Status"]["State"]
            if state == "IDLE":
                break
            if state in ("FAILED", "TERMINATED", "DEGRADED"):
                raise RuntimeError(f"Athena session entered state {state}")
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for Athena session to become IDLE")
            time.sleep(3)
        self._session_id = session_id
        bootstrap = _SPARK_BOOTSTRAP.format(bucket=self.bucket, prefix=self.prefix)
        outcome = self._calculate(bootstrap)
        if outcome["isError"]:
            raise RuntimeError(f"Athena session bootstrap failed: {outcome['stderr'][:500]}")

    def _s3_text(self, uri: str) -> str:
        if not uri:
            return ""
        bucket, key = uri.replace("s3://", "").split("/", 1)
        try:
            body = boto3.client("s3", region_name=self.region).get_object(
                Bucket=bucket, Key=key)["Body"].read()
            return body.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _calculate(self, code: str) -> dict:
        calc_id = self._athena.start_calculation_execution(
            SessionId=self._session_id, CodeBlock=code)["CalculationExecutionId"]
        deadline = time.time() + 600
        while True:
            resp = self._athena.get_calculation_execution(CalculationExecutionId=calc_id)
            state = resp["Status"]["State"]
            if state in ("COMPLETED", "FAILED", "CANCELED"):
                break
            if time.time() > deadline:
                try:
                    self._athena.stop_calculation_execution(CalculationExecutionId=calc_id)
                finally:
                    return {"stdout": "", "stderr": "Calculation timed out after 600s",
                            "exitCode": 1, "isError": True}
            time.sleep(3)
        result = resp.get("Result", {})
        failed = state != "COMPLETED"
        return {
            "stdout": self._s3_text(result.get("StdOutS3Uri")),
            "stderr": self._s3_text(result.get("StdErrorS3Uri")),
            "exitCode": 1 if failed else 0,
            "isError": failed,
        }

    def run(self, code: str) -> dict:
        """Execute code, restarting the session once if it has gone away.

        Returns {stdout, stderr, exitCode, isError} — same shape as CodeSessionManager.
        """
        if self._session_id is None:
            self._start()
        try:
            return self._calculate(code)
        except Exception:
            # session likely idle-terminated - start a fresh one (rebootstraps) and retry once
            self._start()
            return self._calculate(code)

    def fetch_artifact(self, path: str, *, bucket: str, prefix: str, region: str):
        """Artifacts are already in S3 (save_artifact uploaded them from inside the
        sandbox); fetch the bytes back for vision review and return (bytes, s3_uri)."""
        key = f"{prefix}/{os.path.basename(path)}"
        data = boto3.client("s3", region_name=region).get_object(
            Bucket=bucket, Key=key)["Body"].read()
        return data, f"s3://{bucket}/{key}"

    def stop(self):
        if self._session_id is not None:
            try:
                self._athena.terminate_session(SessionId=self._session_id)
            except Exception:
                pass
            finally:
                self._session_id = None


def _upload_artifact(data, filename: str, *, bucket: str, prefix: str, region: str) -> str:
    """Put raw artifact bytes into S3 and return the s3:// uri."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    key = f"{prefix}/{os.path.basename(filename)}"
    boto3.client("s3", region_name=region).put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


def _plotly_png_bytes(data) -> bytes:
    """Render a .plotly json artifact to PNG bytes app-side, so the vision sub-agent
    can inspect the figure (the sandbox has no kaleido, so it can't render its own)."""
    import plotly.io as pio
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    return pio.from_json(data).to_image(format="png")


def _make_execute_code_tool(session, sink: dict, *, bucket: str,
                            prefix: str, region: str, vision: bool = True,
                            status_fn: Optional[Callable[[str], None]] = None):
    """Build the execute_code tool bound to one execution session + artifact sink.

    session is either backend (CodeSessionManager | AthenaSparkSessionManager) — both
    expose run(code) -> {stdout, stderr, exitCode, isError} and
    fetch_artifact(path, ...) -> (bytes, s3_uri).

    sink accumulates {"image_output": [s3uri...], "plotly": [s3uri...]} across all
    execute_code calls in the turn; the app drains it into turn_meta afterwards.
    """

    @tool
    def execute_code(code: str) -> dict:
        """Run Python code in the stateful sandbox and return its output.

        Save charts to relative filenames (plotly json preferred, PNG as fallback) and
        print [[ARTIFACT:plotly:name.plotly]] / [[ARTIFACT:image:name.png]] after each
        so they are captured. On error, the traceback is returned so you can fix the
        code and try again.

        Args:
            code: The Python code to execute.

        Returns:
            The captured stdout (and, on failure, the error traceback).
        """
        if status_fn:
            status_fn("⚙️ Running analysis code…")
        outcome = session.run(code)

        if outcome["isError"] or outcome["exitCode"] != 0:
            err = outcome["stderr"] or outcome["stdout"] or "Unknown execution error"
            return {"status": "error", "content": [{"text": err}]}

        stdout = outcome["stdout"]
        artifacts = _ARTIFACT_RE.findall(stdout)
        clean_stdout = _ARTIFACT_RE.sub("", stdout).strip()

        image_blocks = []
        captured = []
        for kind, path in artifacts:
            try:
                data, uri = session.fetch_artifact(path, bucket=bucket, prefix=prefix,
                                                   region=region)
            except Exception as e:  # a bad artifact must not sink the whole result
                captured.append(f"(failed to capture {path}: {e})")
                continue
            if kind == "image":
                sink["image_output"].append(uri)
                ext = os.path.splitext(path)[1].lower()
                if vision and ext in _IMAGE_EXTS and isinstance(data, (bytes, bytearray)):
                    fmt = "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")
                    image_blocks.append({"image": {"format": fmt, "source": {"bytes": bytes(data)}}})
                captured.append(f"image: {os.path.basename(path)}")
            elif kind == "document":  # generated docs (docx/pptx/pdf/...) for the artifacts expander
                sink.setdefault("doc_output", []).append(uri)
                captured.append(f"document: {os.path.basename(path)}")
            else:  # plotly
                sink["plotly"].append(uri)
                captured.append(f"plotly: {os.path.basename(path)}")
                if vision:
                    try:
                        png = _plotly_png_bytes(data)
                        image_blocks.append({"image": {"format": "png", "source": {"bytes": png}}})
                    except Exception as e:
                        captured.append(f"(could not render {path} to png for review: {e})")

        text = clean_stdout or "(code ran with no stdout)"
        if captured:
            text += "\n\nCaptured artifacts: " + ", ".join(captured)
            if image_blocks:
                text += "\n(The PNG image(s) are shown below for your review.)"
        content = [{"text": text}] + image_blocks
        return {"status": "success", "content": content}

    return execute_code


def make_data_analysis_tool(*, dataset_uris: list, model_id: str, region: str, bucket: str,
                            upload_prefix: str, session,
                            artifact_sink: dict, message_store: Optional[dict] = None,
                            vision: bool = True, reasoning: bool = True,
                            status_fn: Optional[Callable[[str], None]] = None,
                            workings_panel=None):
    """Build the data_analysis tool the orchestrator calls.

    Closes over everything the sub-agent needs (dataset uris, execution session,
    artifact sink, prior sub-agent messages for continuity). The model only supplies
    `request` and optional `dataset_names`. The session's backend (CodeSessionManager
    for "python", AthenaSparkSessionManager for "pyspark") selects the system prompt.

    message_store: optional {"messages": [...]} dict cached in st.session_state so the
        ephemeral sub-agent keeps context across orchestrator turns within a live session.
    workings_panel: optional SubAgentWorkingsPanel - streams the sub-agent's live
        reasoning/text into an 'Agent workings' expander for eye engagement while the
        analysis runs (transient; never persisted).
    """

    @tool
    def data_analysis(request: str, dataset_names: Optional[list] = None) -> str:
        """Analyze the attached structured dataset(s) (CSV/XLSX/Parquet) by writing and
        running Python code — statistics, aggregations, transformations and plots.

        Use this whenever the user asks a question or requests analysis about their
        uploaded tabular data. Give a complete, self-contained description of the
        analysis to perform; the analyst will return a finished written answer.

        Args:
            request: A self-contained description of the analysis to perform, including
                any relevant context from the conversation.
            dataset_names: Optional subset of dataset file names to use (default: all).
                Analyzing the wrong file is expensive - whenever the conversation
                identifies which file(s) the user means, pass their names here.

        Returns:
            A written markdown analysis of the requested results, or a clarifying
            question to relay to the user when the target dataset is ambiguous.
        """
        if status_fn:
            status_fn("🔬 Analyzing data…")

        available = [os.path.basename(u) for u in dataset_uris]
        if not dataset_uris:
            return ("Error: no datasets are attached to this conversation. Ask the user "
                    "to attach the file (upload widget or Files dropdown) and retry.")
        selected = dataset_uris
        if dataset_names:
            wanted = {os.path.basename(n) for n in dataset_names}
            selected = [u for u in dataset_uris if os.path.basename(u) in wanted]
            if not selected:
                # deliberate hard error (no silent fall-back to all datasets): with
                # several accumulated datasets a typo'd filter must not quietly hand
                # the analyst the wrong file. The orchestrator self-corrects on retry.
                return (f"Error: dataset_names {sorted(wanted)} matched none of the "
                        f"available datasets {available}. Retry with names from that list.")

        dataset_lines = "\n".join(f"- {u}" for u in selected)
        if getattr(session, "engine", "python") == "pyspark":
            system_prompt = _SPARK_SYSTEM_PROMPT + (
                f"\n\nDATASETS AVAILABLE IN S3 (load with spark.read as shown above):\n{dataset_lines}"
            )
        else:
            system_prompt = _SYSTEM_PROMPT.format(region=region) + (
                f"\n\nDATASETS AVAILABLE IN S3 (read with boto3 as shown above):\n{dataset_lines}"
            )

        prior = message_store.get("messages", []) if message_store is not None else []
        execute_code = _make_execute_code_tool(
            session, artifact_sink, bucket=bucket, prefix=upload_prefix,
            region=region, vision=vision, status_fn=status_fn,
        )
        model = resolve_model(model_id, region, reasoning=reasoning, max_tokens=8000)
        sub_agent = Agent(
            model=model,
            agent_id="data_analysis",
            system_prompt=system_prompt,
            messages=prior,
            tools=[execute_code],
            # live workings panel if provided; the ANSWER is still relayed by the
            # orchestrator - the panel is transient UI, never part of the turn.
            callback_handler=workings_panel,
        )
        if workings_panel is not None:
            workings_panel.begin()
        try:
            result = sub_agent(request)
        except Exception:
            if workings_panel is not None:
                workings_panel.complete(error=True)
            raise
        if workings_panel is not None:
            workings_panel.complete()
        if message_store is not None:
            message_store["messages"] = sub_agent.messages
        return str(result)

    return data_analysis
