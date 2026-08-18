"""Document-generator sub-agent (agent-as-tool) for the Strands chat orchestrator.

Generates Word/PowerPoint/PDF/Excel files by writing and running Python code in an
AgentCore Code Interpreter sandbox, reusing the data-analysis module's session
manager, execute_code tool and [[ARTIFACT:document:...]] sentinel pipeline.

Topology:
    orchestrator (persisted chat agent)
      └─ tool: document_generator(request, filenames?)   [this module]
            └─ ephemeral sub-agent (own model)
                  └─ tool: execute_code(code)            [AgentCore Code Interpreter]

Generated files land in the turn's S3 prefix via the shared artifact flow (sandbox
download → S3 upload); their uris accumulate in the sink's "doc_output" list, persist
in turn_meta, and render in the UI's "artifacts" expander as presigned links (minted
at render time, so every Streamlit rerun re-signs them - no refresh button needed).

Doc libraries verified in the interpreter (custom image test-y20M8PrsxO):
python-docx 1.1.2, python-pptx 1.0.2, reportlab 4.2.0, fpdf 1.7.2, pypdf 6.2.0,
xlsxwriter 3.2.0, openpyxl, PIL 10.3.0. markdown/weasyprint are MISSING.
"""
import os
from typing import Callable, Optional

from strands import Agent, tool

from .chat_agent import resolve_model
from .data_analysis import _make_execute_code_tool

_SYSTEM_PROMPT = """You are a document generation and editing agent. You produce \
polished, professional files (Word, PowerPoint, PDF, Excel) - and modify existing \
ones - by writing and running Python code in a sandboxed code interpreter.

EXECUTION ENVIRONMENT (read carefully):
- You have one tool, execute_code(code), running a STATEFUL IPython kernel. Variables \
and files PERSIST across calls, so build documents incrementally and fix errors \
without starting over.
- Pre-installed document libraries (use these exact ones):
    Word:       python-docx 1.1.2   (from docx import Document)
    PowerPoint: python-pptx 1.0.2   (from pptx import Presentation)
    PDF:        reportlab 4.2.0 (preferred) or fpdf; pypdf for merging/reading
    Excel:      xlsxwriter / openpyxl
    Charts/images to embed: matplotlib, plotly (write_image is NOT available - render \
charts with matplotlib for embedding), PIL
  NOT installed: weasyprint, markdown, pandoc. You CANNOT pip install anything.
- If source material lives in S3 (s3:// uris), read it with boto3 into a buffer, e.g.:
    import boto3, io
    obj = boto3.client("s3", region_name="{region}").get_object(Bucket=BKT, Key=KEY)
  Always pass region_name="{region}" (other endpoints are unreachable).

EDITING EXISTING FILES:
- To edit a document, download it from its s3:// uri into the sandbox first:
    obj = boto3.client("s3", region_name="{region}").get_object(Bucket=BKT, Key=KEY)
    with open("report.docx", "wb") as f: f.write(obj["Body"].read())
  then open it with the matching library (Document/Presentation/PdfReader), apply the \
requested changes, and save. Preserve everything not asked to change - styles, \
ordering, untouched sections.
- Save the edited file under its ORIGINAL basename unless the user asks for a copy or \
a different name.
- PDFs are edit-hostile: pypdf can merge/split/rotate/watermark pages but not reliably \
restyle text. For content rewrites of a PDF, rebuild it with reportlab and say so in \
your answer.

DELIVERING FILES (required, or the user receives nothing):
- Save every deliverable to a RELATIVE filename with the proper extension (e.g. \
"report.docx"). Absolute paths like /tmp/x.docx are REJECTED.
- Immediately AFTER saving each deliverable, print its sentinel on its own line:
    print("[[ARTIFACT:document:report.docx]]")
  Only sentinel the FINAL deliverables - not intermediate scratch files.
- Before the sentinel, VERIFY the file structurally: re-open it (Document(...), \
Presentation(...), pypdf.PdfReader(...)) and print a short outline (headings/slide \
titles/page count) to confirm the content is really there.

QUALITY:
- Make documents genuinely presentation-ready: title pages/slides, headings, \
consistent styles, tables where tabular, charts embedded as images where they help.
- Follow the request's content faithfully; do not pad with filler.

ANSWER:
- When done, reply with a short markdown summary: each file produced, its purpose, \
and an outline of its contents. Do not paste the full document text."""


def make_document_generator_tool(*, model_id: str, region: str, bucket: str,
                                 upload_prefix: str, session,
                                 artifact_sink: dict, message_store: Optional[dict] = None,
                                 available_uris: Optional[list] = None,
                                 generated_uris: Optional[list] = None,
                                 reasoning: bool = True,
                                 status_fn: Optional[Callable[[str], None]] = None,
                                 workings_panel=None):
    """Build the document_generator tool the orchestrator calls.

    Mirrors make_data_analysis_tool: session is a CodeSessionManager (its own, cached
    per chat session under the "docgen" engine key - kept separate from the analysis
    kernel so neither prompt/context contaminates the other); artifact_sink accumulates
    doc_output/image_output/plotly uris for turn_meta; message_store keeps sub-agent
    continuity across turns ("now add a slide on X" reuses the built document).

    available_uris: attachments still visible in the conversation window (shared, by
        reference, with the data-analysis tool) - always offered as source material so
        the generator can read them even if the orchestrator forgets to pass them.
    generated_uris: previously generated artifacts still referenced in the window
        (from sessions.generated_uris_in_window) - offered as editable outputs, so
        "add two slides to the deck" works even after an app restart wiped the
        sub-agent's message_store and the sandbox kernel.
    """

    @tool
    def document_generator(request: str, source_uris: Optional[list] = None,
                           artifact_names: Optional[list] = None) -> str:
        """Generate or edit document files - Word (.docx), PowerPoint (.pptx), PDF,
        Excel - and deliver them to the user as downloadable artifacts.

        Use this whenever the user asks for a document, report, deck/presentation,
        one-pager or spreadsheet FILE, or asks to modify one (attached or previously
        generated). Give a complete, self-contained brief: the document type(s), the
        full content or where to source it, the exact changes for an edit, and any
        structure, tone or branding requirements. The generator returns a summary of
        what was produced; the files themselves are delivered to the user
        automatically.

        Args:
            request: A self-contained brief for the document(s) to produce or the
                edits to apply, including all content and any relevant context from
                the conversation.
            source_uris: Optional list of s3:// uris of source files the document
                should draw from.
            artifact_names: Optional file names of previously generated artifacts this
                request targets (e.g. ["report.docx"]). Editing the wrong file wastes
                a run - whenever the conversation identifies which artifact the user
                means, pass its name here.

        Returns:
            A short markdown summary of the file(s) generated/edited and their
            contents, ending with a machine-readable [generated artifacts: ...] line.
        """
        if status_fn:
            status_fn("📄 Generating document…")

        system_prompt = _SYSTEM_PROMPT.format(region=region)
        sources = list(available_uris or [])
        for u in (source_uris or []):  # model-supplied extras
            if u not in sources:
                sources.append(u)
        if sources:
            lines = "\n".join(f"- {u}" for u in sources)
            system_prompt += (f"\n\nATTACHED SOURCE MATERIAL IN S3 (read with boto3 as "
                              f"shown above; treat as inputs - if asked to edit one, "
                              f"save the result as a new artifact):\n{lines}")

        prior_artifacts = list(generated_uris or [])
        if artifact_names:
            wanted = {os.path.basename(n) for n in artifact_names}
            available = [os.path.basename(u) for u in prior_artifacts]
            prior_artifacts = [u for u in prior_artifacts
                               if os.path.basename(u) in wanted]
            if not prior_artifacts:
                # strict, like data_analysis.dataset_names: never guess which file
                return (f"Error: artifact_names {sorted(wanted)} matched none of the "
                        f"previously generated artifacts {available}. Retry with names "
                        f"from that list, or omit artifact_names.")
        if prior_artifacts:
            lines = "\n".join(f"- {u}" for u in prior_artifacts)
            system_prompt += (f"\n\nPREVIOUSLY GENERATED ARTIFACTS (your earlier outputs; "
                              f"download from S3 to edit or extend - the sandbox may "
                              f"have been reset):\n{lines}")

        prior = message_store.get("messages", []) if message_store is not None else []
        docs_before = len(artifact_sink.get("doc_output") or [])
        execute_code = _make_execute_code_tool(
            session, artifact_sink, bucket=bucket, prefix=upload_prefix,
            region=region, vision=False, status_fn=status_fn,
        )
        model = resolve_model(model_id, region, reasoning=reasoning, max_tokens=8000)
        sub_agent = Agent(
            model=model,
            agent_id="document_generator",
            system_prompt=system_prompt,
            messages=prior,
            tools=[execute_code],
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
        answer = str(result)
        # Deterministic artifact marker, appended OUTSIDE the model from what the
        # sentinel pipeline actually captured this call. It persists in the
        # orchestrator's toolResult, so artifact names survive restarts word-for-word
        # (sessions.generated_uris_in_window matches them back to turn_meta uris).
        new_docs = (artifact_sink.get("doc_output") or [])[docs_before:]
        if new_docs:
            names = ", ".join(os.path.basename(u) for u in new_docs)
            answer += f"\n\n[generated artifacts: {names}]"
        return answer

    return document_generator
