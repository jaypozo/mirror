"""Draft replies as the owner using local RAG over their own past replies.

Pipeline (all local + on-box):
  1. Embed the incoming message with the local sentence-transformers model.
  2. Retrieve top-K of the owner's OWN past replies (direction='out') plus the
     messages they were answering — real few-shot examples of their voice.
  3. Build a system+user prompt and call GPT-5.5 once via `codex exec`
     (fresh, stateless — no session maintained between drafts).
  4. Return the drafted reply text + the examples used.
"""

from __future__ import annotations

import asyncio
import json
import sys

from agent.feedback import EditCorrection, fetch_recent_edits
from agent.llm import LLMClient, build_llm_client
from agent.threads import ThreadContext, resolve_thread
from agent.retrieve import RetrievedExample, retrieve_examples
from agent.style_sheet import get_active_style_sheet
from agent.summarize import summarize_thread
from agent.types import Brief, DraftRequest, DraftResult


DRAFT_SYSTEM_PROMPT = """You are drafting the owner's reply to an incoming Telegram message, in their voice and style.

Below are REAL examples of how the owner actually replies (each shows the message(s) they were answering, then their reply). Study them and match:
- their tone (direct, pragmatic, plainspoken, no hype, no filler)
- their length (usually short; they do not pad)
- their directness (leads with the answer/outcome, says plainly when a fact is missing)
- their punctuation and casing habits

Rules:
- Write ONLY the reply text the owner would send. No preamble, no quotes, no "Here's a draft".
- Do not mention you are an AI or that this is a draft.
- Do not invent facts. If something is unknown, ask for it or state the constraint plainly, the way the owner does.
- Do not copy an example verbatim; write a fresh reply that fits THIS message in their style."""


def format_examples(examples: list[RetrievedExample]) -> str:
    if not examples:
        return "(No retrieved examples available. Write concisely and directly, as the owner would.)"
    blocks = []
    for index, example in enumerate(examples, 1):
        blocks.append(f"Example {index}:\n{example.format_for_prompt()}")
    return "\n\n".join(blocks)


def format_corrections(corrections: list[EditCorrection]) -> str:
    """Render past owner edits as explicit correction examples: what was drafted
    vs. what the owner actually sent. Empty string when there are none."""
    if not corrections:
        return ""
    blocks = []
    for index, c in enumerate(corrections, 1):
        goal = str((c.summary or {}).get("goal") or "").strip()
        context = f" (context: {goal})" if goal else ""
        blocks.append(
            f"Correction {index}{context}:\n"
            f"You previously drafted: {c.original_draft}\n"
            f"The owner corrected it to: {c.final_text}"
        )
    body = "\n\n".join(blocks)
    return (
        "\nThe owner has corrected past drafts. Learn from these — match the "
        "voice, length, and decisions of the CORRECTED version, not the original:\n"
        f"{body}\n"
    )


async def _load_corrections(limit: int = 8) -> list[EditCorrection]:
    """Fetch recent owner edits for the prompt. Guarded: any failure yields no
    examples so a DB hiccup never breaks drafting."""
    try:
        return await fetch_recent_edits(limit=limit)
    except Exception as exc:
        print(f"[draft] could not load edit corrections (ignored): {exc}", file=sys.stderr)
        return []


async def _load_style_sheet() -> str:
    """Fetch the active living style sheet. Guarded: any failure yields '' so a
    missing/failed style sheet never breaks drafting."""
    try:
        return (await get_active_style_sheet()) or ""
    except Exception as exc:
        print(f"[draft] could not load style sheet (ignored): {exc}", file=sys.stderr)
        return ""


def format_style_sheet(style_sheet: str) -> str:
    """Render the living style sheet as a prompt block. Empty when absent."""
    guide = (style_sheet or "").strip()
    if not guide:
        return ""
    return (
        "\nThe owner's living style sheet (distilled from their whole message "
        "history and refined by their edits) — follow it:\n"
        f"{guide}\n"
    )


async def _resolve_thread(request: DraftRequest) -> ThreadContext | None:
    """Segment the incoming message to a thread and refresh its state. Guarded:
    any failure yields None so drafting falls back to the flat summary."""
    try:
        return await resolve_thread(request.incoming_message, request.thread)
    except Exception as exc:
        print(f"[draft] thread segmentation failed (ignored): {exc}", file=sys.stderr)
        return None


def format_thread_context(context: ThreadContext | None) -> str:
    """Render the matched thread's state + recent decisions as a prompt block so
    the draft understands the high-level objective and where in the task we are.
    Empty when there's no matched thread."""
    if context is None:
        return ""
    state = context.state
    lines = [
        "\nThe message belongs to this ongoing thread — draft with its objective "
        "and current stage in mind:",
        f"- Thread: {state.title}",
    ]
    if state.goal:
        lines.append(f"- Goal: {state.goal}")
    if state.current_task:
        lines.append(f"- Current task: {state.current_task}")
    lines.append(f"- Stage: {state.stage}")
    if context.intent_notes:
        lines.append("- Decisions the owner has already locked in (honor these):")
        lines.extend(f"  - {note}" for note in context.intent_notes)
    return "\n".join(lines) + "\n"


def heuristic_draft(request: DraftRequest) -> str:
    if "?" in request.incoming_message:
        return "Let me check one detail and get back to you."
    return "Got it, on it."


async def draft_reply(
    request: DraftRequest,
    summary: Brief | None = None,
    llm: LLMClient | None = None,
    top_k: int | None = None,
    include_summary: bool = False,
) -> DraftResult:
    # Retrieval query: the incoming message (plus thread tail if present) is the
    # semantic anchor for finding how the owner replied to similar things.
    query_text = request.incoming_message
    if request.thread:
        tail = " ".join(m.text for m in request.thread[-3:] if m.text)
        query_text = f"{request.incoming_message}\n{tail}".strip()

    examples = await retrieve_examples(
        query_text=query_text,
        top_k=top_k,
        context_tag=request.context_tag,
    )

    # Learning signal: recent cases where the owner edited a draft before sending.
    corrections = await _load_corrections()

    # Living style sheet: the always-injected, distilled guide to how the owner
    # writes (primary signal = their whole corpus, refined by edits).
    style_sheet = await _load_style_sheet()

    # Thread-aware state (Build 2). When summaries are enabled we segment the
    # message to the owner's matching thread, refresh its goal/current_task/stage,
    # and use that as the Goal/Now/Next summary — replacing the flat window
    # summarize call (so it's the same cost, not an extra one). The matched
    # thread's recent decisions (intent notes) are injected into the draft. An
    # agent-supplied summary is still honored. Falls back to the flat summary if
    # segmentation is off or fails.
    resolved_summary = summary
    thread_context: ThreadContext | None = None
    if include_summary:
        thread_context = await _resolve_thread(request)
        if thread_context is not None and resolved_summary is None:
            resolved_summary = thread_context.state.to_thread_summary()
        if resolved_summary is None and request.thread:
            resolved_summary = await summarize_thread(request.thread)

    thread_block = ""
    if request.thread:
        transcript = "\n".join(m.to_prompt_line() for m in request.thread[-40:])
        thread_block = f"\nCurrent thread so far:\n{transcript}\n"

    summary_block = ""
    if resolved_summary is not None:
        summary_block = f"\nThread summary:\n{resolved_summary.format_for_telegram()}\n"

    corrections_block = format_corrections(corrections)
    style_block = format_style_sheet(style_sheet)
    thread_context_block = format_thread_context(thread_context)

    user_prompt = f"""{style_block}{thread_context_block}Here are real examples of how the owner replies:

{format_examples(examples)}
{corrections_block}{thread_block}{summary_block}
Now draft the owner's reply to this incoming message:
{request.incoming_message}

Reply as the owner. Output only the reply text."""

    client = llm or build_llm_client()
    try:
        response = await client.complete(
            [
                {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        print(f"[draft] LLM draft failed; using heuristic fallback: {exc}", file=sys.stderr)
        response = heuristic_draft(request)

    draft = response.strip() or heuristic_draft(request)
    return DraftResult(
        draft=draft,
        summary=resolved_summary or Brief.empty(),
        style_examples=[ex.reply_text for ex in examples],
        thread_id=thread_context.state.id if thread_context else None,
    )


async def main() -> None:
    payload = json.load(sys.stdin)
    request = DraftRequest.from_dict(payload)
    result = await draft_reply(request)
    print(
        json.dumps(
            {
                "summary": result.summary.to_dict(),
                "draft": result.draft,
                "style_examples": result.style_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
