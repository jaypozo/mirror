"""Thread-aware goal / per-thread state.

The owner runs MULTIPLE threads interleaved in ONE conversation. A flat summary
of the last N messages cannot tell which thread a message belongs to, nor track
where in a task we are. This module segments each incoming message to the
best-matching active thread (or opens a new one) and maintains per-thread
`goal`, `current_task`, and `stage`, so the Goal/Now/Next summary and the draft
reflect the RIGHT thread's objective — not a linear window.

Design (pragmatic, one extra LLM call per draft — it REPLACES the flat summary
call, so drafting stays at ~one exec for the reply + one for the state/summary):

  1. Embed the incoming message locally (the same on-box model retrieval uses).
  2. Pre-rank candidate threads by cosine distance to their stored
     `anchor_embedding` (falling back to most-recently-updated), so the labeler
     only sees a short, relevant candidate list.
  3. ONE LLM call does segmentation + state update + summary together: given the
     incoming message, recent thread, the candidate threads, and the matched
     thread's recent intent notes (decisions), it returns which thread this is
     (or a new one) AND the refreshed goal/current_task/stage AND a thread-aware
     goal/now/next/open.
  4. Upsert the thread (new row or update the matched one), recomputing its
     anchor embedding.

Everything is guarded: on any failure `resolve_thread` returns None and the
caller falls back to the flat `summarize_thread`, so drafting always works.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

import asyncpg
from dotenv import load_dotenv

from agent.llm import LLMClient, build_llm_client
from agent.types import ChatMessage, ThreadSummary

VALID_STAGES = {"mid-step", "awaiting-owner", "done"}

# How many candidate threads to show the labeler, and how many recent intent
# notes to feed the state update / draft.
MAX_CANDIDATES = 10
MAX_INTENT_NOTES = 6


@dataclass(frozen=True)
class ThreadState:
    id: int
    title: str
    goal: str
    current_task: str
    stage: str
    # goal/now/next/open the labeler produced for THIS message (thread-aware).
    summary: ThreadSummary = field(
        default_factory=lambda: ThreadSummary.empty()
    )

    def to_thread_summary(self) -> ThreadSummary:
        return self.summary


@dataclass(frozen=True)
class ThreadContext:
    state: ThreadState
    intent_notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Embedding helpers (reuse the on-box retrieval encoder).
# --------------------------------------------------------------------------- #
async def _embed(text: str) -> list[float] | None:
    try:
        from agent.retrieve import embed_query

        vec = await embed_query(text)
        return list(vec) if vec else None
    except Exception as exc:  # pragma: no cover - embedding is best-effort
        print(f"[threads] embed failed (ignored): {exc}", file=sys.stderr)
        return None


def _vector_literal(vec: list[float] | None) -> str | None:
    if not vec:
        return None
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


# --------------------------------------------------------------------------- #
# DB helpers.
# --------------------------------------------------------------------------- #
async def _connect() -> asyncpg.Connection | None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    try:
        return await asyncpg.connect(database_url)
    except Exception as exc:
        print(f"[threads] connect failed (ignored): {exc}", file=sys.stderr)
        return None


async def _fetch_candidates(
    conn: asyncpg.Connection, query_vec: str | None, limit: int
) -> list[dict]:
    """Candidate threads for the labeler: nearest by anchor embedding when we
    have a query vector, else the most-recently-updated. Done threads are
    included (a message can reopen one) but rank last by recency."""
    if query_vec is not None:
        rows = await conn.fetch(
            """
            SELECT id, title, goal, current_task, stage
            FROM threads
            ORDER BY anchor_embedding <=> $1::vector NULLS LAST, updated_at DESC
            LIMIT $2
            """,
            query_vec,
            limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id, title, goal, current_task, stage
            FROM threads
            ORDER BY updated_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def _recent_intent_notes(
    conn: asyncpg.Connection, thread_id: int, limit: int = MAX_INTENT_NOTES
) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT note
        FROM intent_notes
        WHERE thread_id = $1 AND note IS NOT NULL AND length(trim(note)) > 0
        ORDER BY ts DESC
        LIMIT $2
        """,
        thread_id,
        limit,
    )
    return [r["note"].strip() for r in rows]


# --------------------------------------------------------------------------- #
# LLM segmentation + state update + summary (one call).
# --------------------------------------------------------------------------- #
_SEGMENT_SYSTEM_PROMPT = """You track the owner's interleaved threads in a single ongoing conversation.

You are given the incoming message, a little recent thread, a list of the owner's KNOWN active threads (with id, title, goal, current task, stage), and recent DECISIONS the owner has locked in on the best-matching thread. Decide which thread the incoming message belongs to, then refresh that thread's state and summarize it.

Return ONLY compact JSON, no prose:
{
  "match_id": <the id of the matching known thread, or null if this starts a NEW thread>,
  "title": "<short thread title>",
  "goal": "<the thread's overall objective, one line>",
  "current_task": "<the specific task in flight right now>",
  "stage": "mid-step" | "awaiting-owner" | "done",
  "keywords": ["<a few distinctive keywords/anchors for this thread>"],
  "now": "<current state or blocker for this thread>",
  "next": "<the next concrete action the owner should take>",
  "open": ["<unresolved question or missing fact>", "..."]
}

Rules:
- Prefer matching an existing thread when the message plausibly continues it; only open a new thread when it clearly does not fit any.
- stage: "mid-step" = actively in progress; "awaiting-owner" = blocked on the owner's input/decision; "done" = the task is complete.
- Honor the locked-in decisions: goal/current_task/now/next must be consistent with them, not with a stale earlier plan.
- Keep every field tight. Do not invent facts the messages do not support."""


def _thread_block(thread: list[ChatMessage], limit: int = 12) -> str:
    if not thread:
        return "(no prior thread)"
    return "\n".join(m.to_prompt_line() for m in thread[-limit:])


def _candidates_block(candidates: list[dict]) -> str:
    if not candidates:
        return "(no known threads yet)"
    lines = []
    for c in candidates:
        lines.append(
            f"- id {c['id']}: {c.get('title') or '?'} "
            f"[stage={c.get('stage')}] goal: {c.get('goal') or '?'} "
            f"| current task: {c.get('current_task') or '?'}"
        )
    return "\n".join(lines)


def _build_segment_prompt(
    incoming: str,
    thread: list[ChatMessage],
    candidates: list[dict],
    intent_notes: list[str],
) -> str:
    notes_block = (
        "\n".join(f"- {n}" for n in intent_notes) if intent_notes else "(none yet)"
    )
    return f"""INCOMING MESSAGE:
{incoming}

RECENT THREAD:
{_thread_block(thread)}

KNOWN ACTIVE THREADS:
{_candidates_block(candidates)}

RECENT LOCKED-IN DECISIONS (for the best-matching known thread):
{notes_block}

Return only the JSON object described in the instructions."""


def _parse_decision(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_stage(value: object) -> str:
    stage = str(value or "").strip().lower()
    return stage if stage in VALID_STAGES else "mid-step"


def _summary_from_decision(decision: dict) -> ThreadSummary:
    open_items = decision.get("open", [])
    if isinstance(open_items, str):
        open_items = [open_items]
    goal = str(decision.get("goal") or "").strip() or "Advance the current thread."
    now = str(decision.get("now") or "").strip() or "Review the latest message."
    nxt = str(decision.get("next") or "").strip() or "Take the next concrete step."
    return ThreadSummary(
        goal=goal,
        now=now,
        next=nxt,
        open=[str(i).strip() for i in open_items if str(i).strip()],
    )


async def _upsert_thread(
    conn: asyncpg.Connection,
    decision: dict,
    candidate_ids: set[int],
) -> ThreadState | None:
    title = str(decision.get("title") or "").strip() or "Untitled thread"
    goal = str(decision.get("goal") or "").strip()
    current_task = str(decision.get("current_task") or "").strip()
    stage = _coerce_stage(decision.get("stage"))
    keywords = decision.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = [str(keywords)]
    keywords = [str(k).strip() for k in keywords if str(k).strip()][:12]
    anchors = {"keywords": keywords}

    # Anchor embedding from the thread's distinctive text (title + goal + keys).
    anchor_text = " ".join([title, goal, " ".join(keywords)]).strip()
    anchor_vec = _vector_literal(await _embed(anchor_text)) if anchor_text else None

    raw_match = decision.get("match_id")
    match_id = None
    try:
        if raw_match is not None:
            match_id = int(raw_match)
    except (TypeError, ValueError):
        match_id = None
    if match_id is not None and match_id not in candidate_ids:
        match_id = None  # only trust an id we actually showed the labeler

    if match_id is not None:
        row = await conn.fetchrow(
            """
            UPDATE threads
               SET title = $2, goal = $3, current_task = $4, stage = $5,
                   anchors = $6::jsonb,
                   anchor_embedding = COALESCE($7::vector, anchor_embedding),
                   updated_at = now()
             WHERE id = $1
            RETURNING id, title, goal, current_task, stage
            """,
            match_id,
            title,
            goal,
            current_task,
            stage,
            json.dumps(anchors),
            anchor_vec,
        )
    else:
        row = await conn.fetchrow(
            """
            INSERT INTO threads
                (title, goal, current_task, stage, anchors, anchor_embedding)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector)
            RETURNING id, title, goal, current_task, stage
            """,
            title,
            goal,
            current_task,
            stage,
            json.dumps(anchors),
            anchor_vec,
        )
    if row is None:
        return None
    return ThreadState(
        id=row["id"],
        title=row["title"],
        goal=row["goal"] or "",
        current_task=row["current_task"] or "",
        stage=row["stage"],
        summary=_summary_from_decision(decision),
    )


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
async def resolve_thread(
    incoming_message: str,
    thread: list[ChatMessage] | None = None,
    llm: LLMClient | None = None,
) -> ThreadContext | None:
    """Segment `incoming_message` to a thread (matching or new), update that
    thread's state, and return it with its recent intent notes. Fully guarded:
    returns None on any failure so the caller falls back to the flat summary."""
    incoming = (incoming_message or "").strip()
    if not incoming:
        return None
    thread = thread or []

    conn = await _connect()
    if conn is None:
        return None
    try:
        tail = " ".join(m.text for m in thread[-3:] if m.text)
        query_vec = _vector_literal(await _embed(f"{incoming}\n{tail}".strip()))
        candidates = await _fetch_candidates(conn, query_vec, MAX_CANDIDATES)
        candidate_ids = {int(c["id"]) for c in candidates}

        # Best candidate's recent decisions steer the state update (and the draft).
        pre_notes: list[str] = []
        if candidates:
            try:
                pre_notes = await _recent_intent_notes(conn, int(candidates[0]["id"]))
            except Exception:
                pre_notes = []

        client = llm or build_llm_client()
        try:
            raw = await client.complete(
                [
                    {"role": "system", "content": _SEGMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_segment_prompt(
                            incoming, thread, candidates, pre_notes
                        ),
                    },
                ]
            )
        except Exception as exc:
            print(f"[threads] segmentation LLM failed (ignored): {exc}", file=sys.stderr)
            return None

        decision = _parse_decision(raw)
        if not decision:
            return None

        state = await _upsert_thread(conn, decision, candidate_ids)
        if state is None:
            return None

        # Intent notes to inject into the draft: the matched thread's own recent
        # decisions (re-fetch in case we just matched a thread we didn't pre-load).
        try:
            notes = await _recent_intent_notes(conn, state.id)
        except Exception:
            notes = pre_notes
        return ThreadContext(state=state, intent_notes=notes)
    except Exception as exc:
        print(f"[threads] resolve_thread failed (ignored): {exc}", file=sys.stderr)
        return None
    finally:
        await conn.close()


async def record_intent_note(
    *,
    note: str,
    thread_id: int | None = None,
    feedback_id: int | None = None,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
) -> None:
    """Persist one decision/intent captured from an INTENT/BOTH edit. Guarded:
    any failure logs and is ignored so it never blocks capture or sending."""
    note = (note or "").strip()
    if not note:
        return
    conn = await _connect()
    if conn is None:
        return
    try:
        await conn.execute(
            """
            INSERT INTO intent_notes
                (thread_id, feedback_id, note, source_chat_id, source_message_id)
            VALUES ($1, $2, $3, $4, $5)
            """,
            thread_id,
            feedback_id,
            note,
            source_chat_id,
            source_message_id,
        )
    except Exception as exc:
        print(f"[threads] record_intent_note failed (ignored): {exc}", file=sys.stderr)
    finally:
        await conn.close()
