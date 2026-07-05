"""Feedback persistence for approve/edit/dismiss signals."""

from __future__ import annotations

import difflib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import asyncpg
from dotenv import load_dotenv

from agent.types import ThreadSummary


@dataclass(frozen=True)
class EditCorrection:
    """One past case where the owner corrected a draft before sending."""

    original_draft: str
    final_text: str
    summary: dict[str, Any]


async def fetch_recent_edits(limit: int = 8, pool: int = 40) -> list[EditCorrection]:
    """Return up to `limit` action='edit' rows (owner corrected the draft before
    sending) as voice-correction examples for drafting, ranked by BOTH recency
    and edit magnitude so the strongest, freshest corrections win.

    We over-fetch the `pool` most-recent edits, then score each by a recency
    weight (newest ranks highest, decaying by position) times an edit-magnitude
    weight (a bigger rewrite is a stronger signal than a one-word tweak, measured
    by normalized difflib distance). The top `limit` are returned — capped so the
    corrections never crowd out the retrieved exemplars or the style sheet.

    Best-effort: any DB hiccup returns [] so drafting is never blocked. Long text
    is truncated to keep the prompt cheap.
    """
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return []

    limit = max(1, int(limit))
    pool = max(limit, int(pool))

    def _trunc(value: str | None, cap: int = 600) -> str:
        text = (value or "").strip()
        return text if len(text) <= cap else text[:cap].rstrip() + "…"

    try:
        conn = await asyncpg.connect(database_url)
    except Exception as exc:
        print(f"[feedback] fetch_recent_edits connect failed: {exc}", file=sys.stderr)
        return []
    try:
        rows = await conn.fetch(
            """
            SELECT original_draft, final_text, summary
            FROM feedback
            WHERE action = 'edit'
              AND final_text IS NOT NULL
              AND length(trim(final_text)) > 0
              AND original_draft IS NOT NULL
              AND final_text <> original_draft
            ORDER BY ts DESC
            LIMIT $1
            """,
            pool,
        )
    except Exception as exc:
        print(f"[feedback] fetch_recent_edits query failed: {exc}", file=sys.stderr)
        return []
    finally:
        await conn.close()

    # Rank the candidate pool by recency (rows arrive newest-first) * magnitude.
    scored: list[tuple[float, EditCorrection]] = []
    for rank, r in enumerate(rows):
        summary = r["summary"]
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except Exception:
                summary = {}
        original = r["original_draft"] or ""
        final = r["final_text"] or ""
        # Normalized edit distance in [0, 1]: 0 = identical, 1 = total rewrite.
        magnitude = 1.0 - difflib.SequenceMatcher(None, original, final).ratio()
        # Geometric recency decay; floor keeps magnitude meaningful for big edits.
        recency = 0.85 ** rank
        score = recency * (0.5 + magnitude)
        scored.append(
            (
                score,
                EditCorrection(
                    original_draft=_trunc(original),
                    final_text=_trunc(final),
                    summary=summary if isinstance(summary, dict) else {},
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [correction for _, correction in scored[:limit]]


async def record_feedback(
    *,
    original_draft: str,
    final_text: str | None,
    action: str,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
    target_chat_id: int | None = None,
    target_thread_id: int | None = None,
    summary: ThreadSummary | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[feedback] DATABASE_URL missing; feedback was not persisted", file=sys.stderr)
        return

    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(
            """
            INSERT INTO feedback (
                original_draft, final_text, action,
                source_chat_id, source_message_id,
                target_chat_id, target_thread_id,
                summary, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb)
            """,
            original_draft,
            final_text,
            action,
            source_chat_id,
            source_message_id,
            target_chat_id,
            target_thread_id,
            json.dumps(summary.to_dict() if summary else {}),
            json.dumps(metadata or {}),
        )
    except Exception as exc:
        print(f"[feedback] failed to persist feedback: {exc}", file=sys.stderr)
    finally:
        await conn.close()
