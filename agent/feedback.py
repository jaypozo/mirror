"""Feedback persistence for approve/edit/dismiss signals."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import asyncpg
from dotenv import load_dotenv

from agent.types import ThreadSummary


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
