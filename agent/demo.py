"""End-to-end demo of the Mirror draft-my-reply pipeline.

Picks a real inbound message from the corpus (or one you pass by
chat_id/message_id), pulls its live thread context, retrieves the owner's most
similar past replies, drafts a reply AS the owner via GPT-5.5 (codex exec), and prints
the retrieved examples + the generated draft.

Usage:
    python -m agent.demo                         # auto-pick a recent inbound question
    python -m agent.demo <chat_id> <message_id>  # target a specific message
"""

from __future__ import annotations

import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv

from agent.draft import draft_reply
from agent.types import ChatMessage, DraftRequest


async def _pick_target(conn: asyncpg.Connection) -> tuple[int, int]:
    row = await conn.fetchrow(
        """
        SELECT chat_id, id
        FROM messages
        WHERE direction = 'in'
          AND text IS NOT NULL
          AND length(trim(text)) BETWEEN 30 AND 400
          AND text ~ '\\?'
          AND sender_name NOT ILIKE '%bot%'
          AND sender_name NOT ILIKE '%| Code%'
          AND sender_name NOT ILIKE '%| Ops%'
          -- only chats where the owner has actually replied, so context is real
          AND EXISTS (
              SELECT 1 FROM messages o
              WHERE o.chat_id = messages.chat_id AND o.direction = 'out'
          )
        ORDER BY ts DESC
        LIMIT 1
        """
    )
    if not row:
        raise SystemExit("No suitable inbound message found in the corpus.")
    return row["chat_id"], row["id"]


async def _load_thread(
    conn: asyncpg.Connection, chat_id: int, message_id: int, window: int = 8
) -> tuple[str, list[ChatMessage], str | None]:
    target = await conn.fetchrow(
        "SELECT chat_id, id, chat_title, sender_name, text FROM messages WHERE chat_id=$1 AND id=$2",
        chat_id,
        message_id,
    )
    if not target:
        raise SystemExit(f"Message {chat_id}/{message_id} not found.")

    rows = await conn.fetch(
        """
        SELECT direction, sender_name, text, id, ts
        FROM messages
        WHERE chat_id = $1
          AND id <= $2
          AND text IS NOT NULL
          AND length(trim(text)) > 0
        ORDER BY id DESC
        LIMIT $3
        """,
        chat_id,
        message_id,
        window,
    )
    rows = list(reversed(rows))
    thread = [
        ChatMessage(
            text=r["text"],
            sender_name=("the owner" if r["direction"] == "out" else r["sender_name"]),
            direction=r["direction"],
            message_id=r["id"],
            ts=r["ts"],
        )
        for r in rows
    ]
    return target["text"], thread, target["chat_title"]


async def run() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(database_url)
    try:
        if len(sys.argv) >= 3:
            chat_id, message_id = int(sys.argv[1]), int(sys.argv[2])
        else:
            chat_id, message_id = await _pick_target(conn)

        incoming, thread, chat_title = await _load_thread(conn, chat_id, message_id)
    finally:
        await conn.close()

    print("=" * 70)
    print(f"TARGET INBOUND MESSAGE  (chat={chat_title!r}, chat_id={chat_id}, id={message_id})")
    print("=" * 70)
    print(incoming)
    print()
    print("THREAD CONTEXT (most recent, oldest first):")
    for m in thread:
        print(f"  {m.sender_name}: {m.text[:120]}")
    print()

    request = DraftRequest(
        incoming_message=incoming,
        thread=thread,
        source_chat_id=chat_id,
        source_message_id=message_id,
    )

    print("Retrieving the owner's similar past replies + drafting via GPT-5.5 (codex exec)...\n")
    try:
        result = await draft_reply(request)
    except Exception as exc:
        print(f"[demo] draft failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print("=" * 70)
    print("RETRIEVED EXAMPLES OF THE OWNER'S VOICE (top-K):")
    print("=" * 70)
    for i, ex in enumerate(result.style_examples, 1):
        print(f"[{i}] {ex[:200]}")
        print()

    print("=" * 70)
    print("GENERATED DRAFT (reply as the owner):")
    print("=" * 70)
    print(result.draft)
    print()


if __name__ == "__main__":
    asyncio.run(run())
