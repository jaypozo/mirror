"""Downward (older-history) backfill for Mirror.

The forward puller (ingest.pull) uses ``min_id = sync_state.last_message_id`` and
only fetches messages *newer* than the cursor. After a recent-window pull that
cursor already points at the newest message, so ingest.pull would never reach
the older history. This module fills the *other* direction: for each dialog it
walks messages *older* than the oldest one already stored, page by page, until
the chat is exhausted.

Resumable with no extra schema: the ``messages`` table itself is the cursor.
Each page pulls messages with ``id < min(stored id for this chat)`` (or from the
newest end if the chat is empty), inserts them (idempotent via ON CONFLICT), and
the next page continues from the new, lower minimum. Kill it any time and re-run;
it never re-pulls what already landed.

Conservative pacing + FloodWait handling are inherited from the same env knobs
as ingest.pull:
  PULL_BATCH_SIZE           page size (default 100)
  PULL_BATCH_SLEEP_SECONDS  sleep between pages (default 2.0 recommended)
  PULL_CHAT_SLEEP_SECONDS   sleep between chats (default 5.0 recommended)
  BACKFILL_MAX_CHATS        cap dialogs processed (0 = all) — for tests
  BACKFILL_ONE_CHAT         only the dialog whose title/username contains this
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
from telethon import TelegramClient, errors

from ingest.pull import (
    Settings,
    ensure_session_parent,
    flush_batch,
    load_settings,
    message_row,
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


async def oldest_stored_id(conn: asyncpg.Connection, chat_id: int) -> int | None:
    return await conn.fetchval(
        "SELECT min(id) FROM messages WHERE chat_id = $1", chat_id
    )


async def backfill_dialog(
    client: TelegramClient,
    conn: asyncpg.Connection,
    dialog: Any,
    settings: Settings,
) -> int:
    chat_id = int(dialog.id)
    chat_title = getattr(dialog, "name", None) or getattr(dialog.entity, "title", None)

    # Start below the oldest message we already have; if the chat is empty,
    # offset_id=0 starts from the newest and walks backward.
    offset_id = await oldest_stored_id(conn, chat_id)
    offset_id = int(offset_id) if offset_id is not None else 0
    print(
        f"[backfill] {chat_title or chat_id}: walking older than id {offset_id or 'newest'}"
    )

    total = 0
    while True:
        rows: list[tuple[Any, ...]] = []
        try:
            async for message in client.iter_messages(
                dialog.entity,
                limit=settings.batch_size,
                offset_id=offset_id,  # strictly older than this id (0 = from newest)
                wait_time=settings.batch_sleep_seconds,
            ):
                if message.id is None:
                    continue
                rows.append(await message_row(message, chat_id, chat_title))
        except errors.FloodWaitError as exc:
            wait_seconds = int(exc.seconds) + 1
            print(f"[backfill] FloodWait {wait_seconds}s; sleeping")
            await asyncio.sleep(wait_seconds)
            continue

        if not rows:
            break  # chat exhausted

        # advance_cursor (inside flush_batch) records the *max* id — harmless for
        # forward sync, and the messages rows are what we actually resume from.
        inserted = await flush_batch(conn, rows, chat_id, chat_title)
        total += inserted
        # Next page continues below the lowest id in this page.
        offset_id = min(int(r[0]) for r in rows)
        print(f"[backfill] {chat_title or chat_id}: stored {total} older messages (next < {offset_id})")
        await asyncio.sleep(settings.batch_sleep_seconds)

    print(f"[backfill] {chat_title or chat_id}: finished, {total} older messages")
    return total


async def run() -> None:
    settings = load_settings()
    ensure_session_parent(settings.session)

    max_chats = _int_env("BACKFILL_MAX_CHATS", 0)
    one_chat = os.getenv("BACKFILL_ONE_CHAT", "").strip()

    conn = await asyncpg.connect(settings.database_url)
    client = TelegramClient(settings.session, settings.api_id, settings.api_hash)
    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise SystemExit(
                "[backfill] session is not authorized — refusing to trigger a login."
            )
        me = await client.get_me()
        print(
            f"[backfill] authenticated as id={me.id} username={getattr(me, 'username', None)}; "
            f"batch={settings.batch_size} batch_sleep={settings.batch_sleep_seconds}s "
            f"chat_sleep={settings.chat_sleep_seconds}s "
            f"started={datetime.now(timezone.utc).isoformat()}"
        )

        dialogs: list[Any] = []
        async for dialog in client.iter_dialogs():
            if one_chat:
                name = (getattr(dialog, "name", None) or "").lower()
                uname = (getattr(getattr(dialog, "entity", None), "username", None) or "").lower()
                if one_chat.lower() not in name and one_chat.lower() not in uname:
                    continue
            dialogs.append(dialog)
            if max_chats and len(dialogs) >= max_chats:
                break
        print(f"[backfill] processing {len(dialogs)} dialog(s)")

        total = 0
        for idx, dialog in enumerate(dialogs):
            total += await backfill_dialog(client, conn, dialog, settings)
            if idx < len(dialogs) - 1:
                await asyncio.sleep(settings.chat_sleep_seconds)

        print(f"[backfill] complete: stored {total} older messages")
    finally:
        await client.disconnect()
        await conn.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[backfill] interrupted; committed batches are safe to resume")


if __name__ == "__main__":
    main()
