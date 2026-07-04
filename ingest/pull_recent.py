"""Windowed, takeout-based recent pull for Mirror.

Pulls only the last PULL_SINCE_HOURS of messages, newest-first per dialog,
stopping as soon as a message is older than the cutoff. Reuses pull.py's row
mapping, batching, and sync_state cursor so it is fully resumable and never
re-pulls already-stored messages.

Safety-first: runs inside a Telegram *takeout* session (official export mode,
lower flood limits) and uses conservative pacing. Handles TakeoutInitDelayError
and FloodWaitError by sleeping rather than pushing through.

Env knobs (all optional, sensible conservative defaults):
  PULL_SINCE_HOURS        window size in hours (default 24)
  PULL_MAX_CHATS          cap number of dialogs processed (0 = all) — for tests
  PULL_ONE_CHAT           only process the dialog whose title/username contains
                          this substring (case-insensitive) — for the smoke test
  PULL_RECENT_LIMIT       hard cap on messages fetched per dialog (0 = no cap)
  PULL_USE_TAKEOUT        1 (default) to use takeout; 0 to use the plain client
  PULL_BATCH_SIZE         batch size for DB flush (default 100)
  PULL_BATCH_SLEEP_SECONDS  sleep between message batches (default 2.0)
  PULL_CHAT_SLEEP_SECONDS   sleep between chats (default 4.0)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
from telethon import TelegramClient, errors

from ingest.pull import (
    Settings,
    advance_cursor,
    ensure_session_parent,
    fetch_last_message_id,
    flush_batch,
    load_settings,
    message_row,
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


async def pull_dialog_windowed(
    conn: asyncpg.Connection,
    message_source: Any,
    dialog: Any,
    settings: Settings,
    cutoff: datetime,
    limit: int,
) -> int:
    """Pull messages newer than ``cutoff`` for one dialog, newest-first.

    ``message_source`` is either the takeout wrapper or the plain client — both
    expose ``iter_messages``.
    """
    chat_id = int(dialog.id)
    chat_title = getattr(dialog, "name", None) or getattr(dialog.entity, "title", None)
    last_seen = await fetch_last_message_id(conn, chat_id)
    print(f"[recent] {chat_title or chat_id}: window since {cutoff.isoformat()} (cursor {last_seen})")

    rows: list[tuple[Any, ...]] = []
    total = 0
    highest_id = last_seen
    stop = False

    while not stop:
        try:
            # Newest-first (default order). Stop once we cross the cutoff.
            async for message in message_source.iter_messages(
                dialog.entity,
                offset_date=None,  # start at the newest message
                wait_time=settings.batch_sleep_seconds,
                limit=limit or None,
            ):
                if message.id is None:
                    continue
                ts = getattr(message, "date", None)
                if ts is not None and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts is not None and ts < cutoff:
                    # Reached messages older than our window; done with this chat.
                    stop = True
                    break
                # Skip anything already stored (resumability).
                if int(message.id) <= last_seen:
                    continue
                rows.append(await message_row(message, chat_id, chat_title))
                highest_id = max(highest_id, int(message.id))
                if len(rows) >= settings.batch_size:
                    inserted = await flush_batch(conn, rows, chat_id, chat_title)
                    total += inserted
                    rows.clear()
                    print(f"[recent] {chat_title or chat_id}: stored {total} messages")
                    await asyncio.sleep(settings.batch_sleep_seconds)
            stop = True
        except errors.FloodWaitError as exc:
            if rows:
                inserted = await flush_batch(conn, rows, chat_id, chat_title)
                total += inserted
                rows.clear()
            wait_seconds = int(exc.seconds) + 1
            print(f"[recent] FloodWait {wait_seconds}s; sleeping")
            await asyncio.sleep(wait_seconds)

    inserted = await flush_batch(conn, rows, chat_id, chat_title)
    total += inserted
    if total == 0:
        # Keep chat_title fresh even when nothing new landed.
        await advance_cursor(conn, chat_id, chat_title, highest_id)
    print(f"[recent] {chat_title or chat_id}: finished with {total} messages")
    return total


def _dialog_matches(dialog: Any, needle: str) -> bool:
    needle = needle.lower()
    name = (getattr(dialog, "name", None) or "").lower()
    username = (getattr(getattr(dialog, "entity", None), "username", None) or "").lower()
    return needle in name or needle in username


async def run() -> None:
    settings = load_settings()
    ensure_session_parent(settings.session)

    since_hours = _float_env("PULL_SINCE_HOURS", 24.0)
    max_chats = _int_env("PULL_MAX_CHATS", 0)
    one_chat = os.getenv("PULL_ONE_CHAT", "").strip()
    per_dialog_limit = _int_env("PULL_RECENT_LIMIT", 0)
    use_takeout = os.getenv("PULL_USE_TAKEOUT", "1") not in ("0", "false", "False")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    conn = await asyncpg.connect(settings.database_url)
    client = TelegramClient(settings.session, settings.api_id, settings.api_hash)
    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise SystemExit(
                "[recent] session is not authorized — refusing to trigger a login. "
                "Authenticate the session out-of-band first."
            )
        me = await client.get_me()
        print(
            f"[recent] authenticated as id={me.id} username={getattr(me, 'username', None)}; "
            f"since_hours={since_hours} takeout={use_takeout} "
            f"batch={settings.batch_size} batch_sleep={settings.batch_sleep_seconds}s "
            f"chat_sleep={settings.chat_sleep_seconds}s"
        )

        # Collect (and cache) dialogs once, so we never re-resolve entities.
        dialogs: list[Any] = []
        async for dialog in client.iter_dialogs():
            if one_chat and not _dialog_matches(dialog, one_chat):
                continue
            dialogs.append(dialog)
            if max_chats and len(dialogs) >= max_chats:
                break
        print(f"[recent] processing {len(dialogs)} dialog(s)")

        async def process(source: Any) -> int:
            grand_total = 0
            for idx, dialog in enumerate(dialogs):
                grand_total += await pull_dialog_windowed(
                    conn, source, dialog, settings, cutoff, per_dialog_limit
                )
                if idx < len(dialogs) - 1:
                    await asyncio.sleep(settings.chat_sleep_seconds)
            return grand_total

        # Max takeout init delay we're willing to wait out before falling back
        # to the plain client. Telegram can hand back a ~24h delay after prior
        # takeout attempts; blocking that long is never acceptable here.
        max_takeout_delay = _float_env("PULL_TAKEOUT_MAX_DELAY_SECONDS", 300.0)

        if use_takeout:
            # Takeout: official export mode, lower flood limits. Retry on a short
            # init delay; fall back to the plain (conservatively paced) client if
            # Telegram demands a long cooldown.
            # If a takeout session is already open for this session (e.g. a prior
            # run crashed mid-export), reuse it instead of requesting a new one —
            # Telegram refuses a second concurrent takeout request.
            total = None
            while True:
                try:
                    existing_takeout = client.session.takeout_id
                    # A stale/empty takeout_id (e.g. b'' from a crashed run)
                    # corrupts the finalize request; treat it as "start fresh".
                    if not isinstance(existing_takeout, int):
                        client.session.takeout_id = None
                        existing_takeout = None
                    if existing_takeout is None:
                        # Telethon 1.44's takeout() covers message history by
                        # default; the flags opt each peer type into the export.
                        cm = client.takeout(
                            users=True, chats=True, channels=True, megagroups=True
                        )
                        print("[recent] starting a new takeout session")
                    else:
                        # Attach to the existing takeout (no new request).
                        cm = client.takeout()
                        print(
                            f"[recent] reusing existing takeout session "
                            f"(id={client.session.takeout_id})"
                        )
                    async with cm as takeout:
                        total = await process(takeout)
                    break
                except errors.TakeoutInitDelayError as exc:
                    wait_seconds = int(exc.seconds) + 1
                    if wait_seconds > max_takeout_delay:
                        print(
                            f"[recent] takeout init delayed {wait_seconds}s "
                            f"(> cap {max_takeout_delay:.0f}s) — falling back to the "
                            "plain client with conservative pacing"
                        )
                        use_takeout = False
                        break
                    print(
                        f"[recent] takeout init delayed {wait_seconds}s; sleeping"
                    )
                    await asyncio.sleep(wait_seconds)

        if not use_takeout:
            total = await process(client)

        print(f"[recent] complete: stored {total} messages in the last {since_hours}h window")
    finally:
        await client.disconnect()
        await conn.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[recent] interrupted; committed batches and sync_state are safe to resume")


if __name__ == "__main__":
    main()
