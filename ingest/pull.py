"""Incrementally pull the owner's Telegram history into Postgres.

This uses a Telethon *user* session. Bots cannot read account history and are
intentionally not supported here.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import asyncpg
from dotenv import load_dotenv
from telethon import TelegramClient, errors


DEFAULT_BATCH_SIZE = 100
DEFAULT_BATCH_SLEEP_SECONDS = 1.0
DEFAULT_CHAT_SLEEP_SECONDS = 2.0


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    session: str
    database_url: str
    batch_size: int
    batch_sleep_seconds: float
    chat_sleep_seconds: float


def load_settings() -> Settings:
    load_dotenv()

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session = os.getenv("TELEGRAM_SESSION", ".telethon/mirror")
    database_url = os.getenv("DATABASE_URL")

    missing = [
        name
        for name, value in {
            "TELEGRAM_API_ID": api_id,
            "TELEGRAM_API_HASH": api_hash,
            "DATABASE_URL": database_url,
        }.items()
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required environment variable(s): {joined}. "
            "Copy .env.example to .env and fill the owner's user-account credentials."
        )

    try:
        parsed_api_id = int(api_id or "")
    except ValueError as exc:
        raise SystemExit("TELEGRAM_API_ID must be an integer from my.telegram.org.") from exc

    return Settings(
        api_id=parsed_api_id,
        api_hash=api_hash or "",
        session=session,
        database_url=database_url or "",
        batch_size=int(os.getenv("PULL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        batch_sleep_seconds=float(
            os.getenv("PULL_BATCH_SLEEP_SECONDS", str(DEFAULT_BATCH_SLEEP_SECONDS))
        ),
        chat_sleep_seconds=float(
            os.getenv("PULL_CHAT_SLEEP_SECONDS", str(DEFAULT_CHAT_SLEEP_SECONDS))
        ),
    )


def ensure_session_parent(session: str) -> None:
    path = Path(session)
    parent = path.parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def compact_raw(message: Any) -> str:
    """Keep enough raw shape for backfills without storing large binary payloads."""
    data = {
        "id": getattr(message, "id", None),
        "date": getattr(message, "date", None),
        "edit_date": getattr(message, "edit_date", None),
        "out": getattr(message, "out", None),
        "post": getattr(message, "post", None),
        "mentioned": getattr(message, "mentioned", None),
        "media_unread": getattr(message, "media_unread", None),
        "silent": getattr(message, "silent", None),
        "message": getattr(message, "message", None),
        "peer_id": repr(getattr(message, "peer_id", None)),
        "sender_id": getattr(message, "sender_id", None),
        "reply_to": repr(getattr(message, "reply_to", None)),
        "grouped_id": getattr(message, "grouped_id", None),
    }
    return json.dumps(data, default=json_default)


def display_name(sender: Any) -> str | None:
    if sender is None:
        return None
    title = getattr(sender, "title", None)
    if title:
        return title
    first_name = getattr(sender, "first_name", None)
    last_name = getattr(sender, "last_name", None)
    username = getattr(sender, "username", None)
    name = " ".join(part for part in [first_name, last_name] if part)
    return name or username


def reply_to_id(message: Any) -> int | None:
    direct = getattr(message, "reply_to_msg_id", None)
    if direct:
        return direct
    reply_to = getattr(message, "reply_to", None)
    if reply_to is None:
        return None
    return getattr(reply_to, "reply_to_msg_id", None)


async def fetch_last_message_id(conn: asyncpg.Connection, chat_id: int) -> int:
    value = await conn.fetchval(
        "SELECT last_message_id FROM sync_state WHERE chat_id = $1",
        chat_id,
    )
    return int(value or 0)


async def upsert_messages(conn: asyncpg.Connection, rows: Iterable[tuple[Any, ...]]) -> None:
    await conn.executemany(
        """
        INSERT INTO messages (
            id, chat_id, chat_title, sender_id, sender_name,
            text, ts, direction, reply_to_id, raw
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
        ON CONFLICT (chat_id, id) DO UPDATE SET
            chat_title = EXCLUDED.chat_title,
            sender_id = EXCLUDED.sender_id,
            sender_name = EXCLUDED.sender_name,
            text = EXCLUDED.text,
            ts = EXCLUDED.ts,
            direction = EXCLUDED.direction,
            reply_to_id = EXCLUDED.reply_to_id,
            raw = EXCLUDED.raw
        """,
        list(rows),
    )


async def advance_cursor(
    conn: asyncpg.Connection,
    chat_id: int,
    chat_title: str | None,
    last_message_id: int,
) -> None:
    await conn.execute(
        """
        INSERT INTO sync_state (chat_id, chat_title, last_message_id, last_pulled_at)
        VALUES ($1, $2, $3, now())
        ON CONFLICT (chat_id) DO UPDATE SET
            chat_title = EXCLUDED.chat_title,
            last_message_id = GREATEST(sync_state.last_message_id, EXCLUDED.last_message_id),
            last_pulled_at = now()
        """,
        chat_id,
        chat_title,
        last_message_id,
    )


async def message_row(message: Any, chat_id: int, chat_title: str | None) -> tuple[Any, ...]:
    sender = getattr(message, "sender", None)
    if sender is None and getattr(message, "sender_id", None):
        sender = await message.get_sender()

    ts = getattr(message, "date", None)
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return (
        int(message.id),
        chat_id,
        chat_title,
        getattr(message, "sender_id", None),
        display_name(sender),
        getattr(message, "raw_text", None) or getattr(message, "message", None),
        ts,
        "out" if getattr(message, "out", False) else "in",
        reply_to_id(message),
        compact_raw(message),
    )


async def flush_batch(
    conn: asyncpg.Connection,
    rows: list[tuple[Any, ...]],
    chat_id: int,
    chat_title: str | None,
) -> int:
    if not rows:
        return 0
    async with conn.transaction():
        await upsert_messages(conn, rows)
        await advance_cursor(conn, chat_id, chat_title, max(row[0] for row in rows))
    return len(rows)


async def pull_dialog(
    client: TelegramClient,
    conn: asyncpg.Connection,
    dialog: Any,
    settings: Settings,
) -> int:
    chat_id = int(dialog.id)
    chat_title = getattr(dialog, "name", None) or getattr(dialog.entity, "title", None)
    last_seen = await fetch_last_message_id(conn, chat_id)
    print(f"[pull] {chat_title or chat_id}: starting after message id {last_seen}")

    rows: list[tuple[Any, ...]] = []
    total = 0

    while True:
        try:
            async for message in client.iter_messages(
                dialog.entity,
                min_id=last_seen,
                reverse=True,
                wait_time=settings.batch_sleep_seconds,
            ):
                if message.id is None:
                    continue
                rows.append(await message_row(message, chat_id, chat_title))
                if len(rows) >= settings.batch_size:
                    inserted = await flush_batch(conn, rows, chat_id, chat_title)
                    total += inserted
                    last_seen = max(row[0] for row in rows)
                    rows.clear()
                    print(f"[pull] {chat_title or chat_id}: stored {total} new messages")
                    await asyncio.sleep(settings.batch_sleep_seconds)
            break
        except errors.FloodWaitError as exc:
            if rows:
                inserted = await flush_batch(conn, rows, chat_id, chat_title)
                total += inserted
                last_seen = max(row[0] for row in rows)
                rows.clear()
            wait_seconds = int(exc.seconds) + 1
            print(f"[pull] Telegram FloodWait for {wait_seconds}s; sleeping")
            await asyncio.sleep(wait_seconds)

    inserted = await flush_batch(conn, rows, chat_id, chat_title)
    total += inserted
    if total == 0:
        await advance_cursor(conn, chat_id, chat_title, last_seen)
    print(f"[pull] {chat_title or chat_id}: finished with {total} new messages")
    return total


async def run() -> None:
    settings = load_settings()
    ensure_session_parent(settings.session)

    conn = await asyncpg.connect(settings.database_url)
    client = TelegramClient(settings.session, settings.api_id, settings.api_hash)
    try:
        await client.start()
        me = await client.get_me()
        print(f"[pull] authenticated as user id={me.id} username={getattr(me, 'username', None)}")

        total = 0
        async for dialog in client.iter_dialogs():
            total += await pull_dialog(client, conn, dialog, settings)
            await asyncio.sleep(settings.chat_sleep_seconds)

        print(f"[pull] complete: stored {total} new messages")
    finally:
        await client.disconnect()
        await conn.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[pull] interrupted; committed batches and sync_state are safe to resume")


if __name__ == "__main__":
    main()
