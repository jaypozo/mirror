"""Corpus writes — the ONE place an owner message enters the retrievable store.

CORE REQUIREMENT (do not violate): the exemplar corpus contains ONLY the owner's
REAL messages. Two sources:
  1. Ingested history (`ingest/`) — real sent/received Telegram messages.
  2. The owner's APPROVED/EDITED FINAL replies — added here at decide-time.

A model DRAFT (`feedback.original_draft`) is NEVER written to `messages` or to
either embedding table. `add_owner_sample()` only ever takes the FINAL text the
owner actually sent, keyed by the REAL Telegram message id returned by the send —
so when normal ingest later re-pulls that message the upsert is a no-op, not a
duplicate. This module never sees a draft.

Everything here is guarded: a failure logs and is swallowed. The message has
already been sent to the owner's chat by the time this runs; failing to index it
must never fail the /decide request (normal ingest would pick it up anyway).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from agent.retrieve import embed_query  # SAME cached MiniLM encoder as retrieval
from agent.style_embed import STYLE_MODEL_DIM, embed_style
from ingest.embed import DEFAULT_LOCAL_MODEL, context_tag

log = logging.getLogger("mirror.corpus")


async def add_owner_sample(sent_message: Any, text: str, request: Any) -> bool:
    """Persist the owner's just-sent FINAL reply as a genuine, retrievable sample.

    Writes the real message row (direction='out'), its TOPIC embedding, and — when
    the active style embedder matches the stored column — its STYLE embedding, so
    the reply is retrievable immediately (not only after the next ingest pass).

    `sent_message` is the Telethon Message returned by the send (real id/chat_id).
    `text` is the FINAL sent text — never a draft. Returns True on success.
    """
    final = (text or "").strip()
    if not final:
        return False
    msg_id = getattr(sent_message, "id", None)
    chat_id = getattr(sent_message, "chat_id", None)
    if msg_id is None or chat_id is None:
        # No real id to key on (send returned nothing usable) — skip; normal
        # ingest will pick this message up on the next pull.
        log.info("add_owner_sample: send returned no id/chat_id; deferring to ingest")
        return False
    msg_id = int(msg_id)
    chat_id = int(chat_id)

    ts = getattr(sent_message, "date", None)
    if ts is None:
        ts = datetime.now(timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    reply_to = getattr(sent_message, "reply_to_msg_id", None)
    owner_id = int(os.getenv("MIRROR_OWNER_USER_ID", "0") or "0") or None

    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        log.warning("add_owner_sample: DATABASE_URL missing")
        return False

    # Embed BEFORE opening the connection (encoders are warm; keeps the txn short).
    topic_vec = await embed_query(final)
    style_vecs, style_embedder = await embed_style([final])
    style_vec = None
    style_model = None
    if (
        style_vecs
        and style_embedder is not None
        and getattr(style_embedder, "persistable", False)
        and style_embedder.dim == STYLE_MODEL_DIM
    ):
        style_vec = style_vecs[0]
        style_model = style_embedder.name

    conn = await asyncpg.connect(database_url)
    try:
        await register_vector(conn)
        async with conn.transaction():
            # 1. The real owner message. Upsert so a later ingest of the same id
            #    is idempotent. chat_title left to ingest (it has the dialog).
            await conn.execute(
                """
                INSERT INTO messages (
                    id, chat_id, chat_title, sender_id, sender_name,
                    text, ts, direction, reply_to_id, raw
                )
                VALUES ($1, $2, NULL, $3, NULL, $4, $5, 'out', $6, $7::jsonb)
                ON CONFLICT (chat_id, id) DO UPDATE SET
                    text = EXCLUDED.text,
                    ts = EXCLUDED.ts,
                    direction = EXCLUDED.direction
                """,
                msg_id,
                chat_id,
                owner_id,
                final,
                ts,
                reply_to,
                json.dumps({"source": "decide", "kind": "owner_final"}),
            )
            # 2. TOPIC embedding (MiniLM) — same model/space as the ingest embed step.
            await conn.execute(
                """
                INSERT INTO message_embeddings (
                    chat_id, message_id, embedding, context_tag, provider, model
                )
                VALUES ($1, $2, $3, $4, 'local', $5)
                ON CONFLICT (chat_id, message_id, context_tag) DO UPDATE SET
                    embedding = EXCLUDED.embedding, embedded_at = now()
                """,
                chat_id,
                msg_id,
                topic_vec,
                context_tag(final),
                DEFAULT_LOCAL_MODEL,
            )
            # 3. STYLE embedding — only when it matches the stored 768-dim column.
            if style_vec is not None:
                await conn.execute(
                    """
                    INSERT INTO message_style_embeddings (chat_id, message_id, embedding, model)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (chat_id, message_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding, model = EXCLUDED.model, embedded_at = now()
                    """,
                    chat_id,
                    msg_id,
                    style_vec,
                    style_model,
                )
        log.info(
            "add_owner_sample: indexed owner final chat=%s id=%s (style=%s)",
            chat_id,
            msg_id,
            bool(style_vec),
        )
        return True
    finally:
        await conn.close()
