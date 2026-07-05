"""Backfill STYLE embeddings for the owner's own messages.

Companion to `ingest/embed.py` (which fills the TOPIC vectors in
`message_embeddings`). This fills `message_style_embeddings` with a
content-independent STYLE vector (StyleDistance, 768-dim) for every one of the
owner's REAL sent messages (direction='out') that doesn't have one yet.

Only `direction='out'` is embedded: retrieval only ever surfaces the owner's own
replies as exemplars, so there is no reason to style-embed inbound messages. As
with the topic backfill, this is idempotent and resumable — it selects only rows
missing a style embedding, so re-running just tops up.

    python -m ingest.embed_style            # backfill everything missing
    STYLE_BACKFILL_LIMIT=500 python -m ingest.embed_style   # cap this run

Guarded: if the style model can't load (offline/unavailable) it exits cleanly
without touching the DB — the live drafter still runs (topic-only retrieval).
"""

from __future__ import annotations

import asyncio
import os

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from agent.style_embed import STYLE_MODEL_DIM, load_style_embedder


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


async def _fetch_batch(conn: asyncpg.Connection, limit: int) -> list[tuple[int, int, str]]:
    rows = await conn.fetch(
        """
        SELECT m.chat_id, m.id AS message_id, m.text
        FROM messages m
        WHERE m.direction = 'out'
          AND m.text IS NOT NULL
          AND length(trim(m.text)) > 0
          AND NOT EXISTS (
              SELECT 1 FROM message_style_embeddings se
              WHERE se.chat_id = m.chat_id AND se.message_id = m.id
          )
        ORDER BY m.ts ASC
        LIMIT $1
        """,
        limit,
    )
    return [(r["chat_id"], r["message_id"], r["text"]) for r in rows]


async def run() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("Missing DATABASE_URL.")

    batch_size = _int_env("STYLE_EMBED_BATCH_SIZE", 128)
    hard_cap = _int_env("STYLE_BACKFILL_LIMIT", 0)  # 0 = no cap

    embedder = await load_style_embedder()
    if embedder is None:
        raise SystemExit(
            "No style embedder available (STYLE_RETRIEVAL_MODE=off or model failed "
            "to load). Nothing embedded."
        )
    if not getattr(embedder, "persistable", False) or embedder.dim != STYLE_MODEL_DIM:
        raise SystemExit(
            f"Active style embedder '{embedder.name}' (dim={embedder.dim}) does not "
            f"match the stored column (dim={STYLE_MODEL_DIM}); refusing to backfill. "
            "Set STYLE_EMBED_MODEL to the 768-dim style model."
        )
    print(f"[embed_style] using {embedder.name} (dim={embedder.dim})")

    conn = await asyncpg.connect(database_url)
    await register_vector(conn)
    try:
        total = 0
        while True:
            remaining = batch_size
            if hard_cap:
                remaining = min(batch_size, hard_cap - total)
                if remaining <= 0:
                    break
            batch = await _fetch_batch(conn, remaining)
            if not batch:
                break

            texts = [t for _, _, t in batch]
            vectors = await asyncio.to_thread(embedder.encode, texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"style embedder returned {len(vectors)} vectors for {len(batch)} texts"
                )

            await conn.executemany(
                """
                INSERT INTO message_style_embeddings (chat_id, message_id, embedding, model)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chat_id, message_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding, model = EXCLUDED.model, embedded_at = now()
                """,
                [
                    (chat_id, message_id, vector, embedder.name)
                    for (chat_id, message_id, _), vector in zip(batch, vectors, strict=True)
                ],
            )
            total += len(batch)
            print(f"[embed_style] stored {total} style embeddings")

        print(f"[embed_style] complete: stored {total} style embeddings")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
