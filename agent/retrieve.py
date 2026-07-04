"""Vector retrieval of the owner's own past replies as few-shot voice examples.

Given a query (an incoming message), embed it with the SAME local
sentence-transformers model used at ingest time, then run a pgvector
nearest-neighbour search over `message_embeddings` restricted to the owner's OWN
outbound replies (direction='out'). Each hit is returned with a little
surrounding thread context so the drafter can see how the owner actually replied in
that situation.

Fully local: no external API key, no network calls.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from functools import lru_cache

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from ingest.embed import DEFAULT_LOCAL_MODEL


@dataclass(frozen=True)
class ContextLine:
    direction: str
    sender_name: str | None
    text: str

    def to_prompt_line(self) -> str:
        who = "the owner" if self.direction == "out" else (self.sender_name or "them")
        return f"{who}: {self.text}".strip()


@dataclass(frozen=True)
class RetrievedExample:
    """One of the owner's past replies plus the message(s) it was answering."""

    chat_id: int
    message_id: int
    chat_title: str | None
    reply_text: str
    distance: float
    context: list[ContextLine] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        lines = [line.to_prompt_line() for line in self.context]
        lines.append(f"the owner: {self.reply_text}")
        return "\n".join(lines)


@lru_cache(maxsize=2)
def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_query_sync(text: str, model_name: str) -> list[float]:
    encoder = _load_encoder(model_name)
    vector = encoder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vector.tolist()


async def embed_query(text: str, model_name: str | None = None) -> list[float]:
    load_dotenv()
    model_name = (
        model_name
        or os.getenv("STYLE_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or DEFAULT_LOCAL_MODEL
    )
    return await asyncio.to_thread(_embed_query_sync, text, model_name)


async def _fetch_context(
    conn: asyncpg.Connection,
    chat_id: int,
    message_id: int,
    window: int,
) -> list[ContextLine]:
    """Return up to `window` messages immediately preceding the owner's reply."""
    rows = await conn.fetch(
        """
        SELECT direction, sender_name, text
        FROM messages
        WHERE chat_id = $1
          AND id < $2
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
    return [
        ContextLine(direction=r["direction"], sender_name=r["sender_name"], text=r["text"])
        for r in rows
    ]


async def retrieve_examples(
    query_text: str,
    top_k: int | None = None,
    context_tag: str | None = None,
    context_window: int = 3,
    database_url: str | None = None,
) -> list[RetrievedExample]:
    """Top-K of the owner's own past replies most similar to `query_text`.

    Cosine distance (`<=>`) over normalized vectors. Restricted to
    direction='out' so every example is genuinely the owner's voice.
    """
    load_dotenv()
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("Missing DATABASE_URL.")

    top_k = top_k or int(os.getenv("RETRIEVE_TOP_K", "6"))
    query_embedding = await embed_query(query_text)

    conn = await asyncpg.connect(database_url)
    try:
        await register_vector(conn)
        rows = await conn.fetch(
            """
            SELECT m.chat_id,
                   m.id AS message_id,
                   m.chat_title,
                   m.text,
                   (e.embedding <=> $1) AS distance
            FROM message_embeddings e
            JOIN messages m
              ON m.chat_id = e.chat_id
             AND m.id = e.message_id
            WHERE m.direction = 'out'
              AND m.text IS NOT NULL
              AND length(trim(m.text)) > 0
              AND ($2::text IS NULL OR e.context_tag = $2)
            ORDER BY e.embedding <=> $1
            LIMIT $3
            """,
            query_embedding,
            context_tag,
            top_k,
        )

        examples: list[RetrievedExample] = []
        for row in rows:
            context = await _fetch_context(
                conn, row["chat_id"], row["message_id"], context_window
            )
            examples.append(
                RetrievedExample(
                    chat_id=row["chat_id"],
                    message_id=row["message_id"],
                    chat_title=row["chat_title"],
                    reply_text=row["text"],
                    distance=float(row["distance"]),
                    context=context,
                )
            )
        return examples
    finally:
        await conn.close()


async def _main() -> None:
    import sys

    query = " ".join(sys.argv[1:]) or "can you get that done by tomorrow?"
    examples = await retrieve_examples(query)
    print(f"Query: {query}\n")
    for i, ex in enumerate(examples, 1):
        print(f"--- Example {i}  (distance={ex.distance:.4f}, chat={ex.chat_title!r}) ---")
        print(ex.format_for_prompt())
        print()


if __name__ == "__main__":
    asyncio.run(_main())
