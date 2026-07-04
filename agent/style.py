"""Optional RAG hook for owner-style examples from the Phase 1 corpus."""

from __future__ import annotations

import os
import sys

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector


BASIC_STYLE_PROFILE = (
    "Write as the owner: concise, direct, pragmatic, specific about next steps, "
    "plainspoken, and comfortable saying when a fact is missing. Avoid hype."
)


async def embed_query(text: str) -> list[float] | None:
    load_dotenv()
    api_key = (
        os.getenv("STYLE_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        return None
    model = os.getenv("STYLE_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = AsyncOpenAI(api_key=api_key)
    response = await client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


async def retrieve_style_examples(
    query_text: str,
    context_tag: str | None = None,
    limit: int | None = None,
) -> list[str]:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return []

    resolved_limit = limit or int(os.getenv("STYLE_EXAMPLE_LIMIT", "5"))
    try:
        query_embedding = await embed_query(query_text)
        if not query_embedding:
            return []

        conn = await asyncpg.connect(database_url)
        try:
            await register_vector(conn)
            rows = await conn.fetch(
                """
                SELECT m.text
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
                resolved_limit,
            )
        finally:
            await conn.close()
    except Exception as exc:
        print(f"[style] style retrieval unavailable; continuing without RAG: {exc}", file=sys.stderr)
        return []

    return [row["text"] for row in rows]
