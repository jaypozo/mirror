"""Embed stored Telegram messages into pgvector.

The provider call is intentionally isolated behind EmbeddingProvider so Phase 1
can run with a clear interface and Phase 2 can swap providers or models later.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

# Default local embedding model: small, fast, free, 384-dim. Matches the
# `vector(384)` column in db/schema.sql.
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIM = 384


@dataclass(frozen=True)
class Settings:
    database_url: str
    provider: str
    model: str
    api_key: str | None
    batch_size: int


@dataclass(frozen=True)
class MessageForEmbedding:
    chat_id: int
    message_id: int
    text: str
    context_tag: str


class EmbeddingProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        # Imported lazily so the module runs without the openai package or a key
        # when EMBEDDING_PROVIDER=local (the default in this deployment).
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]


class LocalEmbeddingProvider(EmbeddingProvider):
    """On-box sentence-transformers provider. No network / API key required."""

    name = "local"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self._encoder = SentenceTransformer(model)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._encoder.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        # Encoding is CPU-bound; run it off the event loop.
        return await asyncio.to_thread(self._encode, texts)


class DryRunEmbeddingProvider(EmbeddingProvider):
    name = "dry-run"

    def __init__(self, model: str = "dry-run-zero-vector", dimensions: int = 8) -> None:
        self.model = model
        self.dimensions = dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]


def load_settings() -> Settings:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("Missing DATABASE_URL. Copy .env.example to .env and fill it in.")

    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower().strip()
    # Pick a sensible default model per provider if EMBEDDING_MODEL is unset.
    default_model = (
        DEFAULT_LOCAL_MODEL if provider in {"local", "sentence-transformers", "st"}
        else "text-embedding-3-small"
    )

    return Settings(
        database_url=database_url,
        provider=provider,
        model=os.getenv("EMBEDDING_MODEL") or default_model,
        api_key=os.getenv("EMBEDDING_API_KEY"),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "100")),
    )


def build_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.provider.lower().strip()
    if provider in {"local", "sentence-transformers", "st"}:
        return LocalEmbeddingProvider(model=settings.model)
    if provider == "openai":
        if not settings.api_key:
            raise SystemExit("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai.")
        return OpenAIEmbeddingProvider(api_key=settings.api_key, model=settings.model)
    if provider in {"dry-run", "dryrun"}:
        return DryRunEmbeddingProvider()
    raise SystemExit(f"Unsupported EMBEDDING_PROVIDER={settings.provider!r}.")


def context_tag(text: str) -> str:
    lowered = text.lower()
    buckets = {
        "code-review": [
            "pr",
            "diff",
            "review",
            "test",
            "tests",
            "bug",
            "stack trace",
            "deploy",
            "merge",
        ],
        "planning": ["plan", "roadmap", "milestone", "goal", "next", "timeline", "scope"],
        "cs": ["customer", "client", "support", "refund", "invoice", "onboarding"],
        "personal": ["family", "dinner", "travel", "home", "birthday", "weekend"],
    }
    for tag, needles in buckets.items():
        if any(needle in lowered for needle in needles):
            return tag
    return "general"


async def fetch_batch(conn: asyncpg.Connection, limit: int) -> list[MessageForEmbedding]:
    rows = await conn.fetch(
        """
        SELECT m.chat_id, m.id AS message_id, m.text
        FROM messages m
        WHERE m.text IS NOT NULL
          AND length(trim(m.text)) > 0
          AND NOT EXISTS (
              SELECT 1
              FROM message_embeddings e
              WHERE e.chat_id = m.chat_id
                AND e.message_id = m.id
          )
        ORDER BY m.ts ASC
        LIMIT $1
        """,
        limit,
    )
    return [
        MessageForEmbedding(
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            text=row["text"],
            context_tag=context_tag(row["text"]),
        )
        for row in rows
    ]


async def insert_embeddings(
    conn: asyncpg.Connection,
    messages: Sequence[MessageForEmbedding],
    embeddings: Sequence[Sequence[float]],
    provider: EmbeddingProvider,
) -> None:
    await conn.executemany(
        """
        INSERT INTO message_embeddings (
            chat_id, message_id, embedding, context_tag, provider, model
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (chat_id, message_id, context_tag) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            provider = EXCLUDED.provider,
            model = EXCLUDED.model,
            embedded_at = now()
        """,
        [
            (
                message.chat_id,
                message.message_id,
                list(embedding),
                message.context_tag,
                provider.name,
                provider.model,
            )
            for message, embedding in zip(messages, embeddings, strict=True)
        ],
    )


async def run() -> None:
    settings = load_settings()
    provider = build_provider(settings)

    conn = await asyncpg.connect(settings.database_url)
    await register_vector(conn)
    try:
        total = 0
        while True:
            messages = await fetch_batch(conn, settings.batch_size)
            if not messages:
                break

            embeddings = await provider.embed_texts([message.text for message in messages])
            if len(embeddings) != len(messages):
                raise RuntimeError(
                    f"Embedding provider returned {len(embeddings)} vectors for "
                    f"{len(messages)} messages."
                )

            await insert_embeddings(conn, messages, embeddings, provider)
            total += len(messages)
            print(f"[embed] stored {total} embeddings")

        print(f"[embed] complete: stored {total} embeddings")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
