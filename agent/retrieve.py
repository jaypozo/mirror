"""Vector retrieval of the owner's own past replies as few-shot voice examples.

Given a query (an incoming message), we surface the owner's OWN outbound replies
(direction='out') as few-shot exemplars — always REAL owner messages, never a
model draft. Selection blends two independent axes plus a diversity pass:

  * TOPIC similarity — MiniLM over the message text (`message_embeddings`,
    pgvector cosine). "What was this reply about."
  * STYLE similarity — a content-independent style vector
    (`message_style_embeddings`, StyleDistance; see agent/style_embed.py) scored
    against the incoming message's own style. "How was this written / what
    register." This is what makes retrieval pick exemplars by the owner's VOICE,
    not just the subject — and it mirrors register (a terse incoming pulls terse
    owner replies; a formal one pulls formal ones).
  * MMR diversity — greedy Maximal-Marginal-Relevance over the style vectors so
    the returned set isn't five near-duplicate lines; it spans different
    lengths/registers.

The blend weights (STYLE_BLEND_TOPIC_WEIGHT / STYLE_BLEND_STYLE_WEIGHT), the MMR
tradeoff (STYLE_MMR_LAMBDA), and the candidate pool size (STYLE_CANDIDATE_POOL)
are all env-configurable with sane defaults. Everything is guarded: if the style
embedder is unavailable or errors, retrieval falls back to the original
topic-only nearest-neighbour order, so drafting never breaks.

Fully local: no external API key, no network calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from agent.style_embed import cosine, embed_style
from ingest.embed import DEFAULT_LOCAL_MODEL

log = logging.getLogger("mirror.retrieve")


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


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _vector_to_list(value) -> list[float] | None:
    if value is None:
        return None
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [float(item) for item in to_list()]
    return [float(item) for item in value]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class _Candidate:
    chat_id: int
    message_id: int
    chat_title: str | None
    text: str
    distance: float          # topic cosine distance (0 = identical)
    style_vec: list[float] | None = None
    topic_sim: float = 0.0   # 1 - distance
    style_sim: float = 0.0   # style cosine to the incoming message
    base_score: float = 0.0  # blended topic+style


def _mmr_select(
    cands: list[_Candidate], top_k: int, lam: float
) -> list[_Candidate]:
    """Greedy MMR over the candidates' STYLE vectors: at each step pick the
    candidate maximizing `lam*base_score - (1-lam)*max_style_sim_to_already_chosen`,
    so we favour high-scoring exemplars while penalizing ones stylistically
    near-identical to those already picked — spreading length/register."""
    selected: list[_Candidate] = []
    remaining = list(cands)
    while remaining and len(selected) < top_k:
        best, best_val = None, float("-inf")
        for c in remaining:
            if not selected or c.style_vec is None:
                val = c.base_score
            else:
                redundancy = max(
                    (cosine(c.style_vec, s.style_vec) for s in selected if s.style_vec is not None),
                    default=0.0,
                )
                val = lam * c.base_score - (1.0 - lam) * redundancy
            if val > best_val:
                best, best_val = c, val
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    return selected


async def retrieve_examples(
    query_text: str,
    top_k: int | None = None,
    context_tag: str | None = None,
    context_window: int = 3,
    database_url: str | None = None,
    style_query_text: str | None = None,
) -> list[RetrievedExample]:
    """Top-K of the owner's own past replies, selected by a TOPIC+STYLE+MMR blend.

    Always restricted to direction='out', so every exemplar is genuinely one of
    the owner's real messages. Over-fetches a topic-nearest candidate pool, then
    re-ranks by blended topic+style similarity and diversifies with MMR. Falls
    back to pure topic order whenever the style path is unavailable.

    `style_query_text` is the text whose WRITING STYLE we match against (the raw
    incoming message — its register); defaults to `query_text`.
    """
    load_dotenv()
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("Missing DATABASE_URL.")

    top_k = top_k or int(os.getenv("RETRIEVE_TOP_K", "6"))
    w_topic = _float_env("STYLE_BLEND_TOPIC_WEIGHT", 0.6)
    w_style = _float_env("STYLE_BLEND_STYLE_WEIGHT", 0.4)
    mmr_lambda = _float_env("STYLE_MMR_LAMBDA", 0.7)
    # Over-fetch a pool to re-rank; wide enough for style+MMR to matter.
    pool_size = max(top_k, _int_env("STYLE_CANDIDATE_POOL", max(40, top_k * 6)))

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
                   (e.embedding <=> $1) AS distance,
                   se.embedding AS style_embedding
            FROM message_embeddings e
            JOIN messages m
              ON m.chat_id = e.chat_id
             AND m.id = e.message_id
            LEFT JOIN message_style_embeddings se
              ON se.chat_id = m.chat_id
             AND se.message_id = m.id
            WHERE m.direction = 'out'
              AND m.text IS NOT NULL
              AND length(trim(m.text)) > 0
              AND ($2::text IS NULL OR e.context_tag = $2)
            ORDER BY e.embedding <=> $1
            LIMIT $3
            """,
            query_embedding,
            context_tag,
            pool_size,
        )

        candidates = [
            _Candidate(
                chat_id=r["chat_id"],
                message_id=r["message_id"],
                chat_title=r["chat_title"],
                text=r["text"],
                distance=float(r["distance"]),
                style_vec=_vector_to_list(r["style_embedding"]),
            )
            for r in rows
        ]

        selected = await _rank_candidates(
            candidates,
            style_query_text=style_query_text or query_text,
            top_k=top_k,
            w_topic=w_topic,
            w_style=w_style,
            mmr_lambda=mmr_lambda,
        )

        examples: list[RetrievedExample] = []
        for c in selected:
            context = await _fetch_context(conn, c.chat_id, c.message_id, context_window)
            examples.append(
                RetrievedExample(
                    chat_id=c.chat_id,
                    message_id=c.message_id,
                    chat_title=c.chat_title,
                    reply_text=c.text,
                    distance=c.distance,
                    context=context,
                )
            )
        return examples
    finally:
        await conn.close()


async def _rank_candidates(
    candidates: list[_Candidate],
    style_query_text: str,
    top_k: int,
    w_topic: float,
    w_style: float,
    mmr_lambda: float,
) -> list[_Candidate]:
    """Score the topic-nearest pool by blended topic+style similarity and pick a
    diverse top-K via MMR. On any style failure, degrade to topic order (the pool
    is already sorted by topic distance)."""
    if not candidates:
        return []

    # Style path: embed the incoming message's register, embed any pool
    # candidates missing a stored style vector on the fly, then blend + MMR.
    try:
        query_style, embedder = await embed_style([style_query_text])
        if not query_style or embedder is None:
            raise RuntimeError("style embedder unavailable")
        qsv = query_style[0]

        # Candidates whose stored style vector is absent (or whose stored vector
        # was made by a different-dim model than the active fallback embedder):
        # embed their text on the fly so every candidate gets a comparable vector.
        need_idx = [
            i
            for i, c in enumerate(candidates)
            if c.style_vec is None or len(c.style_vec) != len(qsv)
        ]
        if need_idx:
            fresh, _ = await embed_style([candidates[i].text for i in need_idx])
            if fresh and len(fresh) == len(need_idx):
                for slot, i in enumerate(need_idx):
                    candidates[i].style_vec = fresh[slot]

        for c in candidates:
            c.topic_sim = 1.0 - c.distance
            c.style_sim = cosine(qsv, c.style_vec) if c.style_vec is not None else 0.0
            c.base_score = w_topic * c.topic_sim + w_style * c.style_sim

        return _mmr_select(candidates, top_k, mmr_lambda)
    except Exception as exc:
        log.warning("style blend unavailable; topic-only retrieval: %s", exc)
        return candidates[:top_k]


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
