"""Content-independent STYLE embeddings of the owner's messages.

Retrieval in `agent/retrieve.py` historically ranked the owner's past replies by
TOPIC similarity only (MiniLM over the message text). That surfaces replies about
the same subject, but not replies written in the same VOICE. This module adds the
missing axis: a vector that captures HOW a message is written — length, register,
punctuation, casing, formality — independent of WHAT it is about.

Two interchangeable local embedders, chosen at runtime (no external API):

  * Preferred — StyleDistance (`StyleDistance/styledistance`): a
    sentence-transformers style-representation model (768-dim). Drop-in like
    MiniLM (`SentenceTransformer(...).encode(...)`), but trained so cosine
    distance reflects writing style, not topic. Verified locally: two casual
    messages on different topics score closer than a casual + formal pair on the
    SAME topic. This is the vector stored in `message_style_embeddings`.

  * Fallback — a normalized stylometric FEATURE vector, reusing the exact feature
    definitions from `agent/style_sheet.py` (length, sentence shape, punctuation
    rates, casing, contractions, emoji, function-word rate). Used only if the
    model can't load offline. Computed on the fly over the (small) candidate pool,
    so style retrieval keeps working with zero model dependency.

Fully local. The model is cached on-box (~242MB) and preloaded at service
warm-up, so call-time never hits the network. Every public path is guarded: if
the style embedder can't be built, callers fall back to topic-only retrieval and
drafting never breaks.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache

# Reuse the SAME stylometric feature primitives the living style sheet uses, so
# the fallback style vector measures exactly what the style sheet describes.
from agent.style_sheet import (
    _CONTRACTION_RE,
    _EMOJI_RE,
    _FUNCTION_WORDS,
    _SENTENCE_SPLIT_RE,
    _WORD_RE,
)

log = logging.getLogger("mirror.style_embed")

# StyleDistance: a sentence-transformers model whose embedding space is writing
# style, not topic. 768-dim — this is the dimension of message_style_embeddings.
DEFAULT_STYLE_MODEL = "StyleDistance/styledistance"
STYLE_MODEL_DIM = 768


# --------------------------------------------------------------------------- #
# Embedder interface + implementations
# --------------------------------------------------------------------------- #
class StyleEmbedder(ABC):
    name: str
    dim: int
    # True when this embedder's vectors match the stored `message_style_embeddings`
    # column (i.e. the same model that produced the backfill). Only then may its
    # vectors be persisted / compared against stored vectors.
    persistable: bool

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized style vector per input text."""


class StyleModelEmbedder(StyleEmbedder):
    """StyleDistance (or any sentence-transformers style model). Local, no key."""

    persistable = True

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"style-model:{model_name}"
        # `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension`
        # in newer sentence-transformers; support both.
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(dim_fn())

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


class StyleFeatureEmbedder(StyleEmbedder):
    """Deterministic stylometric feature vector — the model-free fallback.

    Not stored (dimension differs from the model column); computed on the fly for
    the query and candidate pool so style retrieval survives a missing model.
    """

    persistable = False
    name = "style-features"
    dim = 13

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [stylo_feature_vector(t) for t in texts]


def stylo_feature_vector(text: str) -> list[float]:
    """A small, content-independent style vector for one message, built from the
    same stylometric features `agent/style_sheet.py` profiles over the corpus.
    Each component is scaled to roughly [0, 1], then the whole vector is
    L2-normalized so cosine similarity is meaningful."""
    t = (text or "").strip()
    if not t:
        return [0.0] * StyleFeatureEmbedder.dim

    words = _WORD_RE.findall(t)
    n_words = max(1, len(words))
    n_chars = max(1, len(t))
    avg_word_len = sum(len(w) for w in words) / n_words

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(t) if s.strip()]
    sent_word_lens = [len(_WORD_RE.findall(s)) for s in sentences if s.strip()]
    avg_sent_len = (sum(sent_word_lens) / len(sent_word_lens)) if sent_word_lens else float(n_words)

    fragment = 0.0 if re.search(r"[.!?]", t) else 1.0
    lower_start = 1.0 if t[:1].islower() else 0.0
    all_lower = 1.0 if (t == t.lower() and t != t.upper()) else 0.0
    upper_chars = sum(1 for c in t if c.isupper()) / n_chars

    comma_rate = t.count(",") / n_words
    question = min(1.0, t.count("?") / n_words * 4)
    exclam = min(1.0, t.count("!") / n_words * 4)
    ellipsis = 1.0 if re.search(r"\.\.\.|…", t) else 0.0
    emoji_rate = min(1.0, len(_EMOJI_RE.findall(t)) / n_words * 4)
    contraction_rate = len(_CONTRACTION_RE.findall(t)) / n_words
    func_rate = sum(1 for w in words if w.lower() in _FUNCTION_WORDS) / n_words

    raw = [
        min(1.0, math.log1p(n_chars) / math.log1p(600)),  # length (log-scaled)
        min(1.0, avg_word_len / 12.0),
        min(1.0, avg_sent_len / 30.0),
        fragment,
        lower_start,
        all_lower,
        min(1.0, upper_chars * 5),
        min(1.0, comma_rate * 5),
        question,
        exclam,
        ellipsis,
        min(1.0, emoji_rate),
        func_rate,
    ][: StyleFeatureEmbedder.dim]
    # Pad defensively in case the feature list and dim ever drift.
    raw += [0.0] * (StyleFeatureEmbedder.dim - len(raw))

    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# --------------------------------------------------------------------------- #
# Runtime selection (cached singleton) + async helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_style_embedder() -> StyleEmbedder | None:
    """Build the active style embedder once per process. Honors STYLE_RETRIEVAL_MODE
    (auto|model|features|off) and STYLE_EMBED_MODEL. Returns None when style
    retrieval is disabled or unavailable — callers then use topic-only ranking.

    Blocking (loads the model); call via `load_style_embedder()` off the loop.
    """
    mode = (os.getenv("STYLE_RETRIEVAL_MODE", "auto") or "auto").strip().lower()
    if mode == "off":
        log.info("style retrieval disabled (STYLE_RETRIEVAL_MODE=off)")
        return None
    if mode == "features":
        return StyleFeatureEmbedder()

    model_name = os.getenv("STYLE_EMBED_MODEL", DEFAULT_STYLE_MODEL).strip()
    try:
        embedder = StyleModelEmbedder(model_name)
        log.info("style embedder ready: %s (dim=%d)", embedder.name, embedder.dim)
        return embedder
    except Exception as exc:
        if mode == "model":
            log.warning("style model %s failed to load and mode=model: %s", model_name, exc)
            return None
        log.warning(
            "style model %s failed to load; falling back to stylometric features: %s",
            model_name,
            exc,
        )
        return StyleFeatureEmbedder()


async def load_style_embedder() -> StyleEmbedder | None:
    """Async-safe accessor: builds/caches the embedder off the event loop."""
    return await asyncio.to_thread(get_style_embedder)


async def embed_style(texts: list[str]) -> tuple[list[list[float]] | None, StyleEmbedder | None]:
    """Embed a batch of texts with the active style embedder. Returns
    (vectors, embedder) or (None, None) if style embedding is unavailable. Never
    raises — a failure degrades to topic-only retrieval upstream."""
    if not texts:
        return [], None
    embedder = await load_style_embedder()
    if embedder is None:
        return None, None
    try:
        vectors = await asyncio.to_thread(embedder.encode, texts)
        return vectors, embedder
    except Exception as exc:
        log.warning("style embedding failed (ignored): %s", exc)
        return None, None


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs are normalized by their producers, but we divide
    by norms anyway so a stray un-normalized vector can't blow up the score."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
