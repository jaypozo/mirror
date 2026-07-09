"""Needs-reply gate for incoming messages before draft generation.

Default is conservative: uncertain, empty, invalid, timed-out, or failed model
classification means SKIP. This protects the owner's attention by avoiding noisy
draft cards.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from agent.llm import LLMClient, build_llm_client

NEEDS_REPLY = "NEEDS_REPLY"
SKIP = "SKIP"

_WORD_RE = re.compile(r"\s+")
_QUESTION_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byour call\b"),
    re.compile(r"\bok(?:ay)?\s+to\b"),
    re.compile(r"\bshould\s+(?:i|we)\b"),
    re.compile(r"\bwant\s+me\s+to\b"),
    re.compile(r"\bconfirm(?:ing|ed)?\b"),
    re.compile(r"\b(?:which|what)\s+do\s+you\b"),
)
_STATUS_PREFIXES = (
    "fyi",
    "fyi:",
    "status:",
    "update:",
    "heads up",
    "heads up:",
)
_STATUS_WORDS = {
    "done",
    "merged",
    "ready",
    "shipped",
    "fixed",
    "complete",
    "completed",
}
_STATUS_STARTS = tuple(sorted(_STATUS_WORDS | {"updated"}))


@dataclass(frozen=True)
class NeedsReplyDecision:
    verdict: str
    stage: str
    reason: str

    @property
    def needs_reply(self) -> bool:
        return self.verdict == NEEDS_REPLY


def _normalize(text: str) -> str:
    return _WORD_RE.sub(" ", (text or "").strip().lower())


def _strip_terminal_punct(text: str) -> str:
    return text.strip(" \t\r\n.!:,;")


def heuristic_needs_reply(text: str) -> NeedsReplyDecision | None:
    normalized = _normalize(text)
    if not normalized:
        return NeedsReplyDecision(SKIP, "heuristic", "empty message")

    if normalized.endswith("?"):
        return NeedsReplyDecision(NEEDS_REPLY, "heuristic", "question mark")

    compact = _strip_terminal_punct(normalized)
    if compact in _STATUS_WORDS:
        return NeedsReplyDecision(SKIP, "heuristic", f"status word: {compact}")
    if any(normalized.startswith(f"{word} ") for word in _STATUS_STARTS):
        return NeedsReplyDecision(SKIP, "heuristic", "status statement")
    if "ready for your merge" in normalized:
        return NeedsReplyDecision(SKIP, "heuristic", "ready-for-merge status")
    if any(normalized == p.rstrip(":") or normalized.startswith(p + " ") for p in _STATUS_PREFIXES):
        return NeedsReplyDecision(SKIP, "heuristic", "status/FYI prefix")

    for pattern in _QUESTION_PHRASES:
        if pattern.search(normalized):
            return NeedsReplyDecision(NEEDS_REPLY, "heuristic", f"phrase: {pattern.pattern}")

    return None


def _timeout_seconds() -> float:
    raw = os.getenv("MIRROR_NEEDS_REPLY_TIMEOUT_SECONDS") or os.getenv(
        "CODEX_DRAFT_TIMEOUT_SECONDS", "10"
    )
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 10.0


def _classifier_client() -> LLMClient:
    return build_llm_client(
        model=os.getenv("DRAFT_MODEL") or os.getenv("LLM_MODEL", "gpt-5.5"),
        reasoning_effort=os.getenv("CODEX_DRAFT_REASONING_EFFORT")
        or os.getenv("CODEX_REASONING_EFFORT", "low"),
        timeout_seconds=int(_timeout_seconds()),
    )


def _parse_classifier_response(text: str) -> str | None:
    token = (text or "").strip().upper()
    if token in {NEEDS_REPLY, SKIP}:
        return token
    first = re.split(r"[^A-Z_]+", token, maxsplit=1)[0] if token else ""
    if first in {NEEDS_REPLY, SKIP}:
        return first
    return None


async def classify_needs_reply(
    text: str,
    *,
    llm: LLMClient | None = None,
    timeout_seconds: float | None = None,
) -> NeedsReplyDecision:
    heuristic = heuristic_needs_reply(text)
    if heuristic is not None:
        return heuristic

    client = llm or _classifier_client()
    prompt = (
        "Does this message require Jay to reply (a question, decision, or "
        "approval directed at him)? Answer NEEDS_REPLY or SKIP.\n\n"
        f"Message:\n{text.strip()}"
    )
    try:
        response = await asyncio.wait_for(
            client.complete(
                [
                    {
                        "role": "system",
                        "content": "Classify conservatively. If uncertain, answer SKIP.",
                    },
                    {"role": "user", "content": prompt},
                ]
            ),
            timeout=timeout_seconds if timeout_seconds is not None else _timeout_seconds(),
        )
    except asyncio.TimeoutError:
        return NeedsReplyDecision(SKIP, "classifier", "classifier timeout")
    except Exception as exc:
        return NeedsReplyDecision(SKIP, "classifier", f"classifier error: {exc}")

    verdict = _parse_classifier_response(response)
    if verdict is None:
        return NeedsReplyDecision(SKIP, "classifier", "empty/invalid classifier response")
    return NeedsReplyDecision(verdict, "classifier", f"classifier returned {verdict}")
