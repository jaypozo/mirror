"""Classify an owner edit so the two learning channels stay separate.

An owner edit (original_draft -> final_text) can change:
  * STYLE   — phrasing, tone, voice, length, punctuation; meaning unchanged.
  * INTENT  — the substance: a decision, fact, number, commitment, or answer
              changed. The voice may be untouched.
  * BOTH    — wording AND meaning changed.
  * TRIVIAL — a negligible tweak (typo, spacing) that changes neither.

Why it matters: treating every edit the same trains the voice model on decision
changes (so a fact correction would teach the drafter a phrasing "rule" it never
was). Classification routes each edit to the right channel:
  * STYLE / BOTH  -> voice (style sheet + edit few-shots).
  * INTENT / BOTH -> intent notes (per-thread decisions).

One guarded LLM call (the same codex-exec path the drafter uses). Any failure
degrades to a local heuristic; classification NEVER breaks capture or sending.
"""

from __future__ import annotations

import difflib
import json
import re
import sys

from agent.llm import LLMClient, build_llm_client

VALID_KINDS = {"style", "intent", "both", "trivial"}


_CLASSIFY_SYSTEM_PROMPT = """You compare a machine-drafted reply with what the owner actually sent before it went out, and label HOW the owner changed it.

Return ONLY compact JSON, no prose: {"kind": "<one label>", "note": "<one short line>"}.

kind is exactly one of:
- "style"   — same meaning, decision, and facts; only phrasing, tone, voice, length, punctuation, casing, or word choice changed.
- "intent"  — the substance changed: a different decision, answer, fact, number, commitment, name, or instruction. The voice may be untouched.
- "both"    — the wording AND the meaning/decision changed.
- "trivial" — a negligible tweak (typo, spacing, single-word swap) that changes neither the voice nor the meaning materially.

note: ONE short line naming what changed. For "intent"/"both", state the decision or fact the owner actually chose (this is stored as a durable decision). For "style", name the voice change. No preamble, no quotes.

Judge MEANING, not length. A shorter reply that says the same thing is "style". Do not invent changes that are not present."""


def _build_user_prompt(original_draft: str, final_text: str) -> str:
    return (
        "Drafted (what the machine proposed):\n"
        f"{original_draft}\n\n"
        "Owner sent (what actually went out):\n"
        f"{final_text}\n\n"
        'Return only the JSON object {"kind": ..., "note": ...}.'
    )


def _parse(raw: str) -> tuple[str, str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    kind = str(obj.get("kind") or "").strip().lower()
    if kind not in VALID_KINDS:
        return None
    note = str(obj.get("note") or "").strip()
    if len(note) > 500:
        note = note[:500].rstrip() + "…"
    return kind, note


def _heuristic(original: str, final: str) -> tuple[str, str]:
    """Local fallback when the LLM is unavailable. It cannot read meaning, so it
    only distinguishes a near-identical tweak from a real rewrite and defaults the
    latter to 'style' — the historical behavior (every edit trained voice). It
    never emits 'intent' on its own, so a guessed decision never poisons the
    intent channel."""
    ratio = difflib.SequenceMatcher(None, original, final).ratio()
    if ratio >= 0.97:
        return "trivial", "near-identical edit (heuristic)"
    return "style", "reworded (heuristic; LLM classification unavailable)"


async def classify_edit(
    original_draft: str | None,
    final_text: str | None,
    llm: LLMClient | None = None,
) -> tuple[str, str]:
    """Return (kind, note) for one owner edit. Fully guarded — always returns a
    valid (kind, note); never raises."""
    original = (original_draft or "").strip()
    final = (final_text or "").strip()
    if not original or not final or original == final:
        return "trivial", ""

    client = llm or build_llm_client()
    try:
        raw = await client.complete(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(original, final)},
            ]
        )
    except Exception as exc:
        print(f"[edit_classify] LLM classify failed; using heuristic: {exc}", file=sys.stderr)
        return _heuristic(original, final)

    return _parse(raw) or _heuristic(original, final)
