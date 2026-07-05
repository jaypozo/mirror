"""Living style sheet — a distilled "how the owner writes" guide.

The style sheet is the always-injected, persistent description of the owner's
voice. It is built PRIMARILY from the owner's own message corpus (the historical
`direction='out'` messages in Postgres are the main signal), then refined by the
edit-feedback loop.

Pipeline (`generate_style_sheet`):
  1. Sample the owner's own sent messages from the corpus.
  2. Compute a local stylometric profile over that sample (length distribution,
     sentence-length variance, fragments, punctuation incl. em-dashes,
     capitalization, contractions, openings/closings, emoji, signature n-grams,
     function-word tendencies). No LLM, no network — pure Python.
  3. Pull the accumulated owner edits (feedback action='edit').
  4. Make ONE LLM call (the same `codex exec` path the drafter uses) that turns
     the stylometry + sample messages + edits into a concise markdown style guide
     AND proposes candidate correction rules attributed to the edits that support
     them.
  5. Stage the candidate rules: a rule only graduates into the ACTIVE guide once
     it is backed by >= STYLE_RULE_PROMOTE_THRESHOLD independent edits. One-offs
     stay as candidates; rules that stop recurring decay out.
  6. Persist the active guide (the LLM guide + a "promoted corrections" section)
     as the newest `style_sheet` row.

`get_active_style_sheet()` is the cheap, LLM-free read used by the drafter on
every draft. It returns the latest active guide markdown (or None), so a missing
or failed style sheet degrades gracefully — drafting still works without it.

Regenerate on demand:

    .venv/bin/python -m agent.style_sheet

Schedule it (e.g. daily) with the systemd units in `deploy/systemd/` — do not run
it inline with drafting; it is a periodic background refresh.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import asyncpg
from dotenv import load_dotenv

from agent.llm import LLMClient, build_llm_client


# --------------------------------------------------------------------------- #
# Config (all overridable via .env)
# --------------------------------------------------------------------------- #

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# How many of the owner's own messages to sample for the stylometric profile.
SAMPLE_SIZE_DEFAULT = 400
# How many sample messages to show the LLM verbatim (kept small to bound cost).
MAX_SAMPLE_IN_PROMPT = 60
# How many recent owner edits to feed the LLM as refinement signal.
MAX_EDITS_DEFAULT = 40
# A candidate rule graduates to the ACTIVE guide once >= this many independent
# edits support it. One-offs never make it into the guide (anti-overfit).
PROMOTE_THRESHOLD_DEFAULT = 3
# A promoted rule decays back out after this many consecutive regenerations in
# which no supporting edit reappears.
DECAY_MISSES_DEFAULT = 3


# Common contraction tokens (owners who write "I'll / don't / that's" read very
# differently from ones who spell everything out).
_CONTRACTION_RE = re.compile(
    r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE
)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoji
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators
    "\U00002190-\U000021FF"  # arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "]",
    flags=re.UNICODE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z']+")

# A small closed-class function-word set; their rates are a classic, highly
# content-independent stylometric fingerprint.
_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "so", "if", "then", "because",
    "i", "you", "we", "they", "it", "he", "she", "me", "us", "them",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "as",
    "is", "are", "was", "be", "been", "do", "did", "does", "have", "has",
    "not", "no", "yes", "just", "really", "very", "maybe", "actually",
    "this", "that", "these", "those", "here", "there", "now", "will",
}


# --------------------------------------------------------------------------- #
# Local stylometry (no LLM, no network)
# --------------------------------------------------------------------------- #

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 2)


def compute_stylometry(messages: list[str]) -> dict[str, Any]:
    """Aggregate a content-independent style profile over the owner's messages.

    Everything here is a measurable stylometric feature: message-length
    distribution, sentence-length mean + variance, fragment share, punctuation
    habits (incl. em-dash usage), casing, contractions, emoji, openings/closings,
    signature n-grams, and function-word rates.
    """
    texts = [t.strip() for t in messages if t and t.strip()]
    n = len(texts)
    if n == 0:
        return {"sample_size": 0}

    char_lens = [len(t) for t in texts]
    word_lists = [_WORD_RE.findall(t) for t in texts]
    word_counts = [len(w) for w in word_lists]

    sentence_word_lens: list[int] = []
    fragment_msgs = 0
    for t in texts:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(t) if s.strip()]
        for s in sentences:
            wl = len(_WORD_RE.findall(s))
            if wl:
                sentence_word_lens.append(wl)
        # Fragment proxy: no terminal sentence punctuation anywhere in the message.
        if not re.search(r"[.!?]", t):
            fragment_msgs += 1

    total_chars = max(1, sum(char_lens))
    total_words = max(1, sum(word_counts))

    def _rate_per_1k_chars(count: int) -> float:
        return round(count / total_chars * 1000, 2)

    em_dashes = sum(t.count("—") + t.count(" - ") for t in texts)
    ellipses = sum(len(re.findall(r"\.\.\.|…", t)) for t in texts)
    exclamations = sum(t.count("!") for t in texts)
    questions = sum(t.count("?") for t in texts)
    commas = sum(t.count(",") for t in texts)
    emoji = sum(len(_EMOJI_RE.findall(t)) for t in texts)
    contractions = sum(len(_CONTRACTION_RE.findall(t)) for t in texts)

    lower_start = sum(1 for t in texts if t[:1].islower())
    all_lower = sum(1 for t in texts if t == t.lower() and t != t.upper())

    # Openings / closings: first and last word (lowercased) across messages.
    openings = Counter(w[0].lower() for w in word_lists if w)
    closings = Counter(w[-1].lower() for w in word_lists if w)

    # Signature phrases: most common 2- and 3-grams (lowercased).
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    func_counter: Counter = Counter()
    for w in word_lists:
        low = [x.lower() for x in w]
        for tok in low:
            if tok in _FUNCTION_WORDS:
                func_counter[tok] += 1
        for i in range(len(low) - 1):
            bigrams[" ".join(low[i : i + 2])] += 1
        for i in range(len(low) - 2):
            trigrams[" ".join(low[i : i + 3])] += 1

    sent_mean = round(statistics.mean(sentence_word_lens), 2) if sentence_word_lens else 0.0
    sent_stdev = (
        round(statistics.pstdev(sentence_word_lens), 2) if len(sentence_word_lens) > 1 else 0.0
    )

    return {
        "sample_size": n,
        "message_length_chars": {
            "mean": round(statistics.mean(char_lens), 1),
            "median": round(statistics.median(char_lens), 1),
            "p10": _percentile([float(c) for c in char_lens], 0.10),
            "p90": _percentile([float(c) for c in char_lens], 0.90),
        },
        "message_length_words": {
            "mean": round(statistics.mean(word_counts), 2),
            "median": round(statistics.median(word_counts), 1),
        },
        "sentence_length_words": {"mean": sent_mean, "stdev": sent_stdev},
        "fragment_share": round(fragment_msgs / n, 3),
        "punctuation_per_1k_chars": {
            "em_dash": _rate_per_1k_chars(em_dashes),
            "ellipsis": _rate_per_1k_chars(ellipses),
            "exclamation": _rate_per_1k_chars(exclamations),
            "question": _rate_per_1k_chars(questions),
            "comma": _rate_per_1k_chars(commas),
        },
        "casing": {
            "lowercase_start_share": round(lower_start / n, 3),
            "all_lowercase_share": round(all_lower / n, 3),
        },
        "contractions_per_1k_words": round(contractions / total_words * 1000, 2),
        "emoji_per_message": round(emoji / n, 3),
        "top_openings": [w for w, _ in openings.most_common(8)],
        "top_closings": [w for w, _ in closings.most_common(8)],
        "signature_bigrams": [p for p, c in bigrams.most_common(12) if c > 1],
        "signature_trigrams": [p for p, c in trigrams.most_common(8) if c > 1],
        "function_word_rate_per_1k_words": {
            w: round(c / total_words * 1000, 1) for w, c in func_counter.most_common(15)
        },
    }


# --------------------------------------------------------------------------- #
# DB helpers
# --------------------------------------------------------------------------- #

async def _connect() -> asyncpg.Connection | None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[style_sheet] DATABASE_URL missing", file=sys.stderr)
        return None
    try:
        return await asyncpg.connect(database_url)
    except Exception as exc:  # pragma: no cover - connection error path
        print(f"[style_sheet] connect failed: {exc}", file=sys.stderr)
        return None


async def _sample_owner_messages(conn: asyncpg.Connection, limit: int) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT text
        FROM messages
        WHERE direction = 'out'
          AND text IS NOT NULL
          AND length(trim(text)) > 0
        ORDER BY random()
        LIMIT $1
        """,
        max(1, int(limit)),
    )
    return [r["text"] for r in rows]


@dataclass(frozen=True)
class _Edit:
    id: int
    original_draft: str
    final_text: str


async def _fetch_edits(conn: asyncpg.Connection, limit: int) -> list[_Edit]:
    # Voice channel (DUAL learning): mine style rules only from edits that carry
    # a voice signal — style/both, plus legacy unclassified rows. Pure 'intent'
    # edits (a decision/fact change) and 'trivial' tweaks are excluded so a
    # decision change never becomes a phrasing "rule".
    rows = await conn.fetch(
        """
        SELECT id, original_draft, final_text
        FROM feedback
        WHERE action = 'edit'
          AND final_text IS NOT NULL
          AND length(trim(final_text)) > 0
          AND original_draft IS NOT NULL
          AND final_text <> original_draft
          AND (edit_kind IS NULL OR edit_kind IN ('style', 'both'))
        ORDER BY ts DESC
        LIMIT $1
        """,
        max(1, int(limit)),
    )

    def _trunc(v: str, cap: int = 600) -> str:
        v = (v or "").strip()
        return v if len(v) <= cap else v[:cap].rstrip() + "…"

    return [
        _Edit(id=r["id"], original_draft=_trunc(r["original_draft"]), final_text=_trunc(r["final_text"]))
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# Staged rule promotion
# --------------------------------------------------------------------------- #

def _rule_key(rule_text: str) -> str:
    """Normalize a rule to a stable key so the same rule re-proposed across runs
    accumulates support instead of duplicating."""
    key = re.sub(r"\s+", " ", (rule_text or "").strip().lower())
    key = re.sub(r"[^a-z0-9 ]+", "", key)
    return key[:120]


async def _apply_candidate_rules(
    conn: asyncpg.Connection,
    candidate_rules: list[dict[str, Any]],
    valid_edit_ids: set[int],
    promote_threshold: int,
    decay_misses: int,
) -> list[str]:
    """Merge this run's LLM-proposed rules into the staging table, accumulating
    the set of independent edits that support each rule. Promote rules with
    enough support; decay ones that stopped recurring. Returns the ACTIVE rule
    texts (promoted), for baking into the guide."""
    seen_keys: set[str] = set()

    for item in candidate_rules:
        rule_text = str(item.get("rule") or "").strip()
        if not rule_text:
            continue
        # Only trust edit ids the LLM was actually shown.
        support = {
            int(e)
            for e in item.get("supporting_edit_ids", [])
            if _safe_int(e) is not None and int(e) in valid_edit_ids
        }
        key = _rule_key(rule_text)
        if not key:
            continue
        seen_keys.add(key)

        existing = await conn.fetchrow(
            "SELECT support_edit_ids, status FROM style_rules WHERE rule_key = $1", key
        )
        if existing is None:
            merged = sorted(support)
            status = "active" if len(merged) >= promote_threshold else "candidate"
            await conn.execute(
                """
                INSERT INTO style_rules
                    (rule_key, rule_text, status, support_edit_ids, support_count, misses,
                     first_seen, last_seen)
                VALUES ($1, $2, $3, $4, $5, 0, now(), now())
                """,
                key, rule_text, status, merged, len(merged),
            )
        else:
            prior = set(existing["support_edit_ids"] or [])
            merged = sorted(prior | support)
            status = "active" if len(merged) >= promote_threshold else existing["status"]
            if status == "decayed" and len(merged) >= promote_threshold:
                status = "active"
            await conn.execute(
                """
                UPDATE style_rules
                   SET rule_text = $2, status = $3, support_edit_ids = $4,
                       support_count = $5, misses = 0, last_seen = now()
                 WHERE rule_key = $1
                """,
                key, rule_text, status, merged, len(merged),
            )

    # Decay: any rule NOT re-proposed this run gets a strike; promoted rules that
    # miss too many runs fall back out of the active guide.
    stale = await conn.fetch("SELECT rule_key, misses, status FROM style_rules")
    for r in stale:
        if r["rule_key"] in seen_keys:
            continue
        misses = (r["misses"] or 0) + 1
        new_status = r["status"]
        if r["status"] == "active" and misses >= decay_misses:
            new_status = "decayed"
        await conn.execute(
            "UPDATE style_rules SET misses = $2, status = $3 WHERE rule_key = $1",
            r["rule_key"], misses, new_status,
        )

    active = await conn.fetch(
        "SELECT rule_text FROM style_rules WHERE status = 'active' ORDER BY support_count DESC, last_seen DESC"
    )
    return [r["rule_text"] for r in active]


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# LLM distillation
# --------------------------------------------------------------------------- #

_STYLE_SYSTEM_PROMPT = """You are a stylometrician building a concise, reusable "writing style sheet" for one person (the owner), so another model can draft messages indistinguishable from how the owner writes.

You are given (1) a measured stylometric profile of the owner's own messages, (2) a sample of their real messages, and (3) recent cases where the owner edited a machine draft before sending (drafted vs. what they actually sent).

Produce a SHORT markdown style guide grounded in the numbers — do not invent traits the data does not support. Cover, only where the evidence is clear: typical message length, sentence length and how much it varies, fragments vs. full sentences, punctuation habits (including em-dash usage), capitalization/casing, contractions, common openings/closings/sign-offs, directness vs. hedging (lead-with-the-point / BLUF), formatting (bullets, line breaks), emoji usage, and any signature phrases or tics. Keep it tight and prescriptive — a drafter should be able to follow it.

Then, from the EDITS ONLY, propose discrete "candidate rules": specific, generalizable corrections the owner tends to make (e.g. "drop greetings and go straight to the point", "prefer 'yeah' over 'yes'"). For each rule, list the ids of the edits that support it, using only the edit ids you were shown. Do not fabricate rules that a single edit barely hints at."""


def _build_user_prompt(
    stylometry: dict[str, Any], sample: list[str], edits: list[_Edit]
) -> str:
    sample_block = "\n".join(f"- {s.strip()[:280]}" for s in sample[:MAX_SAMPLE_IN_PROMPT])
    if edits:
        edit_block = "\n\n".join(
            f"Edit id {e.id}:\n  drafted: {e.original_draft}\n  owner sent: {e.final_text}"
            for e in edits
        )
    else:
        edit_block = "(no edits recorded yet)"

    return f"""MEASURED STYLOMETRIC PROFILE (JSON):
{json.dumps(stylometry, indent=2)}

SAMPLE OF THE OWNER'S OWN MESSAGES:
{sample_block or "(no messages sampled)"}

RECENT OWNER EDITS (drafted vs. what the owner actually sent):
{edit_block}

Return ONLY a JSON object, no prose, of exactly this shape:
{{
  "style_guide_md": "<a short markdown style guide, grounded in the profile>",
  "candidate_rules": [
    {{"rule": "<one specific correction rule>", "supporting_edit_ids": [<edit ids that support it>]}}
  ]
}}
If there are no edits, return an empty candidate_rules array."""


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    """Best-effort: pull the JSON object out of the LLM output (it may wrap it in
    prose or a ```json fence)."""
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _fallback_guide(stylometry: dict[str, Any]) -> str:
    """Deterministic guide built straight from the numbers, used when the LLM is
    unavailable or returns unparseable output — the drafter still gets signal."""
    if stylometry.get("sample_size", 0) == 0:
        return "# Owner style sheet\n\n(Not enough corpus data yet.)"
    ml = stylometry.get("message_length_chars", {})
    sl = stylometry.get("sentence_length_words", {})
    casing = stylometry.get("casing", {})
    punct = stylometry.get("punctuation_per_1k_chars", {})
    lines = [
        "# Owner style sheet (auto, from corpus stylometry)",
        "",
        f"- Typical message length: ~{ml.get('median', '?')} chars (p10 {ml.get('p10', '?')}, p90 {ml.get('p90', '?')}). Keep replies short.",
        f"- Sentence length: ~{sl.get('mean', '?')} words, stdev {sl.get('stdev', '?')} — vary sentence length, don't be uniform.",
        f"- Fragments: {round(stylometry.get('fragment_share', 0) * 100)}% of messages have no terminal punctuation — fragments are fine.",
        f"- Casing: starts lowercase {round(casing.get('lowercase_start_share', 0) * 100)}% of the time; all-lowercase {round(casing.get('all_lowercase_share', 0) * 100)}%.",
        f"- Em-dash usage: {punct.get('em_dash', 0)} per 1k chars — {'rare, avoid' if punct.get('em_dash', 0) < 0.5 else 'used'}.",
        f"- Contractions: {stylometry.get('contractions_per_1k_words', 0)} per 1k words.",
        f"- Emoji: {stylometry.get('emoji_per_message', 0)} per message.",
    ]
    openings = stylometry.get("top_openings") or []
    if openings:
        lines.append(f"- Common openings: {', '.join(openings[:6])}.")
    sigs = stylometry.get("signature_bigrams") or []
    if sigs:
        lines.append(f"- Signature phrases: {', '.join(sigs[:8])}.")
    lines.append("- Lead with the point (BLUF). No hype, no filler, no preamble.")
    return "\n".join(lines)


def _compose_active_guide(guide_md: str, active_rules: list[str]) -> str:
    guide = (guide_md or "").strip()
    if active_rules:
        bullets = "\n".join(f"- {r}" for r in active_rules)
        guide += (
            "\n\n## Learned corrections (promoted from the owner's edits)\n"
            "These recur across multiple edits — always apply them:\n"
            f"{bullets}\n"
        )
    return guide.strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

async def get_active_style_sheet() -> str | None:
    """Return the current active style guide markdown, or None. Cheap, LLM-free,
    fully guarded — safe to call on every draft."""
    conn = await _connect()
    if conn is None:
        return None
    try:
        row = await conn.fetchrow(
            "SELECT guide_md FROM style_sheet WHERE active ORDER BY generated_at DESC LIMIT 1"
        )
        if row and (row["guide_md"] or "").strip():
            return row["guide_md"].strip()
        return None
    except Exception as exc:
        print(f"[style_sheet] read failed (ignored): {exc}", file=sys.stderr)
        return None
    finally:
        await conn.close()


async def generate_style_sheet(
    llm: LLMClient | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """(Re)generate the living style sheet from the corpus + edits and persist it
    as the newest active row. Returns a small result dict for the CLI/logs."""
    load_dotenv()
    sample_size = sample_size or _int_env("STYLE_SHEET_SAMPLE_SIZE", SAMPLE_SIZE_DEFAULT)
    max_edits = _int_env("STYLE_SHEET_MAX_EDITS", MAX_EDITS_DEFAULT)
    promote_threshold = _int_env("STYLE_RULE_PROMOTE_THRESHOLD", PROMOTE_THRESHOLD_DEFAULT)
    decay_misses = _int_env("STYLE_RULE_DECAY_MISSES", DECAY_MISSES_DEFAULT)

    conn = await _connect()
    if conn is None:
        raise SystemExit("Missing/unreachable DATABASE_URL — cannot generate style sheet.")

    try:
        sample = await _sample_owner_messages(conn, sample_size)
        edits = await _fetch_edits(conn, max_edits)
        stylometry = compute_stylometry(sample)

        client = llm or build_llm_client()
        guide_md = ""
        candidate_rules: list[dict[str, Any]] = []
        try:
            raw = await client.complete(
                [
                    {"role": "system", "content": _STYLE_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(stylometry, sample, edits)},
                ]
            )
            parsed = _parse_llm_json(raw)
            if parsed:
                guide_md = str(parsed.get("style_guide_md") or "").strip()
                rules = parsed.get("candidate_rules")
                if isinstance(rules, list):
                    candidate_rules = [r for r in rules if isinstance(r, dict)]
        except Exception as exc:
            print(f"[style_sheet] LLM distillation failed; using stylometry fallback: {exc}", file=sys.stderr)

        if not guide_md:
            guide_md = _fallback_guide(stylometry)

        valid_edit_ids = {e.id for e in edits}
        active_rules = await _apply_candidate_rules(
            conn, candidate_rules, valid_edit_ids, promote_threshold, decay_misses
        )

        active_guide = _compose_active_guide(guide_md, active_rules)

        # Newest row wins; retire the previous active guide(s).
        async with conn.transaction():
            await conn.execute("UPDATE style_sheet SET active = false WHERE active")
            row = await conn.fetchrow(
                """
                INSERT INTO style_sheet (guide_md, rubric, sample_size, active, model, generated_at)
                VALUES ($1, $2::jsonb, $3, true, $4, now())
                RETURNING id
                """,
                active_guide,
                json.dumps(stylometry),
                stylometry.get("sample_size", 0),
                getattr(client, "model", None),
            )

        return {
            "style_sheet_id": row["id"],
            "sample_size": stylometry.get("sample_size", 0),
            "edits_considered": len(edits),
            "candidate_rules_proposed": len(candidate_rules),
            "active_rules": len(active_rules),
            "guide_chars": len(active_guide),
        }
    finally:
        await conn.close()


async def _main() -> None:
    result = await generate_style_sheet()
    print(json.dumps(result, indent=2))
    guide = await get_active_style_sheet()
    print("\n----- ACTIVE STYLE SHEET -----\n")
    print(guide or "(none)")


if __name__ == "__main__":
    asyncio.run(_main())
