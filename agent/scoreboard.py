"""Scoreboard metrics — is the drafter getting better at mimicking the owner?

Computes, from the `feedback` table (and the owner's real style corpus), a small
set of "am I improving" signals for the Telegram Mini App dashboard:

  * Approve-without-edit rate over time — approves / (approves + edits) per
    bucket; dismisses tracked separately. Rising = the raw draft is landing
    more often without a rewrite.
  * Edit-magnitude trend — for action='edit' rows, the normalized edit distance
    (1 - difflib ratio) between original_draft and final_text, averaged per
    bucket. Falling = when the owner does edit, they change less.
  * Style-similarity trend — cosine of each draft's STYLE embedding to the
    owner's style centroid (their real messages, from message_style_embeddings).
    Rising = the model's drafts are written more in the owner's voice.
  * Totals — approve/edit/dismiss counts and the edit_kind split (style vs
    intent vs both vs trivial).

Everything is guarded and returns a sane empty-state shape when there is little
data. The style axis is optional: if the style embedder or the owner's style
corpus is unavailable it degrades to `null` rather than failing the endpoint.

Also hosts `validate_init_data`: the Telegram WebApp initData HMAC check used to
gate the Mini App to the owner (mirrors the fleet Mini App's validation).
"""

from __future__ import annotations

import difflib
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from typing import Any

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from agent.style_embed import STYLE_MODEL_DIM, cosine, embed_style

log = logging.getLogger("mirror.scoreboard")

# How many of the most recent drafts to style-embed for the style-similarity
# trend (kept bounded so the endpoint stays fast even as feedback grows).
STYLE_DRAFT_SAMPLE = 300
# How many of the owner's real style vectors to average into the centroid.
STYLE_CENTROID_SAMPLE = 2000
# initData is considered stale after this many seconds.
INIT_DATA_TTL_SECONDS = 24 * 60 * 60


# --------------------------------------------------------------------------- #
# Telegram WebApp initData auth (owner-only gate)
# --------------------------------------------------------------------------- #
def validate_init_data(init_data: str, bot_token: str, owner_id: int) -> bool:
    """Return True iff `init_data` is a valid, fresh Telegram WebApp initData
    HMAC-signed by `bot_token` AND belongs to `owner_id`. Never raises."""
    if not init_data or not bot_token or not owner_id:
        return False
    try:
        pairs = urllib.parse.parse_qsl(init_data, keep_blank_values=True)
        data = dict(pairs)
        recv_hash = data.pop("hash", None)
        if not recv_hash:
            return False
        dcs = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        comp = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(comp, recv_hash):
            return False
        auth_date = int(data.get("auth_date", "0") or "0")
        if not auth_date or (time.time() - auth_date) > INIT_DATA_TTL_SECONDS:
            return False
        user = json.loads(data.get("user", "null") or "null")
        return bool(user) and int(user.get("id", 0)) == int(owner_id)
    except Exception as exc:
        log.warning("initData validation error (rejecting): %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Bucketing helpers
# --------------------------------------------------------------------------- #
def _bucket_key(ts: datetime, granularity: str) -> str:
    d = ts.astimezone(timezone.utc).date()
    if granularity == "week":
        monday = d - timedelta(days=d.weekday())
        return monday.isoformat()
    return d.isoformat()


def _pick_granularity(rows: list[asyncpg.Record]) -> str:
    if not rows:
        return "day"
    first = rows[0]["ts"]
    last = rows[-1]["ts"]
    span_days = (last - first).days
    return "week" if span_days > 28 else "day"


def _magnitude(original: str, final: str) -> float:
    """Normalized edit distance in [0, 1]: 0 = identical, 1 = total rewrite."""
    return 1.0 - difflib.SequenceMatcher(None, original or "", final or "").ratio()


# --------------------------------------------------------------------------- #
# Style centroid (owner's real voice) + per-draft style similarity
# --------------------------------------------------------------------------- #
async def _owner_style_centroid(conn: asyncpg.Connection) -> list[float] | None:
    """Average (L2-normalized) of the owner's real message style vectors. Returns
    None if the style corpus is empty or unreadable."""
    try:
        rows = await conn.fetch(
            """
            SELECT embedding
            FROM message_style_embeddings
            ORDER BY embedded_at DESC
            LIMIT $1
            """,
            STYLE_CENTROID_SAMPLE,
        )
    except Exception as exc:
        log.warning("style centroid query failed (skipping style axis): %s", exc)
        return None
    vecs = [list(r["embedding"]) for r in rows if r["embedding"] is not None]
    if not vecs:
        return None
    dim = len(vecs[0])
    acc = [0.0] * dim
    used = 0
    for v in vecs:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            acc[i] += x
        used += 1
    if not used:
        return None
    centroid = [x / used for x in acc]
    norm = sum(x * x for x in centroid) ** 0.5 or 1.0
    return [x / norm for x in centroid]


async def _style_similarity_series(
    conn: asyncpg.Connection,
    draft_rows: list[asyncpg.Record],
    granularity: str,
) -> dict[str, Any] | None:
    """Per-bucket average cosine(draft style, owner centroid). Returns None if the
    style axis is unavailable (no centroid / embedder off)."""
    centroid = await _owner_style_centroid(conn)
    if centroid is None:
        return None

    # Most recent N drafts (rows arrive oldest-first; take the tail).
    sample = draft_rows[-STYLE_DRAFT_SAMPLE:]
    texts = [(r["original_draft"] or "").strip() for r in sample]
    idx = [i for i, t in enumerate(texts) if t]
    if not idx:
        return None

    vecs, embedder = await embed_style([texts[i] for i in idx])
    if not vecs or embedder is None or embedder.dim != len(centroid):
        # Embedder unavailable or dimension mismatch with the stored centroid
        # (e.g. stylometric-feature fallback vs the 768-dim model column).
        log.info(
            "style axis skipped: embedder=%s dim=%s centroid_dim=%s",
            getattr(embedder, "name", None),
            getattr(embedder, "dim", None),
            len(centroid),
        )
        return None

    buckets: dict[str, list[float]] = {}
    for slot, i in enumerate(idx):
        # cosine over pgvector-sourced vectors can be numpy float32; force a
        # native float so the result is JSON-serializable.
        sim = float(cosine(vecs[slot], centroid))
        key = _bucket_key(sample[i]["ts"], granularity)
        buckets.setdefault(key, []).append(sim)

    series = [
        {"bucket": k, "value": round(sum(v) / len(v), 4), "n": len(v)}
        for k, v in sorted(buckets.items())
    ]
    all_sims = [s for v in buckets.values() for s in v]
    return {
        "series": series,
        "current": series[-1]["value"] if series else None,
        "delta": round(series[-1]["value"] - series[-2]["value"], 4)
        if len(series) >= 2
        else None,
        "overall": round(sum(all_sims) / len(all_sims), 4) if all_sims else None,
        "model": getattr(embedder, "name", None),
        "n": len(all_sims),
    }


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #
def _empty_scoreboard() -> dict[str, Any]:
    return {
        "ok": True,
        "empty": True,
        "granularity": "day",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "approve": 0,
            "edit": 0,
            "dismiss": 0,
            "total": 0,
            "edit_kind": {},
        },
        "approve_rate": {"series": [], "current": None, "delta": None, "overall": None},
        "edit_magnitude": {"series": [], "current": None, "delta": None, "overall": None},
        "style_similarity": None,
    }


async def compute_scoreboard(
    database_url: str | None = None,
    granularity: str | None = None,
) -> dict[str, Any]:
    """Compute all scoreboard metrics. Never raises — returns an empty-state shape
    on any failure so the endpoint always answers with valid JSON."""
    load_dotenv()
    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        out = _empty_scoreboard()
        out["reason"] = "no DATABASE_URL"
        return out

    try:
        conn = await asyncpg.connect(database_url)
    except Exception as exc:
        log.warning("scoreboard DB connect failed: %s", exc)
        out = _empty_scoreboard()
        out["reason"] = "db unavailable"
        return out

    try:
        await register_vector(conn)
        rows = await conn.fetch(
            """
            SELECT id, action, original_draft, final_text, edit_kind, ts
            FROM feedback
            ORDER BY ts ASC
            """
        )

        if not rows:
            return _empty_scoreboard()

        gran = (granularity or _pick_granularity(rows)).lower()
        if gran not in ("day", "week"):
            gran = "day"

        # ---- Totals ---------------------------------------------------------
        totals = {"approve": 0, "edit": 0, "dismiss": 0}
        edit_kind_counts: dict[str, int] = {}
        for r in rows:
            action = r["action"]
            if action in totals:
                totals[action] += 1
            if action == "edit":
                kind = r["edit_kind"] or "unclassified"
                edit_kind_counts[kind] = edit_kind_counts.get(kind, 0) + 1
        totals["total"] = len(rows)
        totals["edit_kind"] = edit_kind_counts

        # ---- Per-bucket accumulation ---------------------------------------
        # counts[bucket] = {approve, edit, dismiss}; mags[bucket] = [magnitudes]
        counts: dict[str, dict[str, int]] = {}
        mags: dict[str, list[float]] = {}
        for r in rows:
            key = _bucket_key(r["ts"], gran)
            c = counts.setdefault(key, {"approve": 0, "edit": 0, "dismiss": 0})
            action = r["action"]
            if action in c:
                c[action] += 1
            if action == "edit" and r["final_text"] and r["original_draft"]:
                mags.setdefault(key, []).append(
                    _magnitude(r["original_draft"], r["final_text"])
                )

        ordered = sorted(counts.keys())

        # Approve-without-edit rate = approve / (approve + edit); dismiss separate.
        approve_series = []
        for k in ordered:
            c = counts[k]
            denom = c["approve"] + c["edit"]
            rate = round(c["approve"] / denom, 4) if denom else None
            approve_series.append(
                {
                    "bucket": k,
                    "value": rate,
                    "approve": c["approve"],
                    "edit": c["edit"],
                    "dismiss": c["dismiss"],
                }
            )
        rated = [s for s in approve_series if s["value"] is not None]
        overall_denom = totals["approve"] + totals["edit"]
        approve_rate = {
            "series": approve_series,
            "current": rated[-1]["value"] if rated else None,
            "delta": round(rated[-1]["value"] - rated[-2]["value"], 4)
            if len(rated) >= 2
            else None,
            "overall": round(totals["approve"] / overall_denom, 4)
            if overall_denom
            else None,
        }

        # Edit magnitude (lower = better).
        mag_series = [
            {"bucket": k, "value": round(sum(mags[k]) / len(mags[k]), 4), "n": len(mags[k])}
            for k in ordered
            if k in mags and mags[k]
        ]
        all_mags = [m for v in mags.values() for m in v]
        edit_magnitude = {
            "series": mag_series,
            "current": mag_series[-1]["value"] if mag_series else None,
            "delta": round(mag_series[-1]["value"] - mag_series[-2]["value"], 4)
            if len(mag_series) >= 2
            else None,
            "overall": round(sum(all_mags) / len(all_mags), 4) if all_mags else None,
        }

        # Style similarity (higher = better) — optional axis.
        try:
            style_similarity = await _style_similarity_series(conn, rows, gran)
        except Exception as exc:
            log.warning("style-similarity trend failed (omitting): %s", exc)
            style_similarity = None

        return {
            "ok": True,
            "empty": False,
            "granularity": gran,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "totals": totals,
            "approve_rate": approve_rate,
            "edit_magnitude": edit_magnitude,
            "style_similarity": style_similarity,
        }
    except Exception as exc:
        log.exception("compute_scoreboard failed: %s", exc)
        out = _empty_scoreboard()
        out["reason"] = "compute error"
        return out
    finally:
        await conn.close()
