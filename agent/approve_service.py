"""Mirror inline-approve HTTP service.

Headless companion to the fleet Telegram bots. The bots render the draft card +
buttons themselves (so the card comes from the same agent the owner is talking to);
this service does the two things a Node bot cannot:

  1. /draft  — retrieve owner-voice examples + codex-draft a reply + summary.
  2. /decide — on approve/edit, SEND AS THE OWNER through the Telethon USER session
               (single owner of .telethon/mirror), threaded to the agent's
               question; log approve/edit/dismiss feedback.

DESIGN CONTRAST WITH agent/service.py:
  service.py drives its OWN Mirror bot DM as the approval UI and edits the card
  in place. This service has NO bot client at all — the fleet bot is the UI. So
  this process is: one Telethon USER client (as the owner, connect-only, never login)
  + one aiohttp loopback server. It is the SINGLE OWNER of the user session and,
  like service.py, can optionally resume the history backfill as a background
  task inside the same client (MIRROR_RESUME_BACKFILL=1).

HARD RULES honoured:
  * Nothing is EVER sent to a chat without an explicit /decide approve|edit call
    that originated from the owner tapping a button on the fleet bot.
  * Only ONE process owns the user session. The run script stops the standalone
    drip first. We never open a second concurrent user client.
  * We .connect() the user client and REFUSE if it is not already authorized —
    never .start()/login.
  * Binds to 127.0.0.1 only. A shared-secret header (MIRROR_APPROVE_SECRET) is
    required on every request so nothing else on the box can drive send-as-owner.
  * the configured excluded group's topic threads are refused even if the bot asks.

This service is INERT until deliberately started via approve_service_run.sh. It
is not wired to any bot until MIRROR_APPROVE_ENABLED=1 is set on that bot.

Drop this file into the mirror repo at agent/approve_service.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from telethon import TelegramClient

# Load .env before the env-derived module constants below evaluate.
load_dotenv()

from agent.corpus import add_owner_sample
from agent.draft import draft_reply
from agent.llm import build_llm_client
from agent.edit_classify import classify_edit
from agent.eligibility import EXCLUDED_TOPIC_CHAT_ID
from agent.feedback import record_feedback
from agent.needs_reply import SKIP, NeedsReplyDecision, classify_needs_reply
from agent.scoreboard import compute_scoreboard, validate_init_data
from agent.threads import record_intent_note
from agent.service import maybe_resume_backfill  # reuse the drip resumer as-is
from agent.types import ChatMessage, DraftRequest, DraftResult

log = logging.getLogger("mirror.approve_service")

OWNER_USER_ID = int(os.getenv("MIRROR_OWNER_USER_ID", "0") or "0")
# Bot token used to sign the Mini App's Telegram WebApp initData. Read from the
# environment ONLY (never hard-coded / logged) — set MIRROR_WEBAPP_BOT_TOKEN in
# the service .env to the token of whichever bot hosts the Scoreboard menu button.
WEBAPP_BOT_TOKEN = (os.getenv("MIRROR_WEBAPP_BOT_TOKEN") or "").strip()
# Absolute path to the self-contained Mini App HTML (served over the tunnel).
WEBAPP_HTML_PATH = Path(__file__).resolve().parent.parent / "webapp" / "scoreboard.html"
PENDING_TTL_SECONDS = 60 * 60  # drafts older than this are swept (nothing sent)
DRAFT_LLM_TIMEOUT_SECONDS = 60

# Pending drafts are persisted here so a service restart doesn't orphan an
# undecided card (which would make Approve silently no-op). Only the Pending
# dataclasses are pickled — never the TelegramClient. Path is relative to the
# service WorkingDirectory (logs/ already exists).
PENDING_STORE_PATH = Path(os.getenv("MIRROR_PENDING_STORE", "logs/pending.pkl"))


# --------------------------------------------------------------------------- #
# Pending drafts. Kept in-memory AND mirrored to disk (PENDING_STORE_PATH) so an
# undecided draft survives a restart; entries older than PENDING_TTL_SECONDS are
# dropped on load. created_at is WALL-CLOCK time.time() (monotonic loop time
# resets on restart and can't be compared across processes). Keyed by approval_id.
# --------------------------------------------------------------------------- #
@dataclass
class Pending:
    approval_id: str
    request: DraftRequest
    result: DraftResult
    created_at: float


@dataclass
class ServiceState:
    user: TelegramClient
    pending: dict[str, Pending] = field(default_factory=dict)


STATE: ServiceState | None = None
SECRET: str = ""

# Set once warm-up (see _warm_up) has eager-loaded every lazy heavy init the
# draft path needs. /draft and /decide gate on this so a request that somehow
# arrives before we're fully warm WAITS instead of running against a cold
# pipeline. In the normal boot sequence this is a no-op guard: _warm_up() runs
# to completion before the HTTP site ever binds, so no request can arrive
# first — but the gate is cheap insurance against that invariant changing.
READY = asyncio.Event()
WARMUP_WAIT_TIMEOUT_SECONDS = 120


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _authorized(request: web.Request) -> bool:
    if not SECRET:
        # No secret configured => refuse everything. Fail closed.
        return False
    return request.headers.get("X-Mirror-Secret") == SECRET


def _client_gone(request: web.Request) -> bool:
    """True if the caller's connection is already closed/closing. A slow /draft
    can outlive the caller's own client-side request timeout (the fleet-bot
    plugin aborts after a fixed timeout); when that happens we still finish the
    draft but the caller never sees it — logged loudly so a "200 0" empty
    response in the access log is never a silent mystery."""
    transport = request.transport
    return transport is None or transport.is_closing()


async def _wait_until_ready(timeout: float = WARMUP_WAIT_TIMEOUT_SECONDS) -> bool:
    if READY.is_set():
        return True
    log.info("request arrived before warm-up finished — waiting for readiness")
    try:
        await asyncio.wait_for(READY.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def _warm_up() -> None:
    """Eager-load the embedding model (and any other lazy heavy init the draft
    path needs) BEFORE the server starts accepting requests.

    Root cause this closes: `agent.retrieve.embed_query` lazily loads the local
    sentence-transformers encoder (all-MiniLM-L6-v2) via an lru_cache on first
    call. Every restart used to pay that load cost (several seconds, sometimes
    tens of seconds while huggingface_hub re-validates the local cache over the
    network) on the FIRST real /draft after the restart — on top of the
    drafting call's own latency. The fleet-bot caller aborts /draft after a
    fixed client-side timeout, so that slow first draft came back to it as a
    connection that was already closed by the time we tried to respond, logged
    as a "200 0" empty body in the access log even though the draft itself
    completed successfully (see `_client_gone`).

    Running the SAME warm-up path drafting uses (embed_query, which populates
    the shared encoder cache) here means the first real /draft after a restart
    is as fast as every later one, comfortably inside the caller's timeout.

    Guarded: any failure here is logged loudly but never blocks startup or
    causes a boot-loop — a slow-but-working first draft beats a service that
    won't come up at all.
    """
    start = time.monotonic()
    try:
        from agent.retrieve import embed_query

        vec = await embed_query("warmup")
        log.info(
            "warm-up: topic encoder loaded (dim=%d) in %.1fs",
            len(vec) if vec else 0,
            time.monotonic() - start,
        )
    except Exception as exc:
        log.warning(
            "warm-up: topic encoder failed to preload in %.1fs (ignored — "
            "the first /draft may be slow/cold): %s",
            time.monotonic() - start,
            exc,
        )

    # Also preload the STYLE embedder (StyleDistance) so the first /draft's
    # blended retrieval and the first /decide corpus-add are warm, not cold.
    style_start = time.monotonic()
    try:
        from agent.style_embed import embed_style

        svec, embedder = await embed_style(["warmup"])
        log.info(
            "warm-up: style embedder ready (%s, dim=%s) in %.1fs",
            getattr(embedder, "name", None),
            (len(svec[0]) if svec else None),
            time.monotonic() - style_start,
        )
    except Exception as exc:
        log.warning(
            "warm-up: style embedder failed to preload in %.1fs (ignored — "
            "retrieval falls back to topic-only): %s",
            time.monotonic() - style_start,
            exc,
        )

    READY.set()
    log.info("warm-up complete")


def _is_excluded_topic(chat_id: int, is_topic: bool) -> bool:
    """the excluded group topic threads are excluded; its General channel is included."""
    return chat_id == EXCLUDED_TOPIC_CHAT_ID and bool(is_topic)


def _log_needs_reply_gate(
    *,
    message_id: int | str | None,
    verdict: str,
    stage: str,
    reason: str,
) -> None:
    log.warning(
        "NEEDS_REPLY_GATE message_id=%s verdict=%s stage=%s reason=%s",
        message_id if message_id is not None else "unknown",
        verdict,
        stage,
        reason.replace("\n", " ")[:240],
    )


async def _classify_needs_reply_gate(text: str) -> NeedsReplyDecision:
    try:
        return await classify_needs_reply(text)
    except Exception as exc:
        log.exception("needs-reply gate failed; skipping draft: %s", exc)
        return NeedsReplyDecision(SKIP, "classifier", f"classifier exception: {exc}")


def _thread_from_payload(items: list[dict]) -> list[ChatMessage]:
    thread: list[ChatMessage] = []
    for it in items or []:
        text = str(it.get("text") or "").strip()
        if not text:
            continue
        thread.append(
            ChatMessage(
                text=text,
                sender_name=it.get("sender_name"),
                direction=it.get("direction"),
                message_id=it.get("message_id"),
            )
        )
    return thread


def _sweep_expired() -> None:
    assert STATE is not None
    now = time.time()
    for aid, p in list(STATE.pending.items()):
        if now - p.created_at > PENDING_TTL_SECONDS:
            STATE.pending.pop(aid, None)


def _save_pending() -> None:
    """Persist STATE.pending (Pending dataclasses only) to disk. Best-effort:
    a store failure logs a warning and is otherwise ignored — never crash the
    service over the persistence layer."""
    if STATE is None:
        return
    try:
        PENDING_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PENDING_STORE_PATH.with_suffix(PENDING_STORE_PATH.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(STATE.pending, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, PENDING_STORE_PATH)
    except Exception as exc:
        log.warning("could not persist pending drafts: %s", exc)


def _load_pending() -> None:
    """Restore STATE.pending from disk at startup, dropping entries older than
    PENDING_TTL_SECONDS. Best-effort: any failure logs a warning and leaves
    pending empty so the service still comes up."""
    assert STATE is not None
    if not PENDING_STORE_PATH.exists():
        return
    try:
        with open(PENDING_STORE_PATH, "rb") as fh:
            restored = pickle.load(fh)
    except Exception as exc:
        log.warning("could not load persisted pending drafts: %s", exc)
        return
    if not isinstance(restored, dict):
        log.warning("persisted pending store had unexpected type %s; ignoring", type(restored))
        return
    now = time.time()
    kept = 0
    for aid, p in restored.items():
        try:
            if now - float(p.created_at) <= PENDING_TTL_SECONDS:
                STATE.pending[aid] = p
                kept += 1
        except Exception:
            continue
    log.info("restored %d pending draft(s)", kept)


# --------------------------------------------------------------------------- #
# POST /draft_gate
#   body: {
#     chat_id: int,
#     question_msg_id: int,
#     question: str,
#     is_topic: bool
#   }
#   -> 200 { ok:true, needs_reply:true, gate:{...} }
#      or 204 No Content when no reply should be drafted/rendered
#      or { ok:false, reason } for auth/readiness/bad-json failures
#
# The fleet bot calls this BEFORE rendering a placeholder. /draft repeats the
# same gate as a belt-and-braces guard, so bypassing this endpoint cannot make a
# SKIP message draft or render.
# --------------------------------------------------------------------------- #
async def handle_draft_gate(request: web.Request) -> web.Response:
    if not _authorized(request):
        return web.json_response({"ok": False, "reason": "unauthorized"}, status=401)
    if not await _wait_until_ready():
        log.warning("handle_draft_gate: still warming up after %.0fs, refusing", WARMUP_WAIT_TIMEOUT_SECONDS)
        return web.json_response(
            {"ok": False, "reason": "service still warming up, try again shortly"},
            status=503,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "reason": "bad json"}, status=400)

    chat_id = int(body.get("chat_id"))
    question_msg_id = body.get("question_msg_id")
    question = str(body.get("question") or "").strip()
    is_topic = bool(body.get("is_topic", False))

    if _is_excluded_topic(chat_id, is_topic):
        _log_needs_reply_gate(
            message_id=question_msg_id,
            verdict="SKIP",
            stage="precheck",
            reason="topic thread excluded",
        )
        return web.Response(status=204)
    if not question:
        _log_needs_reply_gate(
            message_id=question_msg_id,
            verdict="SKIP",
            stage="precheck",
            reason="no question text",
        )
        return web.Response(status=204)

    gate = await _classify_needs_reply_gate(question)
    _log_needs_reply_gate(
        message_id=question_msg_id,
        verdict=gate.verdict,
        stage=gate.stage,
        reason=gate.reason,
    )
    if not gate.needs_reply:
        return web.Response(status=204)

    return web.json_response(
        {
            "ok": True,
            "needs_reply": True,
            "gate": {
                "verdict": gate.verdict,
                "stage": gate.stage,
                "reason": gate.reason,
            },
        }
    )


# --------------------------------------------------------------------------- #
# POST /draft
#   body: {
#     chat_id: int,            # the chat the agent messaged the owner in
#     question_msg_id: int,    # the agent's question message id (thread target)
#     question: str,           # the agent's question text (drafting anchor)
#     thread: [{text, sender_name, direction, message_id}, ...],  # optional
#     is_topic: bool           # true if inside a the excluded group topic thread
#   }
#   -> { ok, approval_id, draft, summary: {goal, now, next, open[]} }
#      or 204 No Content when the needs-reply gate decides no reply is needed
#      or 204 No Content when drafting fails after the gate (log-only; callers
#         should delete any placeholder and render nothing)
#      or { ok:false, reason } for auth/readiness/bad-json failures
# --------------------------------------------------------------------------- #
async def handle_draft(request: web.Request) -> web.Response:
    if not _authorized(request):
        return web.json_response({"ok": False, "reason": "unauthorized"}, status=401)
    if not await _wait_until_ready():
        log.warning("handle_draft: still warming up after %.0fs, refusing", WARMUP_WAIT_TIMEOUT_SECONDS)
        return web.json_response(
            {"ok": False, "reason": "service still warming up, try again shortly"},
            status=503,
        )
    assert STATE is not None
    _sweep_expired()

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "reason": "bad json"}, status=400)

    chat_id = int(body.get("chat_id"))
    question_msg_id = body.get("question_msg_id")
    question = str(body.get("question") or "").strip()
    is_topic = bool(body.get("is_topic", False))

    if _is_excluded_topic(chat_id, is_topic):
        _log_needs_reply_gate(
            message_id=question_msg_id,
            verdict="SKIP",
            stage="precheck",
            reason="topic thread excluded",
        )
        return web.Response(status=204)
    if not question:
        _log_needs_reply_gate(
            message_id=question_msg_id,
            verdict="SKIP",
            stage="precheck",
            reason="no question text",
        )
        return web.Response(status=204)

    gate = await _classify_needs_reply_gate(question)
    _log_needs_reply_gate(
        message_id=question_msg_id,
        verdict=gate.verdict,
        stage=gate.stage,
        reason=gate.reason,
    )
    if not gate.needs_reply:
        return web.Response(status=204)

    thread = _thread_from_payload(body.get("thread", []))

    draft_request = DraftRequest(
        incoming_message=question,
        thread=thread,
        source_chat_id=chat_id,
        source_message_id=int(question_msg_id) if question_msg_id is not None else None,
        target_chat_id=chat_id,
        topic_id=None,  # reply threads to the agent's question, not a topic
        metadata={
            "origin": "fleet-bot",
            "chat_id": chat_id,
            "bot_username": (str(body.get("bot_username")).strip() or None)
            if body.get("bot_username")
            else None,
        },
    )

    try:
        result = await draft_reply(
            draft_request,
            include_summary=True,
            llm=build_llm_client(timeout_seconds=DRAFT_LLM_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        log.exception("draft failed; returning no content: %s", exc)
        return web.Response(status=204)

    # Belt-and-braces: draft_reply raises on empty output today, but keep this
    # endpoint contract silent even if a future drafter returns an unusable value.
    if not (result.draft or "").strip():
        log.error("draft failed; LLM returned empty/unusable output")
        return web.Response(status=204)

    approval_id = uuid.uuid4().hex[:12]
    STATE.pending[approval_id] = Pending(
        approval_id=approval_id,
        request=draft_request,
        result=result,
        created_at=time.time(),
    )
    _save_pending()
    log.info("drafted approval_id=%s chat=%s", approval_id, chat_id)

    if _client_gone(request):
        # The draft is real and is now pending (Approve/Edit/Dismiss still work
        # if the caller retries), but THIS response will land on a closed
        # socket — 0 bytes will actually reach the caller. Logged loudly so
        # this is never mistaken for the app silently returning an empty 200.
        log.warning(
            "handle_draft: caller disconnected before the draft finished "
            "(approval_id=%s) — likely their own client-side request timeout; "
            "draft is still pending server-side, nothing is lost",
            approval_id,
        )

    return web.json_response(
        {
            "ok": True,
            "approval_id": approval_id,
            "draft": result.draft,
            "summary": result.summary.to_dict(),
        }
    )


# --------------------------------------------------------------------------- #
# POST /decide
#   body: { approval_id, action: approve|edit|dismiss, edited_text?: str }
#   -> { ok, action, sent: bool } or { ok:false, reason }
#
# approve  -> send result.draft AS THE OWNER, log, drop pending.
# edit     -> send edited_text  AS THE OWNER, log (draft vs final), drop pending.
# dismiss  -> log, drop pending, send nothing.
#
# The fleet bot deletes the draft card only after ok:true.
# --------------------------------------------------------------------------- #
async def handle_decide(request: web.Request) -> web.Response:
    if not _authorized(request):
        return web.json_response({"ok": False, "reason": "unauthorized"}, status=401)
    if not await _wait_until_ready():
        log.warning("handle_decide: still warming up after %.0fs, refusing", WARMUP_WAIT_TIMEOUT_SECONDS)
        return web.json_response(
            {"ok": False, "reason": "service still warming up, try again shortly"},
            status=503,
        )
    assert STATE is not None

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "reason": "bad json"}, status=400)

    approval_id = str(body.get("approval_id") or "")
    action = str(body.get("action") or "")
    edited_text = body.get("edited_text")

    pending = STATE.pending.get(approval_id)
    if pending is None:
        # Unambiguous 410 so the caller (and the plugin) can tell a lost/expired
        # draft apart from a normal decline, even before rendering the reason.
        return web.json_response(
            {
                "ok": False,
                "reason": "expired: this draft was lost (service restarted) — ask again for a fresh one",
            },
            status=410,
        )

    if action == "dismiss":
        await record_feedback(
            original_draft=pending.result.draft,
            final_text=None,
            action="dismiss",
            source_chat_id=pending.request.source_chat_id,
            source_message_id=pending.request.source_message_id,
            target_chat_id=pending.request.target_chat_id,
            topic_id=pending.request.topic_id,
            summary=pending.result.summary,
            metadata=pending.request.metadata,
        )
        STATE.pending.pop(approval_id, None)
        _save_pending()
        return web.json_response({"ok": True, "action": "dismiss", "sent": False})

    if action in ("approve", "edit"):
        final = pending.result.draft if action == "approve" else str(edited_text or "").strip()
        if action == "edit" and not final:
            return web.json_response({"ok": False, "reason": "empty edit"})

        try:
            sent_message = await send_as_owner(pending.request, final)
        except Exception as exc:
            log.exception("send-as-owner failed: %s", exc)
            return web.json_response({"ok": False, "reason": f"send failed: {exc}"}, status=500)

        # CORE REQUIREMENT: the FINAL text the owner just sent (approved as-is, or
        # their edit) is a genuine owner message — add it to the retrievable
        # topic+style corpus immediately so it can be an exemplar right away. Only
        # the sent final is ever indexed; the model's original_draft is NEVER a
        # positive voice sample (it may only appear as the "before" of a
        # contrastive edit-correction, handled separately in feedback). Fully
        # guarded — the message is already sent, so indexing must never fail the
        # request; normal ingest would pick it up regardless.
        try:
            await add_owner_sample(sent_message, final, pending.request)
        except Exception as exc:
            log.warning("add_owner_sample failed (ignored): %s", exc)

        # DUAL learning (Build 1): classify what the owner changed so the two
        # channels stay separate. Sending already succeeded above, so this is
        # fully guarded — a classify/DB failure must not fail the request.
        edit_kind: str | None = None
        edit_note: str | None = None
        if action == "edit":
            try:
                edit_kind, edit_note = await classify_edit(pending.result.draft, final)
            except Exception as exc:
                log.warning("edit classification failed (ignored): %s", exc)

        feedback_id = await record_feedback(
            original_draft=pending.result.draft,
            final_text=final,
            action=action,
            source_chat_id=pending.request.source_chat_id,
            source_message_id=pending.request.source_message_id,
            target_chat_id=pending.request.target_chat_id,
            topic_id=pending.request.topic_id,
            summary=pending.result.summary,
            metadata=pending.request.metadata,
            edit_kind=edit_kind,
            edit_note=edit_note,
        )

        # INTENT component (Build 1 -> Build 2): a decision/substance change is
        # captured as a durable intent note against the matched thread, so future
        # summaries + drafts reflect the real decision. Guarded.
        if action == "edit" and edit_kind in ("intent", "both"):
            try:
                await record_intent_note(
                    note=(edit_note or final),
                    thread_id=pending.result.thread_id,
                    feedback_id=feedback_id,
                    source_chat_id=pending.request.source_chat_id,
                    source_message_id=pending.request.source_message_id,
                )
            except Exception as exc:
                log.warning("record_intent_note failed (ignored): %s", exc)

        STATE.pending.pop(approval_id, None)
        _save_pending()
        log.info(
            "sent-as-owner approval_id=%s action=%s edit_kind=%s",
            approval_id,
            action,
            edit_kind,
        )
        return web.json_response({"ok": True, "action": action, "sent": True})

    return web.json_response({"ok": False, "reason": f"unknown action: {action}"})


# send-as-owner: the ONLY path that touches a chat. We call Telethon directly on
# THIS service's own user client (its own STATE), rather than agent.service's
# send_as_owner which reads that module's global STATE — keeps service.py
# untouched. Behaviour is identical: send text as the owner, threaded to the agent's
# original question message.
async def send_as_owner(request: DraftRequest, text: str):
    """Send `text` as the owner and return the sent Telethon Message (so the
    caller can index the real message id into the corpus)."""
    assert STATE is not None
    if not request.target_chat_id:
        raise RuntimeError("no target_chat_id")
    # A bot DM's API chat_id equals the owner's own user id; sending there from the owner's
    # user session routes to Saved Messages, not the DM. Address the bot's peer
    # instead so the reply lands inline in the conversation. Groups (negative
    # chat_id) are consistent across accounts, so send those as-is.
    peer = request.target_chat_id
    bot_username = (request.metadata or {}).get("bot_username")
    if request.target_chat_id > 0 and bot_username:
        peer = bot_username
    try:
        return await STATE.user.send_message(peer, text, reply_to=request.source_message_id)
    except Exception:
        # reply_to ids differ between the bot API and the user session in a
        # private chat; if threading fails, still deliver the message inline.
        return await STATE.user.send_message(peer, text)


# --------------------------------------------------------------------------- #
# GET /scoreboard  — metrics for the owner-only Telegram Mini App.
#
# Auth (either):
#   * X-Mirror-Secret header (local testing / same shared secret as /draft), OR
#   * a valid Telegram WebApp initData, HMAC-signed by MIRROR_WEBAPP_BOT_TOKEN
#     AND whose user id == the owner. The Mini App sends it in the
#     `X-Telegram-Init-Data` header (or `?init_data=` query for convenience).
#
# Returns the full scoreboard JSON (see agent/scoreboard.compute_scoreboard).
# Read-only: touches no chat, sends nothing. Optional ?bucket=day|week.
# --------------------------------------------------------------------------- #
def _scoreboard_authorized(request: web.Request) -> bool:
    if _authorized(request):
        return True
    init_data = request.headers.get("X-Telegram-Init-Data") or request.query.get("init_data")
    if not init_data:
        return False
    return validate_init_data(init_data, WEBAPP_BOT_TOKEN, OWNER_USER_ID)


async def handle_scoreboard(request: web.Request) -> web.Response:
    if not _scoreboard_authorized(request):
        return web.json_response({"ok": False, "reason": "unauthorized"}, status=401)
    bucket = request.query.get("bucket")
    data = await compute_scoreboard(granularity=bucket)
    return web.json_response(data)


async def handle_webapp(request: web.Request) -> web.Response:
    """Serve the self-contained Mini App HTML. Public (no data leaks — every
    number comes from the auth-gated /scoreboard call the page makes itself)."""
    try:
        return web.FileResponse(WEBAPP_HTML_PATH)
    except Exception as exc:
        log.warning("could not serve scoreboard html: %s", exc)
        return web.Response(status=404, text="not found")


# --------------------------------------------------------------------------- #
# Health.
# --------------------------------------------------------------------------- #
async def handle_health(request: web.Request) -> web.Response:
    assert STATE is not None
    return web.json_response(
        {
            "ok": True,
            "session_owner": OWNER_USER_ID,
            "pending": len(STATE.pending),
        }
    )


# --------------------------------------------------------------------------- #
# Wiring + run.
# --------------------------------------------------------------------------- #
async def build_and_run() -> None:
    global STATE, SECRET
    load_dotenv()

    api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
    api_hash = os.getenv("TELEGRAM_API_HASH", "")
    session = os.getenv("TELEGRAM_SESSION", ".telethon/mirror")
    database_url = os.getenv("DATABASE_URL")
    host = os.getenv("MIRROR_APPROVE_HOST", "127.0.0.1")
    port = int(os.getenv("MIRROR_APPROVE_PORT", "8791"))
    SECRET = (os.getenv("MIRROR_APPROVE_SECRET") or "").strip()

    if not api_id or not api_hash:
        raise SystemExit("Missing TELEGRAM_API_ID / TELEGRAM_API_HASH in .env")
    if not SECRET:
        raise SystemExit(
            "Missing MIRROR_APPROVE_SECRET — refusing to run an unauthenticated "
            "send-as-owner endpoint. Set a random secret in .env (and the matching "
            "value on the fleet bot)."
        )

    # USER client — connect only, never .start()/login.
    user = TelegramClient(session, api_id, api_hash)
    await user.connect()
    if not await user.is_user_authorized():
        raise SystemExit(
            "User session is not authorized — refusing to trigger a login. "
            "Re-auth outside this service."
        )
    me = await user.get_me()
    log.info("user session owner: id=%s username=%s", me.id, getattr(me, "username", None))
    if me.id != OWNER_USER_ID:
        raise SystemExit(
            f"Session owner id={me.id} is not the owner ({OWNER_USER_ID}). Refusing to run."
        )

    STATE = ServiceState(user=user)
    # Restore any undecided drafts orphaned by a restart (best-effort; drops
    # entries past the TTL). Must happen before the server starts serving.
    _load_pending()

    app = web.Application()
    app.add_routes(
        [
            web.post("/draft", handle_draft),
            web.post("/draft_gate", handle_draft_gate),
            web.post("/decide", handle_decide),
            web.get("/health", handle_health),
            web.get("/scoreboard", handle_scoreboard),
            web.get("/scoreboard.html", handle_webapp),
            web.get("/", handle_webapp),
        ]
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)

    stop = asyncio.Event()

    def _sig(*_):
        log.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    async with user:
        # Eager-load the embedding model (and any other lazy heavy init the
        # draft path needs) BEFORE binding, so the very first /draft after this
        # restart is never a cold, slow one. See _warm_up for why.
        await _warm_up()
        await site.start()
        log.info("Mirror approve service listening on http://%s:%s (loopback)", host, port)
        # Optional: resume older-history backfill inside THIS user client, so the
        # single session owner can keep building the corpus. Reuses service.py.
        await maybe_resume_backfill(user, database_url)
        await stop.wait()

    await runner.cleanup()
    log.info("Mirror approve service stopped.")


def main() -> None:
    logging.basicConfig(
        level=os.getenv("MIRROR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(build_and_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
