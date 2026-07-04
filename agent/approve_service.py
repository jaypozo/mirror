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
import signal
import uuid
from dataclasses import dataclass, field

from aiohttp import web
from dotenv import load_dotenv
from telethon import TelegramClient

# Load .env before the env-derived module constants below evaluate.
load_dotenv()

from agent.draft import draft_reply
from agent.eligibility import EXCLUDED_TOPIC_CHAT_ID
from agent.feedback import record_feedback
from agent.service import maybe_resume_backfill  # reuse the drip resumer as-is
from agent.types import ChatMessage, DraftRequest, DraftResult

log = logging.getLogger("mirror.approve_service")

OWNER_USER_ID = int(os.getenv("MIRROR_OWNER_USER_ID", "0") or "0")
PENDING_TTL_SECONDS = 60 * 60  # drafts older than this are swept (nothing sent)


# --------------------------------------------------------------------------- #
# Pending drafts (in-memory; a restart drops undecided drafts, which is safe —
# nothing was sent). Keyed by approval_id.
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


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _authorized(request: web.Request) -> bool:
    if not SECRET:
        # No secret configured => refuse everything. Fail closed.
        return False
    return request.headers.get("X-Mirror-Secret") == SECRET


def _is_excluded_topic(chat_id: int, is_topic: bool) -> bool:
    """the excluded group topic threads are excluded; its General channel is included."""
    return chat_id == EXCLUDED_TOPIC_CHAT_ID and bool(is_topic)


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
    now = asyncio.get_event_loop().time()
    for aid, p in list(STATE.pending.items()):
        if now - p.created_at > PENDING_TTL_SECONDS:
            STATE.pending.pop(aid, None)


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
#      or { ok:false, reason }
# --------------------------------------------------------------------------- #
async def handle_draft(request: web.Request) -> web.Response:
    if not _authorized(request):
        return web.json_response({"ok": False, "reason": "unauthorized"}, status=401)
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
        return web.json_response({"ok": False, "reason": "topic thread excluded"})
    if not question:
        return web.json_response({"ok": False, "reason": "no question text"})

    thread = _thread_from_payload(body.get("thread", []))

    draft_request = DraftRequest(
        incoming_message=question,
        thread=thread,
        source_chat_id=chat_id,
        source_message_id=int(question_msg_id) if question_msg_id is not None else None,
        target_chat_id=chat_id,
        target_thread_id=None,  # reply threads to the agent's question, not a topic
        metadata={
            "origin": "fleet-bot",
            "chat_id": chat_id,
            "bot_username": (str(body.get("bot_username")).strip() or None)
            if body.get("bot_username")
            else None,
        },
    )

    try:
        result = await draft_reply(draft_request, include_summary=True)
    except Exception as exc:
        log.exception("draft failed: %s", exc)
        return web.json_response({"ok": False, "reason": f"draft failed: {exc}"}, status=500)

    approval_id = uuid.uuid4().hex[:12]
    STATE.pending[approval_id] = Pending(
        approval_id=approval_id,
        request=draft_request,
        result=result,
        created_at=asyncio.get_event_loop().time(),
    )
    log.info("drafted approval_id=%s chat=%s", approval_id, chat_id)

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
        return web.json_response({"ok": False, "reason": "no such pending draft"})

    if action == "dismiss":
        await record_feedback(
            original_draft=pending.result.draft,
            final_text=None,
            action="dismiss",
            source_chat_id=pending.request.source_chat_id,
            source_message_id=pending.request.source_message_id,
            target_chat_id=pending.request.target_chat_id,
            target_thread_id=pending.request.target_thread_id,
            summary=pending.result.summary,
            metadata=pending.request.metadata,
        )
        STATE.pending.pop(approval_id, None)
        return web.json_response({"ok": True, "action": "dismiss", "sent": False})

    if action in ("approve", "edit"):
        final = pending.result.draft if action == "approve" else str(edited_text or "").strip()
        if action == "edit" and not final:
            return web.json_response({"ok": False, "reason": "empty edit"})

        try:
            await send_as_owner(pending.request, final)
        except Exception as exc:
            log.exception("send-as-owner failed: %s", exc)
            return web.json_response({"ok": False, "reason": f"send failed: {exc}"}, status=500)

        await record_feedback(
            original_draft=pending.result.draft,
            final_text=final,
            action=action,
            source_chat_id=pending.request.source_chat_id,
            source_message_id=pending.request.source_message_id,
            target_chat_id=pending.request.target_chat_id,
            target_thread_id=pending.request.target_thread_id,
            summary=pending.result.summary,
            metadata=pending.request.metadata,
        )
        STATE.pending.pop(approval_id, None)
        log.info("sent-as-owner approval_id=%s action=%s", approval_id, action)
        return web.json_response({"ok": True, "action": action, "sent": True})

    return web.json_response({"ok": False, "reason": f"unknown action: {action}"})


# send-as-owner: the ONLY path that touches a chat. We call Telethon directly on
# THIS service's own user client (its own STATE), rather than agent.service's
# send_as_owner which reads that module's global STATE — keeps service.py
# untouched. Behaviour is identical: send text as the owner, threaded to the agent's
# original question message.
async def send_as_owner(request: DraftRequest, text: str) -> None:
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
        await STATE.user.send_message(peer, text, reply_to=request.source_message_id)
    except Exception:
        # reply_to ids differ between the bot API and the user session in a
        # private chat; if threading fails, still deliver the message inline.
        await STATE.user.send_message(peer, text)


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

    app = web.Application()
    app.add_routes(
        [
            web.post("/draft", handle_draft),
            web.post("/decide", handle_decide),
            web.get("/health", handle_health),
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
