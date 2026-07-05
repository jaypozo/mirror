"""Mirror LIVE approve-UI service — the draft-my-reply loop.

One asyncio process, single owner of the Telethon USER session, running two
Telethon clients on the SAME event loop:

  * USER client  (authed as the owner, .telethon/mirror) — watches incoming messages,
    and is the ONLY thing that ever sends AS the owner (on Approve/Edit). It also,
    optionally, resumes the older-history backfill as a background asyncio task
    inside this same client, so we never open a second concurrent client on the
    single-writer session.

  * BOT client   (MIRROR_BOT_TOKEN) — the approve UI: DMs the owner each draft with
    Approve / Edit / Dismiss inline buttons and handles the callbacks.

HARD RULES honoured here:
  * Nothing is EVER sent to the original chat without the owner's Approve / Edit tap.
  * Only ONE process owns the user session — the run script stops the standalone
    backfill drip before starting this. We do not open a second user client.
  * We never call .start() on the user client (that could trigger a login); we
    .connect() and refuse if the session is not already authorized.

If MIRROR_BOT_TOKEN is unset, the user watcher still runs (drafts are computed
and logged, but there is nowhere to approve them, so they are NOT sent). The
service logs "waiting for MIRROR_BOT_TOKEN" and stays up, so Nat can add the
token and restart to go live.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid
from dataclasses import dataclass, field

import asyncpg
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.custom import Button

from agent.draft import draft_reply
from agent.eligibility import evaluate, info_from_event
from agent.feedback import record_feedback
from agent.types import ChatMessage, DraftRequest, DraftResult

log = logging.getLogger("mirror.service")

OWNER_USER_ID = int(os.getenv("MIRROR_OWNER_USER_ID", "0") or "0")  # where approval prompts are DM'd
THREAD_CONTEXT_LIMIT = 8   # incoming-thread messages fed to the drafter


# --------------------------------------------------------------------------- #
# Pending-approval state (in-memory; a restart drops undecided drafts, which is
# safe — nothing was sent).
# --------------------------------------------------------------------------- #
@dataclass
class Pending:
    approval_id: str
    request: DraftRequest
    result: DraftResult
    sender_name: str
    where: str            # human label: chat title / "DM" etc.
    approval_msg_id: int | None = None


@dataclass
class ServiceState:
    user: TelegramClient
    bot: TelegramClient | None
    database_url: str | None
    pending: dict[str, Pending] = field(default_factory=dict)
    # the owner is mid-edit for this approval_id (their next DM text is the final reply).
    awaiting_edit: str | None = None


STATE: ServiceState | None = None


# --------------------------------------------------------------------------- #
# Approve-UI rendering (a Goal/Now/Next/Open card).
# --------------------------------------------------------------------------- #
def _truncate(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def approval_text(pending: Pending) -> str:
    r = pending.result
    incoming = _truncate(pending.request.incoming_message, 700)
    return (
        "🪞 Mirror draft\n"
        f"From: {pending.sender_name}\n"
        f"Where: {pending.where}\n\n"
        f"Incoming:\n{incoming}\n\n"
        f"{r.summary.format_for_telegram()}\n\n"
        f"Draft reply:\n{r.draft}"
    )


def approval_buttons(approval_id: str):
    return [
        [
            Button.inline("✅ Approve", data=f"approve:{approval_id}"),
            Button.inline("✏️ Edit", data=f"edit:{approval_id}"),
            Button.inline("🗑 Dismiss", data=f"dismiss:{approval_id}"),
        ]
    ]


# --------------------------------------------------------------------------- #
# Building the draft request from a live event (+ a little thread context).
# --------------------------------------------------------------------------- #
async def _thread_context(user: TelegramClient, event) -> list[ChatMessage]:
    """The last few messages in this chat, oldest→newest, as voice context."""
    thread: list[ChatMessage] = []
    try:
        async for msg in user.iter_messages(event.chat_id, limit=THREAD_CONTEXT_LIMIT):
            text = getattr(msg, "raw_text", None) or getattr(msg, "message", None)
            if not text:
                continue
            thread.append(
                ChatMessage(
                    text=text,
                    sender_name=None if getattr(msg, "out", False) else "them",
                    direction="out" if getattr(msg, "out", False) else "in",
                    message_id=int(msg.id) if msg.id is not None else None,
                )
            )
    except Exception as exc:  # non-fatal: draft still works without context
        log.warning("thread context fetch failed: %s", exc)
    thread.reverse()
    return thread


async def _describe(user: TelegramClient, event) -> tuple[str, str]:
    """(sender_name, where_label) for the approval card."""
    sender_name = "someone"
    where = "chat"
    try:
        sender = await event.get_sender()
        if sender is not None:
            sender_name = (
                getattr(sender, "title", None)
                or " ".join(
                    p for p in [getattr(sender, "first_name", None), getattr(sender, "last_name", None)] if p
                )
                or getattr(sender, "username", None)
                or str(getattr(sender, "id", "someone"))
            )
    except Exception:
        pass
    try:
        chat = await event.get_chat()
        if event.is_private:
            where = f"DM with {sender_name}"
        else:
            where = getattr(chat, "title", None) or "group"
    except Exception:
        pass
    return sender_name, where


# --------------------------------------------------------------------------- #
# USER client: incoming watcher.
# --------------------------------------------------------------------------- #
async def on_incoming(event) -> None:
    assert STATE is not None
    decision = evaluate(info_from_event(event))
    if not decision.draft:
        log.debug("skip (%s) chat=%s", decision.reason, event.chat_id)
        return

    log.info("eligible (%s) chat=%s — drafting", decision.reason, event.chat_id)

    thread = await _thread_context(STATE.user, event)
    incoming_text = getattr(event.message, "raw_text", None) or getattr(event.message, "message", None) or ""

    request = DraftRequest(
        incoming_message=incoming_text,
        thread=thread,
        source_chat_id=int(event.chat_id),
        source_message_id=int(event.message.id) if event.message.id is not None else None,
        target_chat_id=int(event.chat_id),
        topic_id=None,  # we reply to the original message, not a topic
        metadata={"reason": decision.reason},
    )

    try:
        result = await draft_reply(request, include_summary=True)
    except Exception as exc:
        log.exception("draft failed: %s", exc)
        return

    sender_name, where = await _describe(STATE.user, event)
    pending = Pending(
        approval_id=uuid.uuid4().hex,
        request=request,
        result=result,
        sender_name=sender_name,
        where=where,
    )

    if STATE.bot is None:
        # No approve UI yet. Record the draft signal but NEVER send.
        log.warning(
            "MIRROR_BOT_TOKEN unset — draft computed but not sendable "
            "(no approval UI). from=%s where=%s", sender_name, where
        )
        return

    STATE.pending[pending.approval_id] = pending
    try:
        sent = await STATE.bot.send_message(
            OWNER_USER_ID,
            approval_text(pending),
            buttons=approval_buttons(pending.approval_id),
            link_preview=False,
        )
        pending.approval_msg_id = int(sent.id)
    except Exception as exc:
        log.exception("failed to post approval prompt: %s", exc)
        STATE.pending.pop(pending.approval_id, None)


# --------------------------------------------------------------------------- #
# Send AS the owner (USER client only). This is the sole path that touches the
# original chat, and only from Approve / Edit callbacks.
# --------------------------------------------------------------------------- #
async def send_as_owner(request: DraftRequest, text: str) -> None:
    assert STATE is not None
    if not request.target_chat_id:
        raise RuntimeError("no target_chat_id")
    await STATE.user.send_message(
        request.target_chat_id,
        text,
        reply_to=request.source_message_id,  # thread the reply to the original
    )


async def _log_feedback(pending: Pending, *, action: str, final: str | None) -> None:
    await record_feedback(
        original_draft=pending.result.draft,
        final_text=final,
        action=action,
        source_chat_id=pending.request.source_chat_id,
        source_message_id=pending.request.source_message_id,
        target_chat_id=pending.request.target_chat_id,
        topic_id=pending.request.topic_id,
        summary=pending.result.summary,
        metadata=pending.request.metadata,
    )


# --------------------------------------------------------------------------- #
# BOT client: callbacks + edit-followup.
# --------------------------------------------------------------------------- #
async def on_callback(event) -> None:
    assert STATE is not None
    if event.sender_id != OWNER_USER_ID:
        await event.answer("Not authorized.", alert=True)
        return

    data = event.data.decode() if isinstance(event.data, bytes) else str(event.data)
    action, _, approval_id = data.partition(":")
    pending = STATE.pending.get(approval_id)
    if pending is None:
        await event.answer("This draft is no longer pending.")
        try:
            await event.edit("(expired) " + (event.text or ""))
        except Exception:
            pass
        return

    if action == "approve":
        try:
            await send_as_owner(pending.request, pending.result.draft)
        except Exception as exc:
            log.exception("send-as-owner failed: %s", exc)
            await event.answer(f"Send failed: {exc}", alert=True)
            return
        await _log_feedback(pending, action="approve", final=pending.result.draft)
        STATE.pending.pop(approval_id, None)
        await event.edit(f"✅ Sent as the owner:\n\n{pending.result.draft}", buttons=None)
        return

    if action == "edit":
        STATE.awaiting_edit = approval_id
        await event.answer("Send the edited reply as your next message.")
        await event.edit(
            approval_text(pending) + "\n\n✏️ Send the edited final text as your next message.",
            buttons=None,
        )
        return

    if action == "dismiss":
        await _log_feedback(pending, action="dismiss", final=None)
        STATE.pending.pop(approval_id, None)
        await event.edit("🗑 Dismissed (nothing sent).", buttons=None)
        return


async def on_edit_followup(event) -> None:
    """the owner's plain DM after tapping Edit becomes the final sent-as-owner text."""
    assert STATE is not None
    if event.sender_id != OWNER_USER_ID:
        return
    if STATE.awaiting_edit is None:
        return
    approval_id = STATE.awaiting_edit
    pending = STATE.pending.get(approval_id)
    if pending is None:
        STATE.awaiting_edit = None
        return

    final = (event.raw_text or "").strip()
    if not final:
        await event.reply("Empty edit ignored — nothing sent.")
        return

    try:
        await send_as_owner(pending.request, final)
    except Exception as exc:
        log.exception("send-as-owner (edited) failed: %s", exc)
        await event.reply(f"Send failed: {exc}")
        return

    await _log_feedback(pending, action="edit", final=final)
    STATE.pending.pop(approval_id, None)
    STATE.awaiting_edit = None
    await event.reply("✅ Edited reply sent as the owner.")


# --------------------------------------------------------------------------- #
# Optional: resume older-history backfill inside THIS user client.
# --------------------------------------------------------------------------- #
async def maybe_resume_backfill(user: TelegramClient, database_url: str | None) -> None:
    if os.getenv("MIRROR_RESUME_BACKFILL", "").strip().lower() not in {"1", "true", "yes"}:
        log.info("history backfill resume disabled (set MIRROR_RESUME_BACKFILL=1 to enable)")
        return
    if not database_url:
        log.warning("cannot resume backfill: DATABASE_URL unset")
        return

    from ingest.backfill import backfill_dialog
    from ingest.pull import load_settings

    async def _run() -> None:
        try:
            settings = load_settings()
            conn = await asyncpg.connect(database_url)
            try:
                log.info("resuming older-history backfill inside the live client")
                async for dialog in user.iter_dialogs():
                    await backfill_dialog(user, conn, dialog, settings)
                    await asyncio.sleep(settings.chat_sleep_seconds)
                log.info("backfill complete")
            finally:
                await conn.close()
        except asyncio.CancelledError:
            log.info("backfill task cancelled")
            raise
        except Exception as exc:
            log.exception("backfill task error: %s", exc)

    asyncio.create_task(_run())


# --------------------------------------------------------------------------- #
# Wiring + run.
# --------------------------------------------------------------------------- #
async def build_and_run() -> None:
    global STATE
    load_dotenv()

    api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
    api_hash = os.getenv("TELEGRAM_API_HASH", "")
    session = os.getenv("TELEGRAM_SESSION", ".telethon/mirror")
    database_url = os.getenv("DATABASE_URL")
    bot_token = (os.getenv("MIRROR_BOT_TOKEN") or "").strip()

    if not api_id or not api_hash:
        raise SystemExit("Missing TELEGRAM_API_ID / TELEGRAM_API_HASH in .env")

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

    bot: TelegramClient | None = None
    if bot_token:
        bot = TelegramClient(f"{session}.bot", api_id, api_hash)
        await bot.start(bot_token=bot_token)
        binfo = await bot.get_me()
        log.info("approve-UI bot online: @%s", getattr(binfo, "username", None))
    else:
        log.warning(
            "waiting for MIRROR_BOT_TOKEN — user watcher will run and drafts will "
            "be computed, but with no approval UI NOTHING is sent. Add the token "
            "to .env and restart to go live."
        )

    STATE = ServiceState(user=user, bot=bot, database_url=database_url)

    # Register handlers.
    user.add_event_handler(on_incoming, events.NewMessage(incoming=True))
    if bot is not None:
        bot.add_event_handler(on_callback, events.CallbackQuery())
        # Edit-followup: the owner's DMs to the bot (private, non-command).
        bot.add_event_handler(
            on_edit_followup,
            events.NewMessage(incoming=True, func=lambda e: e.is_private),
        )

    await maybe_resume_backfill(user, database_url)

    log.info("Mirror service running. Watching incoming messages.")

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
        if bot is not None:
            async with bot:
                await stop.wait()
        else:
            await stop.wait()

    log.info("Mirror service stopped.")


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
