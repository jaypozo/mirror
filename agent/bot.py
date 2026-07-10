"""Telegram approval bot for Phase 2 draft review."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.draft import draft_reply
from agent.feedback import record_feedback
from agent.types import DraftRequest, DraftResult

log = logging.getLogger("mirror.bot")


@dataclass
class PendingApproval:
    request: DraftRequest
    result: DraftResult
    approval_chat_id: int
    approval_message_id: int | None = None


PENDING_APPROVALS: dict[str, PendingApproval] = {}
PENDING_EDITS: dict[int, str] = {}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}. Copy .env.example to .env and fill it in.")
    return value


def approval_text(result: DraftResult) -> str:
    return (
        "Mirror draft\n\n"
        f"{result.summary.format_for_telegram()}\n\n"
        "Draft:\n"
        f"{result.draft}"
    )


def approval_keyboard(approval_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Approve & Send", callback_data=f"approve:{approval_id}"),
                InlineKeyboardButton("Edit", callback_data=f"edit:{approval_id}"),
            ],
            [InlineKeyboardButton("Dismiss", callback_data=f"dismiss:{approval_id}")],
        ]
    )


def editing_text(result: DraftResult) -> str:
    return f"{approval_text(result)}\n\nEditing - send your version."


def editing_keyboard(approval_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Cancel Edit", callback_data=f"cancel_edit:{approval_id}")]]
    )


def enter_edit_state(chat_id: int, approval_id: str) -> None:
    PENDING_EDITS[chat_id] = approval_id


def cancel_edit_state(chat_id: int, approval_id: str) -> bool:
    if PENDING_EDITS.get(chat_id) != approval_id:
        return False
    PENDING_EDITS.pop(chat_id, None)
    return True


async def send_to_target(bot: Any, request: DraftRequest, text: str) -> None:
    if not request.target_chat_id:
        raise RuntimeError("No target_chat_id was provided by the message-source integration.")
    kwargs: dict[str, Any] = {"chat_id": request.target_chat_id, "text": text}
    if request.topic_id:
        kwargs["message_thread_id"] = request.topic_id
    await bot.send_message(**kwargs)


async def post_approval_request(
    application: Application,
    request: DraftRequest,
    approval_chat_id: int,
) -> str | None:
    try:
        result = await draft_reply(request)
    except Exception as exc:
        log.exception("draft failed: %s", exc)
        return None

    approval_id = uuid.uuid4().hex
    message = await application.bot.send_message(
        chat_id=approval_chat_id,
        text=approval_text(result),
        reply_markup=approval_keyboard(approval_id),
    )
    PENDING_APPROVALS[approval_id] = PendingApproval(
        request=request,
        result=result,
        approval_chat_id=approval_chat_id,
        approval_message_id=message.message_id,
    )
    return approval_id


async def draft_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    raw_payload = " ".join(context.args)
    if not raw_payload:
        await update.message.reply_text(
            "Send /draft followed by a JSON payload from the message-source integration."
        )
        return

    try:
        payload = json.loads(raw_payload)
        request = DraftRequest.from_dict(payload)
    except Exception as exc:
        await update.message.reply_text(f"Could not parse draft payload: {exc}")
        return

    approval_id = await post_approval_request(context.application, request, update.effective_chat.id)
    if approval_id is None:
        return
    await update.message.reply_text(f"Draft queued: {approval_id}")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()

    action, _, approval_id = query.data.partition(":")
    pending = PENDING_APPROVALS.get(approval_id)
    if not pending:
        await query.edit_message_text("This draft is no longer pending.")
        return

    if action == "approve":
        try:
            await send_to_target(context.bot, pending.request, pending.result.draft)
            await record_feedback(
                original_draft=pending.result.draft,
                final_text=pending.result.draft,
                action="approve",
                source_chat_id=pending.request.source_chat_id,
                source_message_id=pending.request.source_message_id,
                target_chat_id=pending.request.target_chat_id,
                topic_id=pending.request.topic_id,
                summary=pending.result.summary,
                metadata=pending.request.metadata,
            )
            PENDING_APPROVALS.pop(approval_id, None)
            if query.message:
                PENDING_EDITS.pop(query.message.chat_id, None)
            await query.edit_message_text(f"Approved and sent.\n\n{pending.result.draft}")
        except Exception as exc:
            await query.edit_message_text(f"Approve failed: {exc}")
        return

    if action == "edit":
        if query.message:
            enter_edit_state(query.message.chat_id, approval_id)
        await query.edit_message_text(
            editing_text(pending.result),
            reply_markup=editing_keyboard(approval_id),
        )
        return

    if action == "cancel_edit":
        if query.message and cancel_edit_state(query.message.chat_id, approval_id):
            await query.edit_message_text(
                approval_text(pending.result),
                reply_markup=approval_keyboard(approval_id),
            )
        elif query.message:
            await query.edit_message_text(
                approval_text(pending.result),
                reply_markup=approval_keyboard(approval_id),
            )
        else:
            await query.answer("Edit cancelled.")
        return

    if action == "dismiss":
        if query.message:
            PENDING_EDITS.pop(query.message.chat_id, None)
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
        PENDING_APPROVALS.pop(approval_id, None)
        await query.edit_message_text("Dismissed.")


async def handle_edit_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message or not update.message.text:
        return
    approval_id = PENDING_EDITS.pop(update.effective_chat.id, None)
    if not approval_id:
        return

    pending = PENDING_APPROVALS.get(approval_id)
    if not pending:
        await update.message.reply_text("That draft is no longer pending.")
        return

    final_text = update.message.text.strip()
    try:
        await send_to_target(context.bot, pending.request, final_text)
        sent_note = "Edited text captured and sent."
    except Exception as exc:
        sent_note = f"Edited text captured, but send failed: {exc}"

    await record_feedback(
        original_draft=pending.result.draft,
        final_text=final_text,
        action="edit",
        source_chat_id=pending.request.source_chat_id,
        source_message_id=pending.request.source_message_id,
        target_chat_id=pending.request.target_chat_id,
        topic_id=pending.request.topic_id,
        summary=pending.result.summary,
        metadata=pending.request.metadata,
    )
    PENDING_APPROVALS.pop(approval_id, None)
    if pending.approval_message_id is not None:
        try:
            await context.bot.edit_message_text(
                chat_id=pending.approval_chat_id,
                message_id=pending.approval_message_id,
                text=f"{sent_note}\n\n{final_text}",
            )
            return
        except Exception:
            pass
    await update.message.reply_text(sent_note)


async def post_payload_file_if_configured(application: Application, approval_chat_id: int) -> None:
    payload_file = os.getenv("MIRROR_DRAFT_PAYLOAD_FILE")
    if not payload_file:
        return
    try:
        with open(payload_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        request = DraftRequest.from_dict(payload)
    except Exception as exc:
        log.exception("could not load draft payload file %s: %s", payload_file, exc)
        return

    approval_id = await post_approval_request(application, request, approval_chat_id)
    if approval_id is None:
        return
    print(f"[bot] posted approval request from {payload_file}: {approval_id}")


async def post_init(application: Application) -> None:
    approval_chat_id = os.getenv("OWNER_APPROVAL_CHAT_ID")
    if approval_chat_id:
        await post_payload_file_if_configured(application, int(approval_chat_id))


def build_application() -> Application:
    load_dotenv()
    token = require_env("TELEGRAM_BOT_TOKEN")
    application = Application.builder().token(token).post_init(post_init).build()
    application.add_handler(CommandHandler("draft", draft_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_text))
    return application


def main() -> None:
    application = build_application()
    print("[bot] running approval bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bot] stopped")
