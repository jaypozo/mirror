"""Eligibility filter: decide whether an incoming message should be drafted.

This is deliberately a PURE function of a small, explicit shape (``IncomingInfo``)
so it can be unit-tested without a live Telethon event. ``service.py`` adapts a
real ``events.NewMessage`` event into an ``IncomingInfo`` and calls
``should_draft``.

Rules (see task brief):

INCLUDE
  - the owner's private DMs.
  - Other groups (basic groups + non-forum supergroups).
  - The the excluded group group's GENERAL channel (forum root, not a topic thread).

EXCLUDE
  - The the excluded group group's TOPIC THREADS (forum topics other than General).
  - the owner's own outgoing messages.
  - Service / action messages (no real text).
  - Channels / broadcasts.
  - Bot commands aimed at Mirror itself (handled upstream by the bot client,
    but we also skip empty/command-only text here).
  - Messages with no text to reply to.

Forum-topic detection (Telethon 1.44):
  A forum supergroup marks a message that lives inside a NON-General topic with
  ``message.reply_to.forum_topic == True``. The General "topic" is the channel
  root: its messages have either ``reply_to is None`` or a reply header whose
  ``forum_topic`` is falsy (a plain reply to another General message). So:

      is_topic_thread = bool(reply_to and getattr(reply_to, "forum_topic", False))

  We only apply this exclusion to the the excluded group chat id; every other chat is
  judged purely on the include rules above.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any

# The the excluded group supergroup. Topic threads here are excluded; the General
# channel is included.
EXCLUDED_TOPIC_CHAT_ID = int(os.getenv("MIRROR_EXCLUDED_TOPIC_CHAT_ID", "0") or "0")


@dataclass(frozen=True)
class IncomingInfo:
    """Minimal, testable view of an incoming Telegram message."""

    chat_id: int
    text: str
    out: bool = False            # True => the owner's own outgoing message
    is_channel: bool = False     # True => broadcast channel / channel post
    is_service: bool = False     # True => service/action message (join, pin, ...)
    is_bot_command: bool = False # True => a /command aimed at Mirror
    # Forum-topic signal, derived from message.reply_to.forum_topic upstream.
    in_forum_topic: bool = False


@dataclass(frozen=True)
class EligibilityDecision:
    draft: bool
    reason: str


def evaluate(info: IncomingInfo) -> EligibilityDecision:
    """Return whether we should draft a reply, with a human-readable reason."""
    if info.out:
        return EligibilityDecision(False, "outgoing (the owner's own message)")
    if info.is_service:
        return EligibilityDecision(False, "service/action message")
    if info.is_channel:
        return EligibilityDecision(False, "broadcast channel")
    if info.is_bot_command:
        return EligibilityDecision(False, "bot command to Mirror")
    if not (info.text or "").strip():
        return EligibilityDecision(False, "no text")

    # the excluded group: include General, exclude topic threads.
    if info.chat_id == EXCLUDED_TOPIC_CHAT_ID:
        if info.in_forum_topic:
            return EligibilityDecision(False, "the excluded group topic thread")
        return EligibilityDecision(True, "the excluded group General channel")

    # Everything else that passed the basic skips is eligible: DMs, other
    # groups, non-forum supergroups.
    return EligibilityDecision(True, "eligible chat")


def should_draft(info: IncomingInfo) -> bool:
    return evaluate(info).draft


def info_from_event(event: Any) -> IncomingInfo:
    """Adapt a live Telethon NewMessage event into an IncomingInfo.

    Kept here (next to the rules) so the mapping from Telethon internals to the
    testable shape lives in one place.
    """
    message = event.message
    reply_to = getattr(message, "reply_to", None)
    in_forum_topic = bool(reply_to is not None and getattr(reply_to, "forum_topic", False))

    text = getattr(message, "raw_text", None) or getattr(message, "message", None) or ""

    return IncomingInfo(
        chat_id=int(event.chat_id),
        text=text,
        out=bool(getattr(message, "out", False)),
        is_channel=bool(getattr(event, "is_channel", False) and not getattr(event, "is_group", False)),
        is_service=getattr(event, "action_message", None) is not None
        or getattr(message, "action", None) is not None,
        is_bot_command=text.strip().startswith("/"),
        in_forum_topic=in_forum_topic,
    )
