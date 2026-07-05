"""Shared data structures for the Phase 2 draft-approval agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    text: str
    sender_name: str | None = None
    direction: str | None = None
    ts: datetime | None = None
    message_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        raw_ts = data.get("ts")
        parsed_ts = None
        if isinstance(raw_ts, str):
            parsed_ts = datetime.fromisoformat(raw_ts)
        elif isinstance(raw_ts, datetime):
            parsed_ts = raw_ts
        return cls(
            text=str(data.get("text") or ""),
            sender_name=data.get("sender_name") or data.get("sender"),
            direction=data.get("direction"),
            ts=parsed_ts,
            message_id=data.get("message_id") or data.get("id"),
        )

    def to_prompt_line(self) -> str:
        name = self.sender_name or self.direction or "unknown"
        return f"{name}: {self.text}".strip()


@dataclass(frozen=True)
class ThreadSummary:
    goal: str
    now: str
    next: str
    open: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "ThreadSummary":
        return cls(
            goal="Understand what this thread is trying to resolve.",
            now="A reply may be needed.",
            next="Draft a concise response for the owner to approve.",
            open=[],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadSummary":
        open_items = data.get("open", [])
        if isinstance(open_items, str):
            open_items = [open_items]
        return cls(
            goal=str(data.get("goal") or "Clarify the thread goal."),
            now=str(data.get("now") or "Review the latest message."),
            next=str(data.get("next") or "Prepare the owner's response."),
            open=[str(item) for item in open_items if str(item).strip()],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "now": self.now,
            "next": self.next,
            "open": self.open,
        }

    def format_for_telegram(self) -> str:
        open_text = "\n".join(f"- {item}" for item in self.open) if self.open else "- None"
        return (
            f"Goal: {self.goal}\n"
            f"Now: {self.now}\n"
            f"Next: {self.next}\n"
            f"Open:\n{open_text}"
        )


@dataclass(frozen=True)
class DraftRequest:
    incoming_message: str
    thread: list[ChatMessage]
    source_chat_id: int | None = None
    source_message_id: int | None = None
    target_chat_id: int | None = None
    target_thread_id: int | None = None
    context_tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DraftRequest":
        thread = [ChatMessage.from_dict(item) for item in data.get("thread", [])]
        incoming = data.get("incoming_message")
        if not incoming and thread:
            incoming = thread[-1].text
        return cls(
            incoming_message=str(incoming or ""),
            thread=thread,
            source_chat_id=data.get("source_chat_id"),
            source_message_id=data.get("source_message_id"),
            target_chat_id=data.get("target_chat_id"),
            target_thread_id=data.get("target_thread_id"),
            context_tag=data.get("context_tag"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class DraftResult:
    draft: str
    summary: ThreadSummary
    style_examples: list[str] = field(default_factory=list)
    # The thread this draft was segmented into (Build 2), so the /decide edit
    # path can attach an intent note to the right thread. None when thread
    # segmentation was off or failed (drafting falls back to the flat summary).
    thread_id: int | None = None
