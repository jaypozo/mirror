"""Build Goal / Now / Next / Open summaries from a live thread."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from agent.llm import LLMClient, build_llm_client
from agent.types import Brief, ChatMessage


SUMMARY_SYSTEM_PROMPT = """You summarize live work threads for the owner.
Return only compact JSON with keys: goal, now, next, open.
Use the Goal / Now / Next / Open shape:
- goal: what the thread is trying to accomplish
- now: the current state or blocker
- next: the next concrete action the owner should take
- open: unresolved questions, risks, or missing facts as a short list
Do not invent facts."""


def parse_summary_json(text: str) -> Brief | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return Brief.from_dict(data)


def heuristic_summary(thread: list[ChatMessage]) -> Brief:
    if not thread:
        return Brief.empty()

    latest = thread[-1].text.strip()
    first = thread[0].text.strip()
    goal = first[:180] if first else "Resolve the current thread."
    now = latest[:220] if latest else "A reply may be needed."
    return Brief(
        goal=goal,
        now=now,
        next="Draft a concise response that moves the thread forward.",
        open=["Confirm any missing facts before sending."] if "?" in latest else [],
    )


async def summarize_thread(
    thread: list[ChatMessage],
    llm: LLMClient | None = None,
) -> Brief:
    if not thread:
        return Brief.empty()

    client = llm or build_llm_client()
    transcript = "\n".join(message.to_prompt_line() for message in thread[-40:])
    prompt = (
        "Summarize this thread as a JSON object with keys goal, now, next, open.\n\n"
        f"{transcript}"
    )
    try:
        response = await client.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        print(f"[summarize] LLM summary failed; using heuristic fallback: {exc}", file=sys.stderr)
        return heuristic_summary(thread)

    parsed = parse_summary_json(response)
    return parsed or heuristic_summary(thread)


def thread_from_payload(payload: dict[str, Any]) -> list[ChatMessage]:
    return [ChatMessage.from_dict(item) for item in payload.get("thread", [])]


async def main() -> None:
    payload = json.load(sys.stdin)
    summary = await summarize_thread(thread_from_payload(payload))
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
