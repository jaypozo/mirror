"""Unit tests for the eligibility filter — especially the the excluded group
General-vs-topic-thread exclusion. Runnable with pytest OR directly:

    .venv/bin/python -m tests.test_eligibility
"""

from __future__ import annotations

from agent.eligibility import (
    EXCLUDED_TOPIC_CHAT_ID,
    IncomingInfo,
    evaluate,
    should_draft,
)

OWNER_DM = 111111111  # arbitrary DM chat id for tests
OTHER_GROUP = -1002222233333


CASES: list[tuple[str, IncomingInfo, bool]] = [
    # --- the excluded group: the crux. General included, topics excluded. ---
    (
        "the excluded group GENERAL (no reply header) -> INCLUDE",
        IncomingInfo(chat_id=EXCLUDED_TOPIC_CHAT_ID, text="hey nat, quick q", in_forum_topic=False),
        True,
    ),
    (
        "the excluded group TOPIC thread (forum_topic=True) -> EXCLUDE",
        IncomingInfo(chat_id=EXCLUDED_TOPIC_CHAT_ID, text="topic message", in_forum_topic=True),
        False,
    ),
    (
        "the excluded group General plain reply (not a topic) -> INCLUDE",
        IncomingInfo(chat_id=EXCLUDED_TOPIC_CHAT_ID, text="re: that", in_forum_topic=False),
        True,
    ),
    # --- DMs and other groups ---
    ("the owner DM -> INCLUDE", IncomingInfo(chat_id=OWNER_DM, text="you around?"), True),
    ("other group -> INCLUDE", IncomingInfo(chat_id=OTHER_GROUP, text="ship it?"), True),
    (
        "other group in a topic-like thread -> INCLUDE (only the excluded group excludes topics)",
        IncomingInfo(chat_id=OTHER_GROUP, text="thread msg", in_forum_topic=True),
        True,
    ),
    # --- universal skips ---
    ("outgoing (the owner's own) -> EXCLUDE", IncomingInfo(chat_id=OWNER_DM, text="my reply", out=True), False),
    ("service message -> EXCLUDE", IncomingInfo(chat_id=OTHER_GROUP, text="", is_service=True), False),
    ("broadcast channel -> EXCLUDE", IncomingInfo(chat_id=-100999, text="post", is_channel=True), False),
    ("bot command to Mirror -> EXCLUDE", IncomingInfo(chat_id=OWNER_DM, text="/start", is_bot_command=True), False),
    ("empty text -> EXCLUDE", IncomingInfo(chat_id=OWNER_DM, text="   "), False),
    # A topic message that is ALSO empty is excluded (empty wins, still excluded).
    (
        "the excluded group topic + empty -> EXCLUDE",
        IncomingInfo(chat_id=EXCLUDED_TOPIC_CHAT_ID, text="", in_forum_topic=True),
        False,
    ),
]


def test_eligibility_cases() -> None:
    for label, info, expected in CASES:
        decision = evaluate(info)
        assert decision.draft is expected, (
            f"{label}: expected draft={expected}, got {decision.draft} ({decision.reason})"
        )


def test_general_vs_topic_pair() -> None:
    """The load-bearing distinction, stated once, crisply."""
    general = IncomingInfo(chat_id=EXCLUDED_TOPIC_CHAT_ID, text="x", in_forum_topic=False)
    topic = IncomingInfo(chat_id=EXCLUDED_TOPIC_CHAT_ID, text="x", in_forum_topic=True)
    assert should_draft(general) is True
    assert should_draft(topic) is False


def _run_directly() -> None:
    passed = 0
    for label, info, expected in CASES:
        decision = evaluate(info)
        ok = decision.draft is expected
        passed += ok
        mark = "PASS" if ok else "FAIL"
        verdict = "DRAFT" if decision.draft else "SKIP "
        print(f"[{mark}] {verdict} ({decision.reason:28s}) — {label}")
    print(f"\n{passed}/{len(CASES)} cases passed")
    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    _run_directly()
