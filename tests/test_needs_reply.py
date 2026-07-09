from __future__ import annotations

import pytest

from agent.needs_reply import NEEDS_REPLY, SKIP, classify_needs_reply


class FailingLLM:
    name = "test"
    model = "test"

    async def complete(self, messages):
        raise RuntimeError("boom")


class EmptyLLM:
    name = "test"
    model = "test"

    async def complete(self, messages):
        return ""


class SlowLLM:
    name = "test"
    model = "test"

    async def complete(self, messages):
        import asyncio

        await asyncio.sleep(1)
        return "NEEDS_REPLY"


@pytest.mark.asyncio
async def test_question_heuristic_needs_reply() -> None:
    decision = await classify_needs_reply("Should we ship this today?")
    assert decision.verdict == NEEDS_REPLY
    assert decision.stage == "heuristic"


@pytest.mark.asyncio
async def test_ready_for_merge_heuristic_skips() -> None:
    decision = await classify_needs_reply("ready for your merge")
    assert decision.verdict == SKIP
    assert decision.stage == "heuristic"


@pytest.mark.asyncio
async def test_status_statement_heuristic_skips() -> None:
    decision = await classify_needs_reply("merged the branch")
    assert decision.verdict == SKIP
    assert decision.stage == "heuristic"


@pytest.mark.asyncio
async def test_classifier_error_skips() -> None:
    decision = await classify_needs_reply("Taking a look at the deploy now", llm=FailingLLM())
    assert decision.verdict == SKIP
    assert decision.stage == "classifier"
    assert "classifier error" in decision.reason


@pytest.mark.asyncio
async def test_classifier_empty_response_skips() -> None:
    decision = await classify_needs_reply("Taking a look at the deploy now", llm=EmptyLLM())
    assert decision.verdict == SKIP
    assert decision.stage == "classifier"


@pytest.mark.asyncio
async def test_classifier_timeout_skips() -> None:
    decision = await classify_needs_reply(
        "Taking a look at the deploy now",
        llm=SlowLLM(),
        timeout_seconds=0.01,
    )
    assert decision.verdict == SKIP
    assert decision.stage == "classifier"
    assert decision.reason == "classifier timeout"
