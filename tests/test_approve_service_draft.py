from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import approve_service
from agent.needs_reply import NEEDS_REPLY, SKIP, NeedsReplyDecision


class FakeTransport:
    def is_closing(self) -> bool:
        return False


class FakeRequest:
    def __init__(self, body: dict, secret: str = "test-secret") -> None:
        self.headers = {"X-Mirror-Secret": secret}
        self.transport = FakeTransport()
        self._body = body

    async def json(self) -> dict:
        return self._body


@pytest.fixture(autouse=True)
def service_ready(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    approve_service.SECRET = "test-secret"
    approve_service.STATE = approve_service.ServiceState(user=SimpleNamespace())
    approve_service.READY.set()
    monkeypatch.setattr(approve_service, "PENDING_STORE_PATH", tmp_path / "pending.pkl")


def draft_payload(question: str) -> dict:
    return {
        "chat_id": 123,
        "question_msg_id": 456,
        "question": question,
        "is_topic": False,
    }


@pytest.mark.asyncio
async def test_handle_draft_skip_returns_no_content_and_does_not_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def classify(_text: str) -> NeedsReplyDecision:
        return NeedsReplyDecision(SKIP, "classifier", "statement")

    async def draft(*_args, **_kwargs):
        raise AssertionError("draft_reply must not run for SKIP")

    monkeypatch.setattr(approve_service, "classify_needs_reply", classify)
    monkeypatch.setattr(approve_service, "draft_reply", draft)

    response = await approve_service.handle_draft(FakeRequest(draft_payload("Merged the branch.")))

    assert response.status == 204
    assert response.body in (None, b"")
    assert approve_service.STATE is not None
    assert approve_service.STATE.pending == {}


@pytest.mark.asyncio
async def test_handle_draft_classifier_exception_skips_and_does_not_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def classify(_text: str) -> NeedsReplyDecision:
        raise RuntimeError("classifier exploded")

    async def draft(*_args, **_kwargs):
        raise AssertionError("draft_reply must not run when classifier fails")

    monkeypatch.setattr(approve_service, "classify_needs_reply", classify)
    monkeypatch.setattr(approve_service, "draft_reply", draft)

    response = await approve_service.handle_draft(FakeRequest(draft_payload("Night, Jay.")))

    assert response.status == 204
    assert response.body in (None, b"")
    assert approve_service.STATE is not None
    assert approve_service.STATE.pending == {}


@pytest.mark.asyncio
async def test_handle_draft_gate_skip_returns_no_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def classify(_text: str) -> NeedsReplyDecision:
        return NeedsReplyDecision(SKIP, "classifier", "statement")

    monkeypatch.setattr(approve_service, "classify_needs_reply", classify)

    response = await approve_service.handle_draft_gate(
        FakeRequest(draft_payload("👍 Night, Jay."))
    )

    assert response.status == 204
    assert response.body in (None, b"")


@pytest.mark.asyncio
async def test_handle_draft_gate_classifier_exception_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def classify(_text: str) -> NeedsReplyDecision:
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(approve_service, "classify_needs_reply", classify)

    response = await approve_service.handle_draft_gate(
        FakeRequest(draft_payload("Taking a look tonight."))
    )

    assert response.status == 204
    assert response.body in (None, b"")


@pytest.mark.asyncio
async def test_handle_draft_gate_needs_reply_returns_gate_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def classify(_text: str) -> NeedsReplyDecision:
        return NeedsReplyDecision(NEEDS_REPLY, "heuristic", "question mark")

    monkeypatch.setattr(approve_service, "classify_needs_reply", classify)

    response = await approve_service.handle_draft_gate(
        FakeRequest(draft_payload("Can you review?"))
    )
    body = json.loads(response.text)

    assert response.status == 200
    assert body == {
        "ok": True,
        "needs_reply": True,
        "gate": {
            "verdict": NEEDS_REPLY,
            "stage": "heuristic",
            "reason": "question mark",
        },
    }


@pytest.mark.asyncio
async def test_handle_draft_needs_reply_draft_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def classify(_text: str) -> NeedsReplyDecision:
        return NeedsReplyDecision(NEEDS_REPLY, "heuristic", "question mark")

    async def draft(*_args, **_kwargs):
        raise RuntimeError("LLM failed after retries")

    monkeypatch.setattr(approve_service, "classify_needs_reply", classify)
    monkeypatch.setattr(approve_service, "draft_reply", draft)

    response = await approve_service.handle_draft(FakeRequest(draft_payload("Can you review?")))
    body = json.loads(response.text)

    assert response.status == 500
    assert body == {"ok": False, "reason": "draft failed: LLM failed after retries"}
    assert approve_service.STATE is not None
    assert approve_service.STATE.pending == {}
