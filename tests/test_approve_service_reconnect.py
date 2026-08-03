from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import approve_service
from agent.types import Brief, DraftRequest, DraftResult


class FakeUserClient:
    def __init__(self, *, authorized: bool = True, connect_failures: int = 0) -> None:
        self.authorized = authorized
        self.connect_failures = connect_failures
        self.connected = False
        self.connect_calls = 0
        self.authorization_calls = 0
        self.login_calls = 0
        self.sent: list[tuple[object, str, int | None]] = []

    def is_connected(self) -> bool:
        return self.connected

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_calls <= self.connect_failures:
            raise ConnectionError("temporary outage")
        self.connected = True

    async def is_user_authorized(self) -> bool:
        self.authorization_calls += 1
        return self.authorized

    async def start(self) -> None:
        self.login_calls += 1
        raise AssertionError("the approve service must never start a login flow")

    async def send_message(self, peer, text: str, *, reply_to: int | None = None):
        self.sent.append((peer, text, reply_to))
        return SimpleNamespace(id=789)


class FakeDecisionRequest:
    def __init__(self, approval_id: str) -> None:
        self.approval_id = approval_id
        self.headers = {"X-Mirror-Secret": "test-secret"}

    async def json(self) -> dict[str, str]:
        return {"approval_id": self.approval_id, "action": "approve"}


def draft_request() -> DraftRequest:
    return DraftRequest(
        incoming_message="Can you confirm?",
        thread=[],
        source_message_id=456,
        target_chat_id=-100123,
    )


@pytest.mark.asyncio
async def test_send_as_owner_reconnects_before_sending() -> None:
    user = FakeUserClient()
    approve_service.STATE = approve_service.ServiceState(user=user)

    sent = await approve_service.send_as_owner(draft_request(), "Confirmed.")

    assert sent.id == 789
    assert user.connect_calls == 1
    assert user.authorization_calls == 1
    assert user.sent == [(-100123, "Confirmed.", 456)]
    assert user.login_calls == 0


@pytest.mark.asyncio
async def test_send_as_owner_refuses_unauthorized_session_without_login() -> None:
    user = FakeUserClient(authorized=False)
    approve_service.STATE = approve_service.ServiceState(user=user)

    with pytest.raises(
        approve_service.UserSessionUnauthorizedError,
        match="not authorized; refusing to log in",
    ):
        await approve_service.send_as_owner(draft_request(), "Confirmed.")

    assert user.connect_calls == 1
    assert user.authorization_calls == 1
    assert user.sent == []
    assert user.login_calls == 0

    # The TCP connection may remain open after authorization is rejected; the
    # service must remember that state and refuse later sends too.
    with pytest.raises(approve_service.UserSessionUnauthorizedError):
        await approve_service.send_as_owner(draft_request(), "Confirmed.")
    assert user.sent == []
    assert user.login_calls == 0


@pytest.mark.asyncio
async def test_send_as_owner_bounds_reconnect_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = FakeUserClient(connect_failures=approve_service.USER_CONNECT_ATTEMPTS)
    approve_service.STATE = approve_service.ServiceState(user=user)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(approve_service.asyncio, "sleep", no_sleep)

    with pytest.raises(
        approve_service.UserReconnectError,
        match=r"reconnect failed after 3 attempts; retry shortly",
    ):
        await approve_service.send_as_owner(draft_request(), "Confirmed.")

    assert user.connect_calls == approve_service.USER_CONNECT_ATTEMPTS
    assert user.authorization_calls == 0
    assert user.sent == []
    assert user.login_calls == 0


@pytest.mark.asyncio
async def test_decide_surfaces_reconnect_failure_and_keeps_approval_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_id = "approval-123"
    user = FakeUserClient(connect_failures=1)
    pending = approve_service.Pending(
        approval_id=approval_id,
        request=draft_request(),
        result=DraftResult(draft="Confirmed.", summary=Brief.empty()),
        created_at=0,
    )
    approve_service.STATE = approve_service.ServiceState(
        user=user,
        pending={approval_id: pending},
    )
    approve_service.SECRET = "test-secret"
    approve_service.READY.set()
    monkeypatch.setattr(approve_service, "USER_CONNECT_ATTEMPTS", 1)

    response = await approve_service.handle_decide(FakeDecisionRequest(approval_id))
    body = json.loads(response.text)

    assert response.status == 503
    assert body["ok"] is False
    assert "reconnect failed after 1 attempt; retry shortly" in body["reason"]
    assert approve_service.STATE.pending == {approval_id: pending}
    assert user.sent == []
    assert user.login_calls == 0
