from __future__ import annotations

import asyncio
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
        self.disconnect_calls = 0
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

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def send_message(self, peer, text: str, *, reply_to: int | None = None):
        self.sent.append((peer, text, reply_to))
        return SimpleNamespace(id=789)


class BlockingReconnectUser(FakeUserClient):
    def __init__(self) -> None:
        super().__init__()
        self.connect_started = asyncio.Event()
        self.allow_connect = asyncio.Event()

    async def connect(self) -> None:
        self.connect_calls += 1
        self.connect_started.set()
        await self.allow_connect.wait()
        self.connected = True


class FakeStartupUser(FakeUserClient):
    async def get_me(self):
        return SimpleNamespace(id=42, username="owner")

    async def __aenter__(self):
        # Telethon's client context manager enters through start(). Retaining
        # this trap makes the regression test fail if startup ever uses it.
        await self.start()
        return self

    async def __aexit__(self, *_args) -> None:
        await self.disconnect()


class FakeRunner:
    def __init__(self, _app) -> None:
        self.setup_calls = 0
        self.cleanup_calls = 0

    async def setup(self) -> None:
        self.setup_calls += 1

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeSite:
    def __init__(self, _runner, _host: str, _port: int) -> None:
        self.start_calls = 0

    async def start(self) -> None:
        self.start_calls += 1


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


@pytest.mark.asyncio
async def test_concurrent_decide_during_reconnect_sends_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_id = "approval-concurrent"
    user = BlockingReconnectUser()
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

    async def no_owner_sample(*_args, **_kwargs) -> None:
        return None

    async def no_feedback(*_args, **_kwargs) -> int:
        return 1

    monkeypatch.setattr(approve_service, "add_owner_sample", no_owner_sample)
    monkeypatch.setattr(approve_service, "record_feedback", no_feedback)
    monkeypatch.setattr(approve_service, "_save_pending", lambda: None)

    first = asyncio.create_task(
        approve_service.handle_decide(FakeDecisionRequest(approval_id))
    )
    await asyncio.wait_for(user.connect_started.wait(), timeout=1)
    second = asyncio.create_task(
        approve_service.handle_decide(FakeDecisionRequest(approval_id))
    )
    await asyncio.sleep(0)

    assert not second.done()
    user.allow_connect.set()
    responses = await asyncio.gather(first, second)

    assert sorted(response.status for response in responses) == [200, 410]
    assert user.connect_calls == 1
    assert user.sent == [(-100123, "Confirmed.", 456)]
    assert user.login_calls == 0
    assert approve_service.STATE.pending == {}


@pytest.mark.asyncio
async def test_startup_uses_connect_only_and_never_starts_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopService(Exception):
        pass

    user = FakeStartupUser()
    runner = FakeRunner(None)
    site = FakeSite(None, "127.0.0.1", 8791)

    async def warm_up() -> None:
        return None

    async def stop_after_start(*_args, **_kwargs) -> None:
        raise StopService

    monkeypatch.setenv("TELEGRAM_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_API_HASH", "api-hash")
    monkeypatch.setenv("MIRROR_APPROVE_SECRET", "test-secret")
    monkeypatch.setattr(approve_service, "OWNER_USER_ID", 42)
    monkeypatch.setattr(approve_service, "TelegramClient", lambda *_args: user)
    monkeypatch.setattr(approve_service.web, "AppRunner", lambda _app: runner)
    monkeypatch.setattr(approve_service.web, "TCPSite", lambda *_args: site)
    monkeypatch.setattr(approve_service, "_load_pending", lambda: None)
    monkeypatch.setattr(approve_service, "_warm_up", warm_up)
    monkeypatch.setattr(approve_service, "maybe_resume_backfill", stop_after_start)

    with pytest.raises(StopService):
        await approve_service.build_and_run()

    assert user.connect_calls == 1
    assert user.authorization_calls == 1
    assert user.login_calls == 0
    assert user.disconnect_calls == 1
    assert runner.setup_calls == 1
    assert runner.cleanup_calls == 1
    assert site.start_calls == 1
