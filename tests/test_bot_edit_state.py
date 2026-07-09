from __future__ import annotations

from agent.bot import PENDING_EDITS, cancel_edit_state, editing_text, enter_edit_state
from agent.types import Brief, DraftResult


def test_edit_state_keeps_draft_visible_and_can_cancel() -> None:
    PENDING_EDITS.clear()
    result = DraftResult(
        draft="Full proposed reply",
        summary=Brief(goal="g", now="n", next="x", open=[]),
    )

    enter_edit_state(123, "abc")

    assert PENDING_EDITS[123] == "abc"
    assert "Full proposed reply" in editing_text(result)
    assert "Editing - send your version." in editing_text(result)
    assert cancel_edit_state(123, "abc") is True
    assert 123 not in PENDING_EDITS
