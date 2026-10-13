"""ACP reconnect must not claim to resume a session under a different ID."""

from types import SimpleNamespace

import acp
import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def make_server(db, calls):
    def run_conversation(*, user_message, conversation_history, task_id, **kwargs):
        calls.append((task_id, user_message, list(conversation_history or [])))
        return {
            "final_response": "Fixture reply",
            "messages": [
                *(conversation_history or []),
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "Fixture reply"},
            ],
        }

    manager = SessionManager(
        db=db,
        agent_factory=lambda: SimpleNamespace(
            model="fixture",
            provider="fixture",
            tools=[],
            enabled_toolsets=[],
            disabled_toolsets=[],
            run_conversation=run_conversation,
        ),
    )
    return HermesACPAgent(session_manager=manager)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["resume_session", "load_session"])
async def test_lost_empty_session_reports_not_found_before_new_session_prompt(
    tmp_path, method
):
    calls = []
    db = SessionDB(tmp_path / "state.db")
    try:
        original = await make_server(db, calls).new_session(
            cwd=str(tmp_path), mcp_servers=[]
        )
        assert db.get_session(original.session_id) is None

        # A new manager has no in-memory session, as after ACPX closes the process.
        restarted = make_server(db, calls)
        with pytest.raises(acp.RequestError) as error:
            await getattr(restarted, method)(
                cwd=str(tmp_path), session_id=original.session_id
            )
        assert error.value.code == -32002
        assert calls == []
        assert restarted.session_manager.list_sessions() == []

        # The client can now explicitly create a session and use its returned ID.
        fresh = await restarted.new_session(cwd=str(tmp_path), mcp_servers=[])
        response = await restarted.prompt(
            session_id=fresh.session_id,
            prompt=[TextContentBlock(type="text", text="First message")],
        )
        assert response.stop_reason == "end_turn"
        assert calls == [(fresh.session_id, "First message", [])]
        assert db.get_session(original.session_id) is None
        assert [
            m["content"] for m in db.get_messages_as_conversation(fresh.session_id)
        ] == [
            "First message",
            "Fixture reply",
        ]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_nonempty_session_resumes_same_id_and_preserves_history(tmp_path):
    calls = []
    db = SessionDB(tmp_path / "state.db")
    try:
        server = make_server(db, calls)
        original = await server.new_session(cwd=str(tmp_path), mcp_servers=[])
        first = await server.prompt(
            session_id=original.session_id,
            prompt=[TextContentBlock(type="text", text="First message")],
        )
        assert first.stop_reason == "end_turn"
        history = db.get_messages_as_conversation(original.session_id)

        restarted = make_server(db, calls)
        await restarted.resume_session(
            cwd=str(tmp_path), session_id=original.session_id
        )
        response = await restarted.prompt(
            session_id=original.session_id,
            prompt=[TextContentBlock(type="text", text="Second message")],
        )
        assert response.stop_reason == "end_turn"
        assert calls[-1][:2] == (original.session_id, "Second message")
        assert [(m["role"], m["content"]) for m in calls[-1][2]] == [
            (m["role"], m["content"]) for m in history
        ]
        assert [
            m["content"] for m in db.get_messages_as_conversation(original.session_id)
        ] == [
            "First message",
            "Fixture reply",
            "Second message",
            "Fixture reply",
        ]
    finally:
        db.close()
