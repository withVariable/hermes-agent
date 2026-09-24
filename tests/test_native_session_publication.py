"""Real native admission and publication share a serialization boundary."""

import threading
from unittest.mock import Mock
from types import SimpleNamespace

from hermes_state import SessionDB
from tui_gateway import server, session_publication


def test_native_acceptance_excludes_publication_before_initialization(
    tmp_path, monkeypatch
):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.ensure_session("teacher", source="tui")
    session = {
        "session_key": "teacher",
        "history_lock": threading.Lock(),
        "running": False,
        "history": [],
    }
    held = []

    def thread(*, target, **kwargs):
        held.append(target)
        return Mock(start=Mock())

    monkeypatch.setattr(server, "_sessions", {"handle": session})
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *args: None)
    monkeypatch.setattr(server.threading, "Thread", thread)
    monkeypatch.setattr(
        server,
        "_wait_agent",
        lambda *args: {"error": {"message": "controlled init failure"}},
    )
    monkeypatch.setattr(server, "_emit", Mock())
    admitted = server._methods["prompt.submit"](
        "submit", {"session_id": "handle", "text": "I am preparing now"}
    )
    assert admitted["result"]["status"] == "streaming"
    assert session["history"] == []
    params = {
        "session_id": "teacher",
        "message_id": "update:1",
        "text": "New recommendation",
        "timestamp": 1,
        "history_fingerprint": session_publication.history_fingerprint([]),
    }
    assert "session.publish" in server._methods, (
        "native publication must arbitrate with prompt admission"
    )
    publish = server._methods["session.publish"]
    assert publish("publish", params)["result"]["status"] == "busy"
    held.pop()()
    assert session["running"] is False
    assert publish("publish", params)["result"]["status"] == "published"
    db.close()


def test_manual_compaction_retains_ownership_through_key_sync(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.ensure_session("teacher", source="tui")
    session = {
        "session_key": "teacher",
        "history_lock": threading.Lock(),
        "running": False,
        "history": [],
        "agent": SimpleNamespace(),
    }
    monkeypatch.setattr(server, "_sessions", {"handle": session})
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_sess", lambda *args: (session, None))
    monkeypatch.setattr(
        server, "_compress_session_history", lambda *args, **kwargs: (0, {})
    )
    monkeypatch.setattr(server, "_status_update", Mock())
    monkeypatch.setattr(server, "_session_info", lambda *args: {})
    monkeypatch.setattr(server, "_emit", Mock())
    observed = []

    def syncing(*args):
        observed.append(
            server._methods["session.publish"](
                "publish",
                {
                    "session_id": "teacher",
                    "message_id": "external",
                    "text": "Update",
                    "timestamp": 1,
                    "history_fingerprint": session_publication.history_fingerprint([]),
                },
            )["result"]["status"]
        )

    monkeypatch.setattr(server, "_sync_session_key_after_compress", syncing)
    assert (
        server._methods["session.compress"]("compress", {"session_id": "handle"})[
            "result"
        ]["status"]
        == "compressed"
    )
    assert observed == ["busy"]
    db.close()
