"""Atomic external delivery against native turn ownership and history."""

import threading
from unittest.mock import Mock

import pytest


def test_owned_worker_excludes_publication_until_terminal(tmp_path):
    from hermes_state import SessionDB
    from tui_gateway import session_publication as publication

    db = SessionDB(db_path=tmp_path / "state.db")
    db.ensure_session("teacher", source="tui")
    session = {
        "session_key": "teacher",
        "history_lock": threading.Lock(),
        "running": False,
        "history": [],
    }
    held = []

    def thread_factory(*, target, **kwargs):
        held.append(target)
        return Mock(start=Mock())

    publication.start_owned_worker(session, lambda: None, thread_factory=thread_factory)
    params = {
        "session_id": "teacher",
        "message_id": "external:one",
        "text": "A useful update",
        "timestamp": 1.0,
        "history_fingerprint": publication.history_fingerprint([]),
    }
    assert publication.publish_message(db, [], params)["status"] == "busy"
    held.pop()()
    assert publication.publish_message(db, [], params)["status"] == "published"
    assert publication.publish_message(db, [], params)["status"] == "already_published"
    assert len(db.get_messages("teacher")) == 1
    db.close()
