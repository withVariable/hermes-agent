"""Publication never mistakes an idle event, child rotation or close for release."""

import threading
from unittest.mock import Mock

import pytest

from hermes_state import SessionDB
from tui_gateway import session_publication as publication


@pytest.fixture
def state(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.ensure_session("parent", source="tui")
    session = {
        "session_key": "parent",
        "history_lock": threading.Lock(),
        "running": False,
        "history": [],
    }
    params = {
        "session_id": "parent",
        "message_id": "external:one",
        "text": "Useful update",
        "timestamp": 1,
        "history_fingerprint": publication.history_fingerprint([]),
    }
    yield db, session, params
    db.close()
    assert not publication._workers


def held_threads():
    targets = []

    def factory(*, target, **kwargs):
        targets.append(target)
        return Mock(start=Mock())

    return targets, factory


def test_automatic_child_claim_survives_parent_exit(state):
    db, session, params = state
    targets, factory = held_threads()
    observed = []

    def parent():
        # message.complete and idle have already been emitted, but the parent
        # is still deciding whether to schedule a continuation.
        observed.append(publication.publish_message(db, [session], params)["status"])
        publication.start_owned_worker(session, lambda: None, thread_factory=factory)

    publication.start_owned_worker(session, parent, thread_factory=factory)
    targets.pop(0)()
    assert observed == ["busy"]
    assert publication.publish_message(db, [session], params)["status"] == "busy"
    targets.pop(0)()
    assert publication.publish_message(db, [session], params)["status"] == "published"


def test_compression_alias_and_closed_handle_retain_parent_ownership(state):
    db, session, params = state
    targets, factory = held_threads()
    publication.start_owned_worker(session, lambda: None, thread_factory=factory)
    db.ensure_session("child", source="tui", parent_session_id="parent")
    params["session_id"] = "child"
    # Neither a new idle handle for the child nor removing the original
    # transport handle may hide the still-running parent worker.
    idle = {"session_key": "child", "history_lock": threading.Lock(), "running": False}
    assert publication.publish_message(db, [idle], params)["status"] == "busy"
    assert publication.publish_message(db, [], params)["status"] == "busy"
    targets.pop()()
    result = publication.publish_message(db, [idle], params)
    assert result == {"status": "published", "session_id": "child"}
    assert len(db.get_messages("child")) == 1
    assert db.get_messages("parent") == []
    assert idle["history"][-1]["content"] == "Useful update"


def test_tip_rotation_between_resolution_and_lock_is_rechecked(state, monkeypatch):
    db, session, params = state
    original_root = publication._root
    rotated = False

    def root(db, key):
        nonlocal rotated
        if not rotated:
            rotated = True
            db.ensure_session("child", source="tui", parent_session_id="parent")
            db.append_message("child", "user", "Compressed conversation")
            session["session_key"] = "child"
        return original_root(db, key)

    monkeypatch.setattr(publication, "_root", root)
    assert publication.publish_message(db, [session], params)["status"] == "stale"
    assert db.get_messages("parent") == []


def test_post_assessment_teacher_message_invalidates_compare_and_swap(state):
    db, session, params = state
    db.append_message("parent", "user", "This has already been prepared")
    assert publication.publish_message(db, [session], params)["status"] == "stale"
    assert len(db.get_messages("parent")) == 1


def test_publication_is_idempotent_but_rejects_message_id_reuse(state):
    db, session, params = state
    assert publication.publish_message(db, [session], params)["status"] == "published"
    assert (
        publication.publish_message(db, [session], params)["status"]
        == "already_published"
    )
    with pytest.raises(ValueError, match="different content"):
        publication.publish_message(
            db, [session], {**params, "text": "Changed recommendation"}
        )
    assert len(db.get_messages("parent")) == 1


@pytest.mark.parametrize("where", ["worker", "start"])
def test_exception_releases_counted_ownership(state, where):
    db, session, params = state
    targets, factory = held_threads()

    def fail():
        raise RuntimeError("controlled failure")

    if where == "worker":
        publication.start_owned_worker(session, fail, thread_factory=factory)
        with pytest.raises(RuntimeError):
            targets.pop()()
    else:
        with pytest.raises(RuntimeError):
            publication.start_owned_worker(
                session, lambda: None, thread_factory=lambda **kwargs: Mock(start=fail)
            )
    assert publication.publish_message(db, [session], params)["status"] == "published"
