"""Serialize external transcript publication with native session workers.

`running` describes UI turn state, not worker termination: an automatic follow-up
may be admitted after an idle event. Retain counted ownership through the full
worker tail, including child dispatch, and after a transport closes its handle.
Call publication under the server's resume lock so a resumed handle cannot read
an old transcript while publication commits.
"""

import hashlib
import json
import math
import threading
from contextlib import ExitStack

_workers_lock = threading.RLock()
_workers = {}


def history_fingerprint(rows):
    encoded = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reserve(session):
    key = id(session)
    with _workers_lock:
        _, count = _workers.get(key, (session, 0))
        _workers[key] = (session, count + 1)

    def release():
        with _workers_lock:
            _, count = _workers[key]
            if count == 1:
                del _workers[key]
            else:
                _workers[key] = (session, count - 1)

    return release


def run_owned_operation(session, target):
    release = _reserve(session)
    try:
        return target()
    finally:
        release()


def start_owned_worker(session, target, *, thread_factory=threading.Thread):
    """Reserve before scheduling; release only after target and its tail exit."""
    release = _reserve(session)

    def owned():
        try:
            target()
        finally:
            release()

    try:
        thread_factory(target=owned, daemon=True).start()
    except BaseException:
        release()
        raise


def _root(db, session_id):
    lineage = db._session_lineage_root_to_tip(session_id)
    return lineage[0] if lineage else session_id


def publish_message(db, sessions, params):
    """Idempotent CAS append. Never infer ownership from client-visible events."""
    key = params.get("session_id")
    message_id = params.get("message_id")
    text = params.get("text")
    expected = params.get("history_fingerprint")
    at = params.get("timestamp")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (key, message_id, text, expected)
    ):
        raise ValueError(
            "session_id, message_id, text and history_fingerprint are required"
        )
    if (
        not isinstance(at, (float, int))
        or isinstance(at, bool)
        or not math.isfinite(at)
    ):
        raise ValueError("timestamp must be finite")
    if not db.get_session(key):
        raise ValueError("session not found")
    target = db.resolve_resume_session_id(key)
    root = _root(db, target)
    with _workers_lock, ExitStack() as locks:
        candidates = {id(session): session for session in sessions}
        candidates.update({key: session for key, (session, _) in _workers.items()})
        matching = [
            session
            for session in candidates.values()
            if _root(db, session.get("session_key") or "") == root
        ]
        for session in matching:
            locks.enter_context(session["history_lock"])
        if any(
            session.get("running") or id(session) in _workers for session in matching
        ):
            return {"status": "busy", "session_id": target}
        # Compression can finish between the preliminary lookup and acquiring
        # ownership locks. Resolve again inside the publication boundary.
        target = db.resolve_resume_session_id(key)
        rows = [
            row
            for ancestor in db._session_lineage_root_to_tip(target)
            for row in db.get_messages(ancestor)
        ]
        existing = next(
            (row for row in rows if row.get("platform_message_id") == message_id), None
        )
        if existing is not None:
            if existing.get("role") != "assistant" or existing.get("content") != text:
                raise ValueError("message_id already belongs to different content")
            return {"status": "already_published", "session_id": target}
        if history_fingerprint(rows) != expected:
            return {"status": "stale", "session_id": target}
        db.append_message(
            target,
            "assistant",
            text,
            platform_message_id=message_id,
            timestamp=at,
            observed=True,
            finish_reason="stop",
        )
        # No accepted worker can be holding an older cache: prompt admission
        # uses these same locks, and resume is serialized by the caller.
        for session in matching:
            session["history"] = db.get_messages_as_conversation(
                target, include_ancestors=True
            )
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return {"status": "published", "session_id": target}
