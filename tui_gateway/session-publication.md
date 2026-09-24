# Native session publication

`session.publish` is an authenticated, default-profile JSON-RPC operation for a
trusted external producer with a completed assistant message. It performs no
inference. Product proxies should not add it to browser RPC allowlists merely
because the native backend implements it.

Required parameters:

- `session_id`: persistent session key, not a transport handle.
- `message_id`: stable external delivery ID; stored as `platform_message_id`.
- `text`: complete assistant message.
- `timestamp`: finite source publication time, reused on retries.
- `history_fingerprint`: `tui_gateway.session_publication.history_fingerprint`
  over the ordered raw message rows of the root-to-tip session lineage used by
  the producer. Clients must retain that original snapshot, not hash a fresh
  read after generating the message.

The server resolves the current compression tip. Under the same history locks
used by prompt admission, it refuses publication while `running` or counted
native work owns any handle in the lineage. Counted ownership spans initialization,
worker terminal/error tails, nested automatic continuations, manual compression,
and transport closure. An idle notification or interrupt acknowledgment is not
proof of termination. The resume lock serializes handle loading with publication.

Responses contain `session_id` and one of `published`, `already_published`,
`busy`, or `stale`. A retry with the same ID but different content is rejected.
Successful publication appends one observed assistant row and refreshes idle
native history caches without rewriting old messages or prompt prefixes. Busy
and stale responses have no transcript side effect. A caller must not fall back
to directly writing the database when this operation is unavailable.

This arbitration covers this native server's work, not independent CLI or
other processes writing the same database. Ownership is in-process, so process
exit releases it; durable idempotency is in the session database. The caller
remains responsible for approval, semantic usefulness, and any non-chat delivery.

Run regression coverage with:

```sh
scripts/run_tests.sh tests/test_session_publication.py tests/test_session_publication_lifecycle.py tests/test_native_session_publication.py tests/test_tui_gateway_server.py tests/test_tui_gateway_ws.py
```
