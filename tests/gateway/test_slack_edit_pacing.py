"""Slack edits share a workspace budget and preserve cooldown/recovery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack import adapter as slack_module
from plugins.platforms.slack.adapter import SlackAdapter


def slack_error(status, headers):
    return SlackApiError(
        "Slack refused the edit",
        AsyncSlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.update",
            req_args={},
            data={
                "ok": False,
                "error": "ratelimited" if status == 429 else "cant_update_message",
            },
            headers=headers,
            status_code=status,
        ),
    )


@pytest.fixture
def setup(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        clock.now += delay
        await asyncio.sleep(0)

    monkeypatch.setattr(
        slack_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    monkeypatch.setattr(
        slack_module, "asyncio", SimpleNamespace(Lock=asyncio.Lock, sleep=sleep)
    )
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = MagicMock()
    adapter._clear_thread_status_quietly = AsyncMock()
    client = MagicMock(spec=AsyncWebClient)
    client.chat_update = AsyncMock(return_value={"ok": True})
    adapter._app.client = client
    return adapter, client, clock, sleeps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Retry-After": "45"}, 45.0),
        ({"retry-after": "9"}, 9.0),
        ({"Retry-After": "invalid"}, None),
    ],
)
async def test_edit_preserves_slack_cooldown(setup, headers, expected):
    adapter, client, _, _ = setup
    client.chat_update.side_effect = slack_error(429, headers)
    result = await adapter.edit_message("C1", "m1", "streaming text")
    assert not result.success
    assert result.retryable is True
    assert result.retry_after == expected
    assert "rate" in result.error


@pytest.mark.asyncio
async def test_concurrent_threads_share_budget_and_workspaces_are_independent(setup):
    adapter, client, clock, sleeps = setup
    updates = []

    async def update(**kwargs):
        updates.append((clock.now, kwargs["ts"]))
        return {"ok": True}

    client.chat_update.side_effect = update
    results = await asyncio.gather(
        adapter.edit_message("C1", "thread-1", "one"),
        adapter.edit_message("C2", "thread-2", "two"),
        adapter.edit_message("C1", "thread-3", "three", finalize=True),
    )
    assert all(result.success for result in results)
    assert [when for when, _ in updates] == [100.0, 101.25, 102.5]
    other = MagicMock(spec=AsyncWebClient)
    other.chat_update = AsyncMock(return_value={"ok": True})
    adapter._team_clients["T2"] = other
    adapter._channel_team["C3"] = "T2"
    assert (await adapter.edit_message("C3", "other", "other workspace")).success
    assert sleeps == [1.25, 1.25]


@pytest.mark.asyncio
async def test_cooldown_blocks_other_threads_and_final_edits_until_full_retry_after(
    setup,
):
    adapter, client, clock, sleeps = setup
    client.chat_update.side_effect = [
        slack_error(429, {"Retry-After": "45"}),
        {"ok": True},
    ]
    assert not (await adapter.edit_message("C1", "m1", "preview")).success
    result = await adapter.edit_message("C2", "m2", "completed answer", finalize=True)
    assert result.success
    assert sleeps == [45.0]  # never truncate Slack's cooldown to the consumer's 30s cap
    assert clock.now == 145.0


@pytest.mark.asyncio
async def test_consumer_recovers_after_real_slack_429_without_fallback(
    setup, monkeypatch
):
    from gateway import stream_consumer

    adapter, client, clock, sleeps = setup
    monkeypatch.setattr(
        stream_consumer, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    client.chat_update.side_effect = [
        slack_error(429, {"Retry-After": "9"}),
        {"ok": True},
        {"ok": True},
    ]
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "preview"})
    consumer = GatewayStreamConsumer(
        adapter, "C1", StreamConsumerConfig(edit_interval=0.8)
    )
    assert await consumer._send_or_edit("start")
    assert not await consumer._send_or_edit("start plus more")
    assert consumer._current_edit_interval == 9.0
    clock.now += 9.0
    assert await consumer._send_or_edit("start plus more and recovered")
    assert await consumer._send_or_edit(
        "start plus more and recovered final", finalize=True
    )
    assert consumer._edit_supported
    assert not consumer._fallback_final_send
    assert consumer._flood_strikes == 0
    assert client.chat_postMessage.await_count == 1
    assert client.chat_update.await_count == 3
    assert sleeps == [1.25]


@pytest.mark.asyncio
async def test_permanent_edit_errors_remain_permanent(setup):
    adapter, client, _, _ = setup
    client.chat_update.side_effect = slack_error(403, {})
    result = await adapter.edit_message("C1", "m1", "text")
    assert not result.success
    assert not result.retryable
    assert result.retry_after is None
