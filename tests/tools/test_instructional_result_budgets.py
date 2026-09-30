"""Budget exemptions are an operator opt-in, not a change to tool defaults."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.tool_executor import _budget_for_agent
from tools.tool_result_storage import enforce_turn_budget, maybe_persist_tool_result


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("storage", ["missing", "fails", "works"])
@pytest.mark.parametrize("context_length", [None, 65_536, 256_000])
def test_configured_results_survive_both_budgets(monkeypatch, enabled, storage, context_length):
    names = ["skill_view"] if enabled else []
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"tool_result_budget": {"exempt_tools": names}})
    agent = SimpleNamespace(context_compressor=SimpleNamespace(context_length=context_length))
    config = _budget_for_agent(agent)
    content = "instruction\n" * 20_000 + "final instruction"
    env = None if storage == "missing" else MagicMock()
    if env is not None:
        env.execute.return_value = {"returncode": int(storage == "fails")}
    output = maybe_persist_tool_result(content, "skill_view", "instructions", env, config)
    messages = [{"name": "skill_view", "tool_call_id": "instructions", "content": output}]
    enforce_turn_budget(messages, env, config)
    assert (messages[0]["content"] == content) is enabled
    if enabled and env is not None:
        env.execute.assert_not_called()


@pytest.mark.parametrize("identity_key", ["name", "tool_name"])
@pytest.mark.parametrize("settings", [{}, {"exempt_tools": []}, {"exempt_tools": "skill_view"}, {"exempt_tools": [3]}, {"exempt_tools": ["skill_view", "custom_instructions"]}])
def test_only_explicit_exemptions_skip_aggregate_budget(monkeypatch, identity_key, settings):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"tool_result_budget": settings})
    config = _budget_for_agent(SimpleNamespace(context_compressor=SimpleNamespace(context_length=65_536)))
    assert config.turn_budget < 200_000
    content = "instruction\n" * 20_000
    names = ["skill_view", "read_file", "custom_instructions", "terminal", ""]
    messages = [{identity_key: name, "tool_call_id": name, "content": content} for name in names]
    enforce_turn_budget(messages, config=config)
    for name, message in zip(names, messages):
        assert (message["content"] == content) is (name in config.exempt_tools)
