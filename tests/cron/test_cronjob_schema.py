"""Tests for the cronjob tool schema shape.

Guards both the description text AND the formal `if`/`then` conditionals that
flag ``schedule`` (and, for LLM-driven jobs, ``prompt``/``skills``) as
required for ``action=create`` — the load-bearing fix for models that only
enforce the formal ``required[]``/conditional schema and don't reliably
follow prose-only requirements. See issue #32427 / PR #32448 (description-only
fix) and the follow-up that added the schema-level conditionals themselves.
"""

from __future__ import annotations


def test_cronjob_schema_action_description_flags_create_requirements():
    """`action` description must state schedule + prompt/skills are required for create."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    action_desc = CRONJOB_SCHEMA["parameters"]["properties"]["action"]["description"]
    assert "action=create" in action_desc
    assert "schedule" in action_desc
    assert "REQUIRED" in action_desc


def test_cronjob_schema_schedule_description_flags_required_for_create():
    """`schedule` description must explicitly state REQUIRED for action=create."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    schedule_desc = CRONJOB_SCHEMA["parameters"]["properties"]["schedule"]["description"]
    assert "REQUIRED" in schedule_desc
    assert "action=create" in schedule_desc


def test_cronjob_schema_required_array_unchanged():
    """`required[]` stays minimal — `action` only.

    `schedule` and `prompt`/`skills` are only mandatory for action=create,
    not for list/remove/pause/etc., so they can't be promoted into the flat
    top-level required array without breaking those other actions. The
    `allOf`/`if`/`then` conditionals (asserted below) carry the conditional
    requirement instead of the flat array or prose alone.
    """
    from tools.cronjob_tools import CRONJOB_SCHEMA

    assert CRONJOB_SCHEMA["parameters"]["required"] == ["action"]


def _create_condition(entry: dict) -> dict:
    """Return the `if` clause of an `allOf` entry that's gated on action=create."""
    return entry["if"]


def test_cronjob_schema_conditionally_requires_schedule_for_create():
    """The formal schema must require `schedule` whenever action=create,
    without requiring it for any other action (list/update/pause/etc.)."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    all_of = CRONJOB_SCHEMA["parameters"]["allOf"]
    schedule_entries = [
        entry for entry in all_of
        if entry.get("then", {}).get("required") == ["schedule"]
    ]
    assert len(schedule_entries) == 1
    condition = _create_condition(schedule_entries[0])
    assert condition["properties"]["action"]["const"] == "create"
    assert condition["required"] == ["action"]


def test_cronjob_schema_conditionally_requires_prompt_or_skills_unless_no_agent():
    """The formal schema must require prompt-or-skills for action=create,
    but that condition must NOT fire when no_agent=true (script-only jobs
    don't need prompt/skills)."""
    from tools.cronjob_tools import CRONJOB_SCHEMA

    all_of = CRONJOB_SCHEMA["parameters"]["allOf"]
    prompt_or_skills_entries = [
        entry for entry in all_of
        if "anyOf" in entry.get("then", {})
    ]
    assert len(prompt_or_skills_entries) == 1
    entry = prompt_or_skills_entries[0]

    any_of = entry["then"]["anyOf"]
    assert {"required": ["prompt"]} in any_of
    assert any(
        clause.get("required") == ["skills"] for clause in any_of
    ), "must accept a non-empty skills list as the prompt alternative"

    condition = entry["if"]
    # Gated on action=create...
    create_clause = condition["allOf"][0]
    assert create_clause["properties"]["action"]["const"] == "create"
    # ...AND NOT no_agent=true.
    no_agent_clause = condition["allOf"][1]
    assert no_agent_clause["not"]["properties"]["no_agent"]["const"] is True
