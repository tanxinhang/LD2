from __future__ import annotations

from tools.summarize_feasibility_first_confirmation import _different_paths


def test_nested_config_difference_is_reported_exactly() -> None:
    control = {"marl": {"enabled": False, "shared": 3}, "scenario": {"K": 2}}
    candidate = {"marl": {"enabled": True, "shared": 3}, "scenario": {"K": 2}}
    assert _different_paths(control, candidate) == ["marl.enabled"]


def test_config_difference_detects_missing_and_extra_fields() -> None:
    assert _different_paths({"a": 1}, {"b": 1}) == ["a", "b"]
