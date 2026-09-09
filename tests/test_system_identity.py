"""Focused tests for the executable-config and frozen-seed identity gate."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tools.check_system_identity import collect_checks


ROOT = Path(__file__).resolve().parents[1]
K16_CONFIG = ROOT / "config" / "exp_strict_distributed_k16q16.yaml"
K16_BANK = ROOT / "config" / "stratified_seeds_1130_k16q16_blind.json"


def _write_runtime_profile(
    tmp_path: Path,
    bank_path: Path,
    *,
    scenario: dict[str, object] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "extends": str(K16_CONFIG),
        "marl": {"eval_seed_bank_path": str(bank_path)},
    }
    if scenario is not None:
        payload["scenario"] = scenario
    profile = tmp_path / "identity_profile.yaml"
    profile.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return profile


def _formal_warnings(messages: list[str]) -> list[str]:
    return [
        message for message in messages
        if message.startswith("FORMAL WARNING: ")
    ]


def test_k16_v2_executable_config_has_exact_seed_bank_identity():
    failures, messages = collect_checks(str(K16_CONFIG), strict=True)

    assert failures == []
    assert any(
        "fingerprint matches its source_config" in message
        for message in messages
    )
    assert any(
        "matches effective K/Q/region/dynamics" in message
        for message in messages
    )


def test_base_semantic_manifest_has_exact_k4q2_seed_bank_identity():
    failures, messages = collect_checks(
        str(ROOT / "config" / "system_manifest.yaml"), strict=True)

    assert failures == []
    assert any(
        "fingerprint matches its source_config" in message
        for message in messages
    )
    assert any(
        "matches effective K/Q/region/dynamics" in message
        for message in messages
    )


def test_base_semantic_manifest_has_no_formal_warnings_in_compatibility_mode():
    failures, messages = collect_checks(
        str(ROOT / "config" / "system_manifest.yaml"), strict=False)

    assert failures == []
    assert _formal_warnings(messages) == []


def test_missing_configured_bank_fails_even_without_strict_git_gate(tmp_path):
    profile = _write_runtime_profile(tmp_path, tmp_path / "missing-bank.json")

    failures, _messages = collect_checks(str(profile), strict=False)

    assert any(
        "configured seed bank does not exist" in failure
        for failure in failures
    )


def test_schema_defect_is_warning_then_strict_failure(tmp_path):
    bank = json.loads(K16_BANK.read_text(encoding="utf-8"))
    bank["schema_version"] = 1
    candidate_bank = tmp_path / "wrong-schema.json"
    candidate_bank.write_text(json.dumps(bank), encoding="utf-8")
    profile = _write_runtime_profile(tmp_path, candidate_bank)

    compatibility_failures, compatibility_messages = collect_checks(
        str(profile), strict=False)
    strict_failures, _strict_messages = collect_checks(
        str(profile), strict=True)

    assert compatibility_failures == []
    assert any(
        "schema_version is 1, expected 2" in warning
        for warning in _formal_warnings(compatibility_messages)
    )
    assert any(
        "schema_version is 1, expected 2" in failure
        for failure in strict_failures
    )


def test_effective_scenario_must_match_bank_source_and_fingerprint(tmp_path):
    profile = _write_runtime_profile(
        tmp_path,
        K16_BANK,
        scenario={"K": 15},
    )

    failures, _messages = collect_checks(str(profile), strict=True)

    joined = "\n".join(failures)
    assert "scenario_fingerprint does not match the effective" in joined
    assert "scenario identity mismatch" in joined
    assert "K: effective=15, bank_source=16" in joined
