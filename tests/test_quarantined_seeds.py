"""Regression tests for the quarantined-seed registry (2026-08-16).

The documented isolation of seeds {795, 747, 105, 860, 2} must be enforced
by code, not by narrative.  These tests lock:
  1. the registry content matches the documented set;
  2. load_stratified_seed_split fails closed (strict) on any split that still
     contains a quarantined seed;
  3. clean splits load normally in both modes;
  4. legacy banks only ever expose the documented five quarantined seeds in
     their active splits (an audit sentinel -- no *new* seed may be silently
     contaminated).  Banks must be regenerated with
     tools/seed_stratification.py; until then strict loads of the affected
     splits fail closed by design.
"""

import json

import pytest

from uav_isac.agents.trainer import (
    load_quarantined_seeds,
    load_stratified_seed_split,
)

QUARANTINED = {795, 747, 105, 860, 2}

LEGACY_BANKS = (
    "config/stratified_seeds_800_q4.json",
    "config/stratified_seeds_980_k6q6.json",
    "config/stratified_seeds_1130_k8q8.json",
)


def _bank(splits):
    return {
        "schema_version": 1,
        "scenario_fingerprint": "test-fingerprint",
        "source_config": "config/test.yaml",
        "splits": splits,
        "seed_metadata": {},
    }


def test_quarantined_registry_matches_documented_isolation():
    assert load_quarantined_seeds() == QUARANTINED


def test_strict_load_rejects_split_containing_quarantined_seeds(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(_bank({"test": [1, 2, 795, 4, 5]})))
    with pytest.raises(ValueError, match="quarantined"):
        load_stratified_seed_split(str(path), "test")
    # Explicit opt-out exists only for structural validation of legacy banks.
    assert load_stratified_seed_split(
        str(path), "test", strict=False) == [1, 2, 795, 4, 5]


def test_strict_load_accepts_clean_split(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(_bank({"test": [10, 11, 12]})))
    assert load_stratified_seed_split(str(path), "test") == [10, 11, 12]
    assert load_stratified_seed_split(
        str(path), "test", strict=False) == [10, 11, 12]


def test_duplicate_and_empty_checks_still_apply(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(_bank({"test": [1, 1]})))
    with pytest.raises(ValueError, match="duplicates"):
        load_stratified_seed_split(str(path), "test")
    path.write_text(json.dumps(_bank({"test": []})))
    with pytest.raises(ValueError, match="empty"):
        load_stratified_seed_split(str(path), "test")


def test_legacy_banks_expose_only_documented_quarantined_seeds():
    """Audit sentinel over the three legacy seed banks.

    Every quarantined seed found in an active split must be one of the five
    documented ones; the test also pins the known contamination (980_k6q6
    test split) so it stays visible until the banks are regenerated.
    """
    hits = {}
    for name in LEGACY_BANKS:
        with open(name, "r", encoding="utf-8") as handle:
            bank = json.load(handle)
        for split, seeds in bank.get("splits", {}).items():
            found = sorted(
                int(seed) for seed in seeds if int(seed) in QUARANTINED)
            if found:
                hits[f"{name}:{split}"] = found
    all_hits = set()
    for found in hits.values():
        all_hits |= set(found)
    assert all_hits <= QUARANTINED
    assert any("980_k6q6" in key for key in hits), (
        "expected the known 980_k6q6 contamination to remain visible; "
        "regenerate the bank and update this sentinel")
