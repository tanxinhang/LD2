"""Configuration schema must fail closed for reproducible experiments."""

from pathlib import Path

import pytest

from config.params import load_config


def test_unknown_configuration_key_is_rejected(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        "marl:\n  comm_deadline_typo: 0.005\n", encoding="utf-8")
    with pytest.raises(
            ValueError, match=r"config\.marl\.comm_deadline_typo"):
        load_config(str(path))


def test_all_versioned_yaml_configs_match_the_schema():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "config").glob("*.yaml")):
        load_config(str(path))
