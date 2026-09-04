"""Configuration schema must fail closed for reproducible experiments."""

from pathlib import Path

import pytest

from config.params import load_config


DEFAULT = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _write_override(tmp_path, body: str) -> Path:
    path = tmp_path / "override.yaml"
    path.write_text(
        f"extends: {DEFAULT.as_posix()}\n{body}",
        encoding="utf-8",
    )
    return path


def test_unknown_configuration_key_is_rejected(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        "marl:\n  comm_deadline_typo: 0.005\n", encoding="utf-8")
    with pytest.raises(
            ValueError, match=r"config\.marl\.comm_deadline_typo"):
        load_config(str(path))


def test_duplicate_yaml_configuration_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "scenario:\n  K: 4\n  K: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate YAML mapping key 'K'"):
        load_config(str(path))


def test_all_versioned_yaml_configs_match_the_schema():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "config").glob("*.yaml")):
        load_config(str(path))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("scenario:\n  K: banana\n", r"scenario\.K must be an integer"),
        ("scenario:\n  K: true\n", r"scenario\.K must be an integer"),
        ("scenario:\n  region_size:\n    400: ignored\n    500: ignored\n",
         r"scenario\.region_size must be a list/tuple sequence"),
        ("scenario:\n  dt: -1\n", r"scenario\.dt must be greater than zero"),
        ("scenario:\n  Q: 1\n",
         r"target\.omega_q must contain exactly config\.scenario\.Q"),
        ("otfs:\n  B: 0\n", r"otfs\.B must be greater than zero"),
        ("marl:\n  distributed_gap_focus_weight: .nan\n",
         r"distributed_gap_focus_weight must be finite"),
        ("marl:\n  distributed_decision_sufficient_max_bits: -7\n",
         r"distributed_decision_sufficient_max_bits must be greater than zero"),
        ("marl:\n  comm_deadline_s: 0.001\n  comm_processing_delay_s: 0.001\n",
         r"comm_processing_delay_s must be less than"),
    ],
)
def test_core_type_range_and_cross_field_errors_fail_closed(
        tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_config(str(_write_override(tmp_path, body)))


def test_no_truth_mode_requires_a_live_local_belief_chain(tmp_path):
    body = (
        "marl:\n"
        "  distributed_no_truth_fail_closed: true\n"
        "  distributed_coordination_use_local_belief_targets: true\n"
        "  tracking_enabled: false\n"
    )
    with pytest.raises(ValueError, match="requires config.marl.tracking_enabled"):
        load_config(str(_write_override(tmp_path, body)))


def test_empty_nested_mapping_does_not_silently_replace_config(tmp_path):
    with pytest.raises(ValueError, match="config.marl must be a configuration mapping"):
        load_config(str(_write_override(tmp_path, "marl:\n")))


def test_evaluation_only_zero_learning_controls_remain_valid(tmp_path):
    cfg = load_config(str(_write_override(
        tmp_path,
        "marl:\n  lr: 0\n  ppo_epochs: 0\n  num_episodes: 0\n",
    )))
    assert cfg.marl.lr == 0
    assert cfg.marl.ppo_epochs == 0
    assert cfg.marl.num_episodes == 0


def test_exp_800_k8q8_ppo_kl_target_is_pinned_and_unique():
    """P0-regression guard: exp_800_k8q8.yaml must hold a unique PPO KL target.

    This file carries the legacy duplicate ``target_kl`` (0.003 @ old line 99,
    0.02 @ old line 104); PyYAML last-wins made 0.02 the historically effective
    value. Coverage here is explicit (beyond the all-configs glob sweep) so the
    regression cannot silently recur while the file is only spot-globbed.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "exp_800_k8q8.yaml"
    text = path.read_text(encoding="utf-8")
    assert text.count("target_kl:") == 1, (
        "duplicate target_kl key regressed in exp_800_k8q8.yaml")
    cfg = load_config(str(path))
    assert cfg.marl.target_kl == 0.02
    assert cfg.marl.num_episodes == 1000
