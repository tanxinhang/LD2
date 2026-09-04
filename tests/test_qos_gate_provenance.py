"""R15 property tests: the formal acceptance gates have ONE source.

The gate tools (tools/assert_gate_thresholds.py, tools/report_blind_certification.py)
read ``marl.qos_acceptance_floors`` / ``marl.qos_acceptance_wilson_lcb_floor``
when given ``--config``; without a config they use canonical defaults that must
equal the config/params.py defaults.  Training reward floors (coord_reward_*)
and communication constraint floors (comm_qos_*) are numerically identical to
the acceptance triple BY DEFAULT, and that equality is locked here as a TEST
(not by a shared reference) -- the three sources remain semantically
independent (C1): acceptance / reward / constraint.
"""

import pytest

from config.params import get_default_config
from tools.assert_gate_thresholds import (
    DEFAULT_ACCEPTANCE_FLOORS,
    MEDIUM_FLOORS,
    QOS_FLOOR,
    acceptance_floors_from_config,
    assert_medium_gate,
    medium_gate_checks,
)


def test_default_acceptance_floors_match_params():
    """Canonical gate defaults == config/params.py marl.qos_acceptance_floors."""
    cfg = get_default_config()
    p = cfg.marl.qos_acceptance_floors
    assert p == pytest.approx([0.60, 0.70, 0.80], abs=1e-12)
    assert MEDIUM_FLOORS["worst"] == pytest.approx(p[0], abs=1e-12)
    assert MEDIUM_FLOORS["weak3"] == pytest.approx(p[1], abs=1e-12)
    assert MEDIUM_FLOORS["steady"] == pytest.approx(p[2], abs=1e-12)
    assert DEFAULT_ACCEPTANCE_FLOORS == MEDIUM_FLOORS


def test_acceptance_lcb_floor_matches_params():
    cfg = get_default_config()
    assert cfg.marl.qos_acceptance_wilson_lcb_floor == pytest.approx(
        QOS_FLOOR, abs=1e-12)


def test_triple_sources_numerically_consistent():
    """By default acceptance == reward == communication floors (all 0.6/0.7/0.8).

    This locks only NUMERICAL equality; the keys stay independent so a training
    reward change does not silently move the formal gate and vice versa.
    """
    cfg = get_default_config()
    accept = cfg.marl.qos_acceptance_floors
    reward = (cfg.marl.coord_reward_worst_floor,
              cfg.marl.coord_reward_weak3_floor,
              cfg.marl.coord_reward_steady_floor)
    comm = (cfg.marl.comm_qos_worst_min,
            cfg.marl.comm_qos_weak3_min,
            cfg.marl.comm_qos_steady_min)
    assert list(reward) == pytest.approx(accept, abs=1e-12)
    assert list(comm) == pytest.approx(accept, abs=1e-12)


def test_gate_reads_floors_from_config():
    """Changing qos_acceptance_floors in a config moves the gate verdict."""
    cfg = get_default_config()
    cfg.marl.qos_acceptance_floors = [0.62, 0.74, 0.86]
    cfg.marl.qos_acceptance_wilson_lcb_floor = 0.74
    floors = acceptance_floors_from_config.__wrapped__(None) if False else None
    # API-level: pass the floors dict directly (same path a --config run uses)
    custom = {
        "worst": cfg.marl.qos_acceptance_floors[0],
        "weak3": cfg.marl.qos_acceptance_floors[1],
        "steady": cfg.marl.qos_acceptance_floors[2],
    }
    # values that clear the DEFAULT gate but not the tightened config gate:
    # steady 0.84 < config 0.86; weak3 0.73 < config 0.74; worst 0.63 >= 0.62;
    # qos/lcb 0.72 > default 0.70 but < config 0.74.
    steady, weak3, worst, qos, lcb = 0.84, 0.73, 0.63, 0.72, 0.72
    with pytest.raises(AssertionError, match="steady"):
        assert_medium_gate(steady, weak3, worst, qos,
                           qos_wilson_lcb=lcb, floors=custom,
                           qos_floor=cfg.marl.qos_acceptance_wilson_lcb_floor)
    # same values pass the default gate (floor 0.80/0.70/0.60, LCB 0.70)
    assert_medium_gate(steady, weak3, worst, qos, qos_wilson_lcb=lcb)


def test_semantic_independence_of_sources():
    """Editing the acceptance triple must NOT touch reward/comm floors."""
    cfg = get_default_config()
    before_reward = cfg.marl.coord_reward_steady_floor
    before_comm = cfg.marl.comm_qos_steady_min
    cfg.marl.qos_acceptance_floors = [0.60, 0.70, 0.95]
    assert cfg.marl.coord_reward_steady_floor == before_reward
    assert cfg.marl.comm_qos_steady_min == before_comm


def test_acceptance_floors_from_config_fails_loud_on_bad_value():
    """A supplied config with an out-of-range floor must raise (no silent
    fallback to defaults -- that would fake a gate verdict)."""
    cfg = get_default_config()
    cfg.marl.qos_acceptance_floors = [0.60, 0.70, 2.0]  # > 1
    from config.params import MasterConfig
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        # write a minimal yaml via the dataclass dump to exercise the loader
        import yaml
        from dataclasses import asdict
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump({"marl": {"qos_acceptance_floors": [0.60, 0.70, 2.0]}},
                           handle)
        with pytest.raises(RuntimeError, match="in \\(0,1\\)"):
            acceptance_floors_from_config(path)
    finally:
        os.remove(path)


def test_cli_config_drives_gate_from_csv():
    """End-to-end: assert_gate_from_csv with --config floors changes verdict.

    A CSV whose aggregates clear the DEFAULT floors but not tightened config
    floors must fail when --config is supplied and pass without it.
    """
    import csv
    import os
    import tempfile

    fd_csv, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd_csv)
    fd_cfg, cfg_path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd_cfg)
    try:
        # per-episode arrays that clear 0.80/0.70/0.60 but not 0.86/0.74/0.62
        steady = [0.84, 0.85]
        weak3 = [0.73, 0.74]
        worst = [0.63, 0.64]
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "eval_episode_seeds", "eval_episode_steady_P_D",
                "eval_episode_weak3_P_D", "eval_episode_worst_P_D"])
            writer.writeheader()
            writer.writerow({
                "eval_episode_seeds": str(list(range(len(steady)))),
                "eval_episode_steady_P_D": repr(steady),
                "eval_episode_weak3_P_D": repr(weak3),
                "eval_episode_worst_P_D": repr(worst),
            })
        with open(cfg_path, "w", encoding="utf-8") as handle:
            import yaml
            yaml.safe_dump({
                "marl": {"qos_acceptance_floors": [0.62, 0.74, 0.86]},
            }, handle)

        from tools.assert_gate_thresholds import main
        # without --config: default floors 0.60/0.70/0.80 clear -> exit 0
        assert main([csv_path]) == 0
        # with --config: worst 0.62 cleared, but weak3 0.74/steady 0.86 fail
        # (mean weak3 0.735 < 0.74, mean steady 0.845 < 0.86) -> exit 2
        assert main([csv_path, "--config", cfg_path]) == 2
    finally:
        os.remove(csv_path)
        os.remove(cfg_path)