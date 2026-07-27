import json

import numpy as np

from config.params import load_config
from tools.seed_stratification import (
    build_disjoint_splits,
    geometry_difficulty,
)
from uav_isac.agents.trainer import (
    PrioritizedGeometrySeedSampler,
    build_comm_qos_checkpoint_key,
    compute_robust_checkpoint_statistics,
    load_stratified_seed_split,
    load_training_seed_pool,
)


def test_geometry_difficulty_includes_second_endpoint_and_matching():
    metrics = geometry_difficulty(
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[0.0, 0.0], [20.0, 0.0]]),
    )
    assert metrics["worst_nearest_m"] == 10.0
    assert metrics["worst_second_nearest_m"] == 20.0
    assert metrics["bottleneck_matching_m"] == 10.0
    assert metrics["difficulty_score_m"] == 15.0


def test_tiered_splits_are_disjoint_and_follow_one_two_one_ratio():
    tiers = {
        "easy": list(range(0, 30)),
        "medium": list(range(100, 160)),
        "hard": list(range(200, 230)),
    }
    splits = build_disjoint_splits(
        tiers, {"selection": 20, "test": 40},
        np.random.default_rng(7))
    assert len(splits["selection"]) == 20
    assert len(splits["test"]) == 40
    assert set(splits["selection"]).isdisjoint(splits["test"])
    for name, total in (("selection", 20), ("test", 40)):
        values = splits[name]
        assert sum(seed < 100 for seed in values) == total // 4
        assert sum(100 <= seed < 200 for seed in values) == total // 2
        assert sum(seed >= 200 for seed in values) == total // 4


def test_bootstrap_lcb_and_cvar_penalize_unstable_worst():
    stable = np.full(20, 0.5)
    volatile = np.array([0.1] * 10 + [0.9] * 10)
    steady = np.full(20, 0.8)
    weak3 = np.full(20, 0.7)
    targets = np.array([0.8, 0.7, 0.4])
    stable_stats = compute_robust_checkpoint_statistics(
        steady, weak3, stable, targets, bootstrap_samples=4000)
    volatile_stats = compute_robust_checkpoint_statistics(
        steady, weak3, volatile, targets, bootstrap_samples=4000)
    assert stable_stats["eval_worst_lcb"] > volatile_stats["eval_worst_lcb"]
    assert stable_stats["eval_worst_cvar"] > volatile_stats["eval_worst_cvar"]
    assert stable_stats["eval_worst_std"] < volatile_stats["eval_worst_std"]
    assert stable_stats["eval_qos_feasible_wilson_lcb"] > 0.0
    assert (stable_stats["eval_qos_feasible_wilson_lcb"]
            > volatile_stats["eval_qos_feasible_wilson_lcb"])


def test_robust_checkpoint_key_prefers_lcb_over_lucky_mean():
    targets = np.array([0.8, 0.7, 0.6])
    lucky = build_comm_qos_checkpoint_key(
        np.array([0.7, 0.6, 0.50]), targets, 1.0,
        feasibility_lcb=0.0, worst_lcb=0.20,
        worst_cvar=0.10, trimmed_worst=0.55)
    stable = build_comm_qos_checkpoint_key(
        np.array([0.7, 0.6, 0.45]), targets, 500.0,
        feasibility_lcb=0.0, worst_lcb=0.35,
        worst_cvar=0.32, trimmed_worst=0.48)
    assert stable > lucky


def test_versioned_seed_bank_is_disjoint_and_loaded_by_canonical_config():
    path = "config/stratified_seeds_800_q4.json"
    with open(path, "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    selection = load_stratified_seed_split(path, "selection")
    confirmation = load_stratified_seed_split(path, "confirmation")
    test = load_stratified_seed_split(path, "test")
    assert len(selection) == 20
    assert len(confirmation) == 60
    assert len(test) == 100
    assert set(selection).isdisjoint(confirmation)
    assert set(selection).isdisjoint(test)
    assert set(confirmation).isdisjoint(test)
    assert bank["schema_version"] == 1

    cfg = load_config("config/exp_800_q4_u2u_target_motion.yaml")
    assert cfg.marl.eval_seed_bank_path == path
    assert cfg.marl.eval_seed_split == "selection"
    assert cfg.marl.eval_episodes == 20
    assert cfg.marl.checkpoint_confirmation_enabled is True
    assert cfg.marl.checkpoint_confirmation_split == "confirmation"


def test_training_seed_pool_excludes_all_evaluation_splits():
    path = "config/stratified_seeds_800_q4.json"
    with open(path, "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    seeds, difficulties = load_training_seed_pool(path)
    reserved = {
        int(seed)
        for split in bank["splits"].values()
        for seed in split
    }
    assert seeds
    assert set(seeds).isdisjoint(reserved)
    assert len(seeds) == len(difficulties)
    assert np.all(np.diff(difficulties) >= 0.0)


def test_prioritized_geometry_sampler_expands_curriculum_and_tracks_deficit():
    sampler = PrioritizedGeometrySeedSampler(
        [1, 2, 3, 4], np.array([100.0, 200.0, 300.0, 400.0]),
        worst_floor=0.6, min_curriculum_fraction=0.5,
        curriculum_frames=100)
    assert sampler.eligible_count(0) == 2
    assert sampler.eligible_count(100) == 4

    before = sampler.priorities.copy()
    sampler.update(1, episode_worst=0.0)
    sampler.update(2, episode_worst=0.8)
    assert sampler.priorities[0] > sampler.priorities[1]
    assert sampler.priorities[0] >= before[0]


def test_bottleneck_method_config_is_explicit_and_backward_compatible():
    cfg = load_config("config/exp_800_q4_u2u_brh_gmappo.yaml")
    assert cfg.marl.capacity_matching_enabled is True
    assert cfg.marl.capacity_matching_row_capacity == 2
    assert cfg.marl.capacity_matching_column_capacity == 2
    assert cfg.marl.advantage_mode == "bottleneck_risk"
    assert cfg.marl.training_seed_replay_enabled is True

    baseline = load_config("config/exp_800_q4_u2u_sparse_claim_top2.yaml")
    assert baseline.marl.capacity_matching_enabled is False
    assert baseline.marl.training_seed_replay_enabled is False
