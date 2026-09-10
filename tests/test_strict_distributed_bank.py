"""Statistical-contract tests for the blind-bank runner."""

from __future__ import annotations

import pytest

from config.params import load_config
from tools.run_strict_distributed_bank import (
    bind_run_spec,
    summarize,
    validate_formal_protocol_arguments,
    validate_resume_policy,
    validate_resume_identity,
    validate_worker_topology,
    wilson_lower_bound,
)


def _episode(seed: int, success: bool) -> dict:
    return {
        "seed": seed,
        "qos_success": success,
        "steady": 0.9,
        "weak3": 0.8,
        "worst": 0.7,
        "belief_position_rmse_m": 4.0,
        "movement_target_coverage": 0.95,
        "step_time_p95_ms": 90.0,
        "closed_loop_critical_path_p95_ms": 35.0,
        "delivery_rate": 1.0,
        "deadline_violation_rate": 0.0,
        "inter_uav_min_distance_m": 25.0,
        "inter_uav_swept_min_distance_m": 24.0,
        "preexecution_swept_min_distance_m": 24.0,
        "isac_max_power_budget_violation_w": 0.0,
        "minimum_battery_j": 1000.0,
        "energy_causality_violation_j": 0.0,
    }


def test_wilson_uses_episode_count_and_is_monotone():
    assert wilson_lower_bound(100, 100) > 0.95
    assert wilson_lower_bound(90, 100) > wilson_lower_bound(80, 100)
    assert wilson_lower_bound(8, 10) < 0.80


def test_bank_summary_does_not_count_frames_or_targets_as_samples():
    episodes = [_episode(seed, seed < 80) for seed in range(100)]
    result = summarize(episodes)

    assert result["completed_episodes"] == 100
    assert result["qos_successes"] == 80
    assert result["qos_rate"] == pytest.approx(0.8)
    assert result["qos_rate_wilson_lower_95_one_sided"] < 0.8
    assert not result["gates"]["qos_wilson_lower_ge_0_80"]
    assert result["worst"]["minimum"] == pytest.approx(0.7)


def test_bank_rejects_nested_episode_and_node_process_pools():
    cfg = load_config("config/exp_strict_distributed_k16q16_process4.yaml")
    with pytest.raises(ValueError, match="nested episode and node pools"):
        validate_worker_topology(cfg, workers=2)
    assert validate_worker_topology(cfg, workers=1)


def test_formal_bank_cannot_resume_mutable_episode_json():
    with pytest.raises(ValueError, match="formal execution cannot resume"):
        validate_resume_policy(resume=True, formal=True)
    validate_resume_policy(resume=True, formal=False)


def test_formal_protocol_arguments_are_frozen():
    with pytest.raises(ValueError, match="protocol arguments are frozen"):
        validate_formal_protocol_arguments(
            formal=True,
            split="selection",
            workers=2,
            tail_window=25,
            carrier_period=1,
        )
    validate_formal_protocol_arguments(
        formal=False,
        split="selection",
        workers=2,
        tail_window=25,
        carrier_period=1,
    )


def test_run_spec_binds_execution_mode_and_protocol_parameters():
    manifest = {
        "algorithm_version": "algorithm-v1",
        "git": {"commit": "a" * 40},
        "source_tree": {"sha256": "b" * 64},
        "config": {
            "path": "config/a.yaml",
            "source_sha256": "c" * 64,
            "effective_sha256": "d" * 64,
        },
        "seeds": {
            "library_path": "config/bank.json",
            "library_sha256": "e" * 64,
            "library_metadata": {"split": "test"},
            "values": [1, 2],
        },
        "runtime": {
            "packages": {"numpy": "2.5.0"},
            "thread_environment": {"OMP_NUM_THREADS": "1"},
        },
    }
    bind_run_spec(
        manifest,
        formal=True,
        split="test",
        workers=1,
        tail_window=50,
        carrier_period=3,
    )
    assert manifest["run_spec"]["execution_mode"] == "formal"
    assert manifest["run_spec"]["carrier_period"] == 3
    assert len(manifest["run_spec_sha256"]) == 64


def _resume_fixture():
    manifest = {
        "schema_version": "strict-distributed-run-manifest/v2",
        "algorithm_version": "algorithm-v1",
        "formal_result_eligible": True,
        "git": {"commit": "abc123", "dirty": False},
        "source_tree": {"sha256": "source"},
        "config": {"effective_sha256": "config"},
        "seeds": {"library_sha256": "bank"},
        "runtime": {
            "python_version": "python",
            "platform": "platform",
            "packages": {"numpy": "2.5.0"},
            "thread_environment": {"OMP_NUM_THREADS": "1"},
        },
    }
    previous = {
        "run_manifest": manifest,
        "seeds": [11, 12],
        "workers": 1,
        "tail_window": 50,
        "carrier_period": 3,
    }
    return previous, manifest


def test_resume_identity_accepts_only_exact_experiment_identity():
    previous, manifest = _resume_fixture()
    validate_resume_identity(
        previous,
        manifest,
        [11, 12],
        workers=1,
        tail_window=50,
        carrier_period=3,
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("run_manifest", "formal_result_eligible"), False, "formal-result"),
        (("run_manifest", "git", "commit"), "def456", "Git commit"),
        (("run_manifest", "git", "dirty"), True, "Git dirty state"),
        (("run_manifest", "source_tree", "sha256"), "changed", "source-tree"),
        (("run_manifest", "seeds", "library_sha256"), "changed", "seed-bank"),
        (("carrier_period",), 4, "carrier period"),
        (("tail_window",), 25, "tail window"),
    ],
)
def test_resume_identity_rejects_mixed_artifacts(path, value, message):
    import copy

    previous, manifest = _resume_fixture()
    changed = copy.deepcopy(previous)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match=message):
        validate_resume_identity(
            changed,
            manifest,
            [11, 12],
            workers=1,
            tail_window=50,
            carrier_period=3,
        )
