import copy
import hashlib
import json

from config.params import load_config
import pytest
import uav_isac.utils.reproducibility as reproducibility

from uav_isac.utils.reproducibility import (
    SCENARIO_FINGERPRINT_VERSION,
    _source_tree_hash,
    build_run_manifest,
    scenario_fingerprint,
    validate_formal_run,
)


def test_run_manifest_binds_effective_config_code_seed_bank_and_runtime():
    cfg = load_config("config/exp_strict_distributed_no_truth_pilot.yaml")
    manifest = build_run_manifest(
        cfg,
        config_path="config/exp_strict_distributed_no_truth_pilot.yaml",
        seeds=[7, 14],
        algorithm_version="test-v1",
    )

    assert manifest["schema_version"].endswith("/v2")
    assert len(manifest["git"]["commit"]) == 40
    assert len(manifest["source_tree"]["sha256"]) == 64
    assert manifest["source_tree"]["file_count"] > 0
    assert len(manifest["config"]["effective_sha256"]) == 64
    assert manifest["seeds"]["values"] == [7, 14]
    assert len(manifest["seeds"]["library_sha256"]) == 64
    assert manifest["runtime"]["python_executable"]
    assert manifest["formal_result_eligible"] is (not manifest["git"]["dirty"])


def test_source_tree_hash_binds_entrypoints_and_dependency_contract(tmp_path):
    for directory in ("config", "scripts", "tools", "uav_isac"):
        (tmp_path / directory).mkdir()
    (tmp_path / "uav_isac" / "model.py").write_text(
        "VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "run.py").write_text(
        "print('run')\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(
        "numpy>=1\n", encoding="utf-8")
    (tmp_path / "constraints-ci.txt").write_text(
        "numpy==2\n", encoding="utf-8")

    initial, count = _source_tree_hash(tmp_path)
    assert count == 4

    (tmp_path / "constraints-ci.txt").write_text(
        "numpy==3\n", encoding="utf-8")
    dependency_changed, _ = _source_tree_hash(tmp_path)
    assert dependency_changed != initial

    (tmp_path / "scripts" / "run.py").write_text(
        "print('changed')\n", encoding="utf-8")
    entrypoint_changed, _ = _source_tree_hash(tmp_path)
    assert entrypoint_changed != dependency_changed

    # The verifier is executable release logic and must be source-bound.  The
    # post-run registry is deliberately outside the experiment source scope so
    # registering a completed artifact does not create a hash cycle.
    (tmp_path / "tools" / "assert_formal_gates.py").write_text(
        "VERIFIER_VERSION = 1\n", encoding="utf-8")
    verifier_bound, verifier_count = _source_tree_hash(tmp_path)
    assert verifier_count == count + 1
    assert verifier_bound != entrypoint_changed

    (tmp_path / "formal_evidence").mkdir()
    (tmp_path / "formal_evidence" / "registry.json").write_text(
        '{"entries": []}\n', encoding="utf-8")
    registry_changed, registry_count = _source_tree_hash(tmp_path)
    assert registry_changed == verifier_bound
    assert registry_count == verifier_count


def test_formal_protocol_fails_closed_on_dirty_workspace():
    cfg = load_config("config/exp_strict_distributed_no_truth_pilot.yaml")
    manifest = build_run_manifest(
        cfg,
        config_path="config/exp_strict_distributed_no_truth_pilot.yaml",
        seeds=list(range(100)),
        algorithm_version="test-v1",
    )
    manifest["formal_result_eligible"] = False
    with pytest.raises(RuntimeError, match="workspace is dirty"):
        validate_formal_run(cfg, list(range(100)), manifest)


def test_k16_seed_bank_fingerprint_rejects_k8_geometry_identity():
    with open(
        "config/stratified_seeds_1130_k16q16_blind.json",
        encoding="utf-8",
    ) as handle:
        bank = json.load(handle)
    cfg16 = load_config("config/exp_strict_distributed_k16q16.yaml")
    cfg8 = load_config("config/exp_strict_distributed_no_truth_pilot.yaml")

    assert bank["fingerprint_version"] == SCENARIO_FINGERPRINT_VERSION
    assert bank["schema_version"] == 2
    assert scenario_fingerprint(
        cfg16, bank["source_config"]) == bank["scenario_fingerprint"]
    assert scenario_fingerprint(
        cfg8, bank["source_config"]) != bank["scenario_fingerprint"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: setattr(cfg.scenario, "height", cfg.scenario.height + 1),
        lambda cfg: setattr(cfg.uav, "d_safe", cfg.uav.d_safe + 1),
        lambda cfg: setattr(cfg.target, "speed_range", (0.0, 7.0)),
        lambda cfg: setattr(cfg.target, "motion_model", "CA"),
        lambda cfg: setattr(cfg.target, "sigma_a", cfg.target.sigma_a + 0.1),
        lambda cfg: setattr(cfg.marl, "tracking_enabled", False),
    ],
)
def test_scenario_fingerprint_binds_reset_distribution(mutation):
    cfg = load_config("config/exp_strict_distributed_k16q16.yaml")
    changed = copy.deepcopy(cfg)
    mutation(changed)
    identity = "config/exp_strict_distributed_k16q16.yaml"
    assert scenario_fingerprint(changed, identity) != scenario_fingerprint(
        cfg, identity)


def test_k8_legacy_bank_is_explicitly_ineligible_for_formal_run():
    cfg = load_config("config/exp_strict_distributed_no_truth_pilot.yaml")
    with open(cfg.marl.eval_seed_bank_path, encoding="utf-8") as handle:
        bank = json.load(handle)
    seeds = [int(seed) for seed in bank["splits"]["test"]]
    manifest = build_run_manifest(
        cfg,
        config_path="config/exp_strict_distributed_no_truth_pilot.yaml",
        seeds=seeds,
        algorithm_version="test-v1",
    )
    manifest["formal_result_eligible"] = True
    manifest["runtime"]["thread_environment"] = {
        name: "1" for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }
    assert bank.get("fingerprint_version") != SCENARIO_FINGERPRINT_VERSION
    legacy_cfg = load_config(bank["source_config"])
    assert scenario_fingerprint(
        legacy_cfg, bank["source_config"]
    ) != scenario_fingerprint(cfg, bank["source_config"])
    with pytest.raises(
        RuntimeError,
        match="seed-bank schema_version 2|reset-distribution/v2",
    ):
        validate_formal_run(cfg, seeds, manifest)


def test_k16_v2_bank_is_formally_compatible_with_strict_config(monkeypatch):
    cfg = load_config("config/exp_strict_distributed_k16q16.yaml")
    with open(cfg.marl.eval_seed_bank_path, encoding="utf-8") as handle:
        bank = json.load(handle)
    seeds = [int(seed) for seed in bank["splits"]["test"]]
    manifest = build_run_manifest(
        cfg,
        config_path="config/exp_strict_distributed_k16q16.yaml",
        seeds=seeds,
        algorithm_version="test-v1",
    )
    manifest["formal_result_eligible"] = True
    manifest["git"]["dirty"] = False
    manifest["git"]["status_sha256"] = hashlib.sha256(b"").hexdigest()
    manifest["runtime"]["thread_environment"] = {
        name: "1" for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }

    real_git = reproducibility._git

    def clean_git(root, *args):
        if args[:2] == ("status", "--porcelain=v1"):
            return ""
        if args[:2] == ("rev-parse", "HEAD"):
            return manifest["git"]["commit"]
        if args and args[0] == "ls-files":
            return str(args[-1])
        return real_git(root, *args)

    monkeypatch.setattr(reproducibility, "_git", clean_git)
    monkeypatch.setattr(
        reproducibility,
        "source_tree_hash_at_commit",
        lambda _root, _commit: (
            manifest["source_tree"]["sha256"],
            manifest["source_tree"]["file_count"],
        ),
    )

    validate_formal_run(cfg, seeds, manifest)

    tampered = copy.deepcopy(manifest)
    tampered["config"]["effective_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="effective configuration hash"):
        validate_formal_run(cfg, seeds, tampered)

    tampered = copy.deepcopy(manifest)
    tampered["source_tree"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source-tree hash"):
        validate_formal_run(cfg, seeds, tampered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda bank: bank.__setitem__("schema_version", 1), "schema_version 2"),
        (lambda bank: bank["splits"]["selection"].append(
            bank["splits"]["test"][0]), "overlaps another split"),
        (lambda bank: bank["quarantined_excluded"].append(
            bank["splits"]["test"][0]), "quarantined/development"),
    ],
)
def test_formal_protocol_rejects_seed_bank_integrity_defects(
    tmp_path, mutation, message,
):
    cfg = load_config("config/exp_strict_distributed_k16q16.yaml")
    with open(cfg.marl.eval_seed_bank_path, encoding="utf-8") as handle:
        bank = json.load(handle)
    seeds = [int(seed) for seed in bank["splits"]["test"]]
    mutation(bank)
    candidate = tmp_path / "candidate-bank.json"
    candidate.write_text(json.dumps(bank), encoding="utf-8")
    manifest = build_run_manifest(
        cfg,
        config_path="config/exp_strict_distributed_k16q16.yaml",
        seeds=seeds,
        algorithm_version="test-v1",
    )
    manifest["formal_result_eligible"] = True
    manifest["seeds"]["library_path"] = str(candidate)
    manifest["seeds"]["library_sha256"] = hashlib.sha256(
        candidate.read_bytes()).hexdigest()
    manifest["runtime"]["thread_environment"] = {
        name: "1" for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }
    with pytest.raises(RuntimeError, match=message):
        validate_formal_run(cfg, seeds, manifest)
