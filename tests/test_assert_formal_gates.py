"""Tests for tools/assert_formal_gates.py (2026-08-16)."""

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import tools.assert_formal_gates as formal_gate_module
from config.params import load_config
from tools.assert_formal_gates import (
    CURRENT_EVIDENCE_EPOCH,
    FORMAL_EVIDENCE_REGISTRY_SCHEMA,
    FORMAL_RESULTS,
    HISTORICAL_FORMAL_RESULTS,
    FormalResult,
    assert_formal_gates,
    load_formal_evidence_registry,
    main,
    render_table,
    results_for_epoch,
)
from uav_isac.utils.reproducibility import build_run_manifest
from tools.run_strict_distributed_bank import bind_run_spec


@pytest.fixture(autouse=True)
def _synthetic_release_source_binding(monkeypatch):
    """Current-evidence fixtures use real config/bank bytes in a dirty tree."""
    monkeypatch.setattr(
        formal_gate_module,
        "_validate_release_source_binding",
        lambda _commit, _source_hash, _file_count: "synthetic-release",
    )
    monkeypatch.setattr(
        formal_gate_module,
        "_sha256_source_at_commit",
        lambda _commit, path: formal_gate_module._sha256_source_path(path),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_envelope(
    entry: FormalResult,
    path: Path,
    mutate,
) -> FormalResult:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return replace(entry, run_manifest_sha256=_sha256(path))


def _write_current_evidence(
    tmp_path: Path,
    *,
    passing: bool = True,
    enforced: bool = True,
    quarantined: bool = False,
) -> tuple[FormalResult, Path]:
    cfg_path = Path("config/exp_strict_distributed_k16q16.yaml")
    cfg = load_config(str(cfg_path))
    bank_path = Path(cfg.marl.eval_seed_bank_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in bank["splits"]["test"]]
    value = 0.95 if passing else 0.10
    csv_path = tmp_path / "paired_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "eval_episode_steady_P_D",
            "eval_episode_weak3_P_D",
            "eval_episode_worst_P_D",
            "eval_episode_seeds",
        ))
        writer.writeheader()
        writer.writerow({
            "eval_episode_steady_P_D": repr([value] * 100),
            "eval_episode_weak3_P_D": repr([value] * 100),
            "eval_episode_worst_P_D": repr([value] * 100),
            "eval_episode_seeds": repr(seeds),
        })
    run_manifest = build_run_manifest(
        cfg,
        config_path=str(cfg_path),
        seeds=seeds,
        algorithm_version="strict-distributed-owner-posterior-bistatic-v3",
    )
    run_manifest["formal_result_eligible"] = True
    run_manifest["formal_result_ineligibility_reason"] = ""
    run_manifest["git"]["dirty"] = False
    run_manifest["git"]["status_sha256"] = hashlib.sha256(b"").hexdigest()
    run_manifest["runtime"]["thread_environment"] = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    bind_run_spec(
        run_manifest,
        formal=True,
        split="test",
        workers=1,
        tail_window=50,
        carrier_period=3,
    )
    successes = 100 if passing else 0
    qos_rate = successes / 100
    qos_lcb = formal_gate_module._one_sided_wilson_lower(successes, 100)
    envelope = {
        "status": "FORMAL_COMPLETE",
        "execution_mode": "formal",
        "workers": 1,
        "tail_window": 50,
        "carrier_period": 3,
        "seed_split": "test",
        "seeds": seeds,
        "run_manifest": run_manifest,
        "summary": {
            "completed_episodes": 100,
            "qos_successes": successes,
            "qos_rate": qos_rate,
            "qos_rate_wilson_lower_95_one_sided": qos_lcb,
            "delivery_rate_mean": 1.0,
            "deadline_violation_rate_mean": 0.0,
            "timing_claim_eligible": True,
            "gates": {
                "qos_wilson_lower_ge_0_80": passing,
                "delivery_rate_ge_0_99": True,
                "deadline_violation_rate_le_0_01": True,
                "every_seed_closed_loop_p95_le_100ms": True,
            },
        },
        "episodes": [{
            "seed": seed,
            "qos_success": passing,
            "delivery_rate": 1.0,
            "deadline_violation_rate": 0.0,
            "closed_loop_critical_path_p95_ms": 20.0,
        } for seed in seeds],
    }
    manifest_path = tmp_path / "completed.json"
    manifest_path.write_text(
        json.dumps(envelope, sort_keys=True), encoding="utf-8")
    return FormalResult(
        name="synthetic current",
        csv=str(csv_path),
        enforced=enforced,
        quarantined=quarantined,
        require_lcb=True,
        evidence_epoch=CURRENT_EVIDENCE_EPOCH,
        artifact_sha256=_sha256(csv_path),
        run_manifest=str(manifest_path),
        run_manifest_sha256=_sha256(manifest_path),
    ), manifest_path


def _historical_inputs_exist(entries: list[FormalResult]) -> bool:
    return all(
        entry.quarantined
        or (formal_gate_module.RESULTS_ROOT / entry.csv).is_file()
        for entry in entries
    )


def _require_historical_inputs(entries: list[FormalResult]) -> None:
    if not _historical_inputs_exist(entries):
        pytest.skip("ignored historical results/ artifacts are unavailable")


def test_registry_covers_documented_historical_results():
    names = [r.name for r in HISTORICAL_FORMAL_RESULTS]
    assert any("4/4 frozen deployment" in n for n in names)
    assert any("8/8 analytical stack" in n for n in names)
    # Quarantined 6/6 rows are present but flagged, never asserted.
    quarantined = [
        r for r in HISTORICAL_FORMAL_RESULTS if r.quarantined
    ]
    assert len(quarantined) == 2
    assert all("6/6" in r.name for r in quarantined)
    assert all(
        result.evidence_epoch != CURRENT_EVIDENCE_EPOCH
        for result in HISTORICAL_FORMAL_RESULTS
    )
    assert FORMAL_RESULTS == (
        HISTORICAL_FORMAL_RESULTS + results_for_epoch("current"))


def _current_registry_document() -> dict:
    return {
        "schema_version": FORMAL_EVIDENCE_REGISTRY_SCHEMA,
        "evidence_epoch": CURRENT_EVIDENCE_EPOCH,
        "entries": [{
            "name": "synthetic registered current evidence",
            "csv": "synthetic/paired_eval.csv",
            "quarantined": False,
            "enforced": True,
            "note": "test fixture",
            "artifact_sha256": "a" * 64,
            "run_manifest": "synthetic/completed.json",
            "run_manifest_sha256": "b" * 64,
        }],
    }


def test_empty_v1_current_registry_loads_without_claiming_evidence(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "schema_version": FORMAL_EVIDENCE_REGISTRY_SCHEMA,
        "evidence_epoch": CURRENT_EVIDENCE_EPOCH,
        "entries": [],
    }), encoding="utf-8")
    assert load_formal_evidence_registry(
        path, require_git_binding=False) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document.__setitem__("unexpected", True),
            "unknown fields: unexpected",
        ),
        (
            lambda document: document.__setitem__(
                "schema_version", "formal-evidence-registry/v0"),
            "schema_version",
        ),
        (
            lambda document: document["entries"][0].__setitem__(
                "enforced", 1),
            "enforced must be boolean",
        ),
        (
            lambda document: document["entries"][0].__setitem__(
                "unexpected", "injection"),
            "unknown fields: unexpected",
        ),
    ],
)
def test_current_registry_rejects_unknown_schema_and_types(
    tmp_path, mutation, message,
):
    document = _current_registry_document()
    mutation(document)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_formal_evidence_registry(path, require_git_binding=False)


def test_current_registry_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        '{"schema_version":"formal-evidence-registry/v1",'
        '"schema_version":"formal-evidence-registry/v1",'
        '"evidence_epoch":"post_g2","entries":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key 'schema_version'"):
        load_formal_evidence_registry(path, require_git_binding=False)


def test_nonempty_current_registry_requires_exact_head_blob(
    tmp_path, monkeypatch,
):
    repository = tmp_path / "repository"
    path = repository / "formal_evidence" / "registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(_current_registry_document()), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "formal-gate@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Formal Gate Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "add", "formal_evidence/registry.json"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "register evidence"],
        cwd=repository,
        check=True,
    )
    monkeypatch.setattr(formal_gate_module, "ROOT", repository)

    entries = load_formal_evidence_registry(path)
    assert len(entries) == 1
    assert entries[0].evidence_epoch == CURRENT_EVIDENCE_EPOCH
    assert entries[0].require_lcb is True

    changed = _current_registry_document()
    changed["entries"][0]["note"] = "uncommitted registry injection"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match its current HEAD blob"):
        load_formal_evidence_registry(path)


def test_cli_fails_closed_when_current_epoch_has_no_evidence(
    capsys, monkeypatch,
):
    monkeypatch.setattr(
        formal_gate_module,
        "FORMAL_RESULTS",
        list(HISTORICAL_FORMAL_RESULTS),
    )
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "NO CURRENT FORMAL EVIDENCE" in captured.err


def test_epoch_filter_keeps_historical_results_auditable():
    historical = results_for_epoch("historical")
    assert historical == HISTORICAL_FORMAL_RESULTS
    assert results_for_epoch("all") == FORMAL_RESULTS


def test_evidence_neutral_release_allowlist_excludes_scientific_runtime():
    neutral = formal_gate_module._EVIDENCE_NEUTRAL_RELEASE_PATHS

    assert "uav_isac/governance/research_programs.py" in neutral
    assert "uav_isac/domain/artifacts.py" in neutral
    assert "uav_isac/legacy/environment_core.py" not in neutral
    assert "uav_isac/agents/networks.py" not in neutral
    assert "uav_isac/physical/deflection.py" not in neutral
    assert "config/exp_strict_distributed_k16q16.yaml" not in neutral


def test_post_g2_label_alone_cannot_reclassify_a_historical_csv():
    historical = next(
        item for item in FORMAL_RESULTS if "D1.9 bottleneck" in item.name)
    relabelled = replace(
        historical,
        name="spoofed current",
        evidence_epoch=CURRENT_EVIDENCE_EPOCH,
    )
    report = assert_formal_gates([relabelled])
    assert report[relabelled.name]["status"] == "FAIL"
    assert "artifact_sha256" in report[relabelled.name]["error"]


def test_content_addressed_current_strict_bank_evidence_passes(tmp_path):
    entry, _manifest = _write_current_evidence(tmp_path)
    item = assert_formal_gates([entry])[entry.name]
    assert item["status"] == "PASS"
    assert item["provenance_validated"] is True
    assert item["qos_wilson_lcb"] >= 0.80


def test_current_evidence_uses_strict_one_sided_080_contract(tmp_path):
    entry, _manifest = _write_current_evidence(tmp_path, passing=False)
    item = assert_formal_gates([entry])[entry.name]
    assert item["status"] == "FAIL"
    assert "strict-bank gate failed" in item["error"]


def test_current_manifest_tampering_breaks_content_binding(tmp_path):
    entry, manifest = _write_current_evidence(tmp_path)
    manifest.write_text("{}", encoding="utf-8")
    item = assert_formal_gates([entry])[entry.name]
    assert item["status"] == "FAIL"
    assert "run-manifest SHA-256 mismatch" in item["error"]


def test_diagnostic_completion_cannot_be_registered_as_formal(tmp_path):
    entry, manifest = _write_current_evidence(tmp_path)
    entry = _rewrite_envelope(
        entry,
        manifest,
        lambda document: document.update({
            "status": "DIAGNOSTIC_ONLY",
            "execution_mode": "diagnostic",
        }),
    )
    item = assert_formal_gates([entry])[entry.name]
    assert item["status"] == "FAIL"
    assert "FORMAL_COMPLETE" in item["error"]


def test_current_metrics_reject_impossible_rates(tmp_path):
    entry, manifest = _write_current_evidence(tmp_path)
    entry = _rewrite_envelope(
        entry,
        manifest,
        lambda document: document["episodes"][0].update(
            {"delivery_rate": 100.0}),
    )
    item = assert_formal_gates([entry])[entry.name]
    assert item["status"] == "FAIL"
    assert "out-of-range delivery_rate" in item["error"]


def test_effective_snapshot_hash_is_recomputed(tmp_path):
    entry, manifest = _write_current_evidence(tmp_path)
    entry = _rewrite_envelope(
        entry,
        manifest,
        lambda document: document["run_manifest"]["config"][
            "effective_snapshot"]["scenario"].update({"K": 8}),
    )
    item = assert_formal_gates([entry])[entry.name]
    assert item["status"] == "FAIL"
    assert "effective configuration hash" in item["error"]


def test_duplicate_evidence_names_are_rejected_before_dict_overwrite(tmp_path):
    entry, _manifest = _write_current_evidence(tmp_path)
    with pytest.raises(ValueError, match="must be unique"):
        assert_formal_gates([entry, entry])


@pytest.mark.parametrize(
    "entry_kwargs",
    [
        {"enforced": False},
        {"quarantined": True},
    ],
)
def test_current_epoch_without_enforced_pass_exits_nonzero(
    tmp_path, monkeypatch, capsys, entry_kwargs,
):
    entry, _manifest = _write_current_evidence(tmp_path, **entry_kwargs)
    monkeypatch.setattr(formal_gate_module, "FORMAL_RESULTS", [entry])
    assert main([]) == 2
    assert "NO ENFORCED CURRENT PASS" in capsys.readouterr().err


def test_assert_formal_gates_marks_quarantined_without_asserting():
    _require_historical_inputs(FORMAL_RESULTS)
    report = assert_formal_gates()
    for name, item in report.items():
        if item.get("quarantined", False):
            assert item["status"] == "QUARANTINED"
        elif not item.get("enforced", True):
            # LCB-enforced rows are audit-standard checks; a FAIL is a valid,
            # honest disclosure of insufficient statistical power (e.g. D1.5
            # blind 100: LCB 0.636 < 0.70), so it must not crash the batch.
            assert item["status"] in ("PASS", "DISCLOSED_FAIL")
        else:
            assert item["status"] == "PASS", name
            assert item["steady"] > 0
            assert item["worst"] > 0


def test_custom_result_failure_is_reported(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    entry = FormalResult(name="synthetic", csv=str(bad))
    report = assert_formal_gates(results=[entry])
    assert report["synthetic"]["status"] == "FAIL"
    assert "lacks column" in report["synthetic"]["error"]


def test_render_table_lists_every_result():
    _require_historical_inputs(FORMAL_RESULTS)
    report = assert_formal_gates()
    table = render_table(report)
    for name in report:
        assert name in table
    assert "QUARANTINED" in table
    assert "PASS" in table


def test_d1_5_blind_registry_point_estimate_passes_lcb_disclosed():
    """D1.5 blind 100: point estimate passes the QoS gate; the LCB-enforced
    row is present and honestly reports FAIL (0.636 < 0.70), matching the
    documented statistical-power limitation (docs/KNOWN_ISSUES.md)."""
    point = [r for r in FORMAL_RESULTS
             if "D1.5 blind (100 seeds)" in r.name and not r.require_lcb]
    lcb = [r for r in FORMAL_RESULTS
           if "D1.5 blind (100 seeds)" in r.name and r.require_lcb]
    assert len(point) == 1 and len(lcb) == 1
    _require_historical_inputs(point + lcb)
    report = assert_formal_gates(results=point + lcb)
    assert report[point[0].name]["status"] == "PASS"
    assert report[point[0].name]["qos_feasible"] >= 0.70
    assert report[lcb[0].name]["status"] == "DISCLOSED_FAIL"
    assert report[lcb[0].name]["error"]  # non-empty failure detail


def test_d1_9_blind_registry_passes_point_and_lcb():
    """D1.9 bottleneck-lookahead blind 100: both the point-estimate and the
    LCB-enforced rows must PASS (QoS 0.950 / LCB 0.888 >= 0.70), closing the
    D1.5 statistical-power gap (docs/OPTIMIZATION_LOG.md D1.9)."""
    point = [r for r in FORMAL_RESULTS
             if "D1.9 bottleneck-lookahead blind (100 seeds)" in r.name
             and not r.require_lcb]
    lcb = [r for r in FORMAL_RESULTS
           if "D1.9 bottleneck-lookahead blind (100 seeds)" in r.name
           and r.require_lcb]
    assert len(point) == 1 and len(lcb) == 1
    _require_historical_inputs(point + lcb)
    report = assert_formal_gates(results=point + lcb)
    assert report[point[0].name]["status"] == "PASS"
    assert report[point[0].name]["qos_feasible"] >= 0.90
    assert report[lcb[0].name]["status"] == "PASS"
    assert report[lcb[0].name]["qos_wilson_lcb"] >= 0.70


def test_d1_10_indep_blind_registry_passes_point_and_lcb():
    """D1.10-A independent-env blind 100: both rows PASS under the
    statistically correct per-seed independent sampling (QoS 0.940 / LCB
    0.875 >= 0.70), confirming the certification is protocol-robust
    (docs/OPTIMIZATION_LOG.md D1.10)."""
    point = [r for r in FORMAL_RESULTS
             if "D1.10 indep-env blind (100 seeds" in r.name
             and not r.require_lcb]
    lcb = [r for r in FORMAL_RESULTS
           if "D1.10 indep-env blind (100 seeds" in r.name
           and r.require_lcb]
    assert len(point) == 1 and len(lcb) == 1
    _require_historical_inputs(point + lcb)
    report = assert_formal_gates(results=point + lcb)
    assert report[point[0].name]["status"] == "PASS"
    assert report[point[0].name]["qos_feasible"] >= 0.90
    assert report[lcb[0].name]["status"] == "PASS"
    assert report[lcb[0].name]["qos_wilson_lcb"] >= 0.70
