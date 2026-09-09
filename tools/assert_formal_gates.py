#!/usr/bin/env python
"""Batch Medium-gate assertion over the formal evaluation results.

Registry-driven re-assertion of every formal result cited in the docs, so
"Gate passed" is verifiable in one command instead of living only in
markdown tables.  Results whose seeds are still under quarantine are listed
but never asserted (they cannot be formal evidence until the banks are
regenerated -- see docs/EXPERIMENT_LOG.md).

Usage:
    python tools/assert_formal_gates.py                # current post-G2 only
    python tools/assert_formal_gates.py --evidence-epoch historical
    python tools/assert_formal_gates.py --evidence-epoch all --json-output x.json

Exit code 0 = every enforced result in the selected evidence epoch cleared its
gate, and any selected post-G2 epoch contains at least one provenance-valid
enforced PASS.  Exit code 2 means no acceptable enforced current evidence is
registered.  This is deliberate: pre-G2 CSVs remain auditable, but cannot make
the current post-G2 release gate appear green.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional

try:
    from tools.assert_gate_thresholds import (
        assert_gate_from_csv,
        read_episode_arrays,
    )
except ModuleNotFoundError as exc:  # direct ``python tools/...py`` execution
    if exc.name != "tools":
        raise
    from assert_gate_thresholds import assert_gate_from_csv, read_episode_arrays

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from uav_isac.utils.reproducibility import (
    FORMAL_RUNTIME_PACKAGES,
    _canonical_json,
    _is_source_path,
    _normalise_source_bytes,
    _strict_json_loads,
    scenario_fingerprint,
    source_tree_hash_at_commit,
)

RESULTS_ROOT = ROOT / "results"
FORMAL_EVIDENCE_REGISTRY = ROOT / "formal_evidence" / "registry.json"
FORMAL_EVIDENCE_REGISTRY_SCHEMA = "formal-evidence-registry/v1"
STRICT_RUN_MANIFEST_SCHEMA = "strict-distributed-run-manifest/v2"
CURRENT_EVIDENCE_EPOCH = "post_g2"
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_ALLOWED_CURRENT_ALGORITHMS = {
    "strict-distributed-owner-posterior-bistatic-v3",
    "strict-distributed-owner-posterior-bistatic-v4-process-parallel",
}

# These files govern discovery, packaging, and evidence presentation but are
# outside the numerical simulator/controller closure that produced a result.
# Changes elsewhere still invalidate current formal evidence. Keep this list
# exact rather than allowing whole source directories.
_EVIDENCE_NEUTRAL_RELEASE_PATHS = {
    ".gitignore",
    "tools/assert_formal_gates.py",
    "tools/audit_architecture_migration_v2.py",
    "tools/audit_data_lifecycle_v2.py",
    "tools/build_legacy_data_catalog_v2.py",
    "tools/summarize_actor_proximal_ablation.py",
    "tools/summarize_feasibility_first_confirmation.py",
    "scripts/run_mappo.py",
    "tools/run_strict_distributed_pilot.py",
    "tools/run_strict_distributed_bank.py",
    "uav_isac/adapters/filesystem_artifacts.py",
    "uav_isac/adapters/__init__.py",
    "uav_isac/adapters/legacy_process.py",
    "uav_isac/application/artifacts.py",
    "uav_isac/domain/__init__.py",
    "uav_isac/domain/artifacts.py",
    "uav_isac/governance/__init__.py",
    "uav_isac/governance/project_phase.py",
    "uav_isac/governance/entrypoint_inventory.py",
    "uav_isac/governance/data_inventory.py",
    "uav_isac/governance/research_programs.py",
    "uav_isac/governance/research_readiness.py",
    "uav_isac/governance/data/research_programs.yaml",
    "uav_isac/governance/data/project_phase.yaml",
    "uav_isac/interfaces/cli.py",
}


@dataclass(frozen=True)
class FormalResult:
    name: str
    csv: str
    qos_tol: float = 0.0
    quarantined: bool = False
    require_lcb: bool = False
    enforced: bool = True
    note: str = ""
    # Every result currently in this legacy registry predates the frozen
    # post-G2 continuous-DD identity.  A future confirmatory artifact must be
    # registered explicitly as ``post_g2`` after provenance validation.
    evidence_epoch: str = "pre_g2"
    # Current-epoch entries are content addressed.  ``run_manifest`` points to
    # the completed result envelope (status/run_manifest/seeds/episodes), not
    # merely to an extracted, unauthenticated manifest fragment.
    artifact_sha256: str = ""
    run_manifest: str = ""
    run_manifest_sha256: str = ""


HISTORICAL_FORMAL_RESULTS: List[FormalResult] = [
    FormalResult(
        name="4/4 frozen deployment (100 seeds, test split)",
        csv=("architecture_v2_structure_student_u2u_resolve_bw50k_adaptive_"
             "b4b8_failclosed_gate100/paired_eval.csv"),
        note="docs/CURRENT_SYSTEM_MODEL.md 4/4 正式版本"),
    FormalResult(
        name="4/4 anchor-preserving residual safety gate (10 seeds)",
        csv=("architecture_v2_structure_student_cardinality_residual46_"
             "bw50k_gate10/paired_eval.csv"),
    ),
    FormalResult(
        name="8/8 analytical stack e2e D0.95 (20 seeds)",
        csv=("architecture_v2_scale_k8q8_analytical_power_l0l1_movement/"
             "paired_eval.csv"),
        qos_tol=1e-6,
        note="0.60-floor float artifact (QoS 0.65 at tol=0), see D1_1A"),
    FormalResult(
        name="8/8 lexicographic L1 live (D1.1-A, 20 seeds)",
        csv="_d095_lex20/paired_eval.csv",
        note="live eval, same seeds/warm-start as D0.95 baseline"),
    FormalResult(
        name="8/8 lex + multi-candidate L3 live (D1.1-B, 20 seeds)",
        csv="_d095_lexcand20/paired_eval.csv",
        note="deployment candidate baseline (D1.1-B, 0.975)"),
    FormalResult(
        name="8/8 lex + multi-candidate L3 (D1.1-B, 20 seeds) -- LCB enforced",
        csv="_d095_lexcand20/paired_eval.csv",
        require_lcb=True,
        note="audit-recommended Wilson LCB standard"),
    FormalResult(
        name="6/6 cardinality residual (10 seeds) -- QUARANTINED",
        csv=("architecture_v2_scale_k6q6_structure_student_cardinality_"
             "residual46_gate10/paired_eval.csv"),
        quarantined=True,
        note=("test split 前 10 个种子含全部 5 个隔离种子；bank 回填前不得作为"
              "正式证据 (docs/EXPERIMENT_LOG.md)"),
    ),
    FormalResult(
        name="6/6 cross-scale student (10 seeds) -- QUARANTINED",
        csv=("architecture_v2_scale_k6q6_structure_student_adaptive_b4b8_"
             "gate10/paired_eval.csv"),
        quarantined=True,
        note="同 980_k6q6 test split 污染；不可作为正式证据",
    ),
    FormalResult(
        name="8/8 frozen deployment candidate D1.5 blind (100 seeds)",
        csv="_d1_5_blind100/paired_eval.csv",
        note=("100 全新 blind seed 认证：QoS 点估计 0.730 过门（advice 013），"
              "docs/EXPERIMENT_LOG.md D1.5"),
    ),
    FormalResult(
        name="8/8 frozen deployment candidate D1.5 blind (100 seeds) -- LCB enforced",
        csv="_d1_5_blind100/paired_eval.csv",
        require_lcb=True,
        enforced=False,
        note=("Wilson LCB 0.636 < 0.70：N=100 统计功效不足，须在论文中如实披露"
              "（点估计 + 置信区间口径）"),
    ),
    FormalResult(
        name="8/8 D1.9 bottleneck-lookahead blind (100 seeds)",
        csv="_d1_9_blind100/paired_eval.csv",
        note=("D1.9 H=40 前瞻 L3：QoS 0.950 / LCB 0.888 双双过门（advice 013），"
              "docs/EXPERIMENT_LOG.md D1.9"),
    ),
    FormalResult(
        name="8/8 D1.9 bottleneck-lookahead blind (100 seeds) -- LCB enforced",
        csv="_d1_9_blind100/paired_eval.csv",
        require_lcb=True,
        note=("QoS 0.950 + Wilson LCB 0.888 ≥ 0.70：历史 pre-fix 结果"
              "（--require-lcb 成立）"),
    ),
    FormalResult(
        name="8/8 D1.10 indep-env blind (100 seeds, independent sampling)",
        csv="_d1_10_blind100_indep/paired_eval.csv",
        note=("D1.10-A 独立 env 协议：每 seed 独立采样（消除共享实例 RNG "
              "漂移），QoS 0.940 / LCB 0.875 双双过门——统计正确的认证值"),
    ),
    FormalResult(
        name="8/8 D1.10 indep-env blind (100 seeds) -- LCB enforced",
        csv="_d1_10_blind100_indep/paired_eval.csv",
        require_lcb=True,
        note=("QoS 0.940 + Wilson LCB 0.875 ≥ 0.70：独立采样协议下历史 "
              "pre-fix 结果"),
    ),
    FormalResult(
        name="8/8 V3-C0 re-cert blind (100 seeds, same seeds as D1.10)",
        csv="_gate0_8x8_v3c0_indep100/paired_eval.csv",
        require_lcb=True,
        note=("Gate 0 (advice/016, 2026-08-18): V3-C0 χ_rep/energy 修复后同 "
              "D1.10 的 100 seed 重认证，QoS 0.910 / LCB 0.838 双过门——V3-C0 "
              "新 baseline（D1.10 保留为历史）"),
    ),
    FormalResult(
        name="6/6 V3-C0 re-cert blind (100 seeds) -- disclosed failure",
        csv="_gate0_6x6_v3c0_indep100/paired_eval.csv",
        require_lcb=True,
        enforced=False,
        note="QoS 0.590 / LCB 0.492；记录失败，不阻断后续正式结果。",
    ),
    FormalResult(
        name="6/6 Gate 1 multi-scale CE blind (100 seeds)",
        csv="_gate1_6x6_multiscale_ce_blind100/paired_eval.csv",
        note="QoS 点估计 0.780 通过；LCB 在下一信息行单独披露。",
    ),
    FormalResult(
        name="6/6 Gate 1 multi-scale CE blind -- LCB disclosed failure",
        csv="_gate1_6x6_multiscale_ce_blind100/paired_eval.csv",
        require_lcb=True,
        enforced=False,
        note="Wilson LCB 0.689，距 0.70 尚差 0.011。",
    ),
]


_REGISTRY_FIELDS = frozenset({
    "schema_version", "evidence_epoch", "entries",
})
_REGISTRY_ENTRY_FIELDS = frozenset({
    "name",
    "csv",
    "quarantined",
    "enforced",
    "note",
    "artifact_sha256",
    "run_manifest",
    "run_manifest_sha256",
})


def _reject_duplicate_json_keys(
    pairs: List[tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"JSON document contains duplicate key {key!r}")
        result[key] = value
    return result


def _require_exact_registry_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise ValueError(
            f"formal evidence registry {label} has " + "; ".join(details))


def _require_registry_string(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or (not allow_empty and not value.strip()):
        raise ValueError(
            f"formal evidence registry {label} must be a"
            + (" string" if allow_empty else " non-empty string"))
    return value


def _require_results_relative_path(value: object, label: str) -> str:
    text = _require_registry_string(value, label).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
        raise ValueError(
            f"formal evidence registry {label} must be results-relative")
    candidate = (RESULTS_ROOT / Path(text)).resolve()
    try:
        candidate.relative_to(RESULTS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"formal evidence registry {label} escapes results/") from exc
    return text


def _require_registry_sha256(value: object, label: str) -> str:
    text = _require_registry_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(
            f"formal evidence registry {label} must be a lowercase SHA-256")
    return text


def _require_git_bound_registry(path: Path) -> None:
    """Reject a non-empty registry unless it exactly matches the HEAD blob."""
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "formal evidence registry must live inside the repository") from exc
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "non-empty formal evidence registry must be Git-tracked and "
            "present in HEAD") from exc
    if _normalise_source_bytes(path.read_bytes()) != (
        _normalise_source_bytes(committed)
    ):
        raise ValueError(
            "non-empty formal evidence registry must exactly match its "
            "current HEAD blob")


def load_formal_evidence_registry(
    path: Path = FORMAL_EVIDENCE_REGISTRY,
    *,
    require_git_binding: bool = True,
) -> List[FormalResult]:
    """Load strictly typed post-G2 registrations from the standalone file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"formal evidence registry is unreadable: {path}") from exc
    try:
        document = json.loads(
            raw, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"formal evidence registry is invalid JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("formal evidence registry root must be an object")
    _require_exact_registry_fields(document, _REGISTRY_FIELDS, "root")
    if document["schema_version"] != FORMAL_EVIDENCE_REGISTRY_SCHEMA:
        raise ValueError(
            "formal evidence registry schema_version must be "
            f"{FORMAL_EVIDENCE_REGISTRY_SCHEMA!r}")
    if document["evidence_epoch"] != CURRENT_EVIDENCE_EPOCH:
        raise ValueError(
            "formal evidence registry evidence_epoch must be "
            f"{CURRENT_EVIDENCE_EPOCH!r}")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("formal evidence registry entries must be a list")

    entries: List[FormalResult] = []
    for index, raw_entry in enumerate(raw_entries):
        label = f"entries[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"formal evidence registry {label} must be an object")
        _require_exact_registry_fields(
            raw_entry, _REGISTRY_ENTRY_FIELDS, label)
        quarantined = raw_entry["quarantined"]
        enforced = raw_entry["enforced"]
        if type(quarantined) is not bool:
            raise ValueError(
                f"formal evidence registry {label}.quarantined must be boolean")
        if type(enforced) is not bool:
            raise ValueError(
                f"formal evidence registry {label}.enforced must be boolean")
        entries.append(FormalResult(
            name=_require_registry_string(raw_entry["name"], f"{label}.name"),
            csv=_require_results_relative_path(
                raw_entry["csv"], f"{label}.csv"),
            quarantined=quarantined,
            require_lcb=True,
            enforced=enforced,
            note=_require_registry_string(
                raw_entry["note"], f"{label}.note", allow_empty=True),
            evidence_epoch=CURRENT_EVIDENCE_EPOCH,
            artifact_sha256=_require_registry_sha256(
                raw_entry["artifact_sha256"], f"{label}.artifact_sha256"),
            run_manifest=_require_results_relative_path(
                raw_entry["run_manifest"], f"{label}.run_manifest"),
            run_manifest_sha256=_require_registry_sha256(
                raw_entry["run_manifest_sha256"],
                f"{label}.run_manifest_sha256",
            ),
        ))
    names = [entry.name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "formal evidence registry entry names must be unique: "
            + ", ".join(duplicates))
    if entries and require_git_binding:
        _require_git_bound_registry(path)
    return entries


# Historical rows remain source-compatible and auditable.  Only current,
# post-G2 registrations live in the independently committed registry, so
# adding a completed result cannot create a verifier/source-tree hash cycle.
FORMAL_RESULTS: List[FormalResult] = (
    HISTORICAL_FORMAL_RESULTS + load_formal_evidence_registry()
)


def _result_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # Current evidence is a compact, versioned artifact bundle. Historical
    # registrations remain relative to results/ for backward reproduction.
    if path.parts and path.parts[0] == "artifacts":
        resolved = (ROOT / path).resolve()
        artifact_root = (ROOT / "artifacts").resolve()
        if not resolved.is_relative_to(artifact_root):
            raise ValueError("formal artifact path escapes artifacts/")
        return resolved
    return RESULTS_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_source_path(path: Path) -> str:
    return hashlib.sha256(
        _normalise_source_bytes(path.read_bytes())).hexdigest()


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return completed.stdout.strip()


def _sha256_source_at_commit(commit: str, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"post-G2 bound file is absent from evidence commit: {path}") from exc
    return hashlib.sha256(_normalise_source_bytes(content)).hexdigest()


def _require_repo_relative_path(raw: object, label: str) -> Path:
    text = str(raw).strip().replace("\\", "/")
    if not text or re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
        raise ValueError(
            f"post-G2 {label} must be a repository-relative POSIX path")
    candidate = (ROOT / Path(text)).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"post-G2 {label} escapes the repository") from exc
    return candidate


def _validate_release_source_binding(
    commit: str,
    expected_hash: str,
    expected_count: int,
) -> str:
    """Bind evidence to real Git blobs and the current release source."""
    try:
        current_commit = _git_text("rev-parse", "HEAD")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, current_commit],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        evidence_hash, evidence_count = source_tree_hash_at_commit(ROOT, commit)
        release_hash, release_count = source_tree_hash_at_commit(
            ROOT, current_commit)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "post-G2 evidence Git commit is missing or not an ancestor of "
            "the current release") from exc
    if (evidence_hash, evidence_count) != (expected_hash, expected_count):
        raise ValueError(
            "post-G2 source-tree hash does not match the evidence Git commit")
    if (release_hash, release_count) != (expected_hash, expected_count):
        changed = subprocess.run(
            ["git", "diff", "--name-only", f"{commit}..{current_commit}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        ).stdout.splitlines()
        changed_source = {
            path.strip().replace("\\", "/")
            for path in changed
            if path.strip() and _is_source_path(path.strip().replace("\\", "/"))
        }
        execution_changes = sorted(
            changed_source - _EVIDENCE_NEUTRAL_RELEASE_PATHS)
        if execution_changes:
            raise ValueError(
                "post-G2 evidence is stale because execution-relevant source "
                "changed: " + ", ".join(execution_changes[:12]))
    return current_commit


def _require_sha256(value: object, label: str) -> str:
    normalized = str(value).strip()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(
            f"post-G2 evidence requires a 64-character lowercase SHA-256 "
            f"for {label}")
    return normalized


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"post-G2 evidence requires mapping {label}")
    return value


def _one_sided_wilson_lower(successes: int, total: int) -> float:
    """Match the strict-bank runner's 95% one-sided Wilson contract."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid QoS success count in post-G2 evidence")
    z = float(NormalDist().inv_cdf(0.95))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * (
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) ** 0.5
    return max(0.0, (centre - radius) / denominator)


def _require_post_g2_identity(snapshot: Mapping[str, Any]) -> None:
    """Validate immutable physical/protocol pins embedded in the manifest."""
    scenario = _require_mapping(snapshot.get("scenario"), "effective scenario")
    uav = _require_mapping(snapshot.get("uav"), "effective uav")
    otfs = _require_mapping(snapshot.get("otfs"), "effective otfs")
    detection = _require_mapping(
        snapshot.get("detection"), "effective detection")
    marl = _require_mapping(snapshot.get("marl"), "effective marl")
    required = {
        "scenario.sensing_energy_mode": (
            scenario.get("sensing_energy_mode"), "cpi_frame"),
        "detection.dd_gain_mode": (
            detection.get("dd_gain_mode"), "continuous"),
        "marl.tracking_enabled": (marl.get("tracking_enabled"), True),
        "marl.ground_communication_enabled": (
            marl.get("ground_communication_enabled"), False),
        "marl.joint_isac_power_enabled": (
            marl.get("joint_isac_power_enabled"), True),
        "marl.distributed_coordination_use_local_belief_targets": (
            marl.get("distributed_coordination_use_local_belief_targets"),
            True),
        "marl.distributed_no_truth_fail_closed": (
            marl.get("distributed_no_truth_fail_closed"), True),
        "marl.detection_fusion_mode": (
            marl.get("detection_fusion_mode"), "local_only"),
    }
    mismatches = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    numeric_pins = {
        "uav.P_isac_total": (uav.get("P_isac_total"), 1.0),
        "otfs.fc": (otfs.get("fc"), 2.8e10),
        "otfs.B": (otfs.get("B"), 1.0e6),
        "otfs.delta_f": (otfs.get("delta_f"), 1.5625e4),
        "otfs.M": (otfs.get("M"), 64),
        "otfs.N": (otfs.get("N"), 16),
        "otfs.n_cpi": (otfs.get("n_cpi"), 1),
        "detection.P_FA": (detection.get("P_FA"), 0.001),
        "detection.c_det": (detection.get("c_det"), 1.0),
        "detection.g_min": (detection.get("g_min"), 0.5),
    }
    for name, (actual, expected) in numeric_pins.items():
        try:
            matches = abs(float(actual) - float(expected)) <= (
                1.0e-12 * max(1.0, abs(float(expected))))
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches.append(
                f"{name}={actual!r}, expected {expected!r}")
    if mismatches:
        raise ValueError(
            "post-G2 effective config violates canonical identity: "
            + "; ".join(mismatches))


def _validate_post_g2_evidence(
    entry: FormalResult,
    artifact_path: Path,
) -> Dict[str, object]:
    """Validate one completed, content-addressed current-epoch artifact."""
    expected_artifact_hash = _require_sha256(
        entry.artifact_sha256, "artifact_sha256")
    if not artifact_path.is_file():
        raise OSError(f"post-G2 artifact does not exist: {artifact_path}")
    actual_artifact_hash = _sha256_file(artifact_path)
    if actual_artifact_hash != expected_artifact_hash:
        raise ValueError(
            "post-G2 artifact SHA-256 mismatch: "
            f"actual={actual_artifact_hash}, expected={expected_artifact_hash}")

    if not str(entry.run_manifest).strip():
        raise ValueError("post-G2 evidence requires a completed run_manifest")
    manifest_path = _result_path(entry.run_manifest)
    expected_manifest_hash = _require_sha256(
        entry.run_manifest_sha256, "run_manifest_sha256")
    if not manifest_path.is_file():
        raise OSError(
            f"post-G2 run manifest does not exist: {manifest_path}")
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "post-G2 run-manifest SHA-256 mismatch: "
            f"actual={actual_manifest_hash}, expected={expected_manifest_hash}")
    try:
        document = _strict_json_loads(
            manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"post-G2 run manifest is invalid JSON: {exc}") from exc
    envelope = _require_mapping(document, "completed result envelope")
    if envelope.get("status") != "FORMAL_COMPLETE":
        raise ValueError(
            "post-G2 result envelope status must be 'FORMAL_COMPLETE', got "
            f"{envelope.get('status')!r}")
    if envelope.get("execution_mode") != "formal":
        raise ValueError(
            "post-G2 result envelope must record execution_mode='formal'")

    manifest = _require_mapping(
        envelope.get("run_manifest"), "run_manifest")
    if manifest.get("schema_version") != STRICT_RUN_MANIFEST_SCHEMA:
        raise ValueError(
            "post-G2 evidence requires run-manifest schema "
            f"{STRICT_RUN_MANIFEST_SCHEMA!r}")
    if manifest.get("formal_result_eligible") is not True:
        raise ValueError(
            "post-G2 run manifest is not formal_result_eligible")
    if str(manifest.get("formal_result_ineligibility_reason", "")).strip():
        raise ValueError(
            "eligible post-G2 run manifest has an ineligibility reason")
    algorithm_version = str(manifest.get("algorithm_version", "")).strip()
    if algorithm_version not in _ALLOWED_CURRENT_ALGORITHMS:
        raise ValueError(
            "post-G2 evidence uses an unapproved algorithm_version: "
            f"{algorithm_version!r}")

    git = _require_mapping(manifest.get("git"), "run_manifest.git")
    if git.get("dirty") is not False:
        raise ValueError("post-G2 run manifest must record git.dirty=false")
    commit = str(git.get("commit", "")).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(
            "post-G2 run manifest requires a full 40-character git commit")
    status_hash = _require_sha256(
        git.get("status_sha256"), "git.status_sha256")
    if status_hash != hashlib.sha256(b"").hexdigest():
        raise ValueError(
            "post-G2 run manifest does not contain the clean-tree status hash")

    source_tree = _require_mapping(
        manifest.get("source_tree"), "run_manifest.source_tree")
    source_tree_hash = _require_sha256(
        source_tree.get("sha256"), "source_tree.sha256")
    file_count = source_tree.get("file_count")
    if (isinstance(file_count, bool) or not isinstance(file_count, int)
            or file_count < 1):
        raise ValueError(
            "post-G2 run manifest requires a positive source_tree.file_count")
    release_commit = _validate_release_source_binding(
        commit, source_tree_hash, file_count)

    config = _require_mapping(manifest.get("config"), "run_manifest.config")
    config_path = _require_repo_relative_path(
        config.get("path"), "config.path")
    if not config_path.is_file():
        raise ValueError("post-G2 config.path does not exist in this release")
    config_source_hash = _require_sha256(
        config.get("source_sha256"), "config.source_sha256")
    if _sha256_source_path(config_path) != config_source_hash:
        raise ValueError(
            "post-G2 config source hash does not match this release")
    if _sha256_source_at_commit(commit, config_path) != config_source_hash:
        raise ValueError(
            "post-G2 config source hash does not match the evidence commit")
    config_effective_hash = _require_sha256(
        config.get("effective_sha256"), "config.effective_sha256")
    effective_snapshot = _require_mapping(
        config.get("effective_snapshot"), "config.effective_snapshot")
    if hashlib.sha256(_canonical_json(effective_snapshot)).hexdigest() != (
        config_effective_hash
    ):
        raise ValueError(
            "post-G2 effective configuration hash disagrees with its snapshot")
    current_config = load_config(str(config_path))
    if hashlib.sha256(_canonical_json(asdict(current_config))).hexdigest() != (
        config_effective_hash
    ):
        raise ValueError(
            "post-G2 effective configuration differs from this release config")
    _require_post_g2_identity(effective_snapshot)

    seed_manifest = _require_mapping(
        manifest.get("seeds"), "run_manifest.seeds")
    bank_path = _require_repo_relative_path(
        seed_manifest.get("library_path"), "seeds.library_path")
    if not bank_path.is_file():
        raise ValueError(
            "post-G2 seed bank does not exist in this release")
    bank_hash = _require_sha256(
        seed_manifest.get("library_sha256"), "seeds.library_sha256")
    if _sha256_source_path(bank_path) != bank_hash:
        raise ValueError(
            "post-G2 seed-bank hash does not match this release")
    if _sha256_source_at_commit(commit, bank_path) != bank_hash:
        raise ValueError(
            "post-G2 seed-bank hash does not match the evidence commit")
    try:
        bank_document = _strict_json_loads(
            bank_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("post-G2 seed bank is not valid JSON") from exc
    bank = _require_mapping(bank_document, "seed bank")
    bank_metadata = _require_mapping(
        seed_manifest.get("library_metadata"),
        "seeds.library_metadata",
    )
    if bank_metadata.get("schema_version") != 2:
        raise ValueError("post-G2 evidence requires seed-bank schema_version 2")
    if bank_metadata.get("fingerprint_version") != "reset-distribution/v2":
        raise ValueError(
            "post-G2 evidence requires reset-distribution/v2 seed bank")
    if re.fullmatch(
        r"[0-9a-f]{16}",
        str(bank_metadata.get("scenario_fingerprint", "")),
    ) is None:
        raise ValueError(
            "post-G2 evidence requires a valid scenario fingerprint")
    if not str(bank_metadata.get("source_config", "")).strip():
        raise ValueError("post-G2 evidence requires seed-bank source_config")
    if bank_metadata.get("split") != "test":
        raise ValueError("post-G2 evidence requires the frozen test split")
    actual_bank_metadata = {
        "schema_version": bank.get("schema_version"),
        "fingerprint_version": bank.get("fingerprint_version"),
        "scenario_fingerprint": bank.get("scenario_fingerprint"),
        "source_config": bank.get("source_config"),
        "split": "test",
    }
    if dict(bank_metadata) != actual_bank_metadata:
        raise ValueError(
            "post-G2 seed-bank metadata does not match the bound bank")
    if scenario_fingerprint(
        current_config, str(bank.get("source_config", ""))
    ) != str(bank.get("scenario_fingerprint", "")):
        raise ValueError(
            "post-G2 scenario fingerprint does not match the release config")
    seed_values = seed_manifest.get("values")
    if not isinstance(seed_values, list) or len(seed_values) != 100:
        raise ValueError(
            "post-G2 evidence requires exactly 100 ordered seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int)
           for seed in seed_values):
        raise ValueError("post-G2 ordered seeds must all be integers")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("post-G2 ordered seeds must be unique")
    bank_splits = _require_mapping(bank.get("splits"), "seed bank splits")
    actual_test_seeds = bank_splits.get("test")
    if actual_test_seeds != seed_values:
        raise ValueError(
            "post-G2 ordered seeds do not exactly match the bound test split")
    excluded_seeds = set(bank.get("quarantined_excluded", []))
    excluded_seeds.update(bank.get("development_excluded", []))
    legacy_excluded = bank.get("excluded", {})
    if isinstance(legacy_excluded, Mapping):
        excluded_seeds.update(legacy_excluded.get("quarantined", []))
    if excluded_seeds.intersection(seed_values):
        raise ValueError(
            "post-G2 test split contains quarantined/development seeds")

    envelope_seeds = envelope.get("seeds")
    if envelope_seeds != seed_values:
        raise ValueError(
            "post-G2 envelope seed order does not match run_manifest.seeds")
    episodes = envelope.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 100:
        raise ValueError(
            "completed post-G2 envelope must contain exactly 100 episodes")
    episode_seeds = [
        episode.get("seed") if isinstance(episode, Mapping) else None
        for episode in episodes
    ]
    if episode_seeds != seed_values:
        raise ValueError(
            "post-G2 episode order does not match run_manifest.seeds")
    summary = _require_mapping(envelope.get("summary"), "result summary")
    if summary.get("completed_episodes") != 100:
        raise ValueError(
            "completed post-G2 summary must report 100 completed_episodes")

    runtime = _require_mapping(
        manifest.get("runtime"), "run_manifest.runtime")
    packages = _require_mapping(runtime.get("packages"), "runtime.packages")
    omitted_packages = [
        name for name in FORMAL_RUNTIME_PACKAGES if name not in packages
    ]
    if omitted_packages:
        raise ValueError(
            "post-G2 evidence omits required runtime packages: "
            + ", ".join(omitted_packages))
    missing_packages = [
        name for name, version in packages.items()
        if version in (None, "", "NOT_INSTALLED")
    ]
    if missing_packages:
        raise ValueError(
            "post-G2 evidence has missing runtime packages: "
            + ", ".join(sorted(missing_packages)))
    thread_environment = _require_mapping(
        runtime.get("thread_environment"),
        "runtime.thread_environment",
    )
    non_unit = {
        name: thread_environment.get(name)
        for name in _THREAD_VARIABLES
        if str(thread_environment.get(name)) != "1"
    }
    if non_unit:
        raise ValueError(
            "post-G2 evidence requires deterministic single-thread runtime: "
            f"{non_unit}")

    run_spec = _require_mapping(
        manifest.get("run_spec"), "run_manifest.run_spec")
    run_spec_hash = _require_sha256(
        manifest.get("run_spec_sha256"), "run_spec_sha256")
    if hashlib.sha256(_canonical_json(run_spec)).hexdigest() != run_spec_hash:
        raise ValueError("post-G2 run-spec hash disagrees with its payload")
    expected_spec_fields = {
        "schema_version": "strict-distributed-run-spec/v1",
        "execution_mode": "formal",
        "algorithm_version": algorithm_version,
        "git_commit": commit,
        "source_tree_sha256": source_tree_hash,
        "config_path": config.get("path"),
        "config_source_sha256": config_source_hash,
        "config_effective_sha256": config_effective_hash,
        "seed_bank_path": seed_manifest.get("library_path"),
        "seed_bank_sha256": bank_hash,
        "seed_bank_metadata": dict(bank_metadata),
        "seeds": seed_values,
        "seed_split": "test",
        "workers": 1,
        "tail_window": 50,
        "carrier_period": 3,
        "runtime_packages": dict(packages),
        "thread_environment": dict(thread_environment),
        "acceptance_thresholds": {
            "steady_min": 0.80,
            "weak3_min": 0.70,
            "worst_min": 0.60,
            "qos_wilson_lower_min": 0.80,
            "delivery_rate_min": 0.99,
            "deadline_violation_rate_max": 0.01,
            "closed_loop_p95_ms_max": 100.0,
        },
    }
    mismatched_spec = [
        name for name, expected in expected_spec_fields.items()
        if run_spec.get(name) != expected
    ]
    if mismatched_spec:
        raise ValueError(
            "post-G2 run spec violates the frozen protocol: "
            + ", ".join(mismatched_spec))
    for name, expected in (
        ("seed_split", "test"),
        ("workers", 1),
        ("tail_window", 50),
        ("carrier_period", 3),
    ):
        if envelope.get(name) != expected:
            raise ValueError(
                f"post-G2 envelope {name} must equal {expected!r}")

    arrays = read_episode_arrays(str(artifact_path))
    artifact_seeds = arrays.get("seeds")
    if artifact_seeds is None:
        raise ValueError(
            "post-G2 evidence CSV requires eval_episode_seeds")
    if artifact_seeds != seed_values:
        raise ValueError(
            "post-G2 evidence CSV seed order does not match run manifest")

    steady_values = [float(value) for value in arrays["steady"]]
    weak3_values = [float(value) for value in arrays["weak3"]]
    worst_values = [float(value) for value in arrays["worst"]]
    for label, values in (
        ("steady", steady_values),
        ("weak3", weak3_values),
        ("worst", worst_values),
    ):
        if any(
            not (value == value and abs(value) != float("inf"))
            or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                f"post-G2 {label} episode metrics must be finite in [0, 1]")
    qos_success = [
        steady >= 0.80 and weak3 >= 0.70 and worst >= 0.60
        for steady, weak3, worst in zip(
            steady_values, weak3_values, worst_values)
    ]
    episode_success = []
    delivery_values = []
    deadline_values = []
    closed_loop_values = []
    for index, episode in enumerate(episodes):
        record = _require_mapping(
            episode, f"episodes[{index}]")
        if type(record.get("qos_success")) is not bool:
            raise ValueError(
                f"post-G2 episode {index} lacks boolean qos_success")
        episode_success.append(bool(record["qos_success"]))
        for key, destination in (
            ("delivery_rate", delivery_values),
            ("deadline_violation_rate", deadline_values),
            ("closed_loop_critical_path_p95_ms", closed_loop_values),
        ):
            try:
                number = float(record[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"post-G2 episode {index} lacks numeric {key}") from exc
            if not (number == number and abs(number) != float("inf")):
                raise ValueError(
                    f"post-G2 episode {index} has non-finite {key}")
            if key in {"delivery_rate", "deadline_violation_rate"} and not (
                0.0 <= number <= 1.0
            ):
                raise ValueError(
                    f"post-G2 episode {index} has out-of-range {key}")
            if key == "closed_loop_critical_path_p95_ms" and number < 0.0:
                raise ValueError(
                    f"post-G2 episode {index} has negative {key}")
            destination.append(number)
    if episode_success != qos_success:
        raise ValueError(
            "post-G2 episode qos_success flags disagree with the frozen "
            "0.80/0.70/0.60 acceptance floors")

    successes = sum(qos_success)
    qos_rate = successes / len(qos_success)
    qos_lcb = _one_sided_wilson_lower(successes, len(qos_success))
    delivery_mean = sum(delivery_values) / len(delivery_values)
    deadline_mean = sum(deadline_values) / len(deadline_values)
    worst_closed_loop = max(closed_loop_values)
    strict_failures = []
    if qos_lcb < 0.80:
        strict_failures.append(f"QoS one-sided Wilson LCB {qos_lcb:.4f} < 0.80")
    if delivery_mean < 0.99:
        strict_failures.append(
            f"delivery rate {delivery_mean:.4f} < 0.99")
    if deadline_mean > 0.01:
        strict_failures.append(
            f"deadline violation rate {deadline_mean:.4f} > 0.01")
    if envelope.get("workers") != 1:
        strict_failures.append(
            "formal timing claim requires one outer worker")
    if worst_closed_loop > 100.0:
        strict_failures.append(
            f"per-seed closed-loop P95 {worst_closed_loop:.4f} ms > 100 ms")
    if strict_failures:
        raise ValueError(
            "post-G2 strict-bank gate failed: " + "; ".join(strict_failures))

    gates = _require_mapping(summary.get("gates"), "summary.gates")
    required_gates = (
        "qos_wilson_lower_ge_0_80",
        "delivery_rate_ge_0_99",
        "deadline_violation_rate_le_0_01",
        "every_seed_closed_loop_p95_le_100ms",
    )
    if any(gates.get(name) is not True for name in required_gates):
        raise ValueError(
            "post-G2 summary does not record every strict-bank gate as true")
    summary_expectations = {
        "qos_successes": successes,
        "qos_rate": qos_rate,
        "qos_rate_wilson_lower_95_one_sided": qos_lcb,
        "delivery_rate_mean": delivery_mean,
        "deadline_violation_rate_mean": deadline_mean,
        "timing_claim_eligible": True,
    }
    for key, expected in summary_expectations.items():
        actual = summary.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        else:
            try:
                matches = abs(float(actual) - float(expected)) <= 1.0e-12
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ValueError(
                f"post-G2 summary field {key} disagrees with episode records")

    return {
        "artifact_sha256": actual_artifact_hash,
        "run_manifest_sha256": actual_manifest_hash,
        "git_commit": commit,
        "release_commit": release_commit,
        "source_tree_sha256": source_tree_hash,
        "config_source_sha256": config_source_hash,
        "config_effective_sha256": config_effective_hash,
        "seed_bank_sha256": bank_hash,
        "provenance_validated": True,
        "episodes": len(seed_values),
        "steady": sum(steady_values) / len(steady_values),
        "weak3": sum(weak3_values) / len(weak3_values),
        "worst": sum(worst_values) / len(worst_values),
        "qos_feasible": qos_rate,
        "qos_wilson_lcb": qos_lcb,
    }


def assert_formal_gates(
    results: Optional[List[FormalResult]] = None,
    require_lcb: bool = False,
) -> Dict[str, Dict[str, object]]:
    results = results if results is not None else FORMAL_RESULTS
    names = [entry.name for entry in results]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "formal evidence names must be unique: " + ", ".join(duplicates))
    report: Dict[str, Dict[str, object]] = {}
    for entry in results:
        item: Dict[str, object] = {
            "name": entry.name,
            "quarantined": entry.quarantined,
            "evidence_epoch": entry.evidence_epoch,
            "enforced": entry.enforced,
            "note": entry.note,
        }
        path = _result_path(entry.csv)
        if entry.evidence_epoch == CURRENT_EVIDENCE_EPOCH:
            try:
                item.update(_validate_post_g2_evidence(entry, path))
            except (ValueError, OSError) as exc:
                # Provenance is not an optional performance gate.  A current
                # artifact with an invalid identity is always a hard failure,
                # even when its scientific result was intended as disclosure.
                item["status"] = "FAIL"
                item["error"] = str(exc).splitlines()[0]
                report[entry.name] = item
                continue
            item["status"] = (
                "QUARANTINED" if entry.quarantined else "PASS")
            report[entry.name] = item
            # Current evidence follows the strict-bank one-sided Wilson 0.80
            # contract validated above.  Never feed it through the historical
            # paired-eval 0.70/two-sided gate below.
            continue
        if entry.quarantined:
            item["status"] = "QUARANTINED"
            report[entry.name] = item
            continue
        try:
            aggregates = assert_gate_from_csv(
                str(path), qos_tol=entry.qos_tol,
                require_wilson_lcb=require_lcb or entry.require_lcb)
            item["status"] = "PASS"
            item.update(aggregates)
        except (AssertionError, ValueError, OSError) as exc:
            item["status"] = (
                "FAIL" if entry.enforced else "DISCLOSED_FAIL")
            item["error"] = str(exc).splitlines()[0]
        report[entry.name] = item
    return report


def render_table(report: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "| result | epoch | status | steady | weak3 | worst | QoS | LCB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, item in report.items():
        epoch = str(item.get("evidence_epoch", "unknown"))
        if item["status"] == "QUARANTINED":
            lines.append(
                f"| {name} | {epoch} | **QUARANTINED** | n/a | n/a | "
                "n/a | n/a | n/a |")
        elif item["status"] in {"FAIL", "DISCLOSED_FAIL"}:
            label = ("**FAIL**" if item["status"] == "FAIL"
                     else "DISCLOSED FAIL")
            lines.append(
                f"| {name} | {epoch} | {label} | n/a | n/a | n/a | n/a | n/a | "
                f"({item.get('error', '')[:60]})")
        else:
            lines.append(
                f"| {name} | {epoch} | PASS | {item['steady']:.4f} | "
                f"{item['weak3']:.4f} "
                f"| {item['worst']:.4f} | {item['qos_feasible']:.2f} "
                f"| {item['qos_wilson_lcb']:.3f} |")
    return "\n".join(lines)


def results_for_epoch(
    evidence_epoch: str,
    results: Optional[List[FormalResult]] = None,
) -> List[FormalResult]:
    """Return a registry view without conflating historical and live evidence."""
    registry = list(FORMAL_RESULTS if results is None else results)
    normalized = str(evidence_epoch).strip().lower()
    if normalized == "all":
        return registry
    if normalized == "current":
        return [
            item for item in registry
            if item.evidence_epoch == CURRENT_EVIDENCE_EPOCH
        ]
    if normalized == "historical":
        return [
            item for item in registry
            if item.evidence_epoch != CURRENT_EVIDENCE_EPOCH
        ]
    raise ValueError(f"unknown evidence epoch: {evidence_epoch!r}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-lcb", action="store_true",
                        help="also require the QoS Wilson LCB to clear 0.70")
    parser.add_argument(
        "--evidence-epoch",
        choices=("current", "historical", "all"),
        default="current",
        help=("current: post-G2 evidence only (default); historical: pre-G2 "
              "audit archive; all: combined diagnostic table"),
    )
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args(argv)

    selected = results_for_epoch(args.evidence_epoch)
    report = assert_formal_gates(
        results=selected,
        require_lcb=args.require_lcb,
    )
    print(render_table(report))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    if not selected:
        print(
            "NO CURRENT FORMAL EVIDENCE: no post-G2 confirmatory artifact is "
            "registered; historical results require --evidence-epoch "
            "historical.",
            file=sys.stderr,
        )
        return 2
    failed = [n for n, item in report.items() if item["status"] == "FAIL"]
    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        return 1
    selected_current = [
        item for item in selected
        if item.evidence_epoch == CURRENT_EVIDENCE_EPOCH
    ]
    if selected_current:
        enforced_current_passes = [
            item for item in report.values()
            if (item.get("evidence_epoch") == CURRENT_EVIDENCE_EPOCH
                and item.get("enforced") is True
                and item.get("status") == "PASS"
                and item.get("provenance_validated") is True)
        ]
        if not enforced_current_passes:
            print(
                "NO ENFORCED CURRENT PASS: post-G2 evidence contains no "
                "provenance-valid enforced PASS.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
