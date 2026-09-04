#!/usr/bin/env python
"""C0 gate: verify the unique post-G2 system identity (audit advice/001.md).

The audit found the repo had three potentially different systems: the code
dataclass defaults, the YAML default, and the formal experiment runner.  This
checker closes that gap by asserting the frozen ``config/system_manifest.yaml``
identity against (a) the code-level parameter defaults and (b) the published
docs pointer, and by reporting the pinned physics contract:

- scope: U2U-only distributed ISAC (``ground=false``),
  strict no-ground fusion (``detection_fusion_mode`` in
  {local_only, u2u_distributed}), joint RF power ``joint_isac_power_enabled``,
  distributed local-belief coordination (no simulator truth).
- OTFS numerology and one-CPI sensing clock ``T_sense = n_cpi*N*T_sym`` with
  ``sensing_energy_mode == cpi_frame``.
- detector convention ``real_gaussian_shift`` (c_det=1, P_FA), continuous DD
  model ``dd_gain_mode == continuous`` (``I_support * |A|^2``, not the binary
  g_dd gate).
- seed scheme (blind 100-seed bank, eval/final on test split).

Usage:
    python tools/check_system_identity.py [--manifest config/system_manifest.yaml]
                                          [--commit <sha>] [--strict]

Exit code 0 = identity consistent; 1 = mismatch.  Compatibility mode reports
legacy seed-bank defects as warnings; ``--strict`` makes every formal bank
schema/fingerprint/scenario defect fatal.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Tuple

import yaml

# Direct ``python tools/check_system_identity.py`` execution puts ``tools/``
# rather than the repository root on sys.path.  Bootstrap the documented entry
# point before importing the first-party config package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.params import (
    MasterConfig,
    get_default_config,
    load_config,
    load_unique_yaml,
)
from uav_isac.utils.reproducibility import (
    SCENARIO_FINGERPRINT_VERSION,
    _strict_json_loads,
    scenario_fingerprint,
)

MANIFEST_DEFAULT = os.path.join(
    ROOT, "config", "system_manifest.yaml")

DOCS_PATH = os.path.join(
    ROOT, "docs", "CURRENT_SYSTEM_MODEL.md")

_FORMAL_WARNING_PREFIX = "FORMAL WARNING: "
_EXPECTED_SEED_BANK_SCHEMA_VERSION = 2


def _workspace_path(value: str) -> Path:
    """Resolve repository-relative metadata paths on Windows and POSIX."""
    portable = str(value).replace("\\", os.sep).replace("/", os.sep)
    path = Path(portable)
    if not path.is_absolute():
        path = Path(ROOT) / path
    return path.resolve()


def _scenario_identity(cfg: Any) -> Dict[str, Any]:
    """Return the human-readable fields covered by the v2 fingerprint.

    ``scenario_fingerprint`` remains the authoritative digest.  Keeping this
    projection alongside it makes a failure actionable: the gate can name a
    K/Q, region, or dynamics mismatch instead of reporting only two hashes.
    """
    return {
        "K": int(cfg.scenario.K),
        "Q": int(cfg.scenario.Q),
        "region_size": tuple(float(x) for x in cfg.scenario.region_size),
        "height": float(cfg.scenario.height),
        "T": int(cfg.scenario.T),
        "dt": float(cfg.scenario.dt),
        "uav.v_max": float(cfg.uav.v_max),
        "uav.d_safe": float(cfg.uav.d_safe),
        "tracking_enabled": bool(cfg.marl.tracking_enabled),
        "target.motion_model": str(cfg.target.motion_model),
        "target.speed_range": tuple(
            float(x) for x in cfg.target.speed_range),
        "target.sigma_a": float(cfg.target.sigma_a),
        "target.ct_turn_rate": float(cfg.target.ct_turn_rate),
    }


def _config_extends_chain(path: str) -> List[Path]:
    """Return the config inheritance chain, starting at ``path``."""
    current = Path(path)
    if not current.is_absolute():
        current = (Path.cwd() / current).resolve()
    else:
        current = current.resolve()
    chain: List[Path] = []
    seen: set[Path] = set()
    while True:
        if current in seen:
            raise ValueError(f"cyclic config inheritance involving {current}")
        seen.add(current)
        chain.append(current)
        raw = load_raw_manifest(str(current))
        parent = raw.get("extends")
        if parent is None:
            return chain
        if not isinstance(parent, str):
            raise ValueError("config extends must be a relative or absolute path")
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = current.parent / parent_path
        current = parent_path.resolve()


def _collect_seed_bank_checks(
    manifest: Any,
    *,
    strict: bool,
) -> Tuple[List[str], List[str], List[str]]:
    """Validate the configured evaluation bank against the effective config.

    Legacy v1 banks remain usable for non-formal historical reproduction.  In
    compatibility mode their defects are warnings; strict mode promotes the
    same findings to failures.  Missing or unreadable files always fail because
    no run can reproduce a bank that is not present.
    """
    failures: List[str] = []
    passes: List[str] = []
    warnings: List[str] = []

    def formal_issue(message: str) -> None:
        (failures if strict else warnings).append(message)

    configured = str(getattr(
        getattr(manifest, "marl", object()), "eval_seed_bank_path", ""
    )).strip()
    if not configured:
        failures.append("eval_seed_bank_path is empty")
        return failures, passes, warnings

    bank_path = _workspace_path(configured)
    if not bank_path.is_file():
        failures.append(
            f"configured seed bank does not exist: {configured!r} "
            f"(resolved to {bank_path})")
        return failures, passes, warnings
    passes.append(f"configured seed bank exists: {configured!r}")

    try:
        loaded = _strict_json_loads(bank_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failures.append(f"configured seed bank is unreadable JSON: {exc}")
        return failures, passes, warnings
    if not isinstance(loaded, Mapping):
        failures.append("configured seed bank root must be a JSON object")
        return failures, passes, warnings
    bank = dict(loaded)

    schema_version = bank.get("schema_version")
    if schema_version != _EXPECTED_SEED_BANK_SCHEMA_VERSION:
        formal_issue(
            "seed bank schema_version is "
            f"{schema_version!r}, expected "
            f"{_EXPECTED_SEED_BANK_SCHEMA_VERSION}")
    else:
        passes.append(
            f"seed bank schema_version: {_EXPECTED_SEED_BANK_SCHEMA_VERSION}")

    fingerprint_version = str(bank.get("fingerprint_version", ""))
    if fingerprint_version != SCENARIO_FINGERPRINT_VERSION:
        formal_issue(
            "seed bank fingerprint_version is "
            f"{fingerprint_version!r}, expected "
            f"{SCENARIO_FINGERPRINT_VERSION!r}")
    else:
        passes.append(
            f"seed bank fingerprint_version: {SCENARIO_FINGERPRINT_VERSION}")

    split_names = {
        str(getattr(manifest.marl, "eval_seed_split", "test")),
        str(getattr(manifest.marl, "final_eval_seed_split", "test")),
    }
    splits = bank.get("splits")
    if not isinstance(splits, Mapping):
        formal_issue("seed bank splits must be a JSON object")
    else:
        for split_name in sorted(split_names):
            seeds = splits.get(split_name)
            if not isinstance(seeds, list) or not seeds:
                formal_issue(
                    f"seed bank split {split_name!r} must be a non-empty list")
                continue
            if any(isinstance(seed, bool) or not isinstance(seed, int)
                   for seed in seeds):
                formal_issue(
                    f"seed bank split {split_name!r} contains non-integer seeds")
                continue
            if len(seeds) != len(set(seeds)):
                formal_issue(
                    f"seed bank split {split_name!r} contains duplicate seeds")
                continue
            passes.append(
                f"seed bank split {split_name!r}: {len(seeds)} unique seeds")

    bank_identity = str(bank.get("source_config", "")).strip()
    if not bank_identity:
        formal_issue("seed bank source_config is missing")
        source_cfg = None
    else:
        source_path = _workspace_path(bank_identity)
        if not source_path.is_file():
            formal_issue(
                "seed bank source_config does not exist: "
                f"{bank_identity!r} (resolved to {source_path})")
            source_cfg = None
        else:
            try:
                source_cfg = load_config(str(source_path))
            except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                formal_issue(
                    f"seed bank source_config is unreadable: {exc}")
                source_cfg = None

    actual_fingerprint = str(bank.get("scenario_fingerprint", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{16}", actual_fingerprint):
        formal_issue(
            "seed bank scenario_fingerprint must be 16 lowercase hex "
            f"characters, got {actual_fingerprint!r}")
    elif bank_identity:
        expected_fingerprint = scenario_fingerprint(manifest, bank_identity)
        if actual_fingerprint != expected_fingerprint:
            formal_issue(
                "seed bank scenario_fingerprint does not match the effective "
                "K/Q/region/dynamics identity: "
                f"bank={actual_fingerprint}, expected={expected_fingerprint}")
        else:
            passes.append(
                "seed bank scenario_fingerprint matches the effective config")

        if source_cfg is not None:
            source_fingerprint = scenario_fingerprint(
                source_cfg, bank_identity)
            if actual_fingerprint != source_fingerprint:
                formal_issue(
                    "seed bank scenario_fingerprint is stale or incompatible "
                    "with source_config: "
                    f"bank={actual_fingerprint}, source={source_fingerprint}")
            else:
                passes.append(
                    "seed bank fingerprint matches its source_config")

    if source_cfg is not None:
        try:
            effective_identity = _scenario_identity(manifest)
            source_identity = _scenario_identity(source_cfg)
        except (AttributeError, TypeError, ValueError) as exc:
            formal_issue(f"seed bank scenario identity is unreadable: {exc}")
        else:
            differences = [
                f"{name}: effective={effective_identity[name]!r}, "
                f"bank_source={source_identity[name]!r}"
                for name in effective_identity
                if effective_identity[name] != source_identity[name]
            ]
            if differences:
                formal_issue(
                    "seed bank scenario identity mismatch "
                    "(K/Q/region/dynamics): " + "; ".join(differences))
            else:
                passes.append(
                    "seed bank source_config matches effective "
                    "K/Q/region/dynamics")

    return failures, passes, warnings


def load_raw_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = load_unique_yaml(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"manifest root must be a mapping: {path}")
    return raw


def collect_checks(
    manifest_path: str,
    *,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Return ``(failures, messages)`` for the C0 identity contract.

    For API compatibility, non-strict formal warnings are returned in the
    second list with :data:`_FORMAL_WARNING_PREFIX`.  The command-line report
    renders those entries as warnings rather than successful checks.
    """
    failures: List[str] = []
    passes: List[str] = []

    manifest = load_config(manifest_path)
    # --- 1. Scope pins (paper identity) ---
    def scope_ok(label: str, actual, expected) -> None:
        if actual != expected:
            failures.append(
                f"{label}: manifest says {actual!r}, expected {expected!r}")
        else:
            passes.append(f"{label}: {actual!r}")

    def close_ok(label: str, actual: float, expected: float,
                 tolerance: float = 1.0e-12) -> None:
        value = float(actual)
        if abs(value - float(expected)) > float(tolerance):
            failures.append(
                f"{label}: manifest says {value!r}, expected {expected!r}")
        else:
            passes.append(f"{label}: {value!r}")

    try:
        scope_ok("ground_communication_enabled", manifest.marl.ground_communication_enabled, False)
        scope_ok("joint_isac_power_enabled", manifest.marl.joint_isac_power_enabled, True)
        scope_ok("distributed_coordination_use_local_belief_targets",
                 manifest.marl.distributed_coordination_use_local_belief_targets, True)
        fusion = manifest.marl.detection_fusion_mode
        if fusion not in ("local_only", "u2u_distributed"):
            failures.append(
                f"detection_fusion_mode: {fusion!r} not in "
                f"{{local_only, u2u_distributed}} (strict no-ground fusion)")
        else:
            passes.append(f"detection_fusion_mode: {fusion!r} (no ground fusion)")
        # Strict no-truth identity: local-belief coordination requires the belief
        # manager, so the canonical manifest must pin tracking_enabled=true
        # together with the no-truth fail-closed gate (the legacy tracking-free
        # default remains OFF only in default.yaml for pre-G2 reproducibility).
        scope_ok("distributed_no_truth_fail_closed",
                 manifest.marl.distributed_no_truth_fail_closed, True)
        scope_ok("distributed_decision_sufficient_comm_enabled",
                 manifest.marl.distributed_decision_sufficient_comm_enabled,
                 False)
        scope_ok("distributed_decision_sufficient_event_trigger_enabled",
                 manifest.marl.distributed_decision_sufficient_event_trigger_enabled,
                 False)
        scope_ok("distributed_decision_sufficient_adaptive_bits_enabled",
                 manifest.marl.distributed_decision_sufficient_adaptive_bits_enabled,
                 False)
        scope_ok("comm_payload_mode", manifest.marl.comm_payload_mode,
                 "target_tokens")
        scope_ok("target_allocation_enabled",
                 manifest.marl.target_allocation_enabled, True)
        scope_ok("comm_rate_bits_per_dim",
                 list(manifest.marl.comm_rate_bits_per_dim), [0, 4])
        if manifest.marl.distributed_no_truth_fail_closed:
            if not manifest.marl.distributed_coordination_use_local_belief_targets:
                failures.append(
                    "distributed_no_truth_fail_closed requires "
                    "distributed_coordination_use_local_belief_targets=true")
            if not manifest.marl.tracking_enabled:
                failures.append(
                    "distributed_no_truth_fail_closed requires "
                    "tracking_enabled=true (belief manager)")
            else:
                passes.append(
                    "tracking_enabled: true (belief manager for local-belief "
                    "coordination)")
    except Exception as exc:  # pragma: no cover - structural
        failures.append(f"manifest scope fields unreadable: {exc}")

    # --- 2. OTFS numerology + sensing clock ---
    ot = manifest.otfs
    n_cpi = int(ot.n_cpi)
    t_sense = float(n_cpi * ot.N * ot.T_sym)
    if int(ot.M) != 64 or int(ot.N) != 16 or n_cpi != 1:
        failures.append(f"OTFS M/N/n_cpi pins violated: ({ot.M},{ot.N},{n_cpi})")
    else:
        passes.append(f"OTFS numerology M={ot.M}, N={ot.N}, n_cpi={n_cpi}")
    close_ok("otfs.fc", ot.fc, 2.8e10, 1.0e-3)
    close_ok("otfs.B", ot.B, 1.0e6, 1.0e-9)
    close_ok("otfs.delta_f", ot.delta_f, 1.5625e4, 1.0e-9)
    close_ok("otfs.T_sym", ot.T_sym, 6.4e-5, 1.0e-12)
    if abs(t_sense - 1.024e-3) > 1e-9:
        failures.append(f"T_sense = n_cpi*N*T_sym = {t_sense:.6e} s != 1.024 ms")
    else:
        passes.append(f"T_sense = {t_sense:.6e} s (one CPI per control action)")
    mode = str(getattr(manifest.scenario, "sensing_energy_mode", "dt_frame"))
    if mode != "cpi_frame":
        failures.append(f"sensing_energy_mode = {mode!r}, expected cpi_frame")
    else:
        passes.append("sensing_energy_mode: cpi_frame (battery on OTFS clock)")

    # --- 3. Detector + DD model ---
    c_det = float(getattr(manifest.detection, "c_det", 1.0))
    if abs(c_det - 1.0) > 1e-12:
        failures.append(f"c_det = {c_det!r}, expected 1.0 (real Gaussian shift)")
    else:
        passes.append(f"c_det: {c_det} (real_gaussian_shift convention)")
    dd_mode = str(getattr(manifest.detection, "dd_gain_mode", "binary"))
    if dd_mode != "continuous":
        failures.append(f"dd_gain_mode = {dd_mode!r}, expected continuous "
                        f"(I_support*|A|^2, no binary g_dd gate)")
    else:
        passes.append("dd_gain_mode: continuous (support * |A|^2)")
    close_ok("detection.P_FA", manifest.detection.P_FA, 0.001)
    close_ok("detection.g_min", manifest.detection.g_min, 0.5)
    close_ok("uav.P_isac_total", manifest.uav.P_isac_total, 1.0)

    # --- 4. Confirmation protocol + effective seed-bank identity ---
    scope_ok("eval_seed_split", manifest.marl.eval_seed_split, "test")
    scope_ok("final_eval_seed_split", manifest.marl.final_eval_seed_split,
             "test")
    scope_ok("checkpoint_confirmation_enabled",
             manifest.marl.checkpoint_confirmation_enabled, False)
    bank_failures, bank_passes, bank_warnings = _collect_seed_bank_checks(
        manifest, strict=strict)
    failures.extend(bank_failures)
    passes.extend(bank_passes)
    passes.extend(
        f"{_FORMAL_WARNING_PREFIX}{warning}" for warning in bank_warnings)

    # --- 5. Code defaults vs YAML default alignment (audit P0 drift) ---
    code_defaults = MasterConfig()
    yaml_defaults = get_default_config()
    drift = []
    if code_defaults.marl.tracking_enabled is not False:
        drift.append("params.py tracking_enabled != false")
    if code_defaults.marl.ground_communication_enabled is not False:
        drift.append("params.py ground_communication_enabled != false")
    if yaml_defaults.marl.tracking_enabled is not False:
        drift.append("default.yaml tracking_enabled != false")
    if yaml_defaults.marl.ground_communication_enabled is not False:
        drift.append("default.yaml ground_communication_enabled != false")
    if drift:
        failures.append("params.py/YAML default drift: " + "; ".join(drift))
    else:
        passes.append(
            "code/YAML legacy defaults align (tracking/ground = false); "
            "formal profile enables tracking explicitly")

    # --- 6. Docs pointer (C0: docs derive from the manifest) ---
    try:
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            doc = f.read()
        if "system_manifest.yaml" not in doc:
            failures.append("docs/CURRENT_SYSTEM_MODEL.md does not reference "
                            "config/system_manifest.yaml (documented system != manifest)")
        else:
            passes.append("docs/CURRENT_SYSTEM_MODEL.md references the manifest")
    except OSError:
        failures.append("docs/CURRENT_SYSTEM_MODEL.md not readable")

    # --- 7. Manifest/profile must inherit the declared single source ---
    try:
        chain = _config_extends_chain(manifest_path)
        canonical = Path(MANIFEST_DEFAULT).resolve()
        default = (Path(ROOT) / "config" / "default.yaml").resolve()
        candidate = chain[0]
        if candidate == canonical:
            if len(chain) < 2 or chain[1] != default:
                failures.append(
                    "canonical system manifest must directly extend "
                    "config/default.yaml")
            else:
                passes.append(
                    "canonical system manifest directly extends default.yaml")
        elif canonical not in chain[1:]:
            failures.append(
                "manifest extends must be 'default.yaml' for the canonical "
                "manifest, or a runtime config must transitively extend "
                "config/system_manifest.yaml")
        else:
            passes.append(
                "runtime config transitively extends system_manifest.yaml")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        failures.append(f"config inheritance chain unreadable: {exc}")

    return failures, passes


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C0 system-identity gate")
    parser.add_argument("--manifest", default=MANIFEST_DEFAULT)
    parser.add_argument("--commit", default=None,
                        help="expected commit SHA; printed for the run manifest")
    parser.add_argument("--strict", action="store_true",
                        help=(
                            "require a v2 seed bank whose fingerprint and "
                            "K/Q/region/dynamics identity match the config"))
    args = parser.parse_args(argv)

    failures, passes = collect_checks(args.manifest, strict=args.strict)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    try:
        actual_commit = git("rev-parse", "HEAD")
        if args.commit:
            expected_commit = str(args.commit).strip()
            if not expected_commit or not actual_commit.startswith(expected_commit):
                failures.append(
                    f"git commit mismatch: actual {actual_commit}, expected "
                    f"{expected_commit!r}")
            else:
                passes.append(f"git commit: {actual_commit}")
        if args.strict:
            status = git("status", "--porcelain=v1", "--untracked-files=all")
            if status:
                failures.append("git workspace is dirty")
            else:
                passes.append("git workspace is clean")
            manifest_relative = os.path.relpath(
                os.path.abspath(args.manifest), ROOT).replace(os.sep, "/")
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", manifest_relative],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if tracked.returncode != 0:
                failures.append(
                    f"manifest is not tracked by git: {manifest_relative}")
            else:
                passes.append(f"manifest is tracked: {manifest_relative}")
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"git identity unavailable: {exc}")
    print(f"System identity check: {args.manifest}")
    warning_count = 0
    for ok in passes:
        if ok.startswith(_FORMAL_WARNING_PREFIX):
            warning_count += 1
            print(f"  [WARN] {ok[len(_FORMAL_WARNING_PREFIX):]}")
        else:
            print(f"  [OK] {ok}")
    for bad in failures:
        print(f"  [FAIL] {bad}")
    hard_pins = [f for f in failures if "docs/" not in f]
    bad = failures if args.strict else hard_pins
    if bad:
        print(f"RESULT: FAIL ({len(bad)} mismatch(es))")
        return 1
    if warning_count:
        print(
            "RESULT: PASS (compatibility mode only; "
            f"{warning_count} formal warning(s))")
    else:
        print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
