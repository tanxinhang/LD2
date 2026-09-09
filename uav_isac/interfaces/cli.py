"""Canonical architecture V2 command-line interface."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from typing import Sequence

from uav_isac.adapters import (
    HoldPositionPolicy,
    build_legacy_environment,
    build_environment_from_resolved,
    build_repository_identity,
    FileSystemArtifactStore,
    LegacyPythonProcessRunner,
    load_resolved_configuration,
    load_registered_configuration,
)
from uav_isac.application import CommandDispatcher, CommandSpec, EpisodeRunner
from uav_isac.domain import EpisodeSpec
from uav_isac.domain import RunManifest
from uav_isac.governance import (
    assert_operation_allowed,
    audit_architecture,
    load_project_phase,
    load_characterization_baselines,
    load_runtime_profiles,
    semantic_fingerprint,
)


def _phase_command() -> int:
    phase = load_project_phase()
    print(json.dumps({
        "phase": phase.phase,
        "completed_gates": phase.completed_gates,
        "operations": phase.operations,
    }, indent=2))
    return 0


def _audit_command() -> int:
    assert_operation_allowed("migration_audit")
    violations = audit_architecture()
    if violations:
        for violation in violations:
            print(violation.format())
        return 1
    print("PASS: architecture V2 boundaries are valid")
    return 0


def _characterize_command(
    profile: str,
    seed: int,
    frames: int,
    artifacts_root: str | None,
) -> int:
    assert_operation_allowed("characterization_tests")
    resolved = load_registered_configuration(profile)
    environment = build_environment_from_resolved(resolved, seed)
    result = EpisodeRunner(environment, HoldPositionPolicy()).run(
        EpisodeSpec(seed=seed, max_frames=frames)
    )
    last = result.frames[-1]
    detection = last.info.get("P_D_q")
    summary = {
        "seed": seed,
        "frames": len(result.frames),
        "completed": result.completed,
        "semantic_sha256": semantic_fingerprint(result),
        "config_sha256": resolved.sha256,
        "final_team_reward": last.info.get("team_reward"),
        "final_detection_probability": (
            detection.tolist() if hasattr(detection, "tolist") else detection
        ),
    }
    if artifacts_root:
        repository = build_repository_identity()
        run_id = "characterization-" + semantic_fingerprint({
            "config_sha256": resolved.sha256,
            "code_sha256": repository.sha256,
            "seed": seed,
            "frames": frames,
        })[:16]
        store = FileSystemArtifactStore(artifacts_root)
        store.create_run(RunManifest(
            run_id=run_id,
            run_type="characterization",
            state="completed",
            created_at=datetime.now(timezone.utc).isoformat(),
            command=(
                "python", "-m", "uav_isac.interfaces.cli", "characterize",
            ),
            config_sha256=resolved.sha256,
            code_identity=repository.sha256,
            seeds=(seed,),
            inputs={"config": resolved.source},
        ))
        artifact = store.write_json(run_id, "derived/summary.json", summary)
        summary["run_id"] = run_id
        summary["artifact_sha256"] = artifact.sha256
    print(json.dumps(summary, indent=2))
    return 0


def _verify_baseline_command() -> int:
    assert_operation_allowed("characterization_tests")
    failed = 0
    for baseline in load_characterization_baselines():
        environment = build_legacy_environment(baseline.config, baseline.seed)
        result = EpisodeRunner(environment, HoldPositionPolicy()).run(
            EpisodeSpec(seed=baseline.seed, max_frames=baseline.frames)
        )
        actual = semantic_fingerprint(result)
        passed = actual == baseline.semantic_sha256
        print(f"{'PASS' if passed else 'FAIL'} {baseline.name}: {actual}")
        failed += int(not passed)
    return 1 if failed else 0


def _profiles_command() -> int:
    print(json.dumps([
        {
            "name": item.name,
            "path": item.path,
            "purpose": item.purpose,
            "formal": item.formal,
            "status": item.status,
        }
        for item in load_runtime_profiles()
    ], indent=2))
    return 0


def _dispatch_command(name: str, operation: str, arguments: list[str]) -> int:
    assert_operation_allowed(operation)
    forwarded = tuple(arguments[1:] if arguments[:1] == ["--"] else arguments)
    return CommandDispatcher(LegacyPythonProcessRunner()).dispatch(CommandSpec(
        operation=operation,
        entrypoint=name,
        arguments=forwarded,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("phase", help="show the current project phase")
    commands.add_parser("audit", help="audit V2 dependency boundaries")
    commands.add_parser("profiles", help="list registered V2 runtime profiles")
    commands.add_parser(
        "verify-baseline",
        help="replay every registered semantic characterization",
    )
    characterize = commands.add_parser(
        "characterize",
        help="run a deterministic hold-policy migration probe",
    )
    characterize.add_argument("--profile", default="default_dev")
    characterize.add_argument("--seed", type=int, default=451)
    characterize.add_argument("--frames", type=int, default=3)
    characterize.add_argument(
        "--artifacts-root",
        default=None,
        help="optional V2 artifact root; existing run IDs are never overwritten",
    )
    for name, help_text in (
        ("train", "run the canonical training backend when optimization is enabled"),
        ("pilot", "run one strict migration-audit pilot"),
        ("refresh-results", "run the frozen result bank after audit approval"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phase":
        return _phase_command()
    if args.command == "audit":
        return _audit_command()
    if args.command == "profiles":
        return _profiles_command()
    if args.command == "verify-baseline":
        return _verify_baseline_command()
    if args.command == "characterize":
        return _characterize_command(
            args.profile,
            args.seed,
            args.frames,
            args.artifacts_root,
        )
    if args.command == "train":
        return _dispatch_command("train", "algorithm_optimization", args.arguments)
    if args.command == "pilot":
        return _dispatch_command("pilot", "migration_audit", args.arguments)
    if args.command == "refresh-results":
        return _dispatch_command(
            "refresh-results", "full_result_refresh", args.arguments
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
