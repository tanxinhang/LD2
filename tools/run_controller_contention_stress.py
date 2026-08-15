"""Run a frozen controller audit under bounded same-core CPU contention.

This is a reproducible workstation stress screen, not a deployment scheduler
or RF-network model.  The wrapper launches deterministic hashing workers,
pins them and the audited child process to the declared logical CPUs, and
always removes its own workers when the audit exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def _worker() -> None:
    value = bytes(range(32))
    while True:
        value = hashlib.sha256(value).digest()


def _parse_cpus(value: str) -> list[int]:
    cpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cpus or len(cpus) != len(set(cpus)) or min(cpus) < 0:
        raise argparse.ArgumentTypeError(
            "--cpus must be a nonempty list of distinct nonnegative integers")
    available = psutil.cpu_count(logical=True)
    if available is not None and max(cpus) >= available:
        raise argparse.ArgumentTypeError(
            f"CPU index exceeds the {available} logical CPUs reported")
    return cpus


def run(
    command: list[str],
    *,
    cpus: list[int],
    worker_count: int,
    metadata_path: Path,
) -> int:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    workers: list[subprocess.Popen[bytes]] = []
    audited: subprocess.Popen[bytes] | None = None
    started = time.perf_counter()
    try:
        for _ in range(worker_count):
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            psutil.Process(worker.pid).cpu_affinity(cpus)
            workers.append(worker)
        audited = subprocess.Popen(command, creationflags=creationflags)
        psutil.Process(audited.pid).cpu_affinity(cpus)
        return_code = int(audited.wait())
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5.0)
        elapsed = float(time.perf_counter() - started)
        metadata = {
            "schema_version": 1,
            "status": "workstation_same_core_contention_screen",
            "logical_cpu_count_reported": psutil.cpu_count(logical=True),
            "affinity_logical_cpus": cpus,
            "contention_worker_count": worker_count,
            "audit_command": command,
            "audit_return_code": (
                None if audited is None else int(audited.returncode)
            ),
            "elapsed_s": elapsed,
            "scope": {
                "deployment_scheduler_jitter_measured": False,
                "network_contention_measured": False,
                "energy_measured": False,
                "commit_authority": False,
            },
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return return_code


def main() -> None:
    if "--worker" in sys.argv[1:]:
        _worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=_parse_cpus, default=_parse_cpus("0,1,2,3"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an audit command is required after --")
    if int(args.workers) < 1:
        parser.error("--workers must be positive")
    raise SystemExit(run(
        command,
        cpus=list(args.cpus),
        worker_count=int(args.workers),
        metadata_path=args.metadata,
    ))


if __name__ == "__main__":
    main()
