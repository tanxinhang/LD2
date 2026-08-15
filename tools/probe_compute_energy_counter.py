"""Probe supported cumulative package-energy counters without estimating joules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.evaluation.energy_counter_window import (
    discover_linux_rapl_package_domains,
)


def probe_document(powercap_root: Path = Path("/sys/class/powercap")) -> dict:
    domains = discover_linux_rapl_package_domains(powercap_root)
    return {
        "schema_version": 1,
        "platform": platform.system(),
        "status": (
            "cumulative_package_counter_discovered"
            if domains else "no_supported_cumulative_package_counter"),
        "domains": [
            {
                "source_kind": "rapl_package",
                "name": domain.name,
                "path": str(domain.path),
                "counter_modulus_j": domain.counter_modulus_j,
                "counter_resolution_j": 1.0e-6,
                "wrap_index_available": False,
            }
            for domain in domains
        ],
        "instantaneous_gpu_power_is_accepted_as_package_energy": False,
        "assumed_power_times_cpu_seconds_is_accepted_as_energy": False,
        "remaining_requirements": [
            "calibrated per-reading meter uncertainty",
            "conservative package energy-rate upper bound for wrap exclusion",
            "stable meter generation identifier",
            "complete controller window instrumentation",
            "disjoint episode calibration and validation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--powercap-root", type=Path, default=Path("/sys/class/powercap"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe_document(args.powercap_root)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
