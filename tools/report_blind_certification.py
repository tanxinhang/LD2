"""D1.5/D1.9 blind certification report (advice 013): QoS feasible rate +
Wilson LCB as the PRIMARY criterion, plus mean/median/worst quantiles instead
of only the mean.  Run after a blind paired_eval.csv exists.

Usage:
    python tools/report_blind_certification.py [--csv results/_d1_9_blind100/paired_eval.csv]
"""
import argparse
import ast
import csv
import math
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from tools.assert_gate_thresholds import (
    MEDIUM_FLOORS,
    acceptance_floors_from_config,
    assert_medium_gate,
    wilson_lower,
)

DEFAULT_PATH = "results/_d1_5_blind100/paired_eval.csv"
DEFAULT_LCB_FLOOR = 0.70


def _lcb_floor_from_config(config_path: Optional[str]) -> float:
    """Read marl.qos_acceptance_wilson_lcb_floor (R15 single source)."""
    if not config_path:
        return DEFAULT_LCB_FLOOR
    import os
    import sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from config.params import load_config
    cfg = load_config(config_path)
    value = float(getattr(
        cfg.marl, "qos_acceptance_wilson_lcb_floor", DEFAULT_LCB_FLOOR))
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise RuntimeError(
            f"{config_path}: qos_acceptance_wilson_lcb_floor must lie in (0,1)")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_PATH)
    ap.add_argument(
        "--config", default=None,
        help="frozen config to read marl.qos_acceptance_floors and "
             "marl.qos_acceptance_wilson_lcb_floor (R15 single source); "
             "absent -> canonical defaults")
    args = ap.parse_args(argv)
    path = args.csv
    floors = acceptance_floors_from_config(args.config)
    lcb_floor = _lcb_floor_from_config(args.config)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows or "eval_episode_seeds" not in rows[0]:
        print(f"{path} not ready (blind run still in progress)", file=sys.stderr)
        return 2
    r = rows[0]
    seeds = ast.literal_eval(r["eval_episode_seeds"])
    steady = np.asarray(ast.literal_eval(r["eval_episode_steady_P_D"]))
    weak3 = np.asarray(ast.literal_eval(r["eval_episode_weak3_P_D"]))
    worst = np.asarray(ast.literal_eval(r["eval_episode_worst_P_D"]))
    n = len(seeds)
    feasible = np.logical_and.reduce([
        steady >= floors["steady"],
        weak3 >= floors["weak3"],
        worst >= floors["worst"],
    ])
    p_hat = float(feasible.mean())
    lcb = wilson_lower(int(feasible.sum()), n)

    print("=" * 62)
    print("D1.5 BLIND CERTIFICATION REPORT (frozen deployment candidate)")
    print("=" * 62)
    print(f"blind seeds          : {n} (never seen in any run/bank)")
    print(f"QoS feasible rate    : {p_hat:.3f}  (PRIMARY criterion)")
    print(f"Wilson lower endpoint (95% two-sided) : {lcb:.3f}  (PRIMARY criterion)")
    print(f"acceptance floors    : steady {floors['steady']:.2f} / "
          f"weak3 {floors['weak3']:.2f} / worst {floors['worst']:.2f} "
          f"(LCB {lcb_floor:.2f})")
    print(f"Gate (point estimate): "
          f"{'PASS' if p_hat >= lcb_floor else 'FAIL'}")
    print(f"Gate (LCB enforced)  : "
          f"{'PASS' if lcb >= lcb_floor else 'FAIL'}")
    print("-" * 62)
    for name, v in (("steady", steady), ("weak3", weak3), ("worst", worst)):
        print(f"{name:8s}: mean={v.mean():.4f} median={np.median(v):.4f} "
              f"q25={np.quantile(v, .25):.4f} q75={np.quantile(v, .75):.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
    print("-" * 62)
    # Verify the point-estimate Medium gate (worst >= 0.60 etc.).
    try:
        assert_medium_gate(steady.mean(), weak3.mean(), worst.mean(), p_hat,
                           qos_wilson_lcb=lcb, label="blind100",
                           floors=floors, qos_floor=lcb_floor)
        print("Medium gate (point + LCB): PASS")
    except AssertionError as exc:
        print(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
