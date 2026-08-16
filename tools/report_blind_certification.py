"""D1.5 blind certification report (advice 013): QoS feasible rate + Wilson
LCB as the PRIMARY criterion, plus mean/median/worst quantiles instead of
only the mean.  Run after results/_d1_5_blind100/paired_eval.csv exists."""
import ast
import csv
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from tools.assert_gate_thresholds import (
    MEDIUM_FLOORS,
    assert_medium_gate,
    wilson_lower,
)

PATH = "results/_d1_5_blind100/paired_eval.csv"


def main() -> int:
    rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
    if not rows or "eval_episode_seeds" not in rows[0]:
        print(f"{PATH} not ready (blind run still in progress)", file=sys.stderr)
        return 2
    r = rows[0]
    seeds = ast.literal_eval(r["eval_episode_seeds"])
    steady = np.asarray(ast.literal_eval(r["eval_episode_steady_P_D"]))
    weak3 = np.asarray(ast.literal_eval(r["eval_episode_weak3_P_D"]))
    worst = np.asarray(ast.literal_eval(r["eval_episode_worst_P_D"]))
    n = len(seeds)
    feasible = np.logical_and.reduce([
        steady >= MEDIUM_FLOORS["steady"],
        weak3 >= MEDIUM_FLOORS["weak3"],
        worst >= MEDIUM_FLOORS["worst"],
    ])
    p_hat = float(feasible.mean())
    lcb = wilson_lower(int(feasible.sum()), n)

    print("=" * 62)
    print("D1.5 BLIND CERTIFICATION REPORT (frozen deployment candidate)")
    print("=" * 62)
    print(f"blind seeds          : {n} (never seen in any run/bank)")
    print(f"QoS feasible rate    : {p_hat:.3f}  (PRIMARY criterion)")
    print(f"Wilson LCB (95%, 1s) : {lcb:.3f}  (PRIMARY criterion)")
    print(f"Gate (point estimate): "
          f"{'PASS' if p_hat >= 0.70 else 'FAIL'}")
    print(f"Gate (LCB enforced)  : "
          f"{'PASS' if lcb >= 0.70 else 'FAIL'}")
    print("-" * 62)
    for name, v in (("steady", steady), ("weak3", weak3), ("worst", worst)):
        print(f"{name:8s}: mean={v.mean():.4f} median={np.median(v):.4f} "
              f"q25={np.quantile(v, .25):.4f} q75={np.quantile(v, .75):.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
    print("-" * 62)
    # Verify the point-estimate Medium gate (worst >= 0.60 etc.).
    try:
        assert_medium_gate(steady.mean(), weak3.mean(), worst.mean(), p_hat,
                           qos_wilson_lcb=lcb, label="blind100")
        print("Medium gate (point + LCB): PASS")
    except AssertionError as exc:
        print(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
