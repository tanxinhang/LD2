"""Pre-registered analysis for the dynamic K12 confirmation (LD3-dyn-k12-conf-1).

Reads the single-row paired_eval.csv produced by run_mappo for the 25-seed
protocol seeds and reports, per config:

  - worst / weak-3 / steady P_D means, minima, 5th percentiles
  - three-gate pass count (worst >= 0.60, weak3 >= 0.70, steady >= 0.80)
  - Wilson 95% lower bound of the pass fraction
  - safety / protocol invariants (separation violation, per-class deadline
    violation, serialized protocol total deadline violation)

With two CSVs (baseline, candidate) it also reports paired same-seed deltas and
an episode-cluster bootstrap 95% lower bound of the paired worst delta mean.

Usage:
    python tools/summarize_dyn_k12_conf.py results/_conf25_baseline/paired_eval.csv
    python tools/summarize_dyn_k12_conf.py --candidate results/_conf25_freshness/paired_eval.csv results/_conf25_baseline/paired_eval.csv
"""

import ast
import csv
import math
import sys

import numpy as np


def load_run(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, row = rows[0], rows[1]
    values = dict(zip(header, row))
    episodes = ast.literal_eval(values.get("eval_episode_seeds", "[]"))
    worst = np.asarray(ast.literal_eval(values["eval_episode_worst_P_D"]))
    weak3 = np.asarray(ast.literal_eval(values["eval_episode_weak3_P_D"]))
    steady = np.asarray(ast.literal_eval(values["eval_episode_steady_P_D"]))
    return episodes, worst, weak3, steady, values


def wilson_lcb(pass_count, total, z=1.96):
    if total <= 0:
        return 0.0
    phat = pass_count / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt(
        phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, centre - margin)


def summarize(path):
    episodes, worst, weak3, steady, v = load_run(path)
    total = worst.size
    passed = int(np.sum(
        (worst >= 0.60) & (weak3 >= 0.70) & (steady >= 0.80)))
    print(f"== {path}  (n={total}) ==")
    for name, arr in (("worst", worst), ("weak-3", weak3), ("steady", steady)):
        print(f"  {name:7s} mean={np.mean(arr):.4f}  min={np.min(arr):.4f}  "
              f"p05={np.percentile(arr, 5):.4f}")
    print(f"  three-gate pass: {passed}/{total}  "
          f"Wilson95-LCB={wilson_lcb(passed, total):.4f}")
    print(f"  min inter-UAV continuous distance: "
          f"{float(v.get('eval_inter_uav_continuous_min_distance_m', 0)):.2f} m  "
          f"separation violation rate: "
          f"{float(v.get('eval_inter_uav_continuous_separation_violation_rate', 0)):.4f}")
    print(f"  evidence deadline violation rate: "
          f"{float(v.get('eval_evidence_comm_deadline_violation_rate', 0)):.4f}  "
          f"delivery: {float(v.get('eval_evidence_comm_delivery_rate', 0)):.4f}")
    print(f"  serialized protocol latency mean/p95: "
          f"{float(v.get('eval_protocol_total_latency_mean_s', 0)):.4f} / "
          f"{float(v.get('eval_protocol_total_p95_latency_s', 0)):.4f} s  "
          f"total deadline violation: "
          f"{float(v.get('eval_protocol_total_deadline_violation_rate', 0)):.4f}")
    return episodes, worst, weak3, steady, passed, total


def paired_bootstrap_lcb(deltas, samples=2000, alpha=0.05, seed=20260825):
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    n = deltas.size
    for s in range(samples):
        idx = rng.integers(0, n, size=n)
        means[s] = np.mean(deltas[idx])
    return float(np.percentile(means, 100.0 * alpha))


def main():
    args = sys.argv[1:]
    candidate_csv = None
    if "--candidate" in args:
        i = args.index("--candidate")
        candidate_csv = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return
    baseline_csv = args[0]

    ep_b, w_b, w3_b, s_b, pass_b, n_b = summarize(baseline_csv)
    if candidate_csv:
        ep_c, w_c, w3_c, s_c, pass_c, n_c = summarize(candidate_csv)
        assert ep_b == ep_c, "paired runs must use the same seed order"
        delta_worst = w_c - w_b
        delta_weak3 = w3_c - w3_b
        delta_steady = s_c - s_b
        print("== paired same-seed deltas (candidate - baseline) ==")
        print(f"  worst  mean={np.mean(delta_worst):+.4f}  "
              f"bootstrap95-LCB={paired_bootstrap_lcb(delta_worst):+.4f}  "
              f"positive seeds={int(np.sum(delta_worst > 0))}/{n_b}")
        print(f"  weak-3 mean={np.mean(delta_weak3):+.4f}  "
              f"positive seeds={int(np.sum(delta_weak3 > 0))}/{n_b}")
        print(f"  steady mean={np.mean(delta_steady):+.4f}  "
              f"positive seeds={int(np.sum(delta_steady > 0))}/{n_b}")
        print(f"  gate pass: baseline {pass_b}/{n_b} -> candidate {pass_c}/{n_c}")
        verdict = (
            "PASS" if (pass_c > pass_b
                       and np.mean(delta_worst) > 0.0
                       and paired_bootstrap_lcb(delta_worst) > 0.0)
            else "NOT PASS"
        )
        print(f"  verdict per PREREG_LD3_DYN_K12_CONF_1: {verdict}")


if __name__ == "__main__":
    main()
