#!/usr/bin/env python
"""Gate 2 (advice/016 §18): offline task-regret audit of a trained Student.

For each validation/test frame of every scale, replays the student's predicted
d_eff through the TEACHER pairing (same projection as the training report) and
computes the P_D-domain task-regret statistics:

  R_gamma      = 1[teacher feasible AND student infeasible]     (feasibility
                 flip -- the harmful miss, advice/016 §4.1)
  R_t          = max(0, teacher_min_PD - student_min_PD) when BOTH feasible
                 (residual max-min loss in P_D space)
  P(gamma_theta > 1 | gamma* <= 1) = P(flip | teacher feasible)

Also reports the per-scale mean student/teacher min-PD and the exact-set /
owner / edge agreement so the regret can be read against structure accuracy.

Usage:
  python tools/audit_task_regret_student.py \
      --student results/_gate2_task_regret/frozen_structure_student_task_regret.pt \
      --trace results/.../teacher_trace.npz [--trace ...] \
      --output results/_gate2_task_regret/offline_regret.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_structure_teacher_trace import (  # noqa: E402
    _infer_observation_slices,
    _per_target_features,
)
from tools.train_multiscale_structure_student import (  # noqa: E402
    _teacher_frame_gain,
    _torch_pd_from_deflection,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)
from uav_isac.coordination.structure_regret import tstar_of  # noqa: E402


def _predict(model: torch.nn.Module, features: np.ndarray,
             student: FrozenStructureStudent, batch_size: int) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            feat = torch.as_tensor(
                features[start:start + batch_size], dtype=torch.float32)
            normalized = (feat - torch.as_tensor(
                student.feature_mean, dtype=torch.float32)) / torch.as_tensor(
                    student.feature_scale, dtype=torch.float32)
            log_value = model(normalized)
            rows.append(torch.expm1(log_value).clamp_min(0).cpu().numpy())
    prediction = np.concatenate(rows, axis=0)
    for k in range(min(prediction.shape[1], prediction.shape[2])):
        prediction[:, k, k, :] = 0.0
    return prediction


def _student_structure_target_pd(
    prediction: np.ndarray,
    data: dict,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (Q,) target deflection + P_D under the teacher pairing."""
    selected = np.asarray(data["teacher_pair"][frame_indices], dtype=np.float64)
    pred_d = np.maximum(prediction, 0.0)
    receiver_d = np.sum(pred_d * selected, axis=1)
    target_d = np.max(receiver_d, axis=1)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    target_pd = compute_detection_probabilities(target_d, p_fa)
    return target_d, target_pd


def _teacher_structure_target_pd(
    data: dict,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(data["teacher_pair"][frame_indices], dtype=np.float64)
    deff = np.asarray(data["privileged_d_eff"][frame_indices], dtype=np.float64)
    receiver_d = np.sum(deff * selected, axis=1)
    target_d = np.max(receiver_d, axis=1)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    target_pd = compute_detection_probabilities(target_d, p_fa)
    return target_d, target_pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True, type=Path)
    ap.add_argument("--trace", action="append", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--split", default="validation",
                    choices=("validation", "test", "fit"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    student = FrozenStructureStudent(args.student)
    report: dict = {"schema": "gate2-task-regret-audit", "student": str(args.student)}

    # seed-grouped split re-use (fit/validation/test by seed)
    from tools.train_multiscale_structure_student import _seed_grouped_split

    for path in args.trace:
        data = np.load(path, allow_pickle=False)
        d = {key: data[key] for key in data.files}
        split_seed, split_index = _seed_grouped_split(d)
        idx = split_index[args.split]
        K = int(np.asarray(d["num_uavs"]).reshape(-1)[0])
        Q = int(np.asarray(d["num_targets"]).reshape(-1)[0])
        slices = _infer_observation_slices(d["local_obs"].shape[-1], K, Q)
        features = _per_target_features(
            d, "local_obs", idx, slices).astype(np.float32)
        prediction = _predict(student.model, features, student, args.batch_size)

        teacher_d, teacher_pd = _teacher_structure_target_pd(d, idx)
        student_d, student_pd = _student_structure_target_pd(prediction, d, idx)

        qos_floor = float(np.asarray(d["qos_floor"]).reshape(-1)[0])
        p_fa = float(np.asarray(d["p_fa"]).reshape(-1)[0])
        teacher_feas = np.min(teacher_pd, axis=-1) >= qos_floor
        student_feas = np.min(student_pd, axis=-1) >= qos_floor

        r_gamma = teacher_feas & ~student_feas
        both = teacher_feas & student_feas
        r_t = np.maximum(0.0, np.min(teacher_pd, axis=-1)
                         - np.min(student_pd, axis=-1))

        # teacher max-min in deflection units (gain reconstruction)
        gain = _teacher_frame_gain(d, idx)
        budget = np.ones(K)
        tstar = np.asarray([tstar_of(gain[t], budget) for t in range(len(idx))])
        d_req = float(np.maximum(
            np.sqrt(2.0) * 0.0, 0.0))  # placeholder, use PD-domain floor below
        from uav_isac.utils.math_utils import Q_inverse
        d_req = float(max(
            Q_inverse(np.asarray(p_fa)) - Q_inverse(np.asarray(qos_floor)), 0.0) ** 2)

        per_scale = {
            "frames": int(len(idx)),
            "seeds": [int(s) for s in np.unique(d["seed"][idx])],
            "teacher_feasible_fraction": float(np.mean(teacher_feas)),
            "student_feasible_fraction": float(np.mean(student_feas)),
            "E_R_gamma": float(np.mean(r_gamma)),
            "P_flip_given_teacher_feasible": float(
                np.mean(r_gamma[teacher_feas])) if teacher_feas.any() else None,
            "E_R_t_both_feasible": float(np.mean(r_t[both])) if both.any() else None,
            "teacher_min_pd_mean": float(np.mean(np.min(teacher_pd, axis=-1))),
            "student_min_pd_mean": float(np.mean(np.min(student_pd, axis=-1))),
            "teacher_tstar_mean": float(np.mean(tstar)),
            "teacher_tstar_feasible_fraction": float(
                np.mean(tstar >= d_req - 1e-9)),
        }
        report[str(path)] = per_scale
        print(f"{path.parent.name}: {json.dumps(per_scale, indent=2)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
