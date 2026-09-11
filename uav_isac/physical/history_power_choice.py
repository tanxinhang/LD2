"""Offline finite-action selector under a per-node 1 W shared RF cap.

All action forecasts must use one destination, detection window, and frozen
H0 evidence model. History consists of destination-local evidence or confirmed
delivery receipts available to the deciding node. No unreceived remote history
is accepted as implicit context. This module cannot verify model provenance.
"""
from dataclasses import dataclass
import numpy as np
from uav_isac.physical.correlated_soft_evidence import conditional_deflection_gain


@dataclass(frozen=True)
class PowerAction:
    name: str
    sensing_w: float
    communication_w: float
    # Evidence indices in a caller-supplied, frozen predictive model.
    candidate: int
    availability_probability: float


def choose_history_conditioned_power(
    actions, mean_shift, covariance, *, evidence_ids, observed_frames,
    known_available_ids, current_frame, window_frames, total_power_w=1.0,
):
    """Rank one-new-evidence actions by expected conditional H0-Deflection.

    Availability probability includes transport/processing success as needed.
    It must not depend on the unrealized evidence value. Probabilities, delta,
    and covariance must be forecast for each declared power action externally.
    Choosing one action does not jointly solve multi-node scheduling.
    """
    if not np.isfinite(total_power_w) or total_power_w <= 0 or window_frames < 1:
        raise ValueError('invalid budget or history window')
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError('duplicate evidence identities in model')
    if len(evidence_ids) != len(mean_shift) or len(observed_frames) != len(evidence_ids):
        raise ValueError('model identity dimensions do not match')
    known = set(known_available_ids)
    if not known.issubset(set(evidence_ids)):
        raise ValueError('known history lacks a model descriptor')
    if any(observed_frames[i] > current_frame for i,e in enumerate(evidence_ids) if e in known):
        raise ValueError('future observations cannot be history')
    history = [i for i,e in enumerate(evidence_ids)
               if e in known and 0 <= current_frame-observed_frames[i] < window_frames]
    scored=[]
    for a in actions:
        values=(a.sensing_w,a.communication_w,a.availability_probability)
        if not all(np.isfinite(x) for x in values) or min(values)<0 or values[2]>1:
            raise ValueError('invalid power or availability probability')
        if a.sensing_w+a.communication_w > total_power_w+1e-12:
            continue
        if not 0 <= a.candidate < len(evidence_ids):
            raise ValueError('invalid candidate index')
        # Old retransmissions cannot replenish an expired detection window.
        if current_frame-observed_frames[a.candidate] >= window_frames:
            continue
        gain=conditional_deflection_gain(mean_shift,covariance,history,a.candidate)
        scored.append((a.availability_probability*gain,a))
    # Explicit idle remains available when all marginal values vanish.
    positive=[item for item in scored if item[0]>1e-12]
    if not positive:
        return dict(action='idle',sensing_w=0.,communication_w=0.,expected_gain=0.,history_count=len(history))
    score,best=max(positive,key=lambda item:(item[0],-item[1].sensing_w-item[1].communication_w))
    return dict(action=best.name,sensing_w=best.sensing_w,communication_w=best.communication_w,
                expected_gain=float(score),history_count=len(history))
