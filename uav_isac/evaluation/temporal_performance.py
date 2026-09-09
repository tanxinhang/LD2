"""Multi-frame detection and closed-loop convergence diagnostics.

The multi-frame metric sums Deflection, not probabilities.  This is the
correct sufficient-statistic composition for independent, whitened Gaussian
frame observations under the detector convention used by this project.

The convergence metric describes stabilization of an episode trajectory.  It
does not claim optimizer/training convergence; the same scalar helper can be
applied separately to evaluation values recorded across training checkpoints.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from uav_isac.physical.detection import compute_detection_probabilities


def rolling_multiframe_detection(
    deflection_history: np.ndarray,
    p_fa: float,
    window: int,
) -> np.ndarray:
    """Detection probability after accumulating each rolling frame window."""
    values = np.asarray(deflection_history, dtype=np.float64)
    width = int(window)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("deflection_history must be a non-empty (T,Q) array")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("deflection_history must be finite and non-negative")
    if width < 1 or width > values.shape[0]:
        raise ValueError("window must lie in [1,T]")
    prefix = np.concatenate([
        np.zeros((1, values.shape[1]), dtype=np.float64),
        np.cumsum(values, axis=0),
    ], axis=0)
    accumulated = prefix[width:] - prefix[:-width]
    return compute_detection_probabilities(accumulated, float(p_fa))


def frame_qos_metrics(pd_history: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-frame steady/weak3/worst trajectories."""
    values = np.asarray(pd_history, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("pd_history must be a non-empty (T,Q) array")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pd_history must contain probabilities in [0,1]")
    ordered = np.sort(values, axis=1)
    return {
        "steady": np.mean(values, axis=1),
        "weak3": np.mean(ordered[:, :min(3, values.shape[1])], axis=1),
        "worst": ordered[:, 0],
    }


def _first_consecutive(mask: np.ndarray, length: int) -> int | None:
    required = int(length)
    if required < 1 or mask.size < required:
        return None
    run = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        run = run + 1 if bool(value) else 0
        if run >= required:
            return int(index - required + 1)
    return None


def convergence_profile(
    trajectory: Iterable[float],
    *,
    threshold: float | None = None,
    rolling_window: int = 10,
    terminal_window: int = 20,
    patience: int = 3,
    absolute_band: float = 0.02,
    std_tolerance: float = 0.02,
    slope_tolerance: float = 0.002,
) -> dict[str, Any]:
    """Quantify settling, terminal stability and threshold persistence.

    Frames are reported one-based.  ``settling_frame`` is the first frame of
    ``patience`` consecutive rolling windows whose mean lies near the terminal
    mean and whose standard deviation is small.  This is deliberately stricter
    than a one-time threshold crossing.
    """
    values = np.asarray(list(trajectory), dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("trajectory must be finite and non-empty")
    width = min(max(1, int(rolling_window)), values.size)
    tail_width = min(max(1, int(terminal_window)), values.size)
    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    rolling_mean = np.mean(windows, axis=1)
    rolling_std = np.std(windows, axis=1)
    terminal = values[-tail_width:]
    terminal_mean = float(np.mean(terminal))
    x = np.arange(tail_width, dtype=np.float64)
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    terminal_slope = (
        float(np.dot(centered, terminal - terminal_mean) / denominator)
        if denominator > 0.0 else 0.0
    )
    stable = (
        (np.abs(rolling_mean - terminal_mean) <= float(absolute_band))
        & (rolling_std <= float(std_tolerance))
    )
    stable_start = _first_consecutive(stable, int(patience))
    settling_frame = (
        None if stable_start is None else int(stable_start + width)
    )
    result: dict[str, Any] = {
        "frames": int(values.size),
        "rolling_window": width,
        "terminal_window": tail_width,
        "terminal_mean": terminal_mean,
        "terminal_std": float(np.std(terminal)),
        "terminal_slope_per_frame": terminal_slope,
        "terminal_oscillation": float(np.mean(np.abs(np.diff(terminal))))
        if tail_width > 1 else 0.0,
        "settling_frame": settling_frame,
        "stable_window_fraction": float(np.mean(stable)),
        "converged": bool(
            settling_frame is not None
            and abs(terminal_slope) <= float(slope_tolerance)
        ),
    }
    if threshold is not None:
        threshold_value = float(threshold)
        passes = rolling_mean >= threshold_value
        sustained_start = _first_consecutive(passes, int(patience))
        result.update({
            "threshold": threshold_value,
            "first_hit_frame": (
                int(np.flatnonzero(values >= threshold_value)[0] + 1)
                if np.any(values >= threshold_value) else None
            ),
            "sustained_threshold_frame": (
                None if sustained_start is None
                else int(sustained_start + width)
            ),
            "terminal_threshold_hold_rate": float(
                np.mean(terminal >= threshold_value)),
        })
    return result


def summarize_closed_loop_convergence(
    pd_history: np.ndarray,
    *,
    floors: tuple[float, float, float] = (0.80, 0.70, 0.60),
    rolling_window: int = 10,
    terminal_window: int = 20,
    patience: int = 3,
) -> dict[str, Any]:
    """Convergence profiles for steady/weak3/worst and joint QoS holding."""
    metrics = frame_qos_metrics(pd_history)
    thresholds = dict(zip(("steady", "weak3", "worst"), floors))
    profiles = {
        name: convergence_profile(
            values,
            threshold=thresholds[name],
            rolling_window=rolling_window,
            terminal_window=terminal_window,
            patience=patience,
        )
        for name, values in metrics.items()
    }
    joint = (
        (metrics["steady"] >= floors[0])
        & (metrics["weak3"] >= floors[1])
        & (metrics["worst"] >= floors[2])
    )
    sustained = _first_consecutive(joint, int(patience))
    tail_width = min(max(1, int(terminal_window)), len(joint))
    return {
        "scope": "closed-loop trajectory stabilization, not training convergence",
        **profiles,
        "joint_qos": {
            "first_hit_frame": int(np.flatnonzero(joint)[0] + 1)
            if np.any(joint) else None,
            "sustained_frame": None if sustained is None else int(sustained + 1),
            "terminal_hold_rate": float(np.mean(joint[-tail_width:])),
            "ever_sustained": bool(sustained is not None),
        },
    }


def summarize_multiframe_detection(
    deflection_history: np.ndarray,
    p_fa: float,
    *,
    windows: Iterable[int] = (1, 5, 10, 20),
    tail_window: int = 20,
    floors: tuple[float, float, float] = (0.80, 0.70, 0.60),
    frame_duration_s: float | None = None,
) -> dict[str, Any]:
    """Summarize rolling accumulated-evidence performance at several horizons."""
    values = np.asarray(deflection_history, dtype=np.float64)
    result: dict[str, Any] = {
        "assumption": (
            "independent whitened frame evidence with fixed target identity; "
            "Deflection is additive across the selected window"
        ),
        "windows": {},
    }
    for raw_width in sorted(set(int(value) for value in windows)):
        if raw_width < 1 or raw_width > values.shape[0]:
            continue
        pd = rolling_multiframe_detection(values, p_fa, raw_width)
        tail = pd[-min(max(1, int(tail_window)), len(pd)):]
        target_mean = np.mean(tail, axis=0)
        ordered = np.sort(target_mean)
        frame_metrics = frame_qos_metrics(pd)
        frame_pass = (
            (frame_metrics["steady"] >= floors[0])
            & (frame_metrics["weak3"] >= floors[1])
            & (frame_metrics["worst"] >= floors[2])
        )
        row: dict[str, Any] = {
            "accumulated_frames": raw_width,
            "valid_windows": int(len(pd)),
            "steady": float(np.mean(target_mean)),
            "weak3": float(np.mean(ordered[:min(3, len(ordered))])),
            "worst": float(ordered[0]),
            "qos_success": bool(
                np.mean(target_mean) >= floors[0]
                and np.mean(ordered[:min(3, len(ordered))]) >= floors[1]
                and ordered[0] >= floors[2]
            ),
            "rolling_qos_rate": float(np.mean(frame_pass)),
        }
        if frame_duration_s is not None:
            duration = float(frame_duration_s)
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError("frame_duration_s must be finite and positive")
            row["observation_time_s"] = raw_width * duration
        result["windows"][str(raw_width)] = row
    return result
