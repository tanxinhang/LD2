"""Shared type aliases and NamedTuples for the UAV-ISAC system."""

from typing import NamedTuple, Optional, List, Tuple
import numpy as np


class Position3D(NamedTuple):
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Position3D":
        return Position3D(float(arr[0]), float(arr[1]), float(arr[2]))


class UAVState(NamedTuple):
    """Full state of one UAV at a given frame."""
    pos: np.ndarray          # (3,) position [x, y, z]
    vel: np.ndarray          # (3,) velocity [vx, vy, vz]
    battery: float           # remaining energy (J)
    role: int                # 0=tx, 1=rx, 2=idle


class TargetState(NamedTuple):
    """True state of one target at a given frame."""
    pos: np.ndarray          # (3,) position [x, y, z]
    vel: np.ndarray          # (3,) velocity [vx, vy, vz]


class BeliefState(NamedTuple):
    """Belief (prior) about one target held by one UAV."""
    mean: np.ndarray         # (4,) estimated [x, y, vx, vy]
    cov_diag: np.ndarray     # (4,) diagonal of covariance matrix
    aoi: int                 # age of information (frames since last observation)


class Action(NamedTuple):
    """Action for one UAV: trajectory increment and role."""
    delta_p: np.ndarray     # (2,) position increment [dx, dy] (z fixed)
    role: int               # 0=tx, 1=rx, 2=idle

    @staticmethod
    def create(delta_p: np.ndarray, role: int) -> "Action":
        return Action(delta_p=np.asarray(delta_p, dtype=np.float64), role=int(role))


class DeflectionEntry(NamedTuple):
    """Single bistatic pair's deflection data."""
    i: int              # tx UAV index
    j: int              # rx UAV index
    q: int              # target index
    tau: float          # delay (s)
    nu: float           # Doppler shift (Hz)
    alpha: float        # path gain (linear)
    d_raw: float        # raw Deflection d_ijq
    g_dd: float         # DD effectiveness (0..1)
    chi_rep: float      # reporting link reliability
    d_eff: float        # effective Deflection = chi_rep * d_raw if g_dd >= g_min else 0


class P0Solution(NamedTuple):
    """Result of the P0 inner solver."""
    z_selected: np.ndarray       # (K, K, Q) binary selection matrix
    D_q_star: np.ndarray         # (Q,) final cumulative Deflection per target
    U_q: np.ndarray              # (Q,) utility per target
    selected_set: List[Tuple[int, int, int]]  # (tx, rx, target) selected entries
    total_bits: float
    total_latency: float
