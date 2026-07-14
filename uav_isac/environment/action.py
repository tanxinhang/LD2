"""Action space definition for UAV agents.

Each UAV action: a_k = (delta_p, role)
  - delta_p: continuous 2D position increment [dx, dy], bounded by v_max * dt
  - role: discrete 3-way choice {tx=0, rx=1, idle=2}
"""

import numpy as np
from typing import Dict, Optional
from uav_isac.utils.types import Action


class ActionSpace:
    """Action space for a single UAV agent."""

    def __init__(
        self,
        v_max: float = 25.0,
        dt: float = 0.1,
        num_roles: int = 3,
        rng: Optional[np.random.Generator] = None,
        learn_roles: bool = True,
    ):
        """
        Args:
            v_max: Maximum flight speed (m/s)
            dt: Frame duration (s)
            num_roles: Number of role choices (3: tx, rx, idle)
            rng: Random generator
            learn_roles: If False, the policy does NOT choose roles — the role is a
                placeholder (idle) reassigned by the env's P0 solver, and the role
                term is dropped from log-prob/entropy so the role head leaves the
                objective entirely. Must match env's role mode for a consistent
                old_log_prob == new_log_prob ratio.
        """
        self.learn_roles = learn_roles
        self.v_max = v_max
        self.dt = dt
        self.max_dp = v_max * dt  # Maximum displacement per frame (2.5m at 25 m/s, 0.1s)
        # Squashing scale = max_dp (FULL axial range 2.5m). Feasibility (||Δp||<=max_dp)
        # is enforced by projecting the squashed action onto the disk inside decode(),
        # so the env's radial clamp becomes a no-op -> stored == executed (fixes P1)
        # WITHOUT shrinking the action range. (Inscribed-box max_dp/√2 was too small
        # -> 265m budget < 283m half-diagonal -> policy collapsed to P_FA.)
        self.dp_scale = self.max_dp
        self.num_roles = num_roles
        self.rng = rng if rng is not None else np.random.default_rng()
        self.structured_actor = True  # flag for MAPPOAgent
        self.num_targets = 0  # overridden by run script / MAPPOAgent config

    @property
    def dp_dim(self) -> int:
        return 2

    @property
    def role_dim(self) -> int:
        return self.num_roles

    def sample(self) -> Action:
        """Sample a random action uniformly.

        Returns:
            Random Action
        """
        # Uniform random direction, random speed up to max
        angle = self.rng.uniform(0, 2 * np.pi)
        speed = self.rng.uniform(0, self.max_dp)
        delta_p = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        role = int(self.rng.integers(0, self.num_roles))

        return Action(delta_p=delta_p, role=role)

    def sample_dp(self) -> np.ndarray:
        """Sample only the continuous position increment."""
        angle = self.rng.uniform(0, 2 * np.pi)
        speed = self.rng.uniform(0, self.max_dp)
        return np.array([speed * np.cos(angle), speed * np.sin(angle)])

    def sample_role(self) -> int:
        """Sample only the role uniformly."""
        return int(self.rng.integers(0, self.num_roles))

    def clamp(self, delta_p: np.ndarray) -> np.ndarray:
        """Clamp position increment to valid range.

        Args:
            delta_p: Proposed increment [dx, dy]

        Returns:
            Clamped increment
        """
        norm = np.linalg.norm(delta_p)
        if norm > self.max_dp:
            return delta_p * (self.max_dp / norm)
        return delta_p.copy()

    def encode(self, action: Action) -> np.ndarray:
        """Encode action to flat array: [dx, dy, role_onehot].

        Args:
            action: Action to encode

        Returns:
            Flat array of shape (2 + num_roles,)
        """
        role_onehot = np.zeros(self.num_roles, dtype=np.float64)
        role_onehot[action.role] = 1.0
        return np.concatenate([action.delta_p, role_onehot])

    def decode(
        self,
        dp_mean: np.ndarray,      # (2,) or scalar for mean
        dp_std: np.ndarray,       # (2,) or scalar for std
        role_logits: np.ndarray,  # (3,) logits
        deterministic: bool = False,
        dp_deterministic: Optional[bool] = None,
        role_deterministic: Optional[bool] = None,
    ) -> tuple:
        """Decode network outputs to Action + log_prob.

        The continuous (delta_p) and discrete (role) heads can be made
        deterministic INDEPENDENTLY. This enables the four diagnostic eval
        modes used to attribute eval collapse to the continuous action vs the
        discrete role:

            A: dp_det=True,  role_det=False   (continuous frozen, role sampled)
            B: dp_det=False, role_det=True    (role frozen, continuous sampled)
            C: dp_det=True,  role_det=True    (fully greedy)  == old deterministic=True
            D: dp_det=False, role_det=False   (fully stochastic) == old deterministic=False

        Args:
            dp_mean: Mean of Gaussian policy for delta_p
            dp_std: Std of Gaussian policy for delta_p
            role_logits: Logits for categorical role distribution
            deterministic: Legacy flag; sets BOTH heads when the two
                fine-grained flags are left as None (backward compatible).
            dp_deterministic: If set, overrides `deterministic` for delta_p.
            role_deterministic: If set, overrides `deterministic` for role.

        Returns:
            (Action, log_prob). log_prob is 0.0 only when BOTH heads are
            deterministic (matches old behavior); otherwise it is the policy
            log-prob of the realized action (used only by the stochastic
            training path, harmless for diagnostic eval).
        """
        dp_det = deterministic if dp_deterministic is None else dp_deterministic
        role_det = deterministic if role_deterministic is None else role_deterministic

        # ---- Continuous head (delta_p) ----
        # Stochastic path squashes with tanh then CLIPs to ±0.999 BEFORE scaling
        # so the action is exactly reconstructible via arctanh at update time
        # (no saturation loss); compute_log_prob() uses the IDENTICAL path so
        # old_log_prob == new_log_prob (fixes P5). Deterministic path matches the
        # legacy mean-mode exactly (no pre-clip) to preserve eval reproducibility.
        if dp_det:
            delta_p = np.tanh(dp_mean) * self.dp_scale
        else:
            dp_std_pos = np.exp(np.clip(dp_std, -20, 2))  # ensure positive
            delta_p_raw = self.rng.normal(dp_mean, dp_std_pos)
            dp01 = np.clip(np.tanh(delta_p_raw), -0.999, 0.999)
            delta_p = dp01 * self.dp_scale
        n = np.linalg.norm(delta_p)
        if n > self.max_dp:
            delta_p = delta_p * (self.max_dp / n)   # project onto disk -> env clamp no-op (P1)

        # ---- Discrete head (role) ----
        if not self.learn_roles:
            # Policy does not choose roles; env's P0 solver assigns them.
            role = self.num_roles - 1  # idle placeholder (overwritten by env)
        elif role_det:
            role = int(np.argmax(role_logits))
        else:
            logits = role_logits - np.max(role_logits)  # stability
            probs = np.exp(logits) / np.sum(np.exp(logits))
            role = int(self.rng.choice(self.num_roles, p=probs))

        action = Action(delta_p=delta_p, role=role)

        if dp_det and (role_det or not self.learn_roles):
            # No stochastic component used in the objective: log_prob not meaningful.
            return action, 0.0

        # Same reconstruction path as evaluate_actions / compute_log_prob:
        log_prob = self.compute_log_prob(action, dp_mean, dp_std, role_logits)
        return action, float(log_prob)

    def decode_deterministic(self, dp_mean: np.ndarray, role_logits: np.ndarray) -> Action:
        """Decode deterministically (for evaluation)."""
        delta_p = np.tanh(dp_mean) * self.dp_scale
        n = np.linalg.norm(delta_p)
        if n > self.max_dp:
            delta_p = delta_p * (self.max_dp / n)   # project onto disk
        role = int(np.argmax(role_logits))
        return Action(delta_p=delta_p, role=role)

    def compute_log_prob(self, action: Action, dp_mean: np.ndarray,
                         dp_std: np.ndarray, role_logits: np.ndarray) -> float:
        """Compute log probability of a given action under the policy.

        Args:
            action: The action taken
            dp_mean, dp_std: Gaussian parameters
            role_logits: Categorical logits

        Returns:
            Log probability (scalar)
        """
        # Inverse tanh: atanh(Δp / dp_scale)  (must use the SAME scale as decode)
        dp_norm = action.delta_p / max(self.dp_scale, 1e-10)
        dp_norm = np.clip(dp_norm, -0.999, 0.999)
        delta_p_raw = np.arctanh(dp_norm)

        dp_std_pos = np.exp(np.clip(dp_std, -20, 2))

        from scipy.stats import norm
        log_prob_dp = np.sum(norm.logpdf(delta_p_raw, dp_mean, dp_std_pos))
        log_prob_dp -= np.sum(np.log(1.0 - dp_norm ** 2 + 1e-6))

        if not self.learn_roles:
            # Role is not a policy decision -> excluded from the objective.
            return float(log_prob_dp)

        # Role log prob
        logits = role_logits - np.max(role_logits)
        log_probs = logits - np.log(np.sum(np.exp(logits)))
        log_prob_role = log_probs[action.role]

        return float(log_prob_dp + log_prob_role)

    def compute_entropy(self, dp_std: np.ndarray, role_logits: np.ndarray) -> float:
        """Compute policy entropy.

        Args:
            dp_std: Log std for delta_p
            role_logits: Logits for role

        Returns:
            Total entropy (scalar)
        """
        # Gaussian entropy: 0.5 * sum(log(2*pi*e*sigma^2))
        dp_std_pos = np.exp(np.clip(dp_std, -20, 2))
        entropy_dp = np.sum(0.5 * np.log(2 * np.pi * np.e * dp_std_pos ** 2))

        if not self.learn_roles:
            return float(entropy_dp)

        # Categorical entropy: -sum(p * log(p))
        logits = role_logits - np.max(role_logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        entropy_role = -np.sum(probs * np.log(probs + 1e-10))

        return float(entropy_dp + entropy_role)
