"""Abstract base agent interface."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract interface for all agents (MAPPO, IPPO, baselines)."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple:
        """Select action given observation.

        Args:
            obs: Local observation vector
            deterministic: If True, return deterministic (mean/mode) action

        Returns:
            (action, log_prob, value) or (action, log_prob) or (action,)
        """
        pass

    @abstractmethod
    def update(self, rollout_data: Dict) -> Dict[str, float]:
        """Update agent from rollout data.

        Args:
            rollout_data: Dict containing batch data for update

        Returns:
            Dict of training metrics
        """
        pass

    def save(self, path: str) -> None:
        """Save agent parameters. Override in subclasses with trainable params."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement save()"
        )

    def load(self, path: str) -> None:
        """Load agent parameters. Override in subclasses with trainable params."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement load()"
        )
