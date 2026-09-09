"""Small deterministic policies used for migration characterization."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class HoldPositionPolicy:
    """Issue zero movement with one explicit role for every observed UAV."""

    def __init__(self, role: int = 2):
        if role not in (0, 1, 2):
            raise ValueError("role must be 0 (tx), 1 (rx), or 2 (idle)")
        self._role = role

    def actions(
        self,
        observations: Mapping[str, Any],
        frame: int,
    ) -> Mapping[str, Any]:
        del frame
        return {
            agent_id: {
                "delta_p": np.zeros(2, dtype=np.float64),
                "role": self._role,
            }
            for agent_id in observations
        }

