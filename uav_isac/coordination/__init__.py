"""Distributed coordination mechanisms.

Public API (deployment execution path)
--------------------------------------
The names re-exported below are the coordination components that are wired
into the deployed execution path (env_core / trainer) or are active
certified-control components.  Everything else in this package is an
audit/research-only module (banner-marked) consumed by tools/ scripts and
tests only; do not treat it as deployed behaviour.  See
docs/EXPERIMENT_LOG.md and docs/CURRENT_SYSTEM_MODEL.md.
"""

# Deployment-path power allocation (L1): fixed-structure max-min sensing-power
# LP used by env_core.step().
from uav_isac.coordination.maxmin_power import (
    NonUniqueFixedOwnerStructureError,
    solve_fixed_structure_maxmin_power_lp,
)

# Deployment-path structure pairing primitives used by env_core P0 / local
# exchange maintenance.
from uav_isac.coordination.hyperedge import (
    plan_local_hyperedges,
)

from uav_isac.coordination.qpd import (
    update_virtual_queue,
)

# Active certified-control components (D0.7-D0.9 chain, default-off until
# event-level calibration completes; see docs/CURRENT_SYSTEM_MODEL.md).
from uav_isac.coordination.dependency_commit import (
    DependencyCommitCertificate,
    dependency_closure,
)

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    assert_local_feasible,
    rebuild_structure,
)

__all__ = [
    "NonUniqueFixedOwnerStructureError",
    "solve_fixed_structure_maxmin_power_lp",
    "plan_local_hyperedges",
    "update_virtual_queue",
    "DependencyCommitCertificate",
    "dependency_closure",
    "LocalMove",
    "assert_local_feasible",
    "rebuild_structure",
]
