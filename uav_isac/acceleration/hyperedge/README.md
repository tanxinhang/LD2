# Hyperedge acceleration service

This package isolates the hot coefficient-reconstruction path from the
environment.  The built-in `numpy` backend preserves the audited physical
semantics and is the reference implementation.

## Direct use

```python
from uav_isac.acceleration.hyperedge import (
    create_hyperedge_acceleration_service,
)

service = create_hyperedge_acceleration_service("numpy")
coefficients = service.reconstruct_selected(
    positions_xy_by_target,
    velocities_xy_by_target,
    target_positions_m,
    target_velocities_mps,
    seen_mask,
    selected_edges,
    **physical_parameters,
)
print(service.stats())
```

When the same frozen COO edge set is refreshed for several viewer-local public
states, call `reconstruct_selected_batch`.  Its public-state arrays have a
leading viewer dimension and its nominal/lower/upper outputs have shape
`(V, E)`.  This is the production hold-frame path and avoids repeating Python
setup once per viewer.

The default service is a zero-overhead direct-kernel path.  For diagnostics or
microbenchmarks, instantiate `NumpyHyperedgeAccelerationService` directly with
`collect_timing=True`; this enables call counters and timing without
contaminating normal simulation latency.

`reconstruct_dense`, `reconstruct_upper`, `reconstruct_selected`, and
`reconstruct_selected_batch` accept the same arguments as the corresponding
audited functions in
`uav_isac.coordination.hyperedge`.

## Replace the backend

A backend must implement `HyperedgeAccelerationService`.  It can be installed
under a short in-process name:

```python
from uav_isac.acceleration.hyperedge import (
    register_hyperedge_acceleration_backend,
)

register_hyperedge_acceleration_backend("numba", NumbaHyperedgeService)
```

or loaded directly from configuration with a zero-argument factory:

```yaml
marl:
  hyperedge_acceleration_backend: my_package.backends:create_service
```

No environment code needs to change.  A replacement must preserve:

- `(K, K, Q)` dense shapes, `(V,E)` batch shape, and selected-edge ordering;
- fail-closed visibility masking and finite `float64` outputs;
- nominal/lower/upper physical definitions and conservative bounds;
- deterministic results for identical inputs;
- public-observation-only inputs (no hidden simulator truth).

Use `numpy` as the numerical oracle when qualifying a new backend.  Compare all
three outputs, run QoS/worst-case regression, and report service timing from
`stats()` before promoting it.
