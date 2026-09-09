# Acceleration services

`uav_isac.acceleration` contains replaceable services for simulation hot paths.
The environment depends on protocols rather than concrete acceleration
libraries, so NumPy, Numba, CuPy, C++ or other implementations can be qualified
and swapped without editing control logic.

| Service | Default | Responsibility | Configuration |
|---|---|---|---|
| `hyperedge` | `numpy` | dense/upper/selected physical coefficient reconstruction | `marl.hyperedge_acceleration_backend` |
| `deflection` | `numpy` | `DenseDeflection` to canonical `DeflectionEntry` materialization | `marl.deflection_materialization_backend` |

Both registries accept a short registered name or a `module:factory` string.
Factories take no arguments and return an object satisfying the corresponding
runtime-checkable protocol.

```python
from uav_isac.acceleration import (
    create_deflection_materialization_service,
    create_hyperedge_acceleration_service,
)

hyperedge = create_hyperedge_acceleration_service("numpy")
materializer = create_deflection_materialization_service("numpy")
```

The NumPy implementations are the numerical oracle.  A replacement is not
eligible for promotion until it passes exact/near-exact kernel comparison,
fixed-seed trajectory comparison, QoS/worst-case gates, and latency reporting.
Backend-specific contracts and examples are documented in the two subpackages.
