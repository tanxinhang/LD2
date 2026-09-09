# Dense deflection materialization service

This service converts `DenseDeflection` tensors to the legacy
`DeflectionEntry` list only at interfaces that still require objects.

```python
from uav_isac.acceleration import create_deflection_materialization_service

service = create_deflection_materialization_service("numpy")
entries = service.materialize(dense, power_scale_w)
```

Register a replacement with
`register_deflection_materialization_backend("name", factory)`, or configure a
zero-argument external factory:

```yaml
marl:
  deflection_materialization_backend: my_package.backends:create_service
```

A replacement must preserve canonical `i -> j -> q` ordering, diagonal
exclusion, valid-mask filtering, Python scalar values, linear power scaling,
and exact `DeflectionEntry` fields.
