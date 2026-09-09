import numpy as np
import pytest

from uav_isac.acceleration.deflection import (
    DeflectionMaterializationService,
    NumpyDeflectionMaterializationService,
    available_deflection_materialization_backends,
    create_deflection_materialization_service,
    register_deflection_materialization_backend,
)
from uav_isac.physical.deflection import DenseDeflection


def _dense() -> DenseDeflection:
    shape = (3, 3, 2)
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    valid = np.ones(shape, dtype=bool)
    valid[0, 1, 1] = False
    return DenseDeflection(
        tau=base + 0.1,
        nu=base + 0.2,
        alpha=base + 0.3,
        d_raw=base + 0.4,
        g_dd=(base + 1.0) / 20.0,
        d_eff=base + 0.5,
        valid=valid,
    )


def test_numpy_materialization_service_matches_canonical_converter():
    dense = _dense()
    power = np.asarray([
        [0.5, 0.25], [1.0, 0.75], [0.2, 0.1],
    ])
    service = create_deflection_materialization_service("numpy")
    assert isinstance(service, DeflectionMaterializationService)
    assert "numpy" in available_deflection_materialization_backends()
    assert service.materialize(dense, power) == dense.to_entries(power)


def test_materialization_service_registry_and_external_factory():
    register_deflection_materialization_backend(
        "test_numpy_materializer",
        NumpyDeflectionMaterializationService,
        replace=True,
    )
    registered = create_deflection_materialization_service(
        "TEST_NUMPY_MATERIALIZER")
    external = create_deflection_materialization_service(
        "uav_isac.acceleration.deflection:"
        "NumpyDeflectionMaterializationService")
    assert registered.materialize(_dense()) == external.materialize(_dense())


def test_materialization_service_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown deflection materialization"):
        create_deflection_materialization_service("does_not_exist")
