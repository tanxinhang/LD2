import pytest

from uav_isac.adapters import load_registered_configuration
from uav_isac.governance import get_runtime_profile, load_runtime_profiles


def test_only_explicit_profiles_are_active_v2_inputs():
    profiles = load_runtime_profiles()
    assert {item.name for item in profiles} == {"default_dev", "strict_k16q16"}
    assert get_runtime_profile("strict_k16q16").formal
    assert len(load_registered_configuration("strict_k16q16").sha256) == 64


def test_arbitrary_config_name_is_rejected():
    with pytest.raises(KeyError, match="unregistered"):
        load_registered_configuration("exp_temporal_feasible_structure_smoke")

