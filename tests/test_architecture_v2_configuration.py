from uav_isac.adapters import load_resolved_configuration
from uav_isac.governance import canonical_sha256


def test_default_configuration_has_one_resolved_identity():
    implicit = load_resolved_configuration()
    explicit = load_resolved_configuration("config/default.yaml")

    assert implicit.sha256 == explicit.sha256
    assert implicit.payload == explicit.payload
    assert len(implicit.sha256) == 64


def test_resolved_hash_covers_nested_runtime_values():
    resolved = load_resolved_configuration()
    changed = dict(resolved.payload)
    changed["scenario"] = dict(changed["scenario"])
    changed["scenario"]["dt"] = changed["scenario"]["dt"] + 0.01

    assert canonical_sha256(changed) != resolved.sha256

