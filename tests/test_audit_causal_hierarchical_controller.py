from types import SimpleNamespace

from tools.audit_causal_hierarchical_controller import (
    _planner_exact_invariants,
    audit,
)


def test_causal_hierarchical_audit_entrypoint_is_importable():
    assert callable(audit)


def test_planner_invariants_do_not_overclaim_peer_set_for_v1():
    v1 = SimpleNamespace(metadata=SimpleNamespace(
        architecture="shared-uav-target-set-invariant-zoh-residual-v1"))
    v2 = SimpleNamespace(metadata=SimpleNamespace(
        architecture=(
            "shared-uav-peer-target-double-set-equivariant-zoh-residual-v2")))

    assert "peer_uav_set_aggregation" not in _planner_exact_invariants(v1)
    assert "peer_uav_set_aggregation" in _planner_exact_invariants(v2)
