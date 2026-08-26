import json

import numpy as np
import pytest

from uav_isac.evaluation.layer_provenance import (
    LAYER_TRACE_SCHEMA_VERSION,
    audit_layer_trace_provenance,
    build_layer_trace_provenance,
    canonical_json_sha256,
    sha256_state_dict,
)


def _valid_trace(frame_count=2):
    physics = {"c_det": 1.0, "n_cpi": 1, "l_eff": 1}
    provenance = build_layer_trace_provenance(
        run_binding={
            "code_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "checkpoint_sha256": "3" * 64,
        },
        physics_contract=physics,
    )
    K, Q = 2, 1
    return {
        "schema_version": np.asarray([LAYER_TRACE_SCHEMA_VERSION]),
        "episode": np.zeros(frame_count, dtype=np.int32),
        "seed": np.ones(frame_count, dtype=np.int64),
        "frame": np.arange(frame_count),
        "uav_positions": np.zeros((frame_count, K, 3)),
        "uav_velocities": np.zeros((frame_count, K, 3)),
        "target_positions": np.zeros((frame_count, Q, 3)),
        "target_velocities": np.zeros((frame_count, Q, 3)),
        "deployed_selected": np.zeros((frame_count, K, K, Q)),
        "deployed_role": np.zeros((frame_count, K)),
        "deployed_receiver_owner": np.zeros((frame_count, K, Q)),
        "deployed_sensing_power_w": np.full((frame_count, K, Q), 0.01),
        "deployed_comm_power_w": np.full((frame_count, K), 0.1),
        "per_watt_coefficient": np.ones((frame_count, K, K, Q)),
        "privileged_g_dd": np.ones((frame_count, K, K, Q)),
        "privileged_chi_rep": np.ones((frame_count, K, K, Q)),
        "physical_pd": np.full((frame_count, Q), 0.8),
        "task_mean_pd": np.full(frame_count, 0.8),
        "task_weak3_pd": np.full(frame_count, 0.8),
        "task_worst_pd": np.full(frame_count, 0.8),
        "frame_sensing_budget_w": np.full((frame_count, K), 0.9),
        "provenance_json": np.asarray([
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        ]),
    }


def test_layer_trace_provenance_accepts_complete_current_trace():
    trace = _valid_trace()
    result = audit_layer_trace_provenance(trace)
    assert result["gate"] == "PASS"
    assert result["frame_count"] == 2


def test_layer_trace_provenance_rejects_legacy_missing_power():
    trace = _valid_trace()
    del trace["deployed_sensing_power_w"]
    with pytest.raises(ValueError, match="deployed_sensing_power_w"):
        audit_layer_trace_provenance(trace)


def test_layer_trace_provenance_rejects_mutated_physics_payload():
    trace = _valid_trace()
    provenance = json.loads(str(trace["provenance_json"][0]))
    provenance["physics_contract"]["c_det"] = 2.0
    trace["provenance_json"] = np.asarray([json.dumps(provenance)])
    with pytest.raises(ValueError, match="physics contract hash"):
        audit_layer_trace_provenance(trace)


def test_layer_trace_provenance_rejects_cross_model_physics():
    trace = _valid_trace()
    wrong = canonical_json_sha256({"different": True})
    with pytest.raises(ValueError, match="requested model"):
        audit_layer_trace_provenance(
            trace, expected_physics_sha256=wrong)


def test_state_dict_hash_is_order_independent_and_value_sensitive():
    first = {"b": np.asarray([2.0]), "a": np.asarray([1.0])}
    second = {"a": np.asarray([1.0]), "b": np.asarray([2.0])}
    changed = {"a": np.asarray([1.0]), "b": np.asarray([3.0])}
    assert sha256_state_dict(first) == sha256_state_dict(second)
    assert sha256_state_dict(first) != sha256_state_dict(changed)
