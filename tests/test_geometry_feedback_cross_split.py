import json

from tools.audit_geometry_feedback_cross_split import audit


def _row(seed, *, total_delay):
    return {
        "seed": seed,
        "frame": 1,
        "deployed_worst": 0.1,
        "causal_routed_available": True,
        "causal_routed_estimated_worst": 0.9,
        "causal_routed_worst": 0.9,
        "causal_routed_noop_estimated_worst": 0.1,
        "causal_routed_calibration_score": 0.0,
        "causal_routed_noop_calibration_score": 0.0,
        "causal_routed_power_total_variation": 0.0,
        "transport_feasible": True,
        "transport_total_bits": 0,
        "transport_max_packet_latency_s": 0.001,
        "transport_total_protocol_latency_s": total_delay,
        "transport_total_energy_j": 0.0,
    }


def test_net_benefit_charges_full_decision_delay_not_max_packet(tmp_path):
    calibration = {
        "seed_order": [1],
        "rows": [_row(1, total_delay=0.001)],
        "structure_sequence_transport_enabled": True,
        "owner_proposal_transport_enabled": True,
    }
    validation = {
        "seed_order": [2],
        "rows": [_row(2, total_delay=0.9)],
        "structure_sequence_transport_enabled": True,
        "owner_proposal_transport_enabled": True,
    }
    calibration_path = tmp_path / "calibration.json"
    validation_path = tmp_path / "validation.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    result = audit(
        calibration_path,
        validation_path,
        alpha=0.5,
        qos_floor=0.6,
        switch_weight=0.0,
        bit_weight=0.0,
        payload_bits=0,
        delay_weight=1.0,
        energy_weight=0.0,
        candidate_prefix="causal_routed",
    )
    record = result["records"][0]
    assert record["transport_max_packet_latency_s"] == 0.001
    assert record["transport_decision_latency_s"] == 0.9
    assert record["delay_cost"] == 0.9
    assert not record["accept"]
    assert result["cost_model"]["delay_cost_uses_full_decision_protocol"]
