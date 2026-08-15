import json

from tools.audit_self_normalized_feedback_cross_split import audit


def _row(seed, *, protocol_delay, candidate_true=0.8):
    return {
        "seed": seed,
        "frame": 1,
        "deployed_worst": 0.5,
        "deployed_pd": [0.5, 0.6],
        "causal_routed_available": True,
        "causal_routed_estimated_pd": [0.8, 0.8],
        "causal_routed_realized_pd": [candidate_true, candidate_true],
        "causal_routed_noop_estimated_pd": [0.5, 0.6],
        "causal_routed_absolute_uncertainty": [0.1, 0.1],
        "causal_routed_noop_upper_uncertainty": [0.1, 0.1],
        "causal_routed_power_total_variation": 0.0,
        "transport_feasible": True,
        "transport_total_bits": 0,
        "transport_max_packet_latency_s": 0.001,
        "transport_total_protocol_latency_s": protocol_delay,
        "transport_total_energy_j": 0.0,
    }


def test_cross_split_gate_uses_episode_score_and_full_protocol_delay(tmp_path):
    calibration = {"seed_order": [1], "rows": [_row(1, protocol_delay=0.0)]}
    validation = {"seed_order": [2], "rows": [_row(2, protocol_delay=0.4)]}
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
    )
    record = result["records"][0]
    assert result["normalized_margin"] == 0.0
    assert record["transport_max_packet_latency_s"] == 0.001
    assert record["transport_decision_latency_s"] == 0.4
    assert record["charged_cost"] == 0.4
    assert not record["accept"]
