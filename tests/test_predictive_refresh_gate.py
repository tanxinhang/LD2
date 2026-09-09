from uav_isac.prediction.refresh_gate import (
    PredictiveRefreshGate,
    RefreshPath,
)


def test_hold_frame_never_uses_prediction_to_change_indices():
    gate = PredictiveRefreshGate(hold_period=5)
    incumbent = [(0, 1, 0)]
    decision = gate.decide(4, incumbent, [(2, 1, 0)])
    assert decision.path == RefreshPath.HOLD_SELECTED_ONLY
    assert decision.candidate_edges == tuple(incumbent)


def test_boundary_fails_closed_without_all_analytical_gates():
    gate = PredictiveRefreshGate(hold_period=5)
    decision = gate.decide(
        5,
        [(0, 1, 0)],
        [(2, 1, 0)],
        selected_physics_valid=True,
        omitted_edge_bound_valid=False,
        solver_residual_valid=True,
    )
    assert decision.path == RefreshPath.FULL_EXACT_FALLBACK


def test_boundary_accepts_sparse_refresh_only_when_fully_certified():
    gate = PredictiveRefreshGate(hold_period=5)
    decision = gate.decide(
        10,
        [(0, 1, 0)],
        [(2, 1, 0)],
        selected_physics_valid=True,
        omitted_edge_bound_valid=True,
        solver_residual_valid=True,
    )
    assert decision.path == RefreshPath.PREDICTED_SPARSE_REFRESH
    assert len(decision.candidate_edges) == 2
