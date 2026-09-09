from tools.benchmark_fixed_lag_smoother import benchmark


def test_fixed_lag_mechanism_benchmark_is_deterministic_and_scoped():
    first = benchmark(seed_count=4)
    second = benchmark(seed_count=4)
    assert first == second
    assert first["scope"] == "mechanism_screen_not_model_selection_evidence"
    assert first["current_l4_acceleration_model"] == "white"
    assert first["white"]["smoothing_history_mse_ratio"] < 1.0
    assert first["persistent"]["smoothing_history_mse_ratio"] < 1.0
