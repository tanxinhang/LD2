from tools.benchmark_markov_graph_assignment import benchmark


def test_markov_graph_microbenchmark_is_finite_and_rejects_harm():
    result = benchmark(
        seed=20260910, cases=2, cardinality=4,
        neighbors=2, blind_candidates=7)
    assert result["cases"] == 2
    assert 0.0 <= result["graph_acceptance_rate"] <= 1.0
    assert result["graph_mean_evaluated_assignments"] <= 7.0
    assert result["blind_mean_evaluated_candidates"] <= 7.0
    assert len(result["paired_improvement_delta_bootstrap95"]) == 2
    assert (
        result["paired_graph_wins"] + result["paired_ties"]
        + result["paired_graph_losses"] == 2)
    for row in result["rows"]:
        assert row["graph_accepted_cost"] <= row["incumbent_cost"] + 1.0e-12
        assert row["blind_best_cost"] <= row["incumbent_cost"] + 1.0e-12
