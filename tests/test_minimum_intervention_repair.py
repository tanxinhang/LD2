import numpy as np

from uav_isac.coordination.minimum_intervention_repair import (
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.safe_structural_reduction import (
    context_free_zero_gain_screen,
)
from uav_isac.coordination.pwl_pd import (
    evaluate_pwl,
    p_d,
    saturating_chord_lower_bound,
)


def test_saturating_chord_remains_a_lower_bound_over_large_dynamic_range():
    slopes, intercepts, breakpoints = saturating_chord_lower_bound(
        0.001, 11.0, 1.0e5, 1.0e-3)
    grid = np.geomspace(11.0, 1.0e5, 2000)
    lower = evaluate_pwl(grid, slopes, intercepts)
    assert len(breakpoints) > 2
    assert np.all(lower <= p_d(grid, 0.001) + 1.0e-12)
    assert lower[-1] >= 0.999 - 1.0e-12


def test_minimum_intervention_keeps_feasible_structure_unchanged():
    gain = np.zeros((4, 4, 2))
    gain[0, 2, 0] = 20.0
    gain[1, 3, 1] = 20.0
    selected = np.zeros_like(gain, dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 3, 1] = True
    result = solve_minimum_intervention_task_repair_milp(
        gain, np.ones(4), selected,
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([0, 0, 1, 1], dtype=bool), np.array([2, 3]),
        p_fa=0.1, task_floors=(0.6, 0.6, 0.6, 1),
        target_pair_limit=1, reports_per_receiver=1)
    assert result.feasible
    assert result.closure_cardinality == 0
    assert result.changed_roles == result.changed_owners == result.toggled_edges == 0


def test_role_flip_dependency_closure_cannot_hide_edge_rewrite():
    gain = np.zeros((3, 3, 2))
    gain[0, 1, 0] = 20.0
    gain[0, 2, 1] = 20.0
    selected = np.zeros_like(gain, dtype=bool)
    selected[1, 0, 0] = True  # infeasible old geometry; both endpoints must change role
    result = solve_minimum_intervention_task_repair_milp(
        gain, np.ones(3), selected,
        np.array([0, 1, 0], dtype=bool),
        np.array([1, 0, 1], dtype=bool), np.array([0, 2]),
        p_fa=0.1, task_floors=(0.6, 0.6, 0.6, 1),
        target_pair_limit=1, reports_per_receiver=2)
    assert result.feasible
    assert result.changed_roles >= 2
    assert result.toggled_edges >= 2
    assert set(result.participants) >= {0, 1}
    assert 0 in result.affected_targets


def test_three_floor_average_can_require_more_than_worst_floor():
    gain = np.zeros((4, 4, 3))
    for q in range(3):
        gain[0, 2, q] = 30.0 if q > 0 else 12.0
    selected = gain > 0
    result = solve_minimum_intervention_task_repair_milp(
        gain, np.array([1.0, 0.0, 0.0, 0.0]), selected,
        np.array([1, 0, 0, 0], dtype=bool),
        np.array([0, 0, 1, 0], dtype=bool), np.array([2, 2, 2]),
        p_fa=0.1, task_floors=(0.55, 0.60, 0.75, 2),
        target_pair_limit=1, reports_per_receiver=3)
    assert result.feasible
    assert np.min(result.target_pd) >= 0.55 - 1e-7
    assert np.mean(result.target_pd) >= 0.75 - 1e-7


def test_context_free_screen_separates_feasibility_from_lex_optimality():
    gain = np.zeros((4, 4, 1))
    gain[0, 2, 0] = 20.0
    selected = np.zeros_like(gain, dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 2, 0] = True  # Zero gain, but retaining it has zero repair cost.
    screen = context_free_zero_gain_screen(gain, selected)
    assert (1, 2, 0) in screen.feasibility_preserving
    assert (1, 2, 0) in screen.selected_zero_gain
    assert (1, 2, 0) not in screen.optimality_preserving

    kwargs = dict(
        p_fa=0.1, task_floors=(0.6, 0.6, 0.6, 1),
        target_pair_limit=2, reports_per_receiver=2,
    )
    baseline = solve_minimum_intervention_task_repair_milp(
        gain, np.ones(4), selected,
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([0, 0, 1, 0], dtype=bool), np.array([2]), **kwargs)
    d_f = solve_minimum_intervention_task_repair_milp(
        gain, np.ones(4), selected,
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([0, 0, 1, 0], dtype=bool), np.array([2]),
        forbidden_edges=screen.feasibility_preserving, **kwargs)
    assert baseline.feasible and d_f.feasible
    assert baseline.closure_cardinality == 0
    assert d_f.closure_cardinality > baseline.closure_cardinality


def test_context_free_optimality_screen_preserves_lex_solution():
    gain = np.zeros((4, 4, 1))
    gain[0, 2, 0] = 20.0
    selected = np.zeros_like(gain, dtype=bool)
    selected[0, 2, 0] = True
    screen = context_free_zero_gain_screen(gain, selected)
    assert (0, 2, 0) not in screen.feasibility_preserving
    baseline = solve_minimum_intervention_task_repair_milp(
        gain, np.ones(4), selected,
        np.array([1, 0, 0, 0], dtype=bool),
        np.array([0, 0, 1, 0], dtype=bool), np.array([2]),
        p_fa=0.1, task_floors=(0.6, 0.6, 0.6, 1),
        target_pair_limit=1, reports_per_receiver=1)
    screened = solve_minimum_intervention_task_repair_milp(
        gain, np.ones(4), selected,
        np.array([1, 0, 0, 0], dtype=bool),
        np.array([0, 0, 1, 0], dtype=bool), np.array([2]),
        p_fa=0.1, task_floors=(0.6, 0.6, 0.6, 1),
        target_pair_limit=1, reports_per_receiver=1,
        forbidden_edges=screen.optimality_preserving)
    assert baseline.feasible and screened.feasible
    assert (screened.closure_cardinality, screened.prepare_bits) == (
        baseline.closure_cardinality, baseline.prepare_bits)
    assert np.isclose(
        screened.total_sensing_power_w, baseline.total_sensing_power_w,
        rtol=1.0e-7, atol=1.0e-9)
