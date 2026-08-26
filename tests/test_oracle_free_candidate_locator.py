import inspect

import numpy as np

from uav_isac.coordination.oracle_free_candidate_locator import (
    _joint_structural_repair_grammar,
    _structural_rewrite_options,
    generate_oracle_free_candidate_pools,
    structure_digest,
    structure_state,
)


def _inputs():
    K, Q = 4, 2
    coefficient = np.ones((K, K, Q), dtype=np.float64)
    coefficient[np.arange(K), np.arange(K), :] = 0.0
    coefficient[:, :, 1] *= 2.0
    edges = ((0, 2, 0), (0, 3, 1))
    role = np.asarray([0, -1, 1, 1], dtype=np.int8)
    return coefficient, np.ones(K), edges, role


def test_structure_state_enforces_unique_owner_and_single_role():
    assert structure_state(((0, 2, 0), (1, 3, 0)), 4, 1) is None
    assert structure_state(((0, 2, 0), (2, 3, 1)), 4, 2) is None
    valid = structure_state(((0, 2, 0), (1, 3, 1)), 4, 2)
    assert valid is not None
    assert valid[1] == (0, 0, 1, 1)


def test_locator_api_has_no_exact_oracle_or_witness_input():
    names = set(inspect.signature(
        generate_oracle_free_candidate_pools).parameters)
    forbidden = {"witness", "oracle", "closure", "milp", "optimum"}
    assert not names & forbidden


def test_candidate_generation_is_deterministic_nested_and_hard_legal():
    coefficient, budget, edges, role = _inputs()
    kwargs = dict(
        target_pair_limit=2,
        reports_per_receiver=4,
        nested_budgets=(1, 2),
    )
    first = generate_oracle_free_candidate_pools(
        coefficient, budget, edges, role,
        np.asarray([1.0, 2.0]), np.asarray([0.2, 0.1, 0.0, 0.0]),
        **kwargs)
    second = generate_oracle_free_candidate_pools(
        coefficient, budget, edges, role,
        np.asarray([1.0, 2.0]), np.asarray([0.2, 0.1, 0.0, 0.0]),
        **kwargs)
    assert first == second
    assert structure_digest(edges, 4, 2) in {item.digest for item in first}
    assert any(item.pool == "A_DUAL_NESTED" for item in first)
    assert any(item.pool == "B_ROLE_GRAMMAR" for item in first)
    for item in first:
        assert structure_state(item.edges, 4, 2) is not None
        counts = np.bincount([edge[2] for edge in item.edges], minlength=2)
        assert np.all(counts <= 2)


def test_structural_grammar_keeps_minimal_addition_and_receiver_promotion():
    coefficient = np.ones((4, 4, 2), dtype=np.float64)
    coefficient[np.arange(4), np.arange(4), :] = 0.0
    coefficient[1, 3, 0] = 9.0
    coefficient[3, 2, 1] = 12.0
    budget = np.ones(4)
    price = np.ones(2)
    scarcity = np.zeros(4)
    additions = _structural_rewrite_options(
        coefficient, budget, price, scarcity, 0, ((0, 3, 0),),
        target_pair_limit=2, per_family_limit=1)
    assert ((0, 3, 0), (1, 3, 0)) in additions
    promotions = _structural_rewrite_options(
        coefficient, budget, price, scarcity, 1, ((0, 3, 1),),
        target_pair_limit=2, per_family_limit=2)
    assert ((3, 2, 1),) in promotions


def test_joint_grammar_expresses_role_flip_plus_cross_target_augmentation():
    K, Q = 4, 2
    coefficient = np.ones((K, K, Q), dtype=np.float64)
    coefficient[np.arange(K), np.arange(K), :] = 0.0
    coefficient[2, 3, 0] = 20.0
    coefficient[1, 3, 1] = 18.0
    deployed = ((0, 2, 0), (0, 3, 1))
    proposals = _joint_structural_repair_grammar(
        coefficient, np.ones(K), np.ones(Q), np.zeros(K), deployed,
        (0, 1), target_pair_limit=2, per_family_limit=2, beam_width=128)
    desired = ((0, 3, 1), (1, 3, 1), (2, 3, 0))
    assert desired in proposals
    state = structure_state(desired, K, Q)
    assert state is not None
    assert state[1] == (0, 0, 0, 1)


def test_endpoint_conditioning_preserves_shared_participant_economy():
    K, Q = 5, 3
    coefficient = np.ones((K, K, Q), dtype=np.float64)
    coefficient[np.arange(K), np.arange(K), :] = 0.0
    # Node 1 is only the third-best local addition, but reusing it for all
    # three targets minimizes the union of changed participants.
    coefficient[2, 4, :] = 9.0
    coefficient[3, 4, :] = 8.0
    coefficient[1, 4, :] = 7.0
    deployed = tuple((0, 4, q) for q in range(Q))
    proposals = _joint_structural_repair_grammar(
        coefficient, np.ones(K), np.ones(Q), np.zeros(K), deployed,
        (0, 1, 2), target_pair_limit=2, per_family_limit=2,
        beam_width=64, required_participant=1)
    shared = tuple(sorted(
        edge for q in range(Q) for edge in ((0, 4, q), (1, 4, q))))
    assert shared in proposals
    for proposal in proposals:
        toggled = set(proposal).symmetric_difference(deployed)
        assert 1 in {node for edge in toggled for node in edge[:2]}

    # One DP scan over all targets may leave target 2 unchanged while emitting
    # the radius-2 shared-endpoint repair for targets 0 and 1.
    radius_two = _joint_structural_repair_grammar(
        coefficient, np.ones(K), np.ones(Q), np.zeros(K), deployed,
        (0, 1, 2), target_pair_limit=2, per_family_limit=2,
        beam_width=128, required_participant=1,
        allow_unchanged_targets=True,
        min_changed_targets=2, max_changed_targets=2)
    partial = tuple(sorted(
        list((0, 4, q) for q in range(Q))
        + [(1, 4, 0), (1, 4, 1)]))
    assert partial in radius_two
    assert all(len({
        edge[2] for edge in set(item).symmetric_difference(deployed)
    }) == 2 for item in radius_two)
