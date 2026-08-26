import networkx as nx
import numpy as np

from uav_isac.coordination.interaction_width import (
    exact_treewidth,
    floor_capable_task_modes,
    interaction_graphs,
)


def test_exact_treewidth_on_canonical_graphs():
    assert exact_treewidth(nx.path_graph(7)).width == 1
    assert exact_treewidth(nx.cycle_graph(7)).width == 2
    assert exact_treewidth(nx.complete_graph(6)).width == 5
    assert exact_treewidth(nx.complete_bipartite_graph(3, 5)).width == 3


def test_floor_capability_pruning_is_a_necessary_ceiling_test():
    coefficient = np.zeros((3, 3, 1))
    coefficient[0, 2, 0] = 10.0
    coefficient[1, 2, 0] = 1.0
    modes = floor_capable_task_modes(
        coefficient, np.asarray([0.1, 0.1, 0.1]), 2, 0.9)
    retained = {(mode.owner, mode.transmitters) for mode in modes[0]}
    assert (2, (0,)) in retained
    assert (2, (1,)) not in retained
    assert (2, (0, 1)) in retained


def test_graph_views_do_not_hide_raw_global_qos_clique():
    coefficient = np.ones((3, 3, 3))
    diagonal = np.arange(3)
    coefficient[diagonal, diagonal, :] = 0.0
    modes = floor_capable_task_modes(
        coefficient, np.ones(3), 1, 0.5)
    graphs = interaction_graphs(3, 3, modes)
    assert exact_treewidth(graphs["raw_target_primal"]).width == 2
    assert set(graphs) == {
        "raw_target_primal", "resource_target_primal",
        "summary_lifted_incidence", "task_mode_clique_primal"}
