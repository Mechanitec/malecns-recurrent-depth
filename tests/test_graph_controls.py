import numpy as np

from malecns_rd.graph import ConnectomeGraph
from malecns_rd.graph_controls import degree_preserving_edge_swap_v2, graph_invariants


def test_degree_preserving_edge_swap_keeps_exact_degree_sequences():
    graph = ConnectomeGraph.from_edges(
        6,
        np.array([0, 0, 1, 1, 2, 2, 3, 4]),
        np.array([1, 2, 2, 3, 3, 4, 4, 5]),
        np.array([1, -2, 3, 4, -1, 2, 5, -3], dtype=np.float32),
        normalize_incoming=False,
    )
    shuffled = degree_preserving_edge_swap_v2(graph, seed=23, swaps=3)
    before = graph_invariants(graph)
    after = graph_invariants(shuffled)
    assert after["n_edges"] == before["n_edges"]
    assert after["in_degree"] == before["in_degree"]
    assert after["out_degree"] == before["out_degree"]
    assert after["weight_sum"] == before["weight_sum"]
    assert after["absolute_weight_sum"] == before["absolute_weight_sum"]
    assert after["self_loops"] == before["self_loops"]
