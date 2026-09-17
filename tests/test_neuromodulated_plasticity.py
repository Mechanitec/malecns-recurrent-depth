import numpy as np

from malecns_rd.graph import ConnectomeGraph
from malecns_rd.neuromodulated_plasticity import (
    PlasticityConfig, apply_to_graph, load_plasticity_checkpoint,
    save_plasticity_checkpoint, select_kc_mbon_edges,
)


def make_state():
    graph = ConnectomeGraph.from_edges(
        4,
        np.array([0, 1, 0, 1]),
        np.array([2, 2, 3, 3]),
        np.array([1.0, -2.0, 0.5, -0.25], dtype=np.float32),
        normalize_incoming=False,
    )
    state = select_kc_mbon_edges(graph, np.array([0, 1]), np.array([2, 3]))
    return graph, state


def test_reward_and_aversive_updates_change_only_existing_edges():
    graph, state = make_state()
    snapshots = {1: np.array([1, 1, 0, 0], dtype=np.float32), 2: np.ones(4, dtype=np.float32)}
    before = state.current_weights.copy()
    eligibility = state.eligibility_from_trajectory(snapshots)
    state.apply(1.0, eligibility)
    rewarded = state.current_weights.copy()
    assert np.all(np.abs(rewarded) > np.abs(before))
    state.apply(-1.0, eligibility)
    assert np.all(np.abs(state.current_weights) < np.abs(rewarded))
    updated = apply_to_graph(graph, state)
    assert updated.n_edges == graph.n_edges
    np.testing.assert_array_equal(updated.weights.indptr, graph.weights.indptr)
    np.testing.assert_array_equal(updated.weights.indices, graph.weights.indices)
    assert np.all(np.sign(updated.weights.data) == np.sign(graph.weights.data))


def test_inactive_edges_remain_stable_and_ratios_are_bounded():
    _, state = make_state()
    initial = state.current_weights.copy()
    state.apply(1.0, np.zeros(state.edge_count))
    np.testing.assert_allclose(state.current_weights, initial)
    state.apply(100.0, np.ones(state.edge_count))
    assert np.all(state.ratios <= 2.0 + 1e-6)
    assert np.all(state.ratios >= 0.5 - 1e-6)


def test_checkpoint_round_trip_is_exact(tmp_path):
    _, state = make_state()
    state.apply(0.5, np.ones(state.edge_count))
    path = tmp_path / "plasticity.npz"
    save_plasticity_checkpoint(path, state, {"checkpoint": "test"})
    restored, metadata = load_plasticity_checkpoint(path, config=PlasticityConfig())
    assert metadata["checkpoint"] == "test"
    assert restored.edge_hash == state.edge_hash
    assert restored.update_count == state.update_count
    np.testing.assert_array_equal(restored.log_ratios, state.log_ratios)
