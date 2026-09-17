import math

import numpy as np
import pytest

from malecns_rd.graph import ConnectomeGraph
from malecns_rd.neuromodulated_plasticity import (
    PlasticityConfig,
    apply_to_graph,
    load_plasticity_checkpoint,
    pairwise_ranking_accuracy,
    save_plasticity_checkpoint,
    select_kc_mbon_edges,
    teaching_signal,
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
    snapshots = {
        1: np.array([1, 1, 0, 0], dtype=np.float32),
        2: np.ones(4, dtype=np.float32),
    }
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


def test_movement_teaching_signal_uses_direct_legality_target():
    assert teaching_signal("movement", 0.0, is_positive=True) == 1.0
    assert teaching_signal("movement", 200.0, is_positive=False) == -1.0
    with pytest.raises(ValueError):
        teaching_signal("movement", 200.0)


def test_chess_teaching_signal_keeps_predeclared_regret_mapping():
    assert teaching_signal("tactics", 0.0) == pytest.approx(1.0)
    assert teaching_signal("tactics", 200.0) == pytest.approx(
        1.0 - 2.0 * math.tanh(0.5)
    )
    assert teaching_signal("tactics", 400.0) < 0.0


def test_pairwise_ranking_accuracy_is_not_top1_accuracy_duplicate():
    teacher = np.array([3.0, 2.0, 1.0])
    predicted = np.array([2.0, 0.0, 1.0])
    # Top-1 is correct, but the 2-vs-1 ordering is wrong: 2/3 pairs concordant.
    assert pairwise_ranking_accuracy(predicted, teacher) == pytest.approx(2.0 / 3.0)


def test_pairwise_ranking_accuracy_ignores_teacher_ties_and_half_scores_predicted_ties():
    teacher = np.array([2.0, 2.0, 1.0])
    predicted = np.array([1.0, 0.0, 1.0])
    # The teacher tie (0,1) is ignored. Pair (0,2) is a prediction tie and
    # contributes 0.5; pair (1,2) is wrong, so accuracy is 0.25.
    assert pairwise_ranking_accuracy(predicted, teacher) == pytest.approx(0.25)
