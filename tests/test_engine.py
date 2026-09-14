import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.graph import ConnectomeGraph
from malecns_rd.engine import RecurrentDepthEngine


def chain_graph(length: int) -> ConnectomeGraph:
    src = np.arange(length, dtype=np.int64)
    dst = src + 1
    w = np.ones(length, dtype=np.float32)
    return ConnectomeGraph.from_edges(length + 1, src, dst, w)


def test_depth_propagates_farther():
    g = chain_graph(6)
    e = RecurrentDepthEngine(g, leak=0.0, recurrent_gain=1.0, input_gain=1.0, activation="relu")
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 1.0

    shallow = e.run(x, max_depth=3, clamp_sensory=False).state
    deep = e.run(x, max_depth=7, clamp_sensory=False).state
    assert shallow[6] == 0.0
    assert deep[6] > 0.0


def test_fixed_depth_is_deterministic():
    g = chain_graph(4)
    e = RecurrentDepthEngine(g, activation="tanh")
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 0.5
    a = e.run(x, max_depth=8).state
    b = e.run(x, max_depth=8).state
    np.testing.assert_allclose(a, b)


def test_adaptive_stops_on_zero_input():
    g = chain_graph(4)
    e = RecurrentDepthEngine(g)
    x = np.zeros(g.n_neurons, dtype=np.float32)
    result = e.run(x, max_depth=50, min_depth=2, adaptive=True, tolerance=1e-8, patience=2)
    assert result.depth_used < 50


def test_confidence_controller_uses_more_depth_for_longer_path():
    # Two candidate readouts; only index 6 is reachable from the stimulated source.
    g = chain_graph(6)
    # Add an isolated distractor by rebuilding with one extra node.
    src = np.arange(6, dtype=np.int64)
    dst = src + 1
    w = np.ones(6, dtype=np.float32)
    g = ConnectomeGraph.from_edges(8, src, dst, w)
    e = RecurrentDepthEngine(g, leak=0.0, recurrent_gain=1.0, input_gain=1.0, activation="relu")
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 1.0
    result = e.run_until_confident(
        x, np.array([6, 7]), max_depth=20, min_depth=2,
        margin=0.5, min_activity=0.5, stable_steps=2, clamp_sensory=True,
    )
    assert result.depth_used >= 8  # 7 to reach target + 1 stability confirmation
    assert result.depth_used < 20


def test_rate_batch_matches_individual_runs():
    graph = ConnectomeGraph.from_edges(
        5,
        np.array([0, 1, 2, 3]),
        np.array([1, 2, 3, 4]),
        np.array([0.4, -0.2, 0.3, 0.5], dtype=np.float32),
        normalize_incoming=False,
    )
    engine = RecurrentDepthEngine(graph)
    batch = np.array(
        [[1, 0], [0, 1], [0.5, -0.5], [0, 0.25], [0, 0]],
        dtype=np.float32,
    )
    result = engine.run_batch(batch, max_depth=4)
    expected = np.column_stack([
        engine.run(batch[:, index], max_depth=4).state
        for index in range(batch.shape[1])
    ])
    assert np.allclose(result, expected, rtol=0, atol=1e-7)


def test_rate_batch_trajectory_matches_each_requested_depth():
    graph = ConnectomeGraph.from_edges(
        5,
        np.array([0, 1, 2, 3]),
        np.array([1, 2, 3, 4]),
        np.array([0.4, -0.2, 0.3, 0.5], dtype=np.float32),
        normalize_incoming=False,
    )
    engine = RecurrentDepthEngine(graph)
    batch = np.array(
        [[1, 0], [0, 1], [0.5, -0.5], [0, 0.25], [0, 0]],
        dtype=np.float32,
    )
    trajectory = engine.run_batch_trajectory(batch, depths=(1, 2, 4, 8))
    for depth, snapshot in trajectory.snapshots.items():
        expected = engine.run_batch(batch, max_depth=depth)
        np.testing.assert_allclose(snapshot, expected, rtol=0, atol=1e-7)
        assert trajectory.latency_s[depth] >= 0.0


def test_scalar_trajectory_matches_requested_depths():
    graph = chain_graph(4)
    engine = RecurrentDepthEngine(graph, activation="tanh")
    x = np.zeros(graph.n_neurons, dtype=np.float32)
    x[0] = 0.5
    snapshots = engine.run_trajectory(x, depths=(1, 3, 6))
    for depth, snapshot in snapshots.items():
        np.testing.assert_allclose(snapshot, engine.run(x, max_depth=depth).state, rtol=0, atol=1e-7)


def test_batch_trajectory_observations_are_per_candidate_and_do_not_change_states():
    graph = ConnectomeGraph.from_edges(
        3,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1, 2, 0], dtype=np.int64),
        np.array([0.2, -0.3, 0.4], dtype=np.float32),
        normalize_incoming=False,
    )
    engine = RecurrentDepthEngine(graph, leak=0.0, recurrent_gain=1.0, input_gain=1.0)
    sensory = np.array([[1.0, 0.5], [0.0, 1.0], [0.2, 0.0]], dtype=np.float32)
    plain = engine.run_batch_trajectory(sensory, depths=(1, 2), clamp_sensory=True)
    observed = engine.run_batch_trajectory(
        sensory, depths=(1, 2), clamp_sensory=True, observe=True
    )
    for depth in (1, 2):
        np.testing.assert_allclose(plain.snapshots[depth], observed.snapshots[depth])
        metrics = observed.observations[depth]
        assert set(metrics) == {
            "state_norm",
            "delta",
            "recurrent_drive_norm",
            "sensory_drive_norm",
            "recurrent_sensory_ratio",
            "saturation_fraction",
        }
        assert all(value.shape == (2,) for value in metrics.values())
