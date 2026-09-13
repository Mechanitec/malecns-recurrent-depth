import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.graph import ConnectomeGraph
from malecns_rd.lif import LIFRecurrentDepthEngine


def chain_graph(length: int, extra_nodes: int = 0) -> ConnectomeGraph:
    src = np.arange(length, dtype=np.int64)
    dst = src + 1
    w = np.ones(length, dtype=np.float32)
    return ConnectomeGraph.from_edges(length + 1 + extra_nodes, src, dst, w)


def test_lif_depth_propagates_spikes_farther():
    g = chain_graph(6)
    e = LIFRecurrentDepthEngine(g, threshold=0.5)
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 1.0
    shallow = e.run(x, max_depth=3, clamp_sensory=True)
    deep = e.run(x, max_depth=7, clamp_sensory=True)
    assert shallow.spike_counts[6] == 0
    assert deep.spike_counts[6] > 0


def test_lif_unclamped_input_is_single_pulse():
    g = chain_graph(3)
    e = LIFRecurrentDepthEngine(g, threshold=0.5)
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 1.0
    result = e.run(x, max_depth=5, clamp_sensory=False)
    np.testing.assert_array_equal(result.spike_counts, np.array([1, 1, 1, 1]))


def test_lif_confidence_controller_waits_for_distant_readout():
    g = chain_graph(6, extra_nodes=1)
    e = LIFRecurrentDepthEngine(g, threshold=0.5)
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 1.0
    result = e.run_until_confident(
        x,
        np.array([6, 7]),
        max_depth=20,
        min_depth=2,
        count_margin=1,
        stable_steps=2,
        clamp_sensory=True,
    )
    assert result.depth_used == 8
    assert result.spike_counts[6] > result.spike_counts[7]


def test_lif_refractory_reduces_repeated_source_spikes():
    g = chain_graph(1)
    x = np.zeros(g.n_neurons, dtype=np.float32)
    x[0] = 1.0
    no_ref = LIFRecurrentDepthEngine(g, threshold=0.5, refractory_steps=0).run(x, max_depth=6)
    ref = LIFRecurrentDepthEngine(g, threshold=0.5, refractory_steps=1).run(x, max_depth=6)
    assert ref.spike_counts[0] < no_ref.spike_counts[0]


def test_lif_signed_inhibition_can_cancel_excitation():
    src = np.array([0, 2], dtype=np.int64)
    dst = np.array([1, 1], dtype=np.int64)
    weight = np.array([1.0, -1.0], dtype=np.float32)
    g = ConnectomeGraph.from_edges(3, src, dst, weight, normalize_incoming=False)
    e = LIFRecurrentDepthEngine(g, threshold=0.5)
    x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    result = e.run(x, max_depth=2, clamp_sensory=False)
    assert result.spike_counts[1] == 0
