from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .graph import ConnectomeGraph


@dataclass(frozen=True)
class RoutingBenchmark:
    graph: ConnectomeGraph
    sources: np.ndarray
    targets: np.ndarray
    path_lengths: np.ndarray


def make_multihop_routing_benchmark(
    path_lengths=(2, 4, 8, 16, 32),
    copies_per_length: int = 24,
    noise_edges_per_neuron: float = 1.5,
    seed: int = 7,
) -> RoutingBenchmark:
    """Build a controlled graph where correct readout requires multi-hop propagation.

    Every source owns a directed chain to one unique target. Random weak signed
    recurrent edges create distractor activity. A target is only reachable after
    enough applications of the shared dynamics block.
    """
    rng = np.random.default_rng(seed)
    chains: list[list[int]] = []
    cursor = 0
    src_edges: list[int] = []
    dst_edges: list[int] = []
    weights: list[float] = []
    sources: list[int] = []
    targets: list[int] = []
    lengths: list[int] = []

    for length in path_lengths:
        for _ in range(copies_per_length):
            nodes = list(range(cursor, cursor + length + 1))
            cursor += length + 1
            chains.append(nodes)
            sources.append(nodes[0])
            targets.append(nodes[-1])
            lengths.append(length)
            for a, b in zip(nodes[:-1], nodes[1:]):
                src_edges.append(a)
                dst_edges.append(b)
                weights.append(1.0)

    n_chain = cursor
    n_noise = max(128, n_chain // 4)
    n = n_chain + n_noise

    # Weak recurrent background connectivity: enough to create distractors,
    # but below the strength of the anatomical chain edges.
    m_noise = int(n * noise_edges_per_neuron)
    src_noise = rng.integers(0, n, size=m_noise)
    dst_noise = rng.integers(0, n, size=m_noise)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=m_noise, p=[0.35, 0.65])
    mag = rng.uniform(0.01, 0.06, size=m_noise)

    src = np.concatenate([np.asarray(src_edges), src_noise]).astype(np.int64)
    dst = np.concatenate([np.asarray(dst_edges), dst_noise]).astype(np.int64)
    w = np.concatenate([np.asarray(weights, dtype=np.float32), (sign * mag).astype(np.float32)])

    graph = ConnectomeGraph.from_edges(
        n, src, dst, w, normalize_incoming=True, max_incoming_abs=1.15
    )
    return RoutingBenchmark(
        graph=graph,
        sources=np.asarray(sources, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        path_lengths=np.asarray(lengths, dtype=np.int64),
    )
