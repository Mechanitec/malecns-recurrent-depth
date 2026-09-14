"""Deterministic graph controls with explicit invariant guarantees."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .graph import ConnectomeGraph


def degree_preserving_edge_swap_v2(
    graph: ConnectomeGraph, seed: int, *, swaps: int | None = None
) -> ConnectomeGraph:
    """Rewire directed edges while preserving every in/out degree exactly.

    Two edges ``u->v`` and ``x->y`` become ``u->y`` and ``x->v``. Swaps that
    create self-loops or duplicate coordinates are rejected. Edge weights stay
    attached to their source edge, so the total signed and absolute weight
    sums are preserved as well.
    """
    weights = graph.weights.tocoo(copy=True)
    src = weights.col.astype(np.int64, copy=True)
    dst = weights.row.astype(np.int64, copy=True)
    data = weights.data.astype(np.float32, copy=True)
    if len(src) < 2:
        return ConnectomeGraph(graph.body_ids.copy(), graph.weights.copy())
    target = int(swaps if swaps is not None else min(len(src), 100_000))
    if target < 0:
        raise ValueError("swaps must be non-negative")
    rng = np.random.default_rng(seed)
    edges = {(int(source), int(destination)) for source, destination in zip(src, dst)}
    movable = np.flatnonzero(src != dst)
    if len(movable) < 2:
        return ConnectomeGraph(graph.body_ids.copy(), graph.weights.copy())
    completed = 0
    attempts = 0
    max_attempts = max(1000, target * 50)
    while completed < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(movable, size=2, replace=False)
        u, v = int(src[i]), int(dst[i])
        x, y = int(src[j]), int(dst[j])
        if u == y or x == v:
            continue
        new_first = (u, y)
        new_second = (x, v)
        if new_first == new_second or new_first in edges or new_second in edges:
            continue
        edges.remove((u, v))
        edges.remove((x, y))
        edges.add(new_first)
        edges.add(new_second)
        dst[i], dst[j] = dst[j], dst[i]
        completed += 1
    if completed < target:
        raise RuntimeError(
            f"could complete only {completed} of {target} degree-preserving swaps"
        )
    rewired = sp.coo_matrix((data, (dst, src)), shape=graph.weights.shape).tocsr()
    return ConnectomeGraph(graph.body_ids.copy(), rewired)


def graph_invariants(graph: ConnectomeGraph) -> dict[str, object]:
    """Return compact structural and weighted invariants for a graph."""
    weights = graph.weights.tocsr()
    return {
        "n_neurons": int(weights.shape[0]),
        "n_edges": int(weights.nnz),
        "weight_sum": float(weights.data.sum()),
        "absolute_weight_sum": float(np.abs(weights.data).sum()),
        "self_loops": int(weights.diagonal().astype(bool).sum()),
        "in_degree": np.diff(weights.indptr).astype(np.int64).tolist(),
        "out_degree": np.diff(weights.tocsc().indptr).astype(np.int64).tolist(),
        "weighted_in_strength": np.asarray(np.abs(weights).sum(axis=1)).ravel().astype(float).tolist(),
        "weighted_out_strength": np.asarray(np.abs(weights).sum(axis=0)).ravel().astype(float).tolist(),
    }
