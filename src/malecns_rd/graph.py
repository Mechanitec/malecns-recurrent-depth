from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ConnectomeGraph:
    """Sparse directed graph.

    The matrix convention is W[post, pre], so ``W @ activity`` computes
    the total incoming drive at every postsynaptic neuron.
    """

    body_ids: np.ndarray
    weights: sp.csr_matrix

    @property
    def n_neurons(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.weights.nnz)

    @classmethod
    def from_edges(
        cls,
        n_neurons: int,
        src: np.ndarray,
        dst: np.ndarray,
        weight: np.ndarray,
        body_ids: np.ndarray | None = None,
        normalize_incoming: bool = True,
        max_incoming_abs: float = 1.0,
    ) -> "ConnectomeGraph":
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
        if not (src.shape == dst.shape == weight.shape):
            raise ValueError("src, dst, and weight must have the same shape")
        if src.size and (src.min() < 0 or dst.min() < 0 or src.max() >= n_neurons or dst.max() >= n_neurons):
            raise ValueError("edge index outside graph")

        w = sp.coo_matrix((weight, (dst, src)), shape=(n_neurons, n_neurons), dtype=np.float32).tocsr()
        w.sum_duplicates()

        if normalize_incoming and w.nnz:
            abs_rowsum = np.asarray(abs(w).sum(axis=1)).ravel().astype(np.float32)
            scale = np.ones(n_neurons, dtype=np.float32)
            mask = abs_rowsum > max_incoming_abs
            scale[mask] = max_incoming_abs / abs_rowsum[mask]
            w = sp.diags(scale, format="csr") @ w
            w = w.tocsr()

        if body_ids is None:
            body_ids = np.arange(n_neurons, dtype=np.int64)
        else:
            body_ids = np.asarray(body_ids, dtype=np.int64)
            if body_ids.shape != (n_neurons,):
                raise ValueError("body_ids must have length n_neurons")
        return cls(body_ids=body_ids, weights=w)
