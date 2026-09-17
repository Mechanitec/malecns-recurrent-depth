"""Localized reward-modulated plasticity for the Plan 3 pilot.

The implementation changes only existing selected edges. It keeps the sparse
graph structure and the sign of every selected synapse fixed while storing a
bounded log-ratio relative to the original connectome weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .graph import ConnectomeGraph


@dataclass(frozen=True)
class PlasticityConfig:
    learning_rate: float = 0.002
    homeostasis: float = 0.0002
    reward_baseline_decay: float = 0.95
    min_weight_ratio: float = 0.5
    max_weight_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.homeostasis < 0.0:
            raise ValueError("learning_rate must be positive and homeostasis non-negative")
        if not 0.0 < self.reward_baseline_decay < 1.0:
            raise ValueError("reward_baseline_decay must be in (0, 1)")
        if not 0.0 < self.min_weight_ratio <= 1.0 <= self.max_weight_ratio:
            raise ValueError("weight ratios must satisfy 0 < min <= 1 <= max")


def teaching_signal(stage: str, regret_cp: float, *, is_positive: bool | None = None) -> float:
    """Return the predeclared Plan 3 teaching signal for one selected candidate.

    Movement lessons are a legality task and therefore use the direct target
    specified by the curriculum: legal -> +1 and illegal -> -1. Ordinary chess
    stages use the bounded centipawn-regret mapping from ``plan3.md``.
    """
    if stage == "movement":
        if is_positive is None:
            raise ValueError("movement teaching signal requires is_positive")
        return 1.0 if bool(is_positive) else -1.0
    regret = max(0.0, float(regret_cp))
    return 1.0 - 2.0 * math.tanh(regret / 400.0)


def pairwise_ranking_accuracy(predicted_scores: np.ndarray, teacher_scores: np.ndarray) -> float:
    """Return concordance across all teacher-ordered candidate pairs.

    Teacher ties are ignored. Predicted ties count as half correct so that the
    metric is well-defined for degenerate readouts instead of duplicating top-1
    accuracy as the original pilot implementation did.
    """
    predicted = np.asarray(predicted_scores, dtype=np.float64)
    teacher = np.asarray(teacher_scores, dtype=np.float64)
    if predicted.ndim != 1 or teacher.shape != predicted.shape:
        raise ValueError("predicted_scores and teacher_scores must be matching vectors")
    correct = 0.0
    total = 0
    for left in range(len(teacher)):
        for right in range(left + 1, len(teacher)):
            teacher_delta = teacher[left] - teacher[right]
            if np.isclose(teacher_delta, 0.0):
                continue
            predicted_delta = predicted[left] - predicted[right]
            total += 1
            if np.isclose(predicted_delta, 0.0):
                correct += 0.5
            elif predicted_delta * teacher_delta > 0.0:
                correct += 1.0
    return float(correct / total) if total else float("nan")


@dataclass
class PlasticityState:
    """Mutable state for a fixed set of graph edges."""

    edge_offsets: np.ndarray
    post_indices: np.ndarray
    pre_indices: np.ndarray
    original_weights: np.ndarray
    log_ratios: np.ndarray
    config: PlasticityConfig = PlasticityConfig()
    update_count: int = 0

    def __post_init__(self) -> None:
        arrays = [self.edge_offsets, self.post_indices, self.pre_indices, self.original_weights, self.log_ratios]
        lengths = {len(np.asarray(value)) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("plastic edge arrays must be non-empty and have equal length")
        self.edge_offsets = np.asarray(self.edge_offsets, dtype=np.int64)
        self.post_indices = np.asarray(self.post_indices, dtype=np.int64)
        self.pre_indices = np.asarray(self.pre_indices, dtype=np.int64)
        self.original_weights = np.asarray(self.original_weights, dtype=np.float32)
        self.log_ratios = np.asarray(self.log_ratios, dtype=np.float64)
        if np.any(self.original_weights == 0.0) or not np.all(np.isfinite(self.original_weights)):
            raise ValueError("original plastic weights must be finite and nonzero")
        if not np.all(np.isfinite(self.log_ratios)):
            raise ValueError("log_ratios must be finite")
        if np.any(self.edge_offsets < 0):
            raise ValueError("edge offsets must be non-negative")

    @property
    def edge_count(self) -> int:
        return int(len(self.edge_offsets))

    @property
    def ratios(self) -> np.ndarray:
        return np.exp(self.log_ratios).astype(np.float32)

    @property
    def current_weights(self) -> np.ndarray:
        return (self.original_weights.astype(np.float64) * np.exp(self.log_ratios)).astype(np.float32)

    @property
    def edge_hash(self) -> str:
        digest = hashlib.sha256()
        for values in (self.post_indices, self.pre_indices, self.original_weights):
            digest.update(np.asarray(values).tobytes())
        return digest.hexdigest()

    def eligibility_from_trajectory(self, snapshots: dict[int, np.ndarray]) -> np.ndarray:
        """Compute mean pre[d-1] * post[d] eligibility for selected edges."""
        depths = tuple(sorted(int(depth) for depth in snapshots))
        if not depths or depths[0] != 1 or depths != tuple(range(1, depths[-1] + 1)):
            raise ValueError("snapshots must contain every depth from 1 through max depth")
        values: list[np.ndarray] = []
        previous = np.zeros_like(np.asarray(snapshots[1], dtype=np.float32))
        for depth in depths:
            current = np.asarray(snapshots[depth], dtype=np.float32)
            if current.ndim != 1:
                raise ValueError("trajectory snapshots must be one-dimensional")
            values.append(previous[self.pre_indices] * current[self.post_indices])
            previous = current
        return np.mean(np.vstack(values), axis=0, dtype=np.float64)

    def apply(self, advantage: float, eligibility: np.ndarray) -> dict[str, float]:
        """Apply one centered reward update and return update diagnostics."""
        eligibility = np.asarray(eligibility, dtype=np.float64)
        if eligibility.shape != (self.edge_count,) or not np.all(np.isfinite(eligibility)):
            raise ValueError("eligibility must be finite and match the plastic edge count")
        scale = float(np.mean(np.abs(eligibility)))
        normalized = eligibility / max(scale, 1e-8)
        delta = self.config.learning_rate * float(advantage) * normalized
        delta -= self.config.homeostasis * self.log_ratios
        self.log_ratios = np.clip(
            self.log_ratios + delta,
            np.log(self.config.min_weight_ratio),
            np.log(self.config.max_weight_ratio),
        )
        self.update_count += 1
        ratios = self.ratios
        return {
            "update_count": float(self.update_count),
            "eligibility_mean_abs": scale,
            "log_ratio_rms": float(np.sqrt(np.mean(self.log_ratios ** 2))),
            "median_abs_log_ratio": float(np.median(np.abs(self.log_ratios))),
            "fraction_changed_gt_5pct": float(np.mean(np.abs(ratios - 1.0) > 0.05)),
            "fraction_changed_gt_10pct": float(np.mean(np.abs(ratios - 1.0) > 0.10)),
            "fraction_lower_bound": float(np.mean(ratios <= self.config.min_weight_ratio + 1e-6)),
            "fraction_upper_bound": float(np.mean(ratios >= self.config.max_weight_ratio - 1e-6)),
        }

    def metrics(self) -> dict[str, float]:
        ratios = self.ratios
        abs_log = np.abs(self.log_ratios)
        return {
            "update_count": float(self.update_count),
            "rms_log_ratio": float(np.sqrt(np.mean(self.log_ratios ** 2))),
            "median_abs_log_ratio": float(np.median(abs_log)),
            "p95_abs_log_ratio": float(np.percentile(abs_log, 95)),
            "fraction_changed_gt_5pct": float(np.mean(np.abs(ratios - 1.0) > 0.05)),
            "fraction_changed_gt_10pct": float(np.mean(np.abs(ratios - 1.0) > 0.10)),
            "fraction_lower_bound": float(np.mean(ratios <= self.config.min_weight_ratio + 1e-6)),
            "fraction_upper_bound": float(np.mean(ratios >= self.config.max_weight_ratio - 1e-6)),
        }


def select_kc_mbon_edges(
    graph: ConnectomeGraph,
    kc_indices: np.ndarray,
    mbon_indices: np.ndarray,
    *,
    config: PlasticityConfig | None = None,
) -> PlasticityState:
    """Select existing graph edges from KC-like inputs to MBON-like outputs."""
    kc = set(np.asarray(kc_indices, dtype=np.int64).tolist())
    mbon = np.asarray(mbon_indices, dtype=np.int64)
    matrix = graph.weights.tocsr()
    offsets: list[int] = []
    posts: list[int] = []
    pres: list[int] = []
    weights: list[float] = []
    for post in np.unique(mbon):
        row_start, row_end = int(matrix.indptr[post]), int(matrix.indptr[post + 1])
        for offset in range(row_start, row_end):
            pre = int(matrix.indices[offset])
            if pre in kc:
                offsets.append(offset)
                posts.append(int(post))
                pres.append(pre)
                weights.append(float(matrix.data[offset]))
    if not offsets:
        raise ValueError("no existing KC-to-MBON edges matched the supplied manifests")
    return PlasticityState(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(posts, dtype=np.int64),
        np.asarray(pres, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np.zeros(len(offsets), dtype=np.float64),
        config or PlasticityConfig(),
    )


def apply_to_graph(graph: ConnectomeGraph, state: PlasticityState) -> ConnectomeGraph:
    """Return the same sparse topology with updated selected edge weights."""
    weights = graph.weights.copy().tocsr()
    weights.data[state.edge_offsets] = state.current_weights
    return ConnectomeGraph(body_ids=graph.body_ids.copy(), weights=weights)


def save_plasticity_checkpoint(path: str | Path, state: PlasticityState, metadata: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.update({"edge_hash": state.edge_hash, "update_count": state.update_count})
    np.savez_compressed(
        output,
        edge_offsets=state.edge_offsets,
        post_indices=state.post_indices,
        pre_indices=state.pre_indices,
        original_weights=state.original_weights,
        log_ratios=state.log_ratios,
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )


def load_plasticity_checkpoint(path: str | Path, *, config: PlasticityConfig | None = None) -> tuple[PlasticityState, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        state = PlasticityState(
            data["edge_offsets"],
            data["post_indices"],
            data["pre_indices"],
            data["original_weights"],
            data["log_ratios"],
            config or PlasticityConfig(),
            int(metadata.get("update_count", 0)),
        )
    if metadata.get("edge_hash") != state.edge_hash:
        raise ValueError("plasticity checkpoint edge hash does not match its payload")
    return state, metadata
