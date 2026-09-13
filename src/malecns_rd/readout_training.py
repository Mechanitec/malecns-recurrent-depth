from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_CACHE_COLUMNS = ("position_id", "move_uci", "teacher_target")


@dataclass(frozen=True)
class ActivationCache:
    features: np.ndarray
    frame: pd.DataFrame
    readout_indices: np.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class ReadoutCheckpoint:
    readout_weights: np.ndarray
    readout_indices: np.ndarray
    recurrent_depth: int
    projector_seed: int | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class PairwiseTrainingResult:
    weights: np.ndarray
    history: pd.DataFrame
    train_pair_accuracy: float
    validation_pair_accuracy: float
    train_top1_accuracy: float
    validation_top1_accuracy: float


def stable_position_split(position_ids: Iterable[object], validation_fraction: float = 0.2, seed: int = 0) -> np.ndarray:
    """Return a boolean validation mask stable across runs and row order.

    Splitting by position rather than candidate row prevents moves from the same
    chess position leaking across train and validation sets.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    threshold = int(validation_fraction * (1 << 64))
    mask: list[bool] = []
    for value in position_ids:
        payload = f"{seed}:{value}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        bucket = int.from_bytes(digest, "little")
        mask.append(bucket < threshold)
    return np.asarray(mask, dtype=bool)


def _validate_cache(features: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if len(frame) != x.shape[0]:
        raise ValueError("frame rows must match feature rows")
    missing = [c for c in REQUIRED_CACHE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
    f = frame.reset_index(drop=True).copy()
    f["teacher_target"] = pd.to_numeric(f["teacher_target"], errors="raise")
    return x, f


def build_best_vs_rest_pairs(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    min_teacher_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create pair vectors best_candidate - other_candidate for each position."""
    x, f = _validate_cache(features, frame)
    pairs: list[np.ndarray] = []
    pair_positions: list[object] = []
    for position_id, group in f.groupby("position_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < 2:
            continue
        targets = f.loc[indices, "teacher_target"].to_numpy(dtype=np.float64)
        moves = f.loc[indices, "move_uci"].astype(str).to_numpy()
        order = sorted(range(len(indices)), key=lambda j: (-targets[j], moves[j]))
        best_local = order[0]
        best_idx = int(indices[best_local])
        best_target = float(targets[best_local])
        for local_j, idx in enumerate(indices):
            if local_j == best_local:
                continue
            margin = best_target - float(targets[local_j])
            if margin < min_teacher_margin:
                continue
            pairs.append((x[best_idx] - x[int(idx)]).astype(np.float32, copy=False))
            pair_positions.append(position_id)
    if not pairs:
        raise ValueError("no training pairs could be constructed")
    return np.vstack(pairs).astype(np.float32), np.asarray(pair_positions, dtype=object)


def pair_accuracy(pair_features: np.ndarray, weights: np.ndarray) -> float:
    p = np.asarray(pair_features, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if p.ndim != 2 or w.shape != (p.shape[1],):
        raise ValueError("incompatible pair_features/weights shapes")
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p @ w) > 0.0))


def top1_accuracy(features: np.ndarray, frame: pd.DataFrame, weights: np.ndarray) -> float:
    x, f = _validate_cache(features, frame)
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (x.shape[1],):
        raise ValueError("weights length must match feature width")
    correct = total = 0
    scores = x @ w
    for _, group in f.groupby("position_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < 2:
            continue
        moves = f.loc[indices, "move_uci"].astype(str).to_numpy()
        targets = f.loc[indices, "teacher_target"].to_numpy(dtype=np.float64)
        teacher_local = sorted(range(len(indices)), key=lambda j: (-targets[j], moves[j]))[0]
        pred_local = sorted(range(len(indices)), key=lambda j: (-float(scores[indices[j]]), moves[j]))[0]
        correct += int(teacher_local == pred_local)
        total += 1
    return float(correct / total) if total else float("nan")


def _logistic_pair_loss(pair_features: np.ndarray, weights: np.ndarray, l2: float) -> float:
    if len(pair_features) == 0:
        return float("nan")
    z = pair_features @ weights
    return float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * l2 * np.dot(weights, weights))


def train_pairwise_readout(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    epochs: int = 100,
    learning_rate: float = 0.03,
    l2: float = 1e-4,
    batch_size: int = 256,
    min_teacher_margin: float = 0.0,
    seed: int = 0,
) -> PairwiseTrainingResult:
    """Train a linear chess readout with best-vs-rest pairwise logistic loss."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if l2 < 0.0:
        raise ValueError("l2 must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    x, f = _validate_cache(features, frame)
    unique_positions = f["position_id"].drop_duplicates().to_numpy(dtype=object)
    val_position_mask = stable_position_split(unique_positions, validation_fraction, seed)
    val_positions = set(unique_positions[val_position_mask].tolist())
    if not val_positions or len(val_positions) == len(unique_positions):
        ordered = list(unique_positions)
        val_positions = {ordered[-1]}
        if len(ordered) == 1:
            raise ValueError("at least two positions are required for train/validation split")

    row_is_val = f["position_id"].map(lambda v: v in val_positions).to_numpy(dtype=bool)
    train_x, train_f = x[~row_is_val], f.loc[~row_is_val].reset_index(drop=True)
    val_x, val_f = x[row_is_val], f.loc[row_is_val].reset_index(drop=True)
    train_pairs, _ = build_best_vs_rest_pairs(train_x, train_f, min_teacher_margin=min_teacher_margin)
    val_pairs, _ = build_best_vs_rest_pairs(val_x, val_f, min_teacher_margin=min_teacher_margin)

    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    history_rows: list[dict[str, float | int]] = []

    train_pairs64 = train_pairs.astype(np.float64)
    val_pairs64 = val_pairs.astype(np.float64)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_pairs64))
        for start in range(0, len(order), batch_size):
            batch = train_pairs64[order[start:start + batch_size]]
            z = batch @ w
            neg_prob = np.empty_like(z)
            positive = z >= 0
            exp_neg = np.exp(-z[positive])
            neg_prob[positive] = exp_neg / (1.0 + exp_neg)
            exp_pos = np.exp(z[~positive])
            neg_prob[~positive] = 1.0 / (1.0 + exp_pos)
            grad = -(batch.T @ neg_prob) / len(batch) + l2 * w

            step += 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)
            w -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        wf = w.astype(np.float32)
        history_rows.append({
            "epoch": epoch,
            "loss": _logistic_pair_loss(train_pairs64, w, l2),
            "validation_loss": _logistic_pair_loss(val_pairs64, w, l2),
            "train_pair_accuracy": pair_accuracy(train_pairs, wf),
            "validation_pair_accuracy": pair_accuracy(val_pairs, wf),
            "train_top1_accuracy": top1_accuracy(train_x, train_f, wf),
            "validation_top1_accuracy": top1_accuracy(val_x, val_f, wf),
        })

    wf = w.astype(np.float32)
    history = pd.DataFrame(history_rows)
    return PairwiseTrainingResult(
        weights=wf,
        history=history,
        train_pair_accuracy=float(history.iloc[-1]["train_pair_accuracy"]),
        validation_pair_accuracy=float(history.iloc[-1]["validation_pair_accuracy"]),
        train_top1_accuracy=float(history.iloc[-1]["train_top1_accuracy"]),
        validation_top1_accuracy=float(history.iloc[-1]["validation_top1_accuracy"]),
    )


def save_activation_cache(
    path: str | Path,
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    readout_indices: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    x, f = _validate_cache(features, frame)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "features": x,
        "position_id": np.asarray(f["position_id"].astype(str).tolist(), dtype="U"),
        "move_uci": np.asarray(f["move_uci"].astype(str).tolist(), dtype="U"),
        "teacher_target": f["teacher_target"].to_numpy(dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    if readout_indices is not None:
        indices = np.asarray(readout_indices, dtype=np.int64)
        if indices.shape != (x.shape[1],):
            raise ValueError("readout_indices must match activation feature width")
        payload["readout_indices"] = indices
    np.savez_compressed(output, **payload)


def load_activation_cache(path: str | Path) -> ActivationCache:
    with np.load(path, allow_pickle=False) as data:
        x = data["features"].astype(np.float32)
        frame = pd.DataFrame({
            "position_id": data["position_id"].astype(str),
            "move_uci": data["move_uci"].astype(str),
            "teacher_target": data["teacher_target"].astype(np.float32),
        })
        raw = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
        indices = data["readout_indices"].astype(np.int64) if "readout_indices" in data else None
    return ActivationCache(x, frame, indices, json.loads(raw))


def save_readout_checkpoint(
    path: str | Path,
    *,
    readout_weights: np.ndarray,
    readout_indices: np.ndarray,
    recurrent_depth: int,
    projector_seed: int | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    weights = np.asarray(readout_weights, dtype=np.float32)
    indices = np.asarray(readout_indices, dtype=np.int64)
    if weights.ndim != 1 or indices.shape != weights.shape:
        raise ValueError("readout_weights and readout_indices must be matching vectors")
    if recurrent_depth < 1:
        raise ValueError("recurrent_depth must be >= 1")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        readout_weights=weights,
        readout_indices=indices,
        recurrent_depth=np.asarray(int(recurrent_depth), dtype=np.int64),
        projector_seed=np.asarray(-1 if projector_seed is None else int(projector_seed), dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )


def load_readout_checkpoint(path: str | Path) -> ReadoutCheckpoint:
    with np.load(path, allow_pickle=False) as data:
        weights = data["readout_weights"].astype(np.float32)
        indices = data["readout_indices"].astype(np.int64)
        depth = int(data["recurrent_depth"].item())
        raw_seed = int(data["projector_seed"].item()) if "projector_seed" in data else -1
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
    return ReadoutCheckpoint(weights, indices, depth, None if raw_seed < 0 else raw_seed, metadata)
