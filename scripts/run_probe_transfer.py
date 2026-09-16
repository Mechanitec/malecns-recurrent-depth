"""Extract depth-specific features and evaluate cross-depth linear probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import minimize
from scipy.special import expit

from malecns_rd.chess_features import encode_board_move
from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.readout_training import build_best_vs_rest_pairs, pair_accuracy, top1_accuracy


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
PROBE_OPTIMIZER = "deterministic scipy L-BFGS-B pairwise logistic ranking (60-iteration budget)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_selected(path: Path, train_positions: int, validation_positions: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_csv(path)
    required = {"position_id", "split", "fen", "move_uci", "teacher_target", "teacher_cp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher data is missing columns: {sorted(missing)}")
    groups = frame.groupby("position_id", sort=True)
    by_split: dict[str, list[str]] = {"train": [], "validation": []}
    for position_id, group in groups:
        splits = set(group["split"].astype(str))
        if len(splits) != 1 or next(iter(splits)) not in by_split:
            continue
        by_split[next(iter(splits))].append(str(position_id))
    selected_train = by_split["train"][:train_positions]
    selected_validation = by_split["validation"][:validation_positions]
    if not selected_train or not selected_validation:
        raise ValueError("selected data must contain at least one train and one validation position")
    selected = frame[frame["position_id"].astype(str).isin(selected_train + selected_validation)].copy()
    selected["position_id"] = selected["position_id"].astype(str)
    selected["split"] = selected["split"].astype(str)
    return selected.reset_index(drop=True), selected_train, selected_validation


def _objective(pair_features: np.ndarray, weights: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    z = pair_features @ weights
    loss = float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * l2 * np.dot(weights, weights))
    gradient = (-pair_features * expit(-z)[:, None]).mean(axis=0) + l2 * weights
    return loss, gradient


def fit_probe(features: np.ndarray, frame: pd.DataFrame, l2: float = 1e-3) -> np.ndarray:
    pairs, _ = build_best_vs_rest_pairs(features, frame)
    pairs64 = pairs.astype(np.float64)
    result = minimize(
        lambda weights: _objective(pairs64, weights, l2),
        np.zeros(features.shape[1], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 60, "ftol": 1e-7, "gtol": 1e-5, "maxls": 20},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"probe optimizer produced non-finite weights: {result.message}")
    return result.x.astype(np.float32)


def _standardize(train: np.ndarray, inference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return ((train - mean) / scale).astype(np.float32), ((inference - mean) / scale).astype(np.float32)


def _metrics(features: np.ndarray, frame: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    scores = features @ weights
    pair_features, _ = build_best_vs_rest_pairs(features, frame)
    teacher_regrets = []
    top3_hits = []
    selected_teacher_ranks = []
    spearman_values = []
    kendall_values = []
    score_margins = []
    for _, group in frame.groupby("position_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        moves = group["move_uci"].astype(str).to_numpy()
        targets = group["teacher_target"].to_numpy(dtype=np.float64)
        teacher_cp = group["teacher_cp"].to_numpy(dtype=np.float64)
        best_local = sorted(range(len(indices)), key=lambda j: (-targets[j], moves[j]))[0]
        prediction_order = sorted(range(len(indices)), key=lambda j: (-float(scores[indices[j]]), moves[j]))
        predicted_local = prediction_order[0]
        teacher_regrets.append(float(teacher_cp[best_local] - teacher_cp[predicted_local]))
        top3_hits.append(float(best_local in prediction_order[:3]))
        teacher_order = sorted(range(len(indices)), key=lambda j: (-targets[j], moves[j]))
        selected_teacher_ranks.append(float(teacher_order.index(predicted_local) + 1))
        if len(indices) > 1:
            score_margins.append(float(scores[indices[prediction_order[0]]] - scores[indices[prediction_order[1]]]))
        score_values = scores[indices]
        if np.ptp(score_values) > 1e-7 and np.ptp(targets) > 1e-7:
            spearman = spearmanr(score_values, targets).statistic
            kendall = kendalltau(score_values, targets).statistic
        else:
            spearman = float("nan")
            kendall = float("nan")
        if np.isfinite(spearman):
            spearman_values.append(float(spearman))
        if np.isfinite(kendall):
            kendall_values.append(float(kendall))
    return {
        "pair_accuracy": pair_accuracy(pair_features, weights),
        "top1_accuracy": top1_accuracy(features, frame, weights),
        "top3_accuracy": float(np.mean(top3_hits)) if top3_hits else float("nan"),
        "mean_teacher_regret_cp": float(np.mean(teacher_regrets)) if teacher_regrets else float("nan"),
        "mean_selected_teacher_rank": float(np.mean(selected_teacher_ranks)) if selected_teacher_ranks else float("nan"),
        "mean_spearman": float(np.mean(spearman_values)) if spearman_values else float("nan"),
        "mean_kendall": float(np.mean(kendall_values)) if kendall_values else float("nan"),
        "mean_score_margin": float(np.mean(score_margins)) if score_margins else float("nan"),
        "near_zero_unit_fraction": float(np.mean(np.abs(features) < 1e-6)) if features.size else float("nan"),
        "saturated_unit_fraction": float(np.mean(np.abs(features) >= 3.0)) if features.size else float("nan"),
    }


def _effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    power = singular * singular
    if not np.any(power > 0):
        return 0.0
    probability = power[power > 0] / power.sum()
    probability = probability[probability > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(probability * np.log(probability))
    return float(np.exp(entropy))


def _geometry(features: np.ndarray, frame: pd.DataFrame) -> tuple[float, float]:
    distances = []
    for _, group in frame.groupby("position_id", sort=False):
        x = features[group.index.to_numpy(dtype=np.int64)].astype(np.float64)
        if len(x) > 1:
            distances.append(float(np.mean(np.linalg.norm(x[:, None] - x[None, :], axis=2))))
    return float(np.mean(distances)) if distances else float("nan"), _effective_rank(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--neurotransmitters", required=True, type=Path)
    parser.add_argument("--connectome-weights", required=True, type=Path)
    parser.add_argument("--sensory-indices", required=True, type=Path)
    parser.add_argument("--teacher-csv", type=Path, default=Path("data/teacher_dataset_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/probe_transfer"))
    parser.add_argument("--train-positions", type=int, default=32)
    parser.add_argument("--validation-positions", type=int, default=16)
    parser.add_argument("--batch-positions", type=int, default=2)
    parser.add_argument("--projector-seed", type=int, default=0)
    parser.add_argument("--fanout", type=int, default=4)
    args = parser.parse_args()
    selected, train_ids, validation_ids = _load_selected(args.teacher_csv, args.train_positions, args.validation_positions)
    args.output.mkdir(parents=True, exist_ok=True)

    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
    )
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("probe transfer requires the rate checkpoint")
    projector = base.projector
    graph = base.engine.graph
    engine = RecurrentDepthEngine(
        graph,
        leak=base.engine.leak,
        recurrent_gain=base.engine.recurrent_gain,
        input_gain=base.engine.input_gain,
        activation=base.engine.activation,
    )

    row_count = len(selected)
    raw_features = np.empty((row_count, 0), dtype=np.float32)
    sensory_features = np.empty((row_count, len(projector.sensory_indices)), dtype=np.float32)
    activation_features = {depth: np.empty((row_count, len(base.readout_indices)), dtype=np.float32) for depth in DEPTHS}
    raw_parts: list[np.ndarray] = []
    cursor = 0
    positions = []
    for position_id, group in selected.groupby("position_id", sort=True):
        board = chess.Board(str(group.iloc[0]["fen"]))
        moves = [chess.Move.from_uci(move) for move in group["move_uci"].astype(str)]
        positions.append((position_id, board, group.index.to_numpy(dtype=np.int64), moves))
    for start in range(0, len(positions), args.batch_positions):
        batch = positions[start:start + args.batch_positions]
        sensory_blocks = []
        raw_blocks = []
        indices = []
        for position_id, board, row_indices, moves in batch:
            raw_blocks.append(np.vstack([encode_board_move(board, move) for move in moves]))
            sensory_blocks.append(np.column_stack([projector.project(encode_board_move(board, move)) for move in moves]))
            indices.append(row_indices)
        raw_batch = np.vstack(raw_blocks).astype(np.float32)
        sensory_batch = np.concatenate(sensory_blocks, axis=1)
        raw_features = np.empty((row_count, raw_batch.shape[1]), dtype=np.float32) if raw_features.shape[1] == 0 else raw_features
        raw_features[np.concatenate(indices)] = raw_batch
        sensory_features[np.concatenate(indices)] = sensory_batch[projector.sensory_indices, :].T
        trajectory = engine.run_batch_trajectory(sensory_batch, depths=DEPTHS, clamp_sensory=True, observe=False)
        for depth in DEPTHS:
            state = trajectory.snapshots[depth][base.readout_indices, :].T
            activation_features[depth][np.concatenate(indices)] = state

    train_mask = selected["split"].eq("train").to_numpy()
    validation_mask = selected["split"].eq("validation").to_numpy()
    train_frame = selected.loc[train_mask].reset_index(drop=True)
    validation_frame = selected.loc[validation_mask].reset_index(drop=True)
    feature_families: dict[str, dict[int, np.ndarray]] = {
        "raw_encoder": {1: raw_features},
        "sensory_projected": {1: sensory_features},
        "malecns": activation_features,
    }
    matrices: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for family, depth_features in feature_families.items():
        train_source = raw_features if family == "raw_encoder" else sensory_features if family == "sensory_projected" else None
        for train_depth, train_matrix in depth_features.items():
            if train_source is not None:
                train_matrix = train_source
            train_x, _ = _standardize(train_matrix[train_mask], train_matrix[train_mask])
            probe_frame = train_frame
            weights = fit_probe(train_x, probe_frame)
            for inference_depth, inference_matrix in depth_features.items():
                if train_source is not None:
                    inference_matrix = train_source
                _, val_x = _standardize(train_matrix[train_mask], inference_matrix[validation_mask])
                row = {"feature_family": family, "probe_depth": train_depth, "inference_depth": inference_depth}
                row.update({f"validation_{key}": value for key, value in _metrics(val_x, validation_frame, weights).items()})
                matrices.append(row)
            for split_name, mask, frame in (("train", train_mask, train_frame), ("validation", validation_mask, validation_frame)):
                standardized, _ = _standardize(train_matrix[train_mask], train_matrix[mask])
                distance, rank_value = _geometry(standardized, frame)
                geometry_rows.append({
                    "feature_family": family, "probe_depth": train_depth, "split": split_name,
                    "effective_rank": rank_value, "within_position_candidate_distance": distance,
                    "score_separability": _metrics(standardized, frame, weights)["pair_accuracy"],
                })

    pd.DataFrame(matrices).to_csv(args.output / "cross_depth_transfer.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(args.output / "feature_geometry.csv", index=False)
    metadata = {
        "status": "complete", "teacher_csv": str(args.teacher_csv), "teacher_csv_sha256": sha256_file(args.teacher_csv),
        "train_positions": len(train_ids), "validation_positions": len(validation_ids),
        "train_position_ids": train_ids, "validation_position_ids": validation_ids,
        "depths": list(DEPTHS), "features": {family: sorted(depths) for family, depths in feature_families.items()},
        "standardization": "training rows only, independently for each probe depth and feature family",
        "optimizer": PROBE_OPTIMIZER,
        "outputs": ["cross_depth_transfer.csv", "feature_geometry.csv"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
