"""Generate diagnostics for the frozen v1 depth-16 readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from malecns_rd.readout_training import (
    build_best_vs_rest_pairs,
    load_activation_cache,
    load_readout_checkpoint,
    stable_position_split,
    top1_accuracy,
    pair_accuracy,
)


def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/chess_activations_depth16_v1.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/training/v1_depth16_rate_large/readout_checkpoint.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/training/v1_depth16_rate_large"))
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    cache = load_activation_cache(args.cache)
    checkpoint = load_readout_checkpoint(args.checkpoint)
    x = cache.features.astype(np.float64)
    weights = checkpoint.readout_weights.astype(np.float64)
    targets = cache.frame["teacher_target"].to_numpy(dtype=np.float64)
    position_ids = cache.frame["position_id"].astype(str).to_numpy()
    unique_positions = np.unique(position_ids)
    validation_positions = set(unique_positions[stable_position_split(unique_positions, seed=args.split_seed)])
    score = x @ weights
    pair_features, pair_positions = build_best_vs_rest_pairs(cache.features, cache.frame)
    z = pair_features.astype(np.float64) @ weights
    probability = 1.0 / (1.0 + np.exp(np.clip(z, -60.0, 60.0)))
    gradient = -(pair_features.astype(np.float64).T @ probability) / len(pair_features) + 1e-4 * weights
    centered = x - np.mean(x, axis=0, keepdims=True)
    sample = centered[: min(len(centered), 5000)]
    singular_values = np.linalg.svd(sample, full_matrices=False, compute_uv=False)
    variance = singular_values ** 2
    effective_rank = float(np.exp(-np.sum((variance / variance.sum()) * np.log(np.maximum(variance / variance.sum(), 1e-12)))))
    group_sizes = cache.frame.groupby("position_id").size().to_numpy()
    within_distances = []
    for _, group in cache.frame.groupby("position_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        for left, right in zip(indices[:-1], indices[1:]):
            within_distances.append(float(np.linalg.norm(x[left] - x[right])))
    target_margins = []
    for _, group in cache.frame.groupby("position_id", sort=False):
        values = np.sort(group["teacher_target"].to_numpy(dtype=float))[::-1]
        if len(values) > 1:
            target_margins.append(float(values[0] - values[1]))
    diagnostics = {
        "cache": str(args.cache),
        "cache_shape": list(cache.features.shape),
        "cache_metadata": cache.metadata,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint.metadata,
        "positions": int(len(unique_positions)),
        "train_positions": int(np.sum(~np.isin(unique_positions, list(validation_positions)))),
        "validation_positions": int(len(validation_positions)),
        "features": {
            "global": finite_stats(x),
            "per_feature_mean_stats": finite_stats(np.mean(x, axis=0)),
            "per_feature_std_stats": finite_stats(np.std(x, axis=0)),
            "near_zero_fraction": float(np.mean(np.isclose(x, 0.0, atol=1e-7))),
            "saturated_fraction_abs_ge_0_99": float(np.mean(np.abs(x) >= 0.99)),
        },
        "within_position_feature_distance": {
            "mean_l2": float(np.mean(within_distances)),
            "median_l2": float(np.median(within_distances)),
            "pairs": len(within_distances),
        },
        "readout": {
            "weight_stats": finite_stats(weights),
            "weight_l2_norm": float(np.linalg.norm(weights)),
            "score_stats": finite_stats(score),
            "score_variance": float(np.var(score)),
            "gradient_l2_norm_at_checkpoint": float(np.linalg.norm(gradient)),
        },
        "teacher": {
            "target_stats": finite_stats(targets),
            "pair_count": int(len(pair_features)),
            "pair_position_count": int(len(np.unique(pair_positions))),
            "best_margin_stats": finite_stats(np.asarray(target_margins)),
            "candidate_count_stats": finite_stats(group_sizes),
        },
        "spectrum": {
            "sample_rows": int(len(sample)),
            "singular_values_top10": [float(value) for value in singular_values[:10]],
            "effective_rank": effective_rank,
            "nonzero_singular_values": int(np.sum(singular_values > singular_values[0] * 1e-6)),
        },
        "train_validation": {
            "checkpoint_train_pair_accuracy": checkpoint.metadata.get("train_pair_accuracy") if checkpoint.metadata else None,
            "checkpoint_validation_pair_accuracy": checkpoint.metadata.get("validation_pair_accuracy") if checkpoint.metadata else None,
            "checkpoint_train_top1_accuracy": checkpoint.metadata.get("train_top1_accuracy") if checkpoint.metadata else None,
            "checkpoint_validation_top1_accuracy": checkpoint.metadata.get("validation_top1_accuracy") if checkpoint.metadata else None,
            "recomputed_all_pair_accuracy": pair_accuracy(pair_features, weights),
            "recomputed_all_top1_accuracy": top1_accuracy(cache.features, cache.frame, weights),
            "split_seed": args.split_seed,
            "note": "The cache contains the frozen v1 activation subset; no independent evaluation positions are used here.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readout_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
