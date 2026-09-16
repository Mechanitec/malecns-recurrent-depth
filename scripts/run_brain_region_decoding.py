"""Decode the development corpus from multiple MaleCNS readout populations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import _geometry_metrics, _metrics, _standardize, fit_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--limit-positions", type=int, default=256)
    parser.add_argument("--batch-positions", type=int, default=2)
    parser.add_argument("--position-start", type=int, default=0)
    parser.add_argument("--position-end", type=int, default=0)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if args.limit_positions < 1 or args.batch_positions < 1:
        parser.error("limit-positions and batch-positions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(args.corpus)
    all_positions = load_positions(args.corpus)[:args.limit_positions]
    position_end = len(all_positions) if args.position_end == 0 else args.position_end
    if args.position_start < 0 or position_end <= args.position_start or position_end > len(all_positions):
        parser.error("position range must be within the selected corpus and non-empty")
    positions = all_positions[args.position_start:position_end]
    selected_ids = {str(position["position_id"]) for position in positions}
    corpus["position_id"] = corpus["position_id"].astype(str)
    frame = corpus[corpus["position_id"].isin(selected_ids)].copy()
    frame["source_split"] = frame["position_id"].map({str(p["position_id"]): str(p["source_split"]) for p in positions})
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    row_keys = [(str(position["position_id"]), move.uci()) for position in positions for move in sorted(position["board"].legal_moves, key=lambda move: move.uci())]
    expected_keys = list(zip(frame["position_id"], frame["move_uci"]))
    if row_keys != expected_keys:
        raise ValueError("corpus row order does not match legal move extraction order")
    row_map = {key: row for row, key in enumerate(expected_keys)}
    train_mask = frame["source_split"].eq("dev_screen").to_numpy()
    validation_mask = frame["source_split"].eq("dev_confirm").to_numpy()
    if not args.extract_only and (not train_mask.any() or not validation_mask.any()):
        raise ValueError("development corpus must contain both source splits")

    manifest_paths = sorted(args.manifests.glob("readout_*.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"no readout manifests in {args.manifests}")
    manifest_data = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths}
    union = np.asarray(sorted({index for payload in manifest_data.values() for index in payload["indices"]}), dtype=np.int64)
    union_positions = {int(index): offset for offset, index in enumerate(union)}
    row_count = len(frame)
    cache_dir = args.output / "_activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    memmaps = {depth: np.memmap(cache_dir / f"union_depth_{depth}.float32", mode="w+", dtype=np.float32, shape=(row_count, len(union))) for depth in DEPTHS}
    progress_path = args.output / "brain_region_decoding_progress.json"
    progress = {
        "status": "running", "positions": len(positions), "candidate_rows": row_count,
        "batch_positions": args.batch_positions, "batches_total": (len(positions) + args.batch_positions - 1) // args.batch_positions,
        "batches_completed": 0, "union_readout_neurons": len(union), "depths": list(DEPTHS),
    }
    progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("brain-region decoding requires the rate checkpoint")
    for batch_number, start in enumerate(range(0, len(positions), args.batch_positions), start=1):
        batch = positions[start:start + args.batch_positions]
        sensory_blocks = []
        batch_keys = []
        for position in batch:
            moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
            sensory_blocks.append(np.column_stack([base.projector.project(encode_board_move(position["board"], move)) for move in moves]))
            batch_keys.extend((str(position["position_id"]), move.uci()) for move in moves)
        rows = np.asarray([row_map[key] for key in batch_keys], dtype=np.int64)
        trajectory = base.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=False)
        for depth in DEPTHS:
            memmaps[depth][rows, :] = trajectory.snapshots[depth][union, :].T
        progress["batches_completed"] = batch_number
        progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        print(f"trajectory batch {batch_number}/{progress['batches_total']} complete", flush=True)
    for memmap in memmaps.values():
        memmap.flush()

    if args.extract_only:
        metadata = {
            "status": "complete", "mode": "extract_only", "position_start": args.position_start,
            "position_end": position_end, "positions": len(positions), "candidate_rows": row_count,
            "union_readout_neurons": len(union), "depths": list(DEPTHS), "input_population": "Population Baseline A",
        }
        (args.output / "activation_shard_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return

    result_rows = []
    geometry_rows = []
    for name, payload in manifest_data.items():
        indices = np.asarray(payload["indices"], dtype=np.int64)
        columns = np.asarray([union_positions[int(index)] for index in indices], dtype=np.int64)
        for depth in DEPTHS:
            features = np.asarray(memmaps[depth][:, columns])
            train_features, _ = _standardize(features[train_mask], features[train_mask])
            _, validation_features = _standardize(features[train_mask], features[validation_mask])
            probe = fit_probe(train_features, frame.loc[train_mask].reset_index(drop=True))
            train_metrics = _metrics(train_features, frame.loc[train_mask].reset_index(drop=True), probe)
            metrics = _metrics(validation_features, frame.loc[validation_mask].reset_index(drop=True), probe)
            geometry = _geometry_metrics(validation_features, frame.loc[validation_mask].reset_index(drop=True))
            result_rows.append({"population_id": name, "anatomical_family": payload["anatomical_family"], "population_role": payload["role"], "population_size": payload["actual_count"], "depth": depth, **{f"validation_{key}": value for key, value in metrics.items()}, **{f"train_{key}": value for key, value in train_metrics.items()}, "train_validation_pair_accuracy_gap": train_metrics["pair_accuracy"] - metrics["pair_accuracy"], "train_validation_top1_accuracy_gap": train_metrics["top1_accuracy"] - metrics["top1_accuracy"], "train_validation_regret_gap_cp": metrics["mean_teacher_regret_cp"] - train_metrics["mean_teacher_regret_cp"], **geometry, "mean_train_feature_std": float(np.mean(np.std(train_features, axis=0))), "mean_validation_score_margin_proxy": float(np.std(validation_features @ probe))})
            geometry_rows.append({"population_id": name, "depth": depth, **geometry, "candidate_dependent_fraction": float(np.mean(np.std(validation_features, axis=0) > 1e-6))})
    pd.DataFrame(result_rows).to_csv(args.output / "brain_region_decoding_atlas.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(args.output / "region_information_by_depth.csv", index=False)
    metadata = {"status": "complete", "positions": len(positions), "candidate_rows": row_count, "train_positions": int(frame.loc[train_mask, "position_id"].nunique()), "validation_positions": int(frame.loc[validation_mask, "position_id"].nunique()), "readout_populations": len(manifest_data), "union_readout_neurons": len(union), "depths": list(DEPTHS), "input_population": "Population Baseline A", "probe": "train-only standardized L-BFGS-B pairwise ranking", "activation_cache": str(cache_dir)}
    (args.output / "brain_region_decoding_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    progress.update({"status": "complete", "batches_completed": progress["batches_total"]})
    progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    if not args.keep_cache:
        for path in cache_dir.glob("union_depth_*.float32"):
            path.unlink()
        cache_dir.rmdir()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
