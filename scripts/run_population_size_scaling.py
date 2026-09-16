"""Measure nested input/readout population-size effects on the development corpus."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import HashedSensoryProjector, encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import _geometry, _metrics, _standardize, fit_probe


READOUT_SIZES = (32, 64, 128, 256, 512, 1024)
INPUT_SIZES = (128, 256, 512, 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--atlas", type=Path, default=Path("results/population_study/malecns_region_atlas.json"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--batch-positions", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_positions(args.corpus)
    frame = pd.read_csv(args.corpus)
    frame["position_id"] = frame["position_id"].astype(str)
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    train_mask = frame["position_id"].map({str(p["position_id"]): str(p["source_split"]) for p in positions}).eq("dev_screen").to_numpy()
    validation_mask = frame["position_id"].map({str(p["position_id"]): str(p["source_split"]) for p in positions}).eq("dev_confirm").to_numpy()
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    manifests = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.manifests.glob("*.json")}
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("population scaling requires the rate checkpoint")
    graph = base.engine.graph
    settings = {"fanout": base.projector.fanout, "seed": base.projector.seed, "amplitude": base.projector.amplitude}
    baseline_input = np.asarray(manifests["input_baseline_a"]["indices"], dtype=np.int64)
    if args.atlas.exists():
        atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
        fb = np.asarray(atlas["families"]["fan_shaped_body"]["graph_indices"], dtype=np.int64)
    else:
        fb = np.asarray(manifests["readout_fan_shaped_body"]["indices"], dtype=np.int64)
    random_readout = np.random.default_rng(8123).choice(graph.n_neurons, 1024, replace=False).astype(np.int64)
    readout_rules = {"fan_shaped_body": fb, "random_matched": random_readout}
    readout_union = np.asarray(sorted(set(random_readout.tolist()) | set(fb.tolist())), dtype=np.int64)
    union_positions = {int(index): offset for offset, index in enumerate(readout_union)}
    row_map = {(str(row.position_id), str(row.move_uci)): index for index, row in frame.iterrows()}
    rows = []
    for input_size in INPUT_SIZES:
        input_indices = baseline_input[:min(input_size, len(baseline_input))]
        projector = HashedSensoryProjector(graph.n_neurons, input_indices, **settings)
        cache_dir = args.output / "_scaling_activation_cache" / f"input_{input_size}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        memmaps = {depth: np.memmap(cache_dir / f"depth_{depth}.float32", mode="w+", dtype=np.float32, shape=(len(frame), len(readout_union))) for depth in DEPTHS}
        for start in range(0, len(positions), args.batch_positions):
            batch = positions[start:start + args.batch_positions]
            sensory_blocks = []
            keys = []
            for position in batch:
                moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
                sensory_blocks.append(np.column_stack([projector.project(encode_board_move(position["board"], move)) for move in moves]))
                keys.extend((str(position["position_id"]), move.uci()) for move in moves)
            trajectory = base.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=False)
            indices = np.asarray([row_map[key] for key in keys], dtype=np.int64)
            for depth in DEPTHS:
                memmaps[depth][indices, :] = trajectory.snapshots[depth][readout_union, :].T
            print(f"input size {input_size}: trajectory batch {min(start + args.batch_positions, len(positions))}/{len(positions)}", flush=True)
        for memmap in memmaps.values():
            memmap.flush()
        for readout_name, candidates in readout_rules.items():
            for readout_size in READOUT_SIZES:
                if readout_size > len(candidates):
                    continue
                selected = np.sort(candidates[:readout_size])
                columns = np.asarray([union_positions[int(index)] for index in selected], dtype=np.int64)
                for depth in DEPTHS:
                    features = np.asarray(memmaps[depth][:, columns])
                    train_x, _ = _standardize(features[train_mask], features[train_mask])
                    _, validation_x = _standardize(features[train_mask], features[validation_mask])
                    weights = fit_probe(train_x, train_frame)
                    metrics = _metrics(validation_x, validation_frame, weights)
                    distance, rank_value = _geometry(validation_x, validation_frame)
                    rows.append({"input_population_id": "input_baseline_a", "input_size": len(input_indices), "readout_rule": readout_name, "readout_size": readout_size, "depth": depth, "effective_rank": rank_value, "within_position_candidate_distance": distance, **{f"validation_{key}": value for key, value in metrics.items()}})
        for memmap in memmaps.values():
            memmap.flush()
        del memmap, features, train_x, validation_x
        del memmaps
        gc.collect()
        for path in cache_dir.glob("depth_*.float32"):
            path.unlink()
        cache_dir.rmdir()
    pd.DataFrame(rows).to_csv(args.output / "population_size_scaling.csv", index=False)
    metadata = {"status": "complete", "positions": len(positions), "input_sizes": list(INPUT_SIZES), "readout_sizes": list(READOUT_SIZES), "readout_rules": list(readout_rules), "outputs": ["population_size_scaling.csv"]}
    (args.output / "population_size_scaling_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
