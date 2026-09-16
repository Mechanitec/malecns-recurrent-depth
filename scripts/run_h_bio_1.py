"""Run the predeclared H-BIO-1 interface against matched controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import CHESS_FEATURE_DIM, HashedSensoryProjector, encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import _geometry, _metrics, _standardize, fit_probe


CONTROL_INPUTS = (
    "input_baseline_a", "input_random_central_brain", "input_random_whole_brain",
)
READOUTS = (
    "readout_fan_shaped_body", "readout_mbon", "readout_smp", "readout_cre", "readout_sip",
    "readout_pfn", "readout_pfr", "readout_pfl", "readout_lal", "readout_descending", "readout_baseline_a",
)


def _corpus(path: Path, positions: list[dict[str, object]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    frame["position_id"] = frame["position_id"].astype(str)
    ids = {str(position["position_id"]) for position in positions}
    frame = frame[frame["position_id"].isin(ids)].copy()
    frame["source_split"] = frame["position_id"].map({str(p["position_id"]): str(p["source_split"]) for p in positions})
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    return frame, frame["source_split"].eq("dev_screen").to_numpy(), frame["source_split"].eq("dev_confirm").to_numpy()


def _features(board: object, move: object) -> np.ndarray:
    return encode_board_move(board, move).astype(np.float32, copy=False)


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
    parser.add_argument("--batch-positions", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_positions(args.corpus)
    frame, train_mask, validation_mask = _corpus(args.corpus, positions)
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    manifests = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.manifests.glob("*.json")}
    readouts = {name: np.asarray(manifests[name]["indices"], dtype=np.int64) for name in READOUTS if name in manifests}
    if not readouts:
        raise FileNotFoundError("no H-BIO-1 readout manifests found")
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("H-BIO-1 requires the rate checkpoint")
    graph = base.engine.graph
    settings = {"fanout": base.projector.fanout, "seed": base.projector.seed, "amplitude": base.projector.amplitude}
    mb = np.asarray(manifests["input_mb_kenyon_cells"]["indices"], dtype=np.int64)
    fb = np.asarray(manifests["input_fan_shaped_body"]["indices"], dtype=np.int64)
    row_map = {(str(row.position_id), str(row.move_uci)): index for index, row in frame.iterrows()}
    rows = []
    architectures = {"H-BIO-1": ("dual", mb, fb)}
    for name in CONTROL_INPUTS:
        if name in manifests:
            architectures[name.removeprefix("input_")] = ("single", np.asarray(manifests[name]["indices"], dtype=np.int64), None)
    for architecture, (mode, input_a, input_b) in architectures.items():
        projector_a = HashedSensoryProjector(graph.n_neurons, input_a, **settings)
        projector_b = HashedSensoryProjector(graph.n_neurons, input_b, **settings) if input_b is not None else None
        readout_union = np.asarray(sorted({index for indices in readouts.values() for index in indices}), dtype=np.int64)
        union_pos = {int(index): offset for offset, index in enumerate(readout_union)}
        cache_dir = args.output / "_h_bio_activation_cache" / architecture
        cache_dir.mkdir(parents=True, exist_ok=True)
        memmaps = {depth: np.memmap(cache_dir / f"depth_{depth}.float32", mode="w+", dtype=np.float32, shape=(len(frame), len(readout_union))) for depth in DEPTHS}
        for start in range(0, len(positions), args.batch_positions):
            batch = positions[start:start + args.batch_positions]
            sensory_blocks = []
            keys = []
            for position in batch:
                moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
                blocks = []
                for move in moves:
                    features = _features(position["board"], move)
                    if mode == "dual":
                        left = features.copy()
                        right = features.copy()
                        left[1::2] = 0.0
                        right[0::2] = 0.0
                        blocks.append(projector_a.project(left) + projector_b.project(right))
                    else:
                        blocks.append(projector_a.project(features))
                sensory_blocks.append(np.column_stack(blocks))
                keys.extend((str(position["position_id"]), move.uci()) for move in moves)
            trajectory = base.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=False)
            row_indices = np.asarray([row_map[key] for key in keys], dtype=np.int64)
            for depth in DEPTHS:
                memmaps[depth][row_indices, :] = trajectory.snapshots[depth][readout_union, :].T
            print(f"{architecture}: trajectory batch {min(start + args.batch_positions, len(positions))}/{len(positions)}", flush=True)
        for memmap in memmaps.values():
            memmap.flush()
        for readout_name, indices in readouts.items():
            columns = np.asarray([union_pos[int(index)] for index in indices], dtype=np.int64)
            for depth in DEPTHS:
                features = np.asarray(memmaps[depth][:, columns])
                train_x, _ = _standardize(features[train_mask], features[train_mask])
                _, validation_x = _standardize(features[train_mask], features[validation_mask])
                weights = fit_probe(train_x, train_frame)
                metrics = _metrics(validation_x, validation_frame, weights)
                distance, rank_value = _geometry(validation_x, validation_frame)
                rows.append({"architecture": architecture, "readout_population_id": readout_name, "depth": depth, "input_mode": mode, "input_size": len(input_a) + (len(input_b) if input_b is not None else 0), "readout_size": len(indices), "effective_rank": rank_value, "within_position_candidate_distance": distance, **{f"validation_{key}": value for key, value in metrics.items()}})
        for path in cache_dir.glob("depth_*.float32"):
            path.unlink()
        cache_dir.rmdir()
    pd.DataFrame(rows).to_csv(args.output / "h_bio_1_results.csv", index=False)
    metadata = {"status": "complete", "positions": len(positions), "architectures": list(architectures), "readouts": list(readouts), "depths": list(DEPTHS), "feature_split": "even chess feature indices to MB/KC; odd indices to FB", "outputs": ["h_bio_1_results.csv"]}
    (args.output / "h_bio_1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
