"""Run the frozen interface comparison on a fresh source-diverse corpus."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--selection", type=Path, default=Path("results/population_study/final_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--batch-positions", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_positions(args.corpus)
    frame = pd.read_csv(args.corpus)
    frame["position_id"] = frame["position_id"].astype(str)
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    source = {str(position["position_id"]): str(position["source_split"]) for position in positions}
    frame["source_split"] = frame["position_id"].map(source)
    train_mask = frame["source_split"].eq("dev_screen").to_numpy()
    validation_mask = frame["source_split"].eq("dev_confirm").to_numpy()
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    manifests = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.manifests.glob("*.json")}
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection.exists() else {}
    selected_input = str(selection.get("selected_input_population_id", "input_baseline_a"))
    selected_readout = str(selection.get("selected_readout_population_id", "readout_baseline_a"))
    architectures = [("baseline_a", "input_baseline_a", "readout_baseline_a"), ("population_v2", selected_input, selected_readout), ("matched_random_central", "input_random_central_brain", "readout_random_central_brain"), ("matched_random_whole", "input_random_whole_brain", "readout_random_whole_brain")]
    architectures = [(name, input_name, readout_name) for name, input_name, readout_name in architectures if (input_name == "h_bio_1" or input_name in manifests) and readout_name in manifests]
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("fresh confirmation requires the rate checkpoint")
    graph = base.engine.graph
    settings = {"fanout": base.projector.fanout, "seed": base.projector.seed, "amplitude": base.projector.amplitude}
    row_map = {(str(row.position_id), str(row.move_uci)): index for index, row in frame.iterrows()}
    output_rows = []
    raw = np.vstack([encode_board_move(position["board"], move) for position in positions for move in sorted(position["board"].legal_moves, key=lambda move: move.uci())]).astype(np.float32)
    projected = np.vstack([base.projector.project(encode_board_move(position["board"], move))[base.projector.sensory_indices] for position in positions for move in sorted(position["board"].legal_moves, key=lambda move: move.uci())]).astype(np.float32)
    for name, features in (("raw_encoder", raw), ("projected_sensory", projected)):
        train_x, _ = _standardize(features[train_mask], features[train_mask])
        _, validation_x = _standardize(features[train_mask], features[validation_mask])
        weights = fit_probe(train_x, train_frame)
        metrics = _metrics(validation_x, validation_frame, weights)
        distance, rank_value = _geometry(validation_x, validation_frame)
        output_rows.append({"interface": name, "input_population_id": name, "readout_population_id": name, "depth": 0, "effective_rank": rank_value, "within_position_candidate_distance": distance, **{f"validation_{key}": value for key, value in metrics.items()}})
    for name, input_name, readout_name in architectures:
        dual_hbio = input_name == "h_bio_1"
        if dual_hbio:
            input_indices = np.asarray(manifests["input_mb_kenyon_cells"]["indices"], dtype=np.int64)
            context_indices = np.asarray(manifests["input_fan_shaped_body"]["indices"], dtype=np.int64)
        else:
            input_indices = np.asarray(manifests[input_name]["indices"], dtype=np.int64)
        readout_indices = np.asarray(manifests[readout_name]["indices"], dtype=np.int64)
        projector = HashedSensoryProjector(graph.n_neurons, input_indices, **settings)
        context_projector = HashedSensoryProjector(graph.n_neurons, context_indices, **settings) if dual_hbio else None
        features = {depth: np.empty((len(frame), len(readout_indices)), dtype=np.float32) for depth in DEPTHS}
        for start in range(0, len(positions), args.batch_positions):
            batch = positions[start:start + args.batch_positions]
            sensory_blocks = []
            keys = []
            for position in batch:
                moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
                blocks = []
                for move in moves:
                    features = encode_board_move(position["board"], move)
                    if dual_hbio:
                        left = features.copy()
                        right = features.copy()
                        left[1::2] = 0.0
                        right[0::2] = 0.0
                        blocks.append(projector.project(left) + context_projector.project(right))
                    else:
                        blocks.append(projector.project(features))
                sensory_blocks.append(np.column_stack(blocks))
                keys.extend((str(position["position_id"]), move.uci()) for move in moves)
            trajectory = base.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=False)
            rows = np.asarray([row_map[key] for key in keys], dtype=np.int64)
            for depth in DEPTHS:
                features[depth][rows] = trajectory.snapshots[depth][readout_indices, :].T
            print(f"{name}: trajectory batch {min(start + args.batch_positions, len(positions))}/{len(positions)}", flush=True)
        for depth in DEPTHS:
            train_x, _ = _standardize(features[depth][train_mask], features[depth][train_mask])
            _, validation_x = _standardize(features[depth][train_mask], features[depth][validation_mask])
            weights = fit_probe(train_x, train_frame)
            metrics = _metrics(validation_x, validation_frame, weights)
            distance, rank_value = _geometry(validation_x, validation_frame)
            output_rows.append({"interface": name, "input_population_id": input_name, "readout_population_id": readout_name, "depth": depth, "effective_rank": rank_value, "within_position_candidate_distance": distance, **{f"validation_{key}": value for key, value in metrics.items()}})
    results = pd.DataFrame(output_rows)
    results.to_csv(args.output / "fresh_confirmation_results.csv", index=False)
    summary = {"status": "complete", "corpus": str(args.corpus), "positions": len(positions), "candidate_rows": len(frame), "interfaces": [row[0] for row in architectures], "selection": selection, "primary_comparison": "population_v2 versus baseline_a and matched random interfaces", "outputs": ["fresh_confirmation_results.csv", "final_confirmation_summary.json"]}
    (args.output / "final_confirmation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
