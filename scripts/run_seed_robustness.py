"""Estimate interface variability across deterministic input/readout seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import HashedSensoryProjector, encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import _geometry, _metrics, _standardize, fit_probe


def _selected_positions(all_positions: list[dict[str, object]], per_split: int) -> list[dict[str, object]]:
    split = [str(position["source_split"]) for position in all_positions]
    screen = [position for position, value in zip(all_positions, split) if value == "dev_screen"][:per_split]
    confirm = [position for position, value in zip(all_positions, split) if value == "dev_confirm"][:per_split]
    return screen + confirm


def _run_seed(
    base: object,
    positions: list[dict[str, object]],
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    input_indices: np.ndarray,
    readout_indices: np.ndarray,
    batch_positions: int,
) -> list[dict[str, object]]:
    projector = HashedSensoryProjector(base.engine.graph.n_neurons, input_indices, fanout=base.projector.fanout, seed=base.projector.seed, amplitude=base.projector.amplitude)
    rows = {(str(row.position_id), str(row.move_uci)): index for index, row in frame.iterrows()}
    features = {depth: np.empty((len(frame), len(readout_indices)), dtype=np.float32) for depth in DEPTHS}
    for start in range(0, len(positions), batch_positions):
        batch = positions[start:start + batch_positions]
        sensory_blocks = []
        keys = []
        for position in batch:
            moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
            sensory_blocks.append(np.column_stack([projector.project(encode_board_move(position["board"], move)) for move in moves]))
            keys.extend((str(position["position_id"]), move.uci()) for move in moves)
        trajectory = base.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=False)
        row_indices = np.asarray([rows[key] for key in keys], dtype=np.int64)
        for depth in DEPTHS:
            features[depth][row_indices] = trajectory.snapshots[depth][readout_indices, :].T
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    output = []
    for depth in DEPTHS:
        train_x, _ = _standardize(features[depth][train_mask], features[depth][train_mask])
        _, validation_x = _standardize(features[depth][train_mask], features[depth][validation_mask])
        weights = fit_probe(train_x, train_frame)
        metrics = _metrics(validation_x, validation_frame, weights)
        distance, rank_value = _geometry(validation_x, validation_frame)
        output.append({"depth": depth, "effective_rank": rank_value, "within_position_candidate_distance": distance, **{f"validation_{key}": value for key, value in metrics.items()}})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--readout-indices", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--positions-per-split", type=int, default=32)
    parser.add_argument("--batch-positions", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_positions = load_positions(args.corpus)
    positions = _selected_positions(all_positions, args.positions_per_split)
    frame = pd.read_csv(args.corpus)
    frame["position_id"] = frame["position_id"].astype(str)
    selected = {str(position["position_id"]) for position in positions}
    frame = frame[frame["position_id"].isin(selected)].sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    source = {str(position["position_id"]): str(position["source_split"]) for position in positions}
    frame["source_split"] = frame["position_id"].map(source)
    train_mask = frame["source_split"].eq("dev_screen").to_numpy()
    validation_mask = frame["source_split"].eq("dev_confirm").to_numpy()
    manifests = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.manifests.glob("*.json")}
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("seed robustness requires the rate checkpoint")
    rng = np.random.default_rng(9211)
    input_seed_rows = []
    readout_seed_rows = []
    for seed in range(args.seeds):
        input_indices = rng.choice(base.engine.graph.n_neurons, len(np.load(args.sensory_indices)), replace=False).astype(np.int64)
        output = _run_seed(base, positions, frame, train_mask, validation_mask, input_indices, np.asarray(manifests["readout_baseline_a"]["indices"], dtype=np.int64), args.batch_positions)
        for row in output:
            input_seed_rows.append({"seed": seed, "rule": "random_whole_brain_input", **row})
        readout_indices = rng.choice(base.engine.graph.n_neurons, len(np.load(args.readout_indices)), replace=False).astype(np.int64)
        output = _run_seed(base, positions, frame, train_mask, validation_mask, np.asarray(manifests["input_baseline_a"]["indices"], dtype=np.int64), readout_indices, args.batch_positions)
        for row in output:
            readout_seed_rows.append({"seed": seed, "rule": "random_whole_brain_readout", **row})
        print(f"seed {seed + 1}/{args.seeds} complete", flush=True)
    pd.DataFrame(input_seed_rows).to_csv(args.output / "input_seed_distribution.csv", index=False)
    pd.DataFrame(readout_seed_rows).to_csv(args.output / "readout_seed_distribution.csv", index=False)
    metadata = {"status": "complete", "seeds": args.seeds, "positions_per_split": args.positions_per_split, "positions": len(positions), "outputs": ["input_seed_distribution.csv", "readout_seed_distribution.csv"]}
    (args.output / "seed_robustness_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
