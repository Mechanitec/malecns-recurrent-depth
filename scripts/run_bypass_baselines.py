"""Evaluate raw and projected chess features without MaleCNS recurrence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import encode_board_move
from run_mechanistic_factorial import load_positions
from run_probe_transfer import _geometry, _metrics, _standardize, fit_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_positions(args.corpus)
    frame = pd.read_csv(args.corpus)
    frame["position_id"] = frame["position_id"].astype(str)
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    source = {str(p["position_id"]): str(p["source_split"]) for p in positions}
    frame["source_split"] = frame["position_id"].map(source)
    train_mask = frame["source_split"].eq("dev_screen").to_numpy()
    validation_mask = frame["source_split"].eq("dev_confirm").to_numpy()
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    projector = loaded.agent.projector
    raw = np.vstack([encode_board_move(position["board"], move) for position in positions for move in sorted(position["board"].legal_moves, key=lambda move: move.uci())]).astype(np.float32)
    projected = np.vstack([projector.project(encode_board_move(position["board"], move))[projector.sensory_indices] for position in positions for move in sorted(position["board"].legal_moves, key=lambda move: move.uci())]).astype(np.float32)
    rows = []
    for name, features in (("raw_encoder", raw), ("projected_sensory", projected)):
        train_x, _ = _standardize(features[train_mask], features[train_mask])
        _, validation_x = _standardize(features[train_mask], features[validation_mask])
        weights = fit_probe(train_x, train_frame)
        metrics = _metrics(validation_x, validation_frame, weights)
        distance, rank_value = _geometry(validation_x, validation_frame)
        rows.append({"representation": name, "depth": 0, "effective_rank": rank_value, "within_position_candidate_distance": distance, **{f"validation_{key}": value for key, value in metrics.items()}})
    pd.DataFrame(rows).to_csv(args.output / "bypass_baseline_results.csv", index=False)
    metadata = {"status": "complete", "positions": len(positions), "candidate_rows": len(frame), "representations": ["raw_encoder", "projected_sensory"], "outputs": ["bypass_baseline_results.csv"]}
    (args.output / "bypass_baseline_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
