"""Run one reproducible member of the corrected MaleCNS control ensemble."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from malecns_rd.chess_features import encode_board_move
from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.graph import ConnectomeGraph
from malecns_rd.graph_controls import degree_preserving_edge_swap_v2
from run_mechanistic_factorial import DEPTHS, effective_rank, load_positions


CONTROL_DEPTHS = (2, 8, 16, 64)
CONTROL_NAMES = ("original", "degree_preserving_edge_swap_v2", "transmitter_sign_shuffle")
RAW_COLUMNS = (
    "position_id", "source_game_id", "source_split", "phase", "ply", "fen", "variant", "seed",
    "depth", "legal_move_count", "teacher_best_move", "predicted_move", "regret_cp",
    "teacher_best_agreement", "fly_rank_of_teacher_best", "score_margin", "score_dispersion",
    "readout_effective_rank", "candidate_dispersion", "saturation_fraction_mean", "latency_s",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_graph(graph: ConnectomeGraph, variant: str, seed: int) -> ConnectomeGraph:
    if variant == "original":
        return graph
    if variant == "degree_preserving_edge_swap_v2":
        return degree_preserving_edge_swap_v2(graph, seed)
    if variant == "transmitter_sign_shuffle":
        rng = np.random.default_rng(seed)
        weights = graph.weights.tocoo(copy=True)
        signs = np.sign(weights.data)
        rng.shuffle(signs)
        return ConnectomeGraph(
            graph.body_ids.copy(),
            sp.coo_matrix(
                (np.abs(weights.data) * signs, (weights.row, weights.col)),
                shape=weights.shape,
            ).tocsr(),
        )
    raise ValueError(f"unknown control variant: {variant}")


def rank(score_map: dict[str, float], move: str) -> int:
    return sorted(score_map, key=lambda key: (-score_map[key], key)).index(move) + 1


def candidate_dispersion(states: np.ndarray) -> float:
    if states.shape[1] < 2:
        return 0.0
    centered = states - states.mean(axis=1, keepdims=True)
    return float(np.mean(np.linalg.norm(centered, axis=0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--neurotransmitters", required=True, type=Path)
    parser.add_argument("--connectome-weights", required=True, type=Path)
    parser.add_argument("--sensory-indices", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/control_ensemble"))
    parser.add_argument("--variant", choices=CONTROL_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--position-start", type=int, default=0)
    parser.add_argument("--position-count", type=int, default=64)
    parser.add_argument("--batch-positions", type=int, default=2)
    args = parser.parse_args()
    if args.position_start < 0 or args.position_count < 1 or args.batch_positions < 1:
        parser.error("position-start must be >= 0 and position-count/batch-positions must be positive")
    positions = load_positions(args.corpus)
    positions = positions[args.position_start:args.position_start + args.position_count]
    if len(positions) < args.position_count:
        raise ValueError("requested control subset exceeds corpus")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_seed{args.seed}"
    raw_path = args.output / f"{stem}.csv"
    if raw_path.exists():
        print(json.dumps({"status": "already_complete", "path": str(raw_path)}))
        return

    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
    )
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("control ensemble requires the rate checkpoint")
    graph = make_graph(base.engine.graph, args.variant, args.seed)
    engine = RecurrentDepthEngine(
        graph,
        leak=base.engine.leak,
        recurrent_gain=base.engine.recurrent_gain,
        input_gain=base.engine.input_gain,
        activation=base.engine.activation,
    )
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for start in range(0, len(positions), args.batch_positions):
            batch = positions[start:start + args.batch_positions]
            legal_by_position = []
            sensory_blocks = []
            for position in batch:
                board = position["board"]
                legal_moves = sorted(board.legal_moves, key=lambda move: move.uci())
                legal_by_position.append(legal_moves)
                sensory_blocks.append(np.column_stack([
                    base.projector.project(encode_board_move(board, move)) for move in legal_moves
                ]))
            sensory = np.concatenate(sensory_blocks, axis=1)
            trajectory = engine.run_batch_trajectory(
                sensory, depths=CONTROL_DEPTHS, clamp_sensory=True, observe=True
            )
            offset = 0
            for position, legal_moves in zip(batch, legal_by_position):
                end = offset + len(legal_moves)
                teacher = position["teacher"]
                assert isinstance(teacher, dict)
                for depth in CONTROL_DEPTHS:
                    states = trajectory.snapshots[depth][:, offset:end]
                    scores = states[base.readout_indices, :].T @ base.readout_weights
                    score_map = {move.uci(): float(value) for move, value in zip(legal_moves, scores)}
                    predicted = max(score_map, key=lambda move: (score_map[move], move))
                    ordered = sorted(score_map, key=lambda move: (-score_map[move], move))
                    observed = trajectory.observations[depth]
                    readout = states[base.readout_indices, :].T
                    writer.writerow({
                        "position_id": position["position_id"], "source_game_id": position["source_game_id"],
                        "source_split": position["source_split"], "phase": position["phase"], "ply": position["ply"],
                        "fen": position["fen"], "variant": args.variant, "seed": args.seed, "depth": depth,
                        "legal_move_count": len(legal_moves), "teacher_best_move": position["teacher_best"],
                        "predicted_move": predicted, "regret_cp": float(position["teacher_best_cp"] - teacher[predicted]),
                        "teacher_best_agreement": int(predicted == position["teacher_best"]),
                        "fly_rank_of_teacher_best": rank(score_map, position["teacher_best"]),
                        "score_margin": float(score_map[ordered[0]] - score_map[ordered[1]]),
                        "score_dispersion": float(np.std(scores)),
                        "readout_effective_rank": effective_rank(readout),
                        "candidate_dispersion": candidate_dispersion(states[base.readout_indices, :]),
                        "saturation_fraction_mean": float(np.mean(observed["saturation_fraction"][offset:end])),
                        "latency_s": trajectory.latency_s[depth],
                    })
                offset = end
    metadata = {
        "status": "complete", "variant": args.variant, "seed": args.seed,
        "positions": len(positions), "depths": list(CONTROL_DEPTHS), "rows": len(positions) * len(CONTROL_DEPTHS),
        "position_start": args.position_start, "corpus": str(args.corpus), "corpus_sha256": sha256_file(args.corpus),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "control": "corrected directed degree-preserving swaps or transmitter sign permutation",
    }
    (args.output / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
