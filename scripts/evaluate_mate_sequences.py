"""Evaluate trained Plan 3 mate lessons under sequential best defense."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from malecns_rd.chess_agent import FlyCandidateMoveAgent
from malecns_rd.chess_features import HashedSensoryProjector
from malecns_rd.chess_teacher import StockfishTeacher
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.malecns import load_malecns_feather
from malecns_rd.neuromodulated_plasticity import apply_to_graph, load_plasticity_checkpoint
from malecns_rd.readout_training import load_readout_checkpoint


def sha256_graph(graph) -> str:
    digest = hashlib.sha256()
    for values in (graph.body_ids, graph.weights.indptr, graph.weights.indices, graph.weights.data):
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def sha256_structure(graph) -> str:
    digest = hashlib.sha256()
    for values in (graph.weights.indptr, graph.weights.indices):
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_rows(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, int]]:
    roots: list[tuple[str, pd.DataFrame, int]] = []
    for position_id, group in frame.groupby("position_id", sort=True):
        ordered = group.sort_values(["teacher_cp", "move_uci"], ascending=[False, True])
        mate_distance = ordered.iloc[0].get("mate_distance")
        if pd.isna(mate_distance) or int(mate_distance) <= 0:
            continue
        roots.append((str(position_id), group, int(mate_distance)))
    return roots


def evaluate_root(
    position_id: str,
    group: pd.DataFrame,
    root_mate_distance: int,
    agent: FlyCandidateMoveAgent,
    teacher: StockfishTeacher,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    board = chess.Board(str(group.iloc[0]["fen"]))
    root_turn = board.turn
    expected_plies = 2 * root_mate_distance - 1
    own_moves = 0
    preserved = 0
    lost_reason = ""
    move_rows: list[dict[str, object]] = []

    for ply in range(expected_plies + 2):
        if board.is_checkmate():
            break
        if board.is_game_over(claim_draw=True):
            lost_reason = "terminal_without_checkmate"
            break
        if board.turn == root_turn:
            own_moves += 1
            selected = agent.choose_move(board)
            selected_uci = selected.uci()
            before_fen = board.fen()
            board.push(selected)
            if board.is_checkmate():
                remains_forced = True
                after_mate_distance = 0
            else:
                defense = teacher.best_move(board)
                after_mate_distance = defense[2] if defense is not None else None
                remains_forced = after_mate_distance is not None and after_mate_distance < 0
            preserved += int(remains_forced)
            move_rows.append({
                "position_id": position_id,
                "root_mate_distance": root_mate_distance,
                "ply": ply + 1,
                "fly_move": selected_uci,
                "fen_before_move": before_fen,
                "forced_mate_preserved": int(remains_forced),
                "opponent_pov_mate_distance_after_move": after_mate_distance,
            })
            if not remains_forced:
                lost_reason = "forced_mate_lost_after_fly_move"
                break
        else:
            defense = teacher.best_move(board)
            if defense is None:
                lost_reason = "stockfish_no_defense_move"
                break
            board.push(defense[0])

    solved = int(board.is_checkmate() and board.turn != root_turn and own_moves <= root_mate_distance)
    if not solved and not lost_reason:
        lost_reason = "maximum_sequence_length_reached"
    summary = {
        "position_id": position_id,
        "root_mate_distance": root_mate_distance,
        "fly_side": "white" if root_turn else "black",
        "fly_moves": own_moves,
        "expected_plies": expected_plies,
        "forced_mate_preserved_fraction": preserved / own_moves if own_moves else 0.0,
        "solved": solved,
        "terminal_checkmate": int(board.is_checkmate()),
        "lost_reason": lost_reason,
    }
    return summary, move_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--readout-checkpoint", type=Path, required=True)
    parser.add_argument("--plasticity-checkpoint", type=Path, required=True)
    parser.add_argument("--curriculum-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--stockfish-nodes", type=int, default=20_000)
    parser.add_argument("--max-positions", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    graph, _ = load_malecns_feather(
        args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=3
    )
    state, checkpoint_meta = load_plasticity_checkpoint(args.plasticity_checkpoint)
    expected_graph_hash = checkpoint_meta.get("graph_hash")
    if expected_graph_hash and sha256_graph(graph) != expected_graph_hash:
        raise ValueError("plasticity checkpoint was not produced from the supplied graph")
    if checkpoint_meta.get("graph_structure_hash") and sha256_structure(graph) != checkpoint_meta["graph_structure_hash"]:
        raise ValueError("plasticity checkpoint graph topology does not match the supplied graph")
    graph = apply_to_graph(graph, state)

    input_indices = np.asarray(read_manifest(args.input_manifest)["indices"], dtype=np.int64)
    checkpoint = load_readout_checkpoint(args.readout_checkpoint)
    projector = HashedSensoryProjector(graph.n_neurons, input_indices, fanout=4, seed=0, amplitude=1.0)
    engine = RecurrentDepthEngine(graph)
    agent = FlyCandidateMoveAgent(
        engine=engine,
        projector=projector,
        readout_indices=checkpoint.readout_indices,
        readout_weights=checkpoint.readout_weights,
        depth=8,
        clamp_sensory=True,
        name="MaleCNS-RD-Plastic",
    )
    frame = pd.read_csv(args.curriculum_dir / "mates.csv")
    roots = _root_rows(frame.loc[frame["split"].astype(str) == "validation"])
    if args.max_positions is not None:
        roots = roots[: args.max_positions]
    summaries: list[dict[str, object]] = []
    moves: list[dict[str, object]] = []
    with StockfishTeacher(str(args.stockfish), nodes=args.stockfish_nodes) as teacher:
        for position_id, group, mate_distance in roots:
            summary, move_rows = evaluate_root(position_id, group, mate_distance, agent, teacher)
            summaries.append(summary)
            moves.extend(move_rows)

    summary_frame = pd.DataFrame(summaries)
    move_frame = pd.DataFrame(moves)
    summary_frame.to_csv(args.output / "mate_validation.csv", index=False)
    move_frame.to_csv(args.output / "mate_sequence_validation.csv", index=False)
    metadata = {
        "status": "complete",
        "evaluation": "sequential_fly_move_stockfish_best_defense",
        "positions_evaluated": len(summary_frame),
        "by_mate_distance": {
            str(distance): int((summary_frame["root_mate_distance"] == distance).sum())
            for distance in sorted(summary_frame["root_mate_distance"].unique())
        } if not summary_frame.empty else {},
        "aggregate_solve_rate": float(summary_frame["solved"].mean()) if not summary_frame.empty else float("nan"),
        "aggregate_preserved_fraction": float(summary_frame["forced_mate_preserved_fraction"].mean()) if not summary_frame.empty else float("nan"),
        "stockfish_nodes": args.stockfish_nodes,
        "plasticity_checkpoint": str(args.plasticity_checkpoint),
        "readout_checkpoint": str(args.readout_checkpoint),
    }
    (args.output / "mate_sequence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
