"""Run the development-only recurrent-gain and depth factorial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.chess_features import encode_board_move


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
GAIN_SCALES = (0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25)
MODES = ("clamped_sensory", "single_pulse")
RAW_COLUMNS = (
    "position_id", "source_game_id", "source_split", "phase", "ply", "fen",
    "gain_scale", "input_mode", "depth", "legal_move_count", "teacher_best_move",
    "predicted_move", "teacher_best_cp", "predicted_cp", "regret_cp",
    "teacher_best_agreement", "selected_teacher_rank", "fly_rank_of_teacher_best",
    "score_margin", "score_dispersion", "spearman_rank_correlation",
    "kendall_rank_correlation", "state_norm_mean", "state_norm_std", "delta_mean",
    "delta_std", "recurrent_drive_norm_mean", "sensory_drive_norm_mean",
    "recurrent_sensory_ratio_mean", "saturation_fraction_mean", "readout_mean",
    "readout_std", "readout_effective_rank", "candidate_cosine_mean",
    "candidate_dispersion", "winner_switch_from_previous", "d64_winner_rank",
    "d2_winner_rank", "d2_winner_displaced", "latency_s", "trajectory_latency_s",
    "recurrent_passes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_positions(path: Path) -> list[dict[str, object]]:
    import chess

    grouped: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = grouped.setdefault(row["position_id"], {
                "position_id": row["position_id"],
                "source_game_id": row["source_game_id"],
                "phase": row["phase"],
                "ply": int(row["ply"]),
                "fen": row["fen"],
                "teacher": {},
                "best_markers": [],
            })
            teacher = item["teacher"]
            assert isinstance(teacher, dict)
            teacher[row["move_uci"]] = float(row["teacher_cp"])
            if row.get("is_best") == "1":
                markers = item["best_markers"]
                assert isinstance(markers, list)
                markers.append(row["move_uci"])
    positions: list[dict[str, object]] = []
    source_groups = sorted({str(item["source_game_id"]) for item in grouped.values()})
    split_at = len(source_groups) // 2
    source_split = {
        source_id: ("dev_screen" if index < split_at else "dev_confirm")
        for index, source_id in enumerate(source_groups)
    }
    for position_id in sorted(grouped):
        item = grouped[position_id]
        board = chess.Board(str(item["fen"]))
        teacher = item["teacher"]
        assert isinstance(teacher, dict)
        legal = {move.uci() for move in board.legal_moves}
        if board.is_game_over(claim_draw=True) or set(teacher) != legal:
            raise ValueError(f"invalid development position {position_id}")
        best_cp = max(teacher.values())
        markers = item["best_markers"]
        assert isinstance(markers, list)
        best_moves = sorted(set(markers)) or sorted(move for move, cp in teacher.items() if cp == best_cp)
        if len(best_moves) != 1:
            raise ValueError(f"teacher best move is not unique: {position_id}")
        item["board"] = board
        item["teacher_best"] = best_moves[0]
        item["teacher_best_cp"] = best_cp
        item["source_split"] = source_split[str(item["source_game_id"])]
        positions.append(item)
    return positions


def rank(values: dict[str, float], move: str) -> int:
    return sorted(values, key=lambda key: (-values[key], key)).index(move) + 1


def correlations(teacher: dict[str, float], scores: dict[str, float]) -> tuple[float | None, float | None]:
    moves = sorted(teacher)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr([teacher[move] for move in moves], [scores[move] for move in moves]).statistic
        tau = kendalltau([teacher[move] for move in moves], [scores[move] for move in moves]).statistic
    return (
        float(rho) if np.isfinite(rho) else None,
        float(tau) if np.isfinite(tau) else None,
    )


def effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    power = singular * singular
    if not np.any(power > 0):
        return 0.0
    probability = power[power > 0] / power.sum()
    return float(np.exp(-np.sum(probability * np.log(probability))))


def cosine_summary(states: np.ndarray, sample_indices: np.ndarray) -> tuple[float, float]:
    sampled = states[sample_indices, :].T.astype(np.float64)
    norms = np.linalg.norm(sampled, axis=1)
    valid = norms > 1e-9
    if valid.sum() < 2:
        return 0.0, 1.0
    normalized = sampled[valid] / norms[valid, None]
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(len(normalized), k=1)]
    mean_cosine = float(np.mean(upper)) if len(upper) else 0.0
    return mean_cosine, 1.0 - mean_cosine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--neurotransmitters", required=True, type=Path)
    parser.add_argument("--connectome-weights", required=True, type=Path)
    parser.add_argument("--sensory-indices", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/gain_factorial"))
    parser.add_argument("--limit-positions", type=int, default=0)
    parser.add_argument("--batch-positions", type=int, default=4)
    parser.add_argument("--gain-scale", type=float, choices=GAIN_SCALES)
    parser.add_argument("--input-mode", choices=MODES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if args.limit_positions < 0 or args.batch_positions < 1:
        parser.error("--limit-positions must be >= 0 and --batch-positions must be positive")
    positions = load_positions(args.corpus)
    if args.limit_positions:
        positions = positions[:args.limit_positions]
    if not positions:
        raise ValueError("development corpus is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    selected_gains = (args.gain_scale,) if args.gain_scale is not None else GAIN_SCALES
    selected_modes = (args.input_mode,) if args.input_mode is not None else MODES
    raw_name = (
        "raw_factorial.csv"
        if args.gain_scale is None and args.input_mode is None
        else f"raw_gain_{args.gain_scale if args.gain_scale is not None else 'all'}_{args.input_mode or 'all'}.csv"
    )
    raw_path = args.output / raw_name
    progress_path = args.output / f"{raw_path.stem}.progress.json"
    completed: set[tuple[str, str, str, int]] = set()
    if args.resume and raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                completed.add((row["position_id"], row["gain_scale"], row["input_mode"], int(row["depth"])))
    mode = "a" if args.resume and raw_path.exists() else "w"
    base = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
    )
    if not isinstance(base.agent.engine, RecurrentDepthEngine):
        raise RuntimeError("mechanistic factorial requires the rate checkpoint")
    graph = base.agent.engine.graph
    sample_indices = np.linspace(0, graph.n_neurons - 1, min(2048, graph.n_neurons), dtype=np.int64)
    writer_handle = raw_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(writer_handle, fieldnames=RAW_COLUMNS)
    if mode == "w":
        writer.writeheader()
    completed_count = len(completed)
    total = len(positions) * len(selected_gains) * len(selected_modes) * len(DEPTHS)
    for gain_scale in selected_gains:
        engine = RecurrentDepthEngine(
            graph,
            leak=base.agent.engine.leak,
            recurrent_gain=base.agent.engine.recurrent_gain * gain_scale,
            input_gain=base.agent.engine.input_gain,
            activation=base.agent.engine.activation,
        )
        for input_mode in selected_modes:
            clamp = input_mode == "clamped_sensory"
            for batch_start in range(0, len(positions), args.batch_positions):
                batch_positions = positions[batch_start:batch_start + args.batch_positions]
                batch_legal_moves = []
                sensory_blocks = []
                for position in batch_positions:
                    board = position["board"]
                    legal_moves = sorted(board.legal_moves, key=lambda move: move.uci())
                    batch_legal_moves.append(legal_moves)
                    sensory_blocks.append(np.column_stack([
                        base.agent.projector.project(encode_board_move(board, move))
                        for move in legal_moves
                    ]))
                sensory = np.concatenate(sensory_blocks, axis=1)
                trajectory = engine.run_batch_trajectory(
                    sensory, depths=DEPTHS, clamp_sensory=clamp, observe=True
                )
                column_start = 0
                for position, legal_moves in zip(batch_positions, batch_legal_moves):
                    column_end = column_start + len(legal_moves)
                    teacher = position["teacher"]
                    assert isinstance(teacher, dict)
                    local_snapshots = {
                        depth: trajectory.snapshots[depth][:, column_start:column_end]
                        for depth in DEPTHS
                    }
                    local_observations = {
                        depth: {
                            name: value[column_start:column_end]
                            for name, value in trajectory.observations[depth].items()
                        }
                        for depth in DEPTHS
                    }
                    scores_by_depth: dict[int, dict[str, float]] = {}
                    moves_by_depth: dict[int, str] = {}
                    for depth in DEPTHS:
                        scores = local_snapshots[depth][base.agent.readout_indices, :].T @ base.agent.readout_weights
                        score_map = {move.uci(): float(score) for move, score in zip(legal_moves, scores)}
                        scores_by_depth[depth] = score_map
                        moves_by_depth[depth] = max(score_map, key=lambda move: (score_map[move], move))
                    d64_move = moves_by_depth[64]
                    d2_move = moves_by_depth[2]
                    for depth in DEPTHS:
                        key = (str(position["position_id"]), f"{gain_scale:.2f}", input_mode, depth)
                        if key in completed:
                            continue
                        score_map = scores_by_depth[depth]
                        predicted = moves_by_depth[depth]
                        observed = local_observations[depth]
                        readout = local_snapshots[depth][base.agent.readout_indices, :].T
                        order = sorted(score_map, key=lambda move: (-score_map[move], move))
                        rho, tau = correlations(teacher, score_map)
                        cosine, dispersion = cosine_summary(local_snapshots[depth], sample_indices)
                        d64_rank = rank(score_map, d64_move)
                        d2_rank = rank(score_map, d2_move)
                        values = {
                        "position_id": position["position_id"],
                        "source_game_id": position["source_game_id"],
                        "source_split": position["source_split"],
                        "phase": position["phase"], "ply": position["ply"], "fen": position["fen"],
                        "gain_scale": f"{gain_scale:.2f}", "input_mode": input_mode, "depth": depth,
                        "legal_move_count": len(legal_moves), "teacher_best_move": position["teacher_best"],
                        "predicted_move": predicted, "teacher_best_cp": position["teacher_best_cp"],
                        "predicted_cp": teacher[predicted], "regret_cp": float(position["teacher_best_cp"] - teacher[predicted]),
                        "teacher_best_agreement": int(predicted == position["teacher_best"]),
                        "selected_teacher_rank": rank(teacher, predicted),
                        "fly_rank_of_teacher_best": rank(score_map, position["teacher_best"]),
                        "score_margin": score_map[order[0]] - score_map[order[1]],
                        "score_dispersion": float(np.std(scores)), "spearman_rank_correlation": rho,
                        "kendall_rank_correlation": tau,
                        "state_norm_mean": float(np.mean(observed["state_norm"])),
                        "state_norm_std": float(np.std(observed["state_norm"])),
                        "delta_mean": float(np.mean(observed["delta"])),
                        "delta_std": float(np.std(observed["delta"])),
                        "recurrent_drive_norm_mean": float(np.mean(observed["recurrent_drive_norm"])),
                        "sensory_drive_norm_mean": float(np.mean(observed["sensory_drive_norm"])),
                        "recurrent_sensory_ratio_mean": float(np.mean(observed["recurrent_sensory_ratio"])),
                        "saturation_fraction_mean": float(np.mean(observed["saturation_fraction"])),
                        "readout_mean": float(np.mean(readout)), "readout_std": float(np.std(readout)),
                        "readout_effective_rank": effective_rank(readout),
                        "candidate_cosine_mean": cosine, "candidate_dispersion": dispersion,
                        "winner_switch_from_previous": int(depth > 1 and predicted != moves_by_depth[max(d for d in DEPTHS if d < depth)]),
                        "d64_winner_rank": d64_rank, "d2_winner_rank": d2_rank,
                        "d2_winner_displaced": int(predicted != d2_move),
                        "latency_s": trajectory.latency_s[depth], "trajectory_latency_s": trajectory.latency_s[depth],
                        "recurrent_passes": len(legal_moves) * depth,
                        }
                        writer.writerow(values)
                        writer_handle.flush()
                        completed.add(key)
                        completed_count += 1
                    column_start = column_end
                progress_path.write_text(json.dumps({
                    "status": "running", "completed_cells": completed_count, "total_cells": total,
                    "positions": len(positions), "batch_positions": args.batch_positions,
                    "source_split": "first half dev_screen, second half dev_confirm",
                }, indent=2) + "\n", encoding="utf-8")
    writer_handle.close()
    metadata = {
        "status": "complete", "positions": len(positions), "source_groups": len({p["source_game_id"] for p in positions}),
        "source_split": "source groups split lexicographically in half before scoring",
        "depths": list(DEPTHS), "gain_scales": list(selected_gains), "input_modes": list(selected_modes),
        "grid_cells": total, "rows": completed_count, "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint), "seed": args.seed,
        "graph_neurons": graph.n_neurons, "graph_edges": graph.n_edges,
        "observability": "rate-engine hidden-state, drive, saturation, readout, candidate geometry, ranking and latency metrics",
    }
    (args.output / f"{raw_path.stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    progress_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
