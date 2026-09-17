"""Run Plan 3: frozen decoder plus localized reward/aversive plasticity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.chess_features import HashedSensoryProjector, encode_board_move_unchecked
from malecns_rd.chess_teacher import StockfishTeacher
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.malecns import load_malecns_feather
from malecns_rd.neuromodulated_plasticity import (
    PlasticityConfig, PlasticityState, apply_to_graph, save_plasticity_checkpoint,
    select_kc_mbon_edges,
)
from malecns_rd.readout_training import save_readout_checkpoint, train_pairwise_readout

import chess


STAGES = ("movement", "endgames", "tactics", "mates")
DEPTHS = (4, 8, 16)
OUTPUT_FILES = {
    "movement": "movement_validation.csv", "endgames": "endgame_validation.csv",
    "tactics": "tactical_validation.csv", "mates": "mate_validation.csv",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def candidate_batch(frame: pd.DataFrame, projector: HashedSensoryProjector) -> tuple[chess.Board, list[chess.Move], np.ndarray]:
    board = chess.Board(str(frame.iloc[0]["fen"]))
    stage = str(frame.iloc[0]["stage"])
    moves = [chess.Move.from_uci(str(value)) for value in frame["move_uci"]]
    projected = np.column_stack([
        projector.project(encode_board_move_unchecked(board, move)) for move in moves
    ])
    return board, moves, projected


def grouped(frame: pd.DataFrame):
    key = "position_id" if "position_id" in frame.columns else "lesson_id"
    return frame.groupby(key, sort=True)


def extract_features(frame: pd.DataFrame, engine: RecurrentDepthEngine, projector: HashedSensoryProjector, readout_indices: np.ndarray, depth: int) -> tuple[np.ndarray, pd.DataFrame]:
    blocks: list[np.ndarray] = []
    pieces: list[pd.DataFrame] = []
    for _, group in grouped(frame):
        _, _, projected = candidate_batch(group, projector)
        state = engine.run_batch(projected, max_depth=depth, clamp_sensory=True)
        blocks.append(state[readout_indices].T.astype(np.float32, copy=False))
        pieces.append(group.reset_index(drop=True))
    if not blocks:
        raise ValueError("curriculum has no rows")
    return np.vstack(blocks), pd.concat(pieces, ignore_index=True)


def stockfish_scores(teacher: StockfishTeacher | None, board: chess.Board, group: pd.DataFrame) -> dict[str, tuple[float, int | None]]:
    if teacher is None:
        return {}
    return {item.move_uci: (item.teacher_cp, item.mate_distance) for item in teacher.evaluate_position(board, position_id=str(group.iloc[0]["position_id"]))}


def validate_stage(frame: pd.DataFrame, engine: RecurrentDepthEngine, projector: HashedSensoryProjector, readout_indices: np.ndarray, readout_weights: np.ndarray, depths: tuple[int, ...], teacher: StockfishTeacher | None, output: Path, stage: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for position_id, group in grouped(frame):
        board, moves, projected = candidate_batch(group, projector)
        sf = stockfish_scores(teacher, board, group) if group["teacher_cp"].isna().any() else {}
        scores_by_depth = engine.run_batch_trajectory(projected, depths=depths, clamp_sensory=True)
        best_cp = max(float(sf.get(move.uci(), (row["teacher_cp"], row.get("mate_distance")))[0]) for move, (_, row) in zip(moves, group.iterrows()))
        for depth in depths:
            scores = scores_by_depth.snapshots[depth][readout_indices].T @ readout_weights
            selected = int(np.argmax(scores))
            selected_move = moves[selected].uci()
            row = group.iloc[selected]
            selected_cp, mate_distance = sf.get(selected_move, (float(row["teacher_cp"]), row.get("mate_distance")))
            rows.append({
                "stage": stage, "position_id": str(position_id), "depth": depth,
                "selected_move": selected_move, "teacher_best_cp": best_cp,
                "selected_cp": selected_cp, "regret_cp": max(0.0, best_cp - selected_cp),
                "top1_correct": int(abs(best_cp - selected_cp) < 1e-6),
                "mate_flag": int(mate_distance is not None), "mate_distance": mate_distance,
            })
    result = pd.DataFrame(rows, columns=[
        "stage", "position_id", "depth", "selected_move", "teacher_best_cp",
        "selected_cp", "regret_cp", "top1_correct", "mate_flag", "mate_distance",
    ])
    result.to_csv(output / OUTPUT_FILES[stage], index=False)
    depth8 = result.loc[result["depth"] == 8] if not result.empty else result
    return {
        "stage": stage, "validation_positions": int(depth8["position_id"].nunique()) if not depth8.empty else 0,
        "top1_accuracy_d8": float(depth8["top1_correct"].mean()) if not depth8.empty else float("nan"),
        "mean_regret_cp_d8": float(depth8["regret_cp"].mean()) if not depth8.empty else float("nan"),
        "mate_solve_rate_d8": float(depth8.loc[depth8["mate_flag"] == 1, "top1_correct"].mean()) if not depth8.loc[depth8["mate_flag"] == 1].empty else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--readout-manifest", type=Path, required=True)
    parser.add_argument("--curriculum-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/neuromodulated_curriculum_v1"))
    parser.add_argument("--stockfish", type=Path)
    parser.add_argument("--stockfish-nodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--control", choices=("reward_aversive", "frozen", "shuffled"), default="reward_aversive")
    parser.add_argument("--max-positions-per-stage", type=int)
    parser.add_argument("--readout-epochs", type=int, default=20)
    args = parser.parse_args()
    if args.depth != 8:
        raise ValueError("Plan 3 plasticity depth is fixed at D8")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    graph, _ = load_malecns_feather(args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=3)
    input_manifest = read_manifest(args.input_manifest)
    readout_manifest = read_manifest(args.readout_manifest)
    sensory_indices = np.asarray(input_manifest["indices"], dtype=np.int64)
    readout_indices = np.asarray(readout_manifest["indices"], dtype=np.int64)
    # Keep the original sensory interface path in metadata, but use the Plan 3
    # anatomical input manifest for the plasticity experiment.
    baseline_sensory = np.load(args.sensory_indices).astype(np.int64)
    if len(sensory_indices) == 0 or len(readout_indices) == 0:
        raise ValueError("input/readout manifests must be non-empty")
    projector = HashedSensoryProjector(graph.n_neurons, sensory_indices, fanout=4, seed=0, amplitude=1.0)
    engine = RecurrentDepthEngine(graph)
    kc = select_kc_mbon_edges(graph, sensory_indices, readout_indices, config=PlasticityConfig())
    topology_hash = sha256_graph(graph)
    structure_hash = sha256_structure(graph)

    frames = {stage: pd.read_csv(args.curriculum_dir / f"{stage}.csv") for stage in STAGES}
    if args.max_positions_per_stage is not None:
        for stage in STAGES:
            stage_frame = frames[stage]
            train_positions = sorted(stage_frame.loc[stage_frame["split"].astype(str) == "train", "position_id"].astype(str).unique())
            validation_positions = sorted(stage_frame.loc[stage_frame["split"].astype(str) == "validation", "position_id"].astype(str).unique())
            train_limit = max(1, int(round(args.max_positions_per_stage * 0.8)))
            positions = train_positions[:train_limit] + validation_positions[:max(1, args.max_positions_per_stage - train_limit)]
            frames[stage] = frames[stage].loc[frames[stage]["position_id"].astype(str).isin(positions)].copy()
    train = pd.concat([frames[stage].loc[frames[stage]["split"].astype(str) == "train"] for stage in STAGES], ignore_index=True)
    features, train_frame = extract_features(train, engine, projector, readout_indices, args.depth)
    fit = train_pairwise_readout(features, train_frame, epochs=args.readout_epochs, seed=args.seed, validation_fraction=0.2)
    decoder_path = args.output / "frozen_decoder_checkpoint.npz"
    decoder_metadata = {
        "status": "frozen_before_plasticity", "decoder_source": "curriculum_train_only",
        "curriculum_dir": str(args.curriculum_dir), "curriculum_metadata_sha256": sha256_file(args.curriculum_dir / "metadata.json"),
        "readout_manifest_sha256": sha256_file(args.readout_manifest), "graph_hash": topology_hash,
        "recurrent_depth": 8, "dynamics": "RecurrentDepthEngine", "projector_seed": 0,
        "projector_fanout": 4, "projector_amplitude": 1.0, "train_top1_accuracy": fit.train_top1_accuracy,
        "validation_top1_accuracy": fit.validation_top1_accuracy, "selected_epoch": fit.best_epoch,
    }
    save_readout_checkpoint(decoder_path, readout_weights=fit.best_weights, readout_indices=readout_indices, recurrent_depth=8, projector_seed=0, metadata=decoder_metadata)

    checkpoint_meta = {"checkpoint": "before_plasticity", "graph_hash": topology_hash, "topology_hash": topology_hash, "control": args.control}
    save_plasticity_checkpoint(args.output / "checkpoint_before_plasticity.npz", kc, checkpoint_meta)
    with_teacher = StockfishTeacher(str(args.stockfish), nodes=args.stockfish_nodes) if args.stockfish else None
    teacher_metadata = with_teacher.metadata() if with_teacher is not None else None
    running_mean = 0.0
    rng = random.Random(args.seed)
    shuffle_buffer: list[float] = []
    training_rows: list[dict[str, object]] = []
    biological_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    prior_train_frames: list[pd.DataFrame] = []
    try:
        for stage in STAGES:
            train_stage = frames[stage].loc[frames[stage]["split"].astype(str) == "train"]
            train_stage = train_stage.assign(pass_kind="curriculum")
            if prior_train_frames:
                prior = pd.concat(prior_train_frames, ignore_index=True)
                prior_positions = sorted(prior["position_id"].astype(str).unique())
                replay_count = max(1, int(math.ceil(0.2 * len(prior_positions))))
                replay_positions = rng.sample(prior_positions, min(replay_count, len(prior_positions)))
                replay = prior.loc[prior["position_id"].astype(str).isin(replay_positions)].copy().assign(pass_kind="replay")
                train_stage = pd.concat([train_stage, replay], ignore_index=True)
            for position_id, group in grouped(train_stage):
                board, moves, projected = candidate_batch(group, projector)
                sf = stockfish_scores(with_teacher, board, group) if group["teacher_cp"].isna().any() else {}
                trajectory = engine.run_batch_trajectory(projected, depths=tuple(range(1, args.depth + 1)), clamp_sensory=True, observe=True)
                scores = trajectory.snapshots[args.depth][readout_indices].T @ fit.best_weights
                selected = int(np.argmax(scores))
                selected_move = moves[selected].uci()
                selected_row = group.iloc[selected]
                fallback_cp = float(selected_row["teacher_cp"]) if not pd.isna(selected_row["teacher_cp"]) else 0.0
                selected_cp, mate_distance = sf.get(selected_move, (fallback_cp, selected_row.get("mate_distance")))
                teacher_values = [float(v) for v in group["teacher_cp"].dropna().tolist()]
                cp_best = max(teacher_values) if teacher_values else selected_cp
                if sf:
                    cp_best = max(value[0] for value in sf.values())
                regret = max(0.0, cp_best - selected_cp)
                raw_signal = 1.0 - 2.0 * math.tanh(regret / 400.0)
                advantage = raw_signal - running_mean
                running_mean = 0.95 * running_mean + 0.05 * raw_signal
                applied = advantage
                if args.control == "shuffled":
                    shuffle_buffer.append(advantage)
                    if len(shuffle_buffer) > 8:
                        applied = shuffle_buffer.pop(rng.randrange(len(shuffle_buffer)))
                eligibility = kc.eligibility_from_trajectory({depth: trajectory.snapshots[depth][:, selected] for depth in range(1, args.depth + 1)})
                diagnostics = kc.apply(applied, eligibility) if args.control != "frozen" else kc.metrics()
                if args.control != "frozen":
                    graph = apply_to_graph(graph, kc)
                    engine = RecurrentDepthEngine(graph)
                row = {
                    "step": len(training_rows), "stage": stage, "position_id": str(position_id),
                    "selected_move": selected_move, "cp_best": cp_best, "cp_move": selected_cp,
                    "regret_cp": regret, "raw_signal": raw_signal, "running_signal_mean": running_mean,
                    "centered_advantage": advantage, "applied_advantage": applied,
                    "mate_flag": int(mate_distance is not None), "mate_distance": mate_distance,
                    "selected_is_teacher_best": int(selected_cp >= cp_best), "control": args.control,
                    "pass_kind": str(group.iloc[0].get("pass_kind", "curriculum")),
                    "eligibility_mean_abs": diagnostics.get("eligibility_mean_abs", float("nan")),
                    "rms_log_ratio": kc.metrics()["rms_log_ratio"], "elapsed_s": time.perf_counter() - started,
                }
                training_rows.append(row)
                signal_rows.append({"stage": stage, "raw_signal": raw_signal, "centered_advantage": advantage, "applied_advantage": applied, "regret_cp": regret, "mate_flag": int(mate_distance is not None)})
            prior_train_frames.append(frames[stage].loc[frames[stage]["split"].astype(str) == "train"].copy())
            save_plasticity_checkpoint(args.output / f"checkpoint_after_{stage}.npz", kc, {"checkpoint": f"after_{stage}", "graph_hash_initial": topology_hash, "control": args.control, "stage": stage})
            save_plasticity_checkpoint(args.output / "checkpoint_replay_fallback.npz", kc, {"checkpoint": "replay_fallback", "stage": stage, "graph_hash_initial": topology_hash, "control": args.control})
            validation = validate_stage(frames[stage].loc[frames[stage]["split"].astype(str) == "validation"], engine, projector, readout_indices, fit.best_weights, (8,), with_teacher, args.output, stage)
            validation["plasticity"] = kc.metrics()
            stage_rows.append(validation)
            biological_rows.append({"checkpoint": f"after_{stage}", "stage": stage, "graph_neurons": graph.n_neurons, "graph_edges": graph.n_edges, "structure_hash_initial": structure_hash, "structure_hash_current": sha256_structure(graph), "topology_preserved": int(sha256_structure(graph) == structure_hash), "sign_preserved": int(np.all(np.sign(kc.current_weights) == np.sign(kc.original_weights))), **kc.metrics()})
    finally:
        if with_teacher is not None:
            with_teacher.close()

    pd.DataFrame(training_rows).to_csv(args.output / "training_log.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(args.output / "stage_metrics.csv", index=False)
    pd.DataFrame(biological_rows).to_csv(args.output / "biological_deviation.csv", index=False)
    pd.DataFrame(signal_rows).groupby("stage", as_index=False).agg({"raw_signal": ["mean", "std", "min", "max"], "centered_advantage": "mean", "applied_advantage": "mean", "regret_cp": "mean", "mate_flag": "mean"}).to_csv(args.output / "reward_signal_distribution.csv", index=False)

    final_depth_rows: list[dict[str, object]] = []
    for stage in STAGES:
        frame = frames[stage].loc[frames[stage]["split"].astype(str) == "validation"]
        if frame.empty:
            continue
        board_count = 0
        for position_id, group in grouped(frame):
            board, moves, projected = candidate_batch(group, projector)
            traj = engine.run_batch_trajectory(projected, depths=DEPTHS, clamp_sensory=True)
            best_cp = float(group["teacher_cp"].max())
            for depth in DEPTHS:
                scores = traj.snapshots[depth][readout_indices].T @ fit.best_weights
                selected_cp = float(group.iloc[int(np.argmax(scores))]["teacher_cp"])
                final_depth_rows.append({"stage": stage, "position_id": str(position_id), "depth": depth, "top1_correct": int(selected_cp >= best_cp), "regret_cp": max(0.0, best_cp - selected_cp)})
            board_count += 1
    pd.DataFrame(final_depth_rows).to_csv(args.output / "depth_metrics.csv", index=False)
    from plot_neuromodulated_results import generate
    generate(args.output)

    metadata = {
        "status": "complete", "control": args.control, "seed": args.seed, "depth_train": 8, "validation_depths": list(DEPTHS),
        "graph_neurons": graph.n_neurons, "graph_edges": graph.n_edges, "graph_hash_initial": topology_hash, "graph_structure_hash": structure_hash,
        "plastic_edge_count": kc.edge_count, "plastic_edge_hash": kc.edge_hash, "plasticity": kc.config.__dict__,
        "curriculum_metadata_sha256": sha256_file(args.curriculum_dir / "metadata.json"),
        "input_manifest_sha256": sha256_file(args.input_manifest), "readout_manifest_sha256": sha256_file(args.readout_manifest),
        "baseline_sensory_indices_sha256": sha256_file(args.sensory_indices), "decoder": str(decoder_path),
        "stockfish": teacher_metadata,
        "training_positions": len(training_rows), "elapsed_s": time.perf_counter() - started,
        "stop_conditions": {"max_runtime_hours": 10, "all_stages_present": True, "passes_per_stage": 1},
        "outputs": ["plastic_edge_audit.csv", "plastic_edge_metadata.json", "training_log.csv", "stage_metrics.csv", "movement_validation.csv", "endgame_validation.csv", "tactical_validation.csv", "mate_validation.csv", "biological_deviation.csv", "reward_signal_distribution.csv", "depth_metrics.csv", "run_metadata.json", "final_summary.md"],
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    summary = pd.DataFrame(stage_rows)
    (args.output / "final_summary.md").write_text("# Plan 3 neuromodulated curriculum\n\n```text\n" + summary.to_string(index=False) + "\n```\n\nControl: `" + args.control + "`. The decoder was fit before plasticity, and only predeclared existing KC-to-MBON edges were updated.\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
