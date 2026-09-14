"""Run the independent all-legal, multi-depth recurrent-depth study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.stats import kendalltau, spearmanr

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_agent import FlyCandidateMoveAgent
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.graph import ConnectomeGraph
from malecns_rd.position_analysis import AnalysisConfig, StockfishPositionEvaluator


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
VARIANTS = (
    "original",
    "degree_preserving_topology_shuffle",
    "transmitter_sign_shuffle",
    "recurrent_edges_attenuated_0.05",
)
RAW_COLUMNS = (
    "position_id", "source_game_id", "phase", "ply", "fen", "side_to_move",
    "variant", "depth", "legal_move_count", "teacher_best_move",
    "predicted_move", "selected_teacher_cp", "teacher_best_cp", "regret_cp",
    "teacher_best_agreement", "top3_teacher_agreement", "selected_teacher_rank",
    "fly_rank_of_teacher_best", "score_margin", "score_dispersion",
    "spearman_rank_correlation", "kendall_rank_correlation", "recurrent_passes",
    "latency_s", "trajectory_latency_s", "stockfish_eval_loss_cp",
    "post_eval_included",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(path: Path):
    import chess

    grouped: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            position_id = row["position_id"]
            item = grouped.setdefault(
                position_id,
                {
                    "position_id": position_id,
                    "source_game_id": row["source_game_id"],
                    "phase": row["phase"],
                    "ply": int(row["ply"]),
                    "fen": row["fen"],
                    "side_to_move": row["side_to_move"],
                    "candidates": {},
                    "best_markers": [],
                },
            )
            if row["fen"] != item["fen"]:
                raise ValueError(f"position {position_id} contains multiple FENs")
            candidates = item["candidates"]
            assert isinstance(candidates, dict)
            candidates[row["move_uci"]] = float(row["teacher_cp"])
            markers = item["best_markers"]
            assert isinstance(markers, list)
            if int(row["is_best"]):
                markers.append(row["move_uci"])
    positions = []
    for position_id in sorted(grouped):
        item = grouped[position_id]
        board = chess.Board(item["fen"])
        candidates = item["candidates"]
        assert isinstance(candidates, dict)
        legal = {move.uci() for move in board.legal_moves}
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"evaluation position is terminal: {position_id}")
        if set(candidates) != legal:
            raise ValueError(f"corpus candidates do not match legal moves: {position_id}")
        if len(candidates) < 2:
            raise ValueError(f"evaluation position has fewer than two legal moves: {position_id}")
        best_cp = max(candidates.values())
        markers = item["best_markers"]
        assert isinstance(markers, list)
        best_moves = sorted(set(markers))
        if not best_moves:
            best_moves = sorted(move for move, cp in candidates.items() if cp == best_cp)
        if len(best_moves) != 1:
            raise ValueError(f"teacher best move is not unique: {position_id}")
        item["board"] = board
        item["teacher_best"] = best_moves[0]
        item["teacher_best_cp"] = best_cp
        positions.append(item)
    if not positions:
        raise ValueError("evaluation corpus is empty")
    return positions


def graph_variant(graph: ConnectomeGraph, variant: str, seed: int) -> ConnectomeGraph:
    if variant == "original":
        return graph
    if variant == "recurrent_edges_attenuated_0.05":
        return ConnectomeGraph(graph.body_ids.copy(), (graph.weights * np.float32(0.05)).tocsr())
    rng = np.random.default_rng(seed)
    weights = graph.weights.tocoo(copy=True)
    if variant == "degree_preserving_topology_shuffle":
        shuffled_dst = weights.row.copy()
        rng.shuffle(shuffled_dst)
        return ConnectomeGraph(
            graph.body_ids.copy(),
            sp.coo_matrix((weights.data, (shuffled_dst, weights.col)), shape=weights.shape).tocsr(),
        )
    if variant == "transmitter_sign_shuffle":
        data = weights.data.copy()
        signs = np.sign(data)
        rng.shuffle(signs)
        return ConnectomeGraph(
            graph.body_ids.copy(),
            sp.coo_matrix((np.abs(data) * signs, (weights.row, weights.col)), shape=weights.shape).tocsr(),
        )
    raise ValueError(f"unknown graph variant: {variant}")


def make_agent(base, graph: ConnectomeGraph) -> FlyCandidateMoveAgent:
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("position depth study requires a rate checkpoint")
    engine = RecurrentDepthEngine(
        graph,
        leak=base.engine.leak,
        recurrent_gain=base.engine.recurrent_gain,
        input_gain=base.engine.input_gain,
        activation=base.engine.activation,
    )
    return FlyCandidateMoveAgent(
        engine=engine,
        projector=base.projector,
        readout_indices=base.readout_indices,
        readout_weights=base.readout_weights,
        depth=64,
        clamp_sensory=base.clamp_sensory,
        max_candidates=None,
        name=base.name,
    )


def _rank(values: dict[str, float], move: str) -> int:
    ordered = sorted(values, key=lambda key: (-values[key], key))
    return ordered.index(move) + 1


def _correlation(teacher: dict[str, float], fly_scores: dict[str, float]) -> tuple[float | None, float | None]:
    moves = sorted(teacher)
    teacher_values = [teacher[move] for move in moves]
    fly_values = [fly_scores[move] for move in moves]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(teacher_values, fly_values).statistic
        tau = kendalltau(teacher_values, fly_values).statistic
    return (
        float(rho) if np.isfinite(rho) else None,
        float(tau) if np.isfinite(tau) else None,
    )


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--neurotransmitters", required=True, type=Path)
    parser.add_argument("--connectome-weights", required=True, type=Path)
    parser.add_argument("--sensory-indices", required=True, type=Path)
    parser.add_argument("--evaluation-corpus", type=Path, default=Path("data/chess_evaluation_corpus_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/position_depth_study_v2"))
    parser.add_argument("--limit-positions", type=int, default=0, help="Development limit; 0 uses the full corpus")
    parser.add_argument("--post-eval-count", type=int, default=32)
    parser.add_argument("--analysis-depth", type=int, default=6)
    parser.add_argument("--analysis-threads", type=int, default=1)
    parser.add_argument("--analysis-hash-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--variant", choices=VARIANTS, help="Run one isolated variant; default runs all variants")
    args = parser.parse_args()
    if args.post_eval_count < 0:
        parser.error("--post-eval-count must be >= 0")
    if args.limit_positions < 0:
        parser.error("--limit-positions must be >= 0")

    positions = load_corpus(args.evaluation_corpus)
    if args.limit_positions:
        positions = positions[:args.limit_positions]
    selected_variants = (args.variant,) if args.variant else VARIANTS
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw_position_metrics.csv"
    progress_path = args.output / "progress.json"
    existing: set[tuple[str, str, int]] = set()
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing.add((row["variant"], row["position_id"], int(row["depth"])))
    mode = "a" if raw_path.exists() else "w"
    selected_for_post_eval = {item["position_id"] for item in positions[:args.post_eval_count]}
    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
        min_synapses=3,
    )
    base = loaded.agent
    variants = {name: graph_variant(base.engine.graph, name, args.seed) for name in selected_variants}
    evaluator = StockfishPositionEvaluator(
        AnalysisConfig(
            str(args.stockfish),
            depth=args.analysis_depth,
            threads=args.analysis_threads,
            hash_mb=args.analysis_hash_mb,
        )
    ) if selected_for_post_eval else None
    baseline: dict[str, float | None] = {}
    if evaluator is not None:
        try:
            import chess
            for index, item in enumerate(positions):
                if item["position_id"] not in selected_for_post_eval:
                    continue
                evaluation = evaluator.evaluate(
                    item["board"],
                    game_index=index,
                    fly_is_white=item["board"].turn == chess.WHITE,
                    last_move=None,
                    last_actor=None,
                )
                baseline[item["position_id"]] = evaluation.fly_cp
        finally:
            evaluator.close()

    with raw_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        if mode == "w":
            writer.writeheader()
        for variant_index, variant in enumerate(selected_variants):
            agent = make_agent(base, variants[variant])
            post_evaluator = StockfishPositionEvaluator(
                AnalysisConfig(
                    str(args.stockfish),
                    depth=args.analysis_depth,
                    threads=args.analysis_threads,
                    hash_mb=args.analysis_hash_mb,
                )
            ) if selected_for_post_eval else None
            try:
                for position_index, item in enumerate(positions):
                    position_id = item["position_id"]
                    needed = [depth for depth in DEPTHS if (variant, position_id, depth) not in existing]
                    if not needed:
                        continue
                    trajectory_started = time.perf_counter()
                    ranked_by_depth = agent.rank_moves_by_depth(item["board"], DEPTHS)
                    trajectory_wall = time.perf_counter() - trajectory_started
                    teacher = item["candidates"]
                    assert isinstance(teacher, dict)
                    teacher_best = item["teacher_best"]
                    teacher_best_cp = float(item["teacher_best_cp"])
                    for depth in DEPTHS:
                        if depth not in needed:
                            continue
                        ranked = ranked_by_depth[depth]
                        fly_scores = {uci: float(score) for score, uci, _ in ranked}
                        predicted = ranked[0][1]
                        predicted_cp = float(teacher[predicted])
                        teacher_rank = _rank(teacher, predicted)
                        fly_best_rank = next(index + 1 for index, row in enumerate(ranked) if row[1] == teacher_best)
                        rho, tau = _correlation(teacher, fly_scores)
                        after_loss = None
                        post_included = position_id in selected_for_post_eval
                        if post_included and post_evaluator is not None:
                            after_board = item["board"].copy(stack=False)
                            after_board.push_uci(predicted)
                            after = post_evaluator.evaluate(
                                after_board,
                                game_index=position_index,
                                fly_is_white=item["board"].turn,
                                last_move=predicted,
                                last_actor="fly",
                            ).fly_cp
                            before = baseline.get(position_id)
                            if before is not None and after is not None:
                                after_loss = max(0.0, float(before - after))
                        row = {
                            "position_id": position_id,
                            "source_game_id": item["source_game_id"],
                            "phase": item["phase"],
                            "ply": item["ply"],
                            "fen": item["fen"],
                            "side_to_move": item["side_to_move"],
                            "variant": variant,
                            "depth": depth,
                            "legal_move_count": len(teacher),
                            "teacher_best_move": teacher_best,
                            "predicted_move": predicted,
                            "selected_teacher_cp": predicted_cp,
                            "teacher_best_cp": teacher_best_cp,
                            "regret_cp": teacher_best_cp - predicted_cp,
                            "teacher_best_agreement": int(predicted == teacher_best),
                            "top3_teacher_agreement": int(teacher_best in [row[1] for row in ranked[:3]]),
                            "selected_teacher_rank": teacher_rank,
                            "fly_rank_of_teacher_best": fly_best_rank,
                            "score_margin": float(ranked[0][0] - ranked[1][0]) if len(ranked) > 1 else None,
                            "score_dispersion": float(np.std(list(fly_scores.values()))),
                            "spearman_rank_correlation": rho,
                            "kendall_rank_correlation": tau,
                            "recurrent_passes": len(teacher) * depth,
                            "latency_s": agent.last_decisions_by_depth[depth].get("latency_s"),
                            "trajectory_latency_s": trajectory_wall,
                            "stockfish_eval_loss_cp": after_loss,
                            "post_eval_included": int(post_included),
                        }
                        writer.writerow(row)
                        handle.flush()
                        existing.add((variant, position_id, depth))
                    atomic_json(progress_path, {
                        "status": "running",
                        "variant": variant,
                        "variant_index": variant_index,
                        "position_id": position_id,
                        "position_index": position_index,
                        "completed_positions": sum(
                            1 for value in positions
                            if all((variant_name, value["position_id"], depth) in existing for variant_name in selected_variants for depth in DEPTHS)
                        ),
                        "expected_positions": len(positions),
                        "rows": len(existing),
                    })
            finally:
                if post_evaluator is not None:
                    post_evaluator.close()
    metadata = {
        "status": "completed",
        "positions": len(positions),
        "variants": list(selected_variants),
        "depths": list(DEPTHS),
        "evaluation_corpus": str(args.evaluation_corpus),
        "evaluation_corpus_sha256": sha256_file(args.evaluation_corpus),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "seed": args.seed,
        "post_eval_count": args.post_eval_count,
        "post_eval_position_ids": sorted(selected_for_post_eval),
        "analysis_depth": args.analysis_depth,
        "analysis_threads": args.analysis_threads,
        "analysis_hash_mb": args.analysis_hash_mb,
        "raw_columns": list(RAW_COLUMNS),
        "raw_rows": len(existing),
        "raw_sha256": sha256_file(raw_path),
        "note": "All legal candidate moves are scored. Stockfish post-move evaluation is an observer metric only and is restricted to a predeclared position subset.",
    }
    atomic_json(args.output / "metadata.json", metadata)
    atomic_json(progress_path, {**metadata, "progress": "complete"})
    print(json.dumps({"positions": len(positions), "rows": len(existing)}, indent=2))


if __name__ == "__main__":
    main()
