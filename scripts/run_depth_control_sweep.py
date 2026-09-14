"""Run the frozen-checkpoint depth sweep and graph-control diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_agent import FlyCandidateMoveAgent
from malecns_rd.chess_benchmark import save_tournament
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.fast_opponents import run_fast_elo_tournament
from malecns_rd.graph import ConnectomeGraph
from malecns_rd.position_analysis import AnalysisConfig, StockfishPositionEvaluator
from malecns_rd.readout_training import load_readout_checkpoint


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
CONTROL_VARIANTS = (
    "original",
    "degree_preserving_topology_shuffle",
    "transmitter_sign_shuffle",
    "recurrent_edges_attenuated",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_positions(path: str | Path, count: int, seed: int):
    import chess

    grouped: dict[str, dict[str, object]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test":
                continue
            fen = row["fen"]
            item = grouped.setdefault(
                fen,
                {"fen": fen, "best": row["move_uci"], "candidates": {}},
            )
            candidates = item["candidates"]
            assert isinstance(candidates, dict)
            candidates[row["move_uci"]] = float(row["teacher_cp"])
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    positions = []
    for fen in keys[:count]:
        item = grouped[fen]
        board = chess.Board(fen)
        candidates = item["candidates"]
        assert isinstance(candidates, dict)
        best = max(candidates, key=candidates.get)
        positions.append((board, best, candidates))
    if not positions:
        raise RuntimeError("teacher dataset contains no test positions")
    return positions


def load_opening_fens(path: Path) -> list[tuple[str, str]]:
    metadata_path = path.parent / "calibration_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [(str(item["opening_pair"]), str(item["fen"])) for item in payload["opening_fens"]]


def graph_variant(graph: ConnectomeGraph, variant: str, seed: int) -> ConnectomeGraph:
    if variant == "original":
        return graph
    rng = np.random.default_rng(seed)
    weights = graph.weights.tocoo(copy=True)
    if variant == "degree_preserving_topology_shuffle":
        # Shuffle destination labels across existing edges. This preserves the
        # source out-degree and destination in-degree multisets exactly.
        shuffled_dst = weights.row.copy()
        rng.shuffle(shuffled_dst)
        return ConnectomeGraph(
            body_ids=graph.body_ids.copy(),
            weights=__import__("scipy.sparse", fromlist=["coo_matrix"])
            .coo_matrix((weights.data, (shuffled_dst, weights.col)), shape=weights.shape)
            .tocsr(),
        )
    if variant == "transmitter_sign_shuffle":
        data = weights.data.copy()
        signs = np.sign(data)
        rng.shuffle(signs)
        data = np.abs(data) * signs
        return ConnectomeGraph(
            body_ids=graph.body_ids.copy(),
            weights=__import__("scipy.sparse", fromlist=["coo_matrix"])
            .coo_matrix((data, (weights.row, weights.col)), shape=weights.shape)
            .tocsr(),
        )
    if variant == "recurrent_edges_attenuated":
        return ConnectomeGraph(
            body_ids=graph.body_ids.copy(),
            weights=(graph.weights * np.float32(0.05)).tocsr(),
        )
    raise ValueError(f"unknown graph variant: {variant}")


def make_agent(base, graph: ConnectomeGraph, depth: int, max_candidates: int | None):
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("depth/control sweep requires the rate checkpoint")
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
        depth=depth,
        clamp_sensory=base.clamp_sensory,
        max_candidates=max_candidates,
        name=base.name,
    )


def _bootstrap_interval(values, seed: int) -> tuple[float | None, float | None]:
    numeric = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=float)
    if numeric.size < 2:
        return (float(numeric[0]), float(numeric[0])) if numeric.size else (None, None)
    rng = np.random.default_rng(seed)
    samples = rng.choice(numeric, size=(500, numeric.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _mean(values) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(numeric)) if numeric else None


def position_metrics(agent, positions, *, evaluator=None, baseline_evals=None):
    import chess

    rows = []
    for index, (board, teacher_best, teacher_candidates) in enumerate(positions):
        started = time.perf_counter()
        ranked = agent.rank_moves(board)
        latency = time.perf_counter() - started
        predicted = ranked[0][1]
        decision = agent.last_decision
        predicted_cp = teacher_candidates.get(predicted)
        best_cp = max(teacher_candidates.values())
        stockfish_loss = None
        if evaluator is not None:
            mover_is_white = board.turn == chess.WHITE
            before = (baseline_evals or {}).get(board.fen())
            if before is None:
                before = evaluator.evaluate(
                    board,
                    game_index=index,
                    fly_is_white=mover_is_white,
                    last_move=None,
                    last_actor=None,
                ).fly_cp
            after_board = board.copy(stack=False)
            after_board.push_uci(predicted)
            after = evaluator.evaluate(
                after_board,
                game_index=index,
                fly_is_white=mover_is_white,
                last_move=predicted,
                last_actor="fly",
            ).fly_cp
            if before is not None and after is not None:
                stockfish_loss = max(0.0, float(before - after))
        rows.append(
            {
                "teacher_best_move": teacher_best,
                "predicted_move": predicted,
                "teacher_best_in_candidate_set": teacher_best in {item[1] for item in ranked},
                "teacher_agreement": predicted == teacher_best,
                "quality_loss_cp": (best_cp - predicted_cp) if predicted_cp is not None else None,
                "latency_s": latency,
                "candidate_count": int(decision.get("candidate_count", len(ranked))),
                "recurrent_passes": int(decision.get("recurrent_passes", 0)),
                "score_margin": decision.get("score_margin"),
                "stockfish_eval_loss_cp": stockfish_loss,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--minic", required=True)
    parser.add_argument("--gaia", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--connectome-weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, default=Path("data/teacher_dataset_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--positions", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--position-max-candidates",
        type=int,
        default=0,
        help="Candidate cap for held-out position metrics; 0 scores all legal moves",
    )
    parser.add_argument("--games-per-elo", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=4)
    parser.add_argument("--move-time", type=float, default=0.01)
    parser.add_argument("--analysis-depth", type=int, default=6)
    parser.add_argument("--analysis-threads", type=int, default=1)
    parser.add_argument("--analysis-hash-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_test_positions(args.teacher_dataset, args.positions, args.seed)
    opening_fens = load_opening_fens(args.calibration)
    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
        min_synapses=3,
    )
    base = loaded.agent
    checkpoint = load_readout_checkpoint(args.checkpoint)
    position_cap = args.position_max_candidates if args.position_max_candidates > 0 else None
    evaluator = StockfishPositionEvaluator(
        AnalysisConfig(args.stockfish, depth=args.analysis_depth, threads=args.analysis_threads, hash_mb=args.analysis_hash_mb)
    )
    baseline_evals = {}
    try:
        import chess
        for index, (board, _, _) in enumerate(positions):
            baseline_evals[board.fen()] = evaluator.evaluate(
                board,
                game_index=index,
                fly_is_white=board.turn == chess.WHITE,
                last_move=None,
                last_actor=None,
            ).fly_cp
    finally:
        evaluator.close()
    depth_rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        agent = make_agent(base, loaded.agent.engine.graph, depth, args.max_candidates)
        evaluator = StockfishPositionEvaluator(
            AnalysisConfig(args.stockfish, depth=args.analysis_depth, threads=args.analysis_threads, hash_mb=args.analysis_hash_mb)
        )
        try:
            metric_agent = make_agent(base, loaded.agent.engine.graph, depth, position_cap)
            rows = position_metrics(metric_agent, positions, evaluator=evaluator, baseline_evals=baseline_evals)
        finally:
            evaluator.close()
        rating_dir = args.output / "depth_sweep" / f"depth_{depth}"
        rating = run_fast_elo_tournament(
            agent,
            stockfish_executable=args.stockfish,
            minic_executable=args.minic,
            gaia_executable=args.gaia,
            opponent_elos=[310, 580],
            games_per_elo=args.games_per_elo,
            move_time_s=args.move_time,
            max_plies=args.max_plies,
            live_state_path=rating_dir / "live_state.json",
            calibration_path=args.calibration,
            opening_fens=opening_fens,
            run_id=f"depth_{depth}",
        )
        save_tournament(rating, rating_dir)
        depth_rows.append(
            {
                "depth": depth,
                "variant": "original",
                "teacher_agreement": _mean([r["teacher_agreement"] for r in rows]),
                "teacher_agreement_ci_low": _bootstrap_interval([r["teacher_agreement"] for r in rows], args.seed + depth)[0],
                "teacher_agreement_ci_high": _bootstrap_interval([r["teacher_agreement"] for r in rows], args.seed + depth)[1],
                "quality_loss_cp": _mean([r["quality_loss_cp"] for r in rows]),
                "stockfish_eval_loss_cp": _mean([r["stockfish_eval_loss_cp"] for r in rows]),
                "latency_s": _mean([r["latency_s"] for r in rows]),
                "recurrent_passes": _mean([r["recurrent_passes"] for r in rows]),
                "score_margin": _mean([r["score_margin"] for r in rows]),
                "rating": rating.elo.rating,
                "rating_ci_low": rating.elo.ci95_low,
                "rating_ci_high": rating.elo.ci95_high,
                "games": len(rating.games),
            }
        )

    control_rows: list[dict[str, object]] = []
    for variant in CONTROL_VARIANTS:
        graph = graph_variant(loaded.agent.engine.graph, variant, args.seed)
        for depth in DEPTHS:
            agent = make_agent(base, graph, depth, args.max_candidates)
            evaluator = StockfishPositionEvaluator(
                AnalysisConfig(args.stockfish, depth=args.analysis_depth, threads=args.analysis_threads, hash_mb=args.analysis_hash_mb)
            )
            try:
                metric_agent = make_agent(base, graph, depth, position_cap)
                rows = position_metrics(metric_agent, positions, evaluator=evaluator, baseline_evals=baseline_evals)
            finally:
                evaluator.close()
            control_rows.append(
                {
                    "depth": depth,
                    "variant": variant,
                    "teacher_agreement": _mean([r["teacher_agreement"] for r in rows]),
                    "teacher_agreement_ci_low": _bootstrap_interval([r["teacher_agreement"] for r in rows], args.seed + depth)[0],
                    "teacher_agreement_ci_high": _bootstrap_interval([r["teacher_agreement"] for r in rows], args.seed + depth)[1],
                    "quality_loss_cp": _mean([r["quality_loss_cp"] for r in rows]),
                    "stockfish_eval_loss_cp": _mean([r["stockfish_eval_loss_cp"] for r in rows]),
                    "latency_s": _mean([r["latency_s"] for r in rows]),
                    "recurrent_passes": _mean([r["recurrent_passes"] for r in rows]),
                    "score_margin": _mean([r["score_margin"] for r in rows]),
                }
            )

    depth_dir = args.output / "depth_sweep"
    control_dir = args.output / "control_sweep"
    control_dir.mkdir(parents=True, exist_ok=True)
    with (depth_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(depth_rows[0]))
        writer.writeheader()
        writer.writerows(depth_rows)
    with (control_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(control_rows[0]))
        writer.writeheader()
        writer.writerows(control_rows)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        [r["depth"] for r in depth_rows],
        [r["rating"] for r in depth_rows],
        yerr=[
            [r["rating"] - r["rating_ci_low"] for r in depth_rows],
            [r["rating_ci_high"] - r["rating"] for r in depth_rows],
        ],
        marker="o",
    )
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Compact diagnostic rating")
    plt.tight_layout()
    plt.savefig(depth_dir / "rating_vs_depth.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    for variant in CONTROL_VARIANTS:
        rows = [r for r in control_rows if r["variant"] == variant]
        x = [r["depth"] for r in rows]
        y = [r["teacher_agreement"] for r in rows]
        low = [r["teacher_agreement_ci_low"] for r in rows]
        high = [r["teacher_agreement_ci_high"] for r in rows]
        plt.plot(x, y, marker="o", label=variant)
        plt.fill_between(x, low, high, alpha=0.12)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Teacher-best agreement")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(control_dir / "control_depth_curves.png", dpi=140)
    plt.close()

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot([r["depth"] for r in depth_rows], [r["quality_loss_cp"] for r in depth_rows], marker="o")
    axes[0].set_ylabel("Teacher quality loss (cp)")
    axes[1].plot([r["depth"] for r in depth_rows], [r["latency_s"] for r in depth_rows], marker="o")
    axes[1].set_ylabel("Fly move latency (s)")
    axes[2].plot([r["depth"] for r in depth_rows], [r["teacher_agreement"] for r in depth_rows], marker="o")
    axes[2].set_ylabel("Teacher-best agreement")
    for axis in axes:
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel("Recurrent depth")
    figure.tight_layout()
    figure.savefig(depth_dir / "position_metrics_vs_depth.png", dpi=140)
    plt.close(figure)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_dataset": str(args.teacher_dataset),
        "teacher_dataset_sha256": sha256_file(args.teacher_dataset),
        "depths": list(DEPTHS),
        "control_variants": list(CONTROL_VARIANTS),
        "positions": len(positions),
        "max_candidates": args.max_candidates,
        "position_max_candidates": position_cap,
        "analysis_depth": args.analysis_depth,
        "games_per_elo": args.games_per_elo,
        "max_plies": args.max_plies,
        "opponent_elos": [310, 580],
        "opening_pairs": [opening_id for opening_id, _ in opening_fens],
        "seed": args.seed,
        "checkpoint_recurrent_depth": checkpoint.recurrent_depth,
        "note": "Compact bounded sweep. Full legal-move and 400-game runs require a longer compute allocation.",
    }
    (args.output / "depth_sweep" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "control_sweep" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"depth_rows": len(depth_rows), "control_rows": len(control_rows)}, indent=2))


if __name__ == "__main__":
    main()
