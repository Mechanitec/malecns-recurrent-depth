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


def position_metrics(agent, positions):
    rows = []
    for board, teacher_best, teacher_candidates in positions:
        started = time.perf_counter()
        ranked = agent.rank_moves(board)
        latency = time.perf_counter() - started
        predicted = ranked[0][1]
        decision = agent.last_decision
        predicted_cp = teacher_candidates.get(predicted)
        best_cp = max(teacher_candidates.values())
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
    parser.add_argument("--games-per-elo", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=4)
    parser.add_argument("--move-time", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_test_positions(args.teacher_dataset, args.positions, args.seed)
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
    depth_rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        agent = make_agent(base, loaded.agent.engine.graph, depth, args.max_candidates)
        rows = position_metrics(agent, positions)
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
            run_id=f"depth_{depth}",
        )
        save_tournament(rating, rating_dir)
        depth_rows.append(
            {
                "depth": depth,
                "variant": "original",
                "teacher_agreement": float(np.mean([r["teacher_agreement"] for r in rows])),
                "quality_loss_cp": float(np.nanmean([r["quality_loss_cp"] for r in rows])),
                "latency_s": float(np.mean([r["latency_s"] for r in rows])),
                "recurrent_passes": float(np.mean([r["recurrent_passes"] for r in rows])),
                "score_margin": float(np.mean([r["score_margin"] for r in rows])),
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
            rows = position_metrics(agent, positions)
            control_rows.append(
                {
                    "depth": depth,
                    "variant": variant,
                    "teacher_agreement": float(np.mean([r["teacher_agreement"] for r in rows])),
                    "quality_loss_cp": float(np.nanmean([r["quality_loss_cp"] for r in rows])),
                    "latency_s": float(np.mean([r["latency_s"] for r in rows])),
                    "recurrent_passes": float(np.mean([r["recurrent_passes"] for r in rows])),
                    "score_margin": float(np.mean([r["score_margin"] for r in rows])),
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
        plt.plot([r["depth"] for r in rows], [r["teacher_agreement"] for r in rows], marker="o", label=variant)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Teacher-best agreement")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(control_dir / "control_depth_curves.png", dpi=140)
    plt.close()

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_dataset": str(args.teacher_dataset),
        "teacher_dataset_sha256": sha256_file(args.teacher_dataset),
        "depths": list(DEPTHS),
        "control_variants": list(CONTROL_VARIANTS),
        "positions": len(positions),
        "max_candidates": args.max_candidates,
        "games_per_elo": args.games_per_elo,
        "max_plies": args.max_plies,
        "opponent_elos": [310, 580],
        "seed": args.seed,
        "checkpoint_recurrent_depth": checkpoint.recurrent_depth,
        "note": "Compact bounded sweep. Full legal-move and 400-game runs require a longer compute allocation.",
    }
    (args.output / "depth_sweep" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "control_sweep" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"depth_rows": len(depth_rows), "control_rows": len(control_rows)}, indent=2))


if __name__ == "__main__":
    main()
