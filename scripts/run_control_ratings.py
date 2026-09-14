"""Run the same compact rating protocol for every graph control and depth."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_depth_control_sweep import CONTROL_VARIANTS, DEPTHS, graph_variant, load_opening_fens, make_agent
from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_benchmark import save_tournament
from malecns_rd.fast_opponents import run_fast_elo_tournament


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
    parser.add_argument("--output", type=Path, default=Path("results/control_sweep"))
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--games-per-elo", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=4)
    parser.add_argument("--move-time", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
        min_synapses=3,
    )
    openings = load_opening_fens(args.calibration)
    rows = []
    for variant in CONTROL_VARIANTS:
        graph = graph_variant(loaded.agent.engine.graph, variant, args.seed)
        for depth in DEPTHS:
            agent = make_agent(loaded.agent, graph, depth, args.max_candidates)
            output = args.output / "ratings" / variant / f"depth_{depth}"
            result = run_fast_elo_tournament(
                agent,
                stockfish_executable=args.stockfish,
                minic_executable=args.minic,
                gaia_executable=args.gaia,
                opponent_elos=[310, 580],
                games_per_elo=args.games_per_elo,
                move_time_s=args.move_time,
                max_plies=args.max_plies,
                live_state_path=output / "live_state.json",
                calibration_path=args.calibration,
                opening_fens=openings,
                run_id=f"{variant}_depth_{depth}",
            )
            save_tournament(result, output)
            wins = sum(r.fly_score >= 0.75 for r in result.games)
            draws = sum(0.25 < r.fly_score < 0.75 for r in result.games)
            losses = sum(r.fly_score <= 0.25 for r in result.games)
            latencies = [r.median_fly_move_latency_s for r in result.games if r.median_fly_move_latency_s is not None]
            rows.append(
                {
                    "variant": variant,
                    "depth": depth,
                    "rating": result.elo.rating,
                    "rating_ci_low": result.elo.ci95_low,
                    "rating_ci_high": result.elo.ci95_high,
                    "games": len(result.games),
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "median_fly_move_latency_s": sorted(latencies)[len(latencies) // 2] if latencies else None,
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "control_ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} control rating rows")


if __name__ == "__main__":
    main()
