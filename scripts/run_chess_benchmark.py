"""Hybrid Alfil + Stockfish tournament entry point with live position analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

from malecns_rd.chess_agent import RandomLegalAgent
from malecns_rd.chess_benchmark import run_elo_tournament, save_tournament
from malecns_rd.position_analysis import AnalysisConfig, run_elo_tournament_with_analysis


def parse_elos(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one Elo")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish executable")
    parser.add_argument(
        "--alfil",
        help="Path to Alfil executable; required when testing below Stockfish's floor",
    )
    parser.add_argument(
        "--elos",
        type=parse_elos,
        default=parse_elos("0,200,400,600,800,1000,1200,1320,1400,1500"),
    )
    parser.add_argument(
        "--stockfish-floor",
        type=int,
        default=None,
        help="Override Stockfish minimum Elo; by default it is detected from UCI_Elo",
    )
    parser.add_argument("--games-per-elo", type=int, default=20)
    parser.add_argument("--move-time", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("results/chess_smoke"))
    parser.add_argument(
        "--live-state",
        type=Path,
        default=None,
        help="Live JSON path; defaults to <output>/live_state.json",
    )
    parser.add_argument(
        "--analysis-stockfish",
        default=None,
        help="Full-strength Stockfish binary used only for position evaluation; defaults to --stockfish",
    )
    parser.add_argument(
        "--analysis-depth",
        type=int,
        default=18,
        help="Fixed Stockfish analysis depth after every move (default: 18)",
    )
    parser.add_argument("--analysis-threads", type=int, default=1)
    parser.add_argument("--analysis-hash-mb", type=int, default=128)
    parser.add_argument(
        "--no-position-eval",
        action="store_true",
        help="Disable the independent Stockfish position observer",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    live_state = args.live_state or (args.output / "live_state.json")

    # Harness smoke test. Replace with FlyCandidateMoveAgent once a chess
    # readout checkpoint is loaded for benchmark play.
    agent = RandomLegalAgent(seed=args.seed)

    if args.no_position_eval:
        result = run_elo_tournament(
            agent,
            stockfish_executable=args.stockfish,
            alfil_executable=args.alfil,
            stockfish_floor=args.stockfish_floor,
            opponent_elos=args.elos,
            games_per_elo=args.games_per_elo,
            move_time_s=args.move_time,
            live_state_path=live_state,
            run_id=args.output.name,
        )
    else:
        analysis_config = AnalysisConfig(
            executable=args.analysis_stockfish or args.stockfish,
            depth=args.analysis_depth,
            threads=args.analysis_threads,
            hash_mb=args.analysis_hash_mb,
        )
        result = run_elo_tournament_with_analysis(
            agent,
            stockfish_executable=args.stockfish,
            alfil_executable=args.alfil,
            stockfish_floor=args.stockfish_floor,
            opponent_elos=args.elos,
            games_per_elo=args.games_per_elo,
            move_time_s=args.move_time,
            live_state_path=live_state,
            evaluation_state_path=args.output / "position_eval.json",
            evaluation_history_path=args.output / "evaluation_history.csv",
            analysis_config=analysis_config,
            run_id=args.output.name,
        )

    save_tournament(result, args.output)
    print(f"Stockfish floor: {result.stockfish_floor}")
    print(f"Live state: {live_state}")
    if not args.no_position_eval:
        print(f"Position eval: {args.output / 'position_eval.json'}")
        print(f"Eval history: {args.output / 'evaluation_history.csv'}")
    print(result.elo)


if __name__ == "__main__":
    main()
