"""Tournament entry point.

This file intentionally keeps model construction explicit. The first real MaleCNS
chess checkpoint will supply sensory/readout populations and trained readout
weights; the tournament harness itself is already complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from malecns_rd.chess_agent import RandomLegalAgent
from malecns_rd.chess_benchmark import run_elo_tournament, save_tournament


def parse_elos(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one Elo")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish executable")
    parser.add_argument("--elos", type=parse_elos, default=parse_elos("1320,1400,1500"))
    parser.add_argument("--games-per-elo", type=int, default=20)
    parser.add_argument("--move-time", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("results/chess_smoke"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Harness smoke test. Replace with FlyCandidateMoveAgent once a chess
    # readout checkpoint is trained.
    agent = RandomLegalAgent(seed=args.seed)
    result = run_elo_tournament(
        agent,
        stockfish_executable=args.stockfish,
        opponent_elos=args.elos,
        games_per_elo=args.games_per_elo,
        move_time_s=args.move_time,
    )
    save_tournament(result, args.output)
    print(result.elo)


if __name__ == "__main__":
    main()
