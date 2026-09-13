"""Measure local UCI engine startup, reliability, and weak-play throughput."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.engine_smoke import default_configurations, run_engine_smoke  # noqa: E402


def _levels(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--minic", type=Path, required=True)
    parser.add_argument("--gaia", type=Path, required=True)
    parser.add_argument("--stockfish-floor", type=int, default=None)
    parser.add_argument("--minic-levels", type=_levels, default=_levels("0,1,5,10,15,20,25,30"))
    parser.add_argument("--gaia-levels", type=_levels, default=_levels("1,2,3,4,5,6,7"))
    parser.add_argument("--games-per-config", type=int, default=10)
    parser.add_argument("--move-time", type=float, default=0.02)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--game-timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("results/engine_smoke"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    configurations = default_configurations(
        stockfish=args.stockfish,
        minic=args.minic,
        gaia=args.gaia,
        stockfish_floor=args.stockfish_floor,
        minic_levels=args.minic_levels,
        gaia_levels=args.gaia_levels,
    )
    summary = run_engine_smoke(
        configurations,
        output_dir=args.output,
        games_per_config=args.games_per_config,
        move_time_s=args.move_time,
        max_plies=args.max_plies,
        game_timeout_s=args.game_timeout,
        seed=args.seed,
    )
    for row in summary:
        print(
            f"{row['engine']} {row['setting']}: "
            f"{row['games_completed']}/{row['games_requested']} games, "
            f"median_move={row['median_move_latency_s']}, "
            f"games/hour={row['games_per_hour']}"
        )


if __name__ == "__main__":
    main()
