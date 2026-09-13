"""Run and fit the reproducible Minic/Gaia low-strength calibration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.low_elo_calibration import (  # noqa: E402
    fit_calibration,
    run_calibration_experiment,
    write_calibration_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minic", type=Path, required=True)
    parser.add_argument("--gaia", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/low_elo_calibration"))
    parser.add_argument("--opening-pairs", type=int, default=20)
    parser.add_argument("--move-time", type=float, default=0.01)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--game-timeout", type=float, default=15.0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--opening-seed", type=int, default=71)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit the existing games.csv without running or retrying games",
    )
    args = parser.parse_args()

    if args.fit_only:
        metadata_path = args.output / "calibration_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    else:
        _, metadata = run_calibration_experiment(
            minic_path=args.minic,
            gaia_path=args.gaia,
            output_dir=args.output,
            opening_pairs=args.opening_pairs,
            move_time_s=args.move_time,
            max_plies=args.max_plies,
            game_timeout_s=args.game_timeout,
            opening_seed=args.opening_seed,
            seed=args.seed,
        )
    games_path = args.output / "games.csv"
    with games_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = fit_calibration(rows, bootstrap_samples=args.bootstrap, seed=args.seed)
    metadata = {
        **metadata,
        "game_rows": len(rows),
        "complete_game_rows": sum(row.get("status", "complete") == "complete" for row in rows),
        "error_game_rows": sum(row.get("status", "complete") != "complete" for row in rows),
        "bootstrap_samples": int(args.bootstrap),
    }
    write_calibration_outputs(result, args.output, metadata=metadata)
    for row in result.rows:
        print(f"{row['engine']} {row['setting']}: {row['raw_calibrated_elo']}")
    print(f"games: {len(rows)}")
    print(f"diagnostics: {args.output / 'calibration_diagnostics.json'}")


if __name__ == "__main__":
    main()
