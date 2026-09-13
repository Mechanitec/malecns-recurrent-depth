"""Write the machine-readable final experiment summary and concise report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--serious-run", type=Path, default=Path("results/first_fly_rating/serious_400_bounded"))
    args = parser.parse_args()
    root = args.repo
    artifacts = {
        "low_elo_calibration": root / "results/low_elo_calibration/low_elo_calibration.csv",
        "calibration_metadata": root / "results/low_elo_calibration/calibration_metadata.json",
        "teacher_dataset": root / "data/teacher_dataset_v1.csv",
        "population_manifest": root / "data/chess_population_manifest.json",
        "activation_cache_depth16": root / "data/chess_activations_depth16_v1.npz",
        "readout_checkpoint": root / "results/training/v1_depth16_rate_large/readout_checkpoint.npz",
        "depth_sweep": root / "results/depth_sweep/metrics.csv",
        "control_sweep": root / "results/control_sweep/metrics.csv",
        "serious_games": root / args.serious_run / "games.csv",
    }
    serious_elo = read_json(root / args.serious_run / "elo.json", {})
    serious_games = read_csv(artifacts["serious_games"])
    depth_rows = read_csv(artifacts["depth_sweep"])
    control_rows = read_csv(artifacts["control_sweep"])
    expected_games = int(serious_elo.get("elo", {}).get("n_games", 0))
    status = "complete_bounded_protocol" if expected_games >= 400 else "provisional_incomplete"
    wins = sum(row.get("fly_score") == "1.0" for row in serious_games)
    draws = sum(row.get("fly_score") == "0.5" for row in serious_games)
    losses = sum(row.get("fly_score") == "0.0" for row in serious_games)
    latencies = sorted(float(row["median_fly_move_latency_s"]) for row in serious_games if row.get("median_fly_move_latency_s"))
    median_latency = latencies[len(latencies) // 2] if latencies else None
    recurrent_passes = sum(int(row.get("total_recurrent_passes", 0) or 0) for row in serious_games)
    summary = {
        "status": status,
        "repo": "https://github.com/Mechanitec/malecns-recurrent-depth",
        "reference_depth": 16,
        "dynamics": "rate",
        "calibration": read_json(artifacts["calibration_metadata"]),
        "serious_run": {
            "path": str(args.serious_run),
            "games_recorded": len(serious_games),
            "expected_games": 400,
            "elo": serious_elo.get("elo"),
            "bounded_candidates": 4,
            "max_plies": 2,
            "analysis_depth": 1,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "median_fly_move_latency_s": median_latency,
            "total_recurrent_passes": recurrent_passes,
        },
        "depth_sweep_rows": len(depth_rows),
        "control_sweep_rows": len(control_rows),
        "artifacts": {name: {"path": str(path), "exists": path.exists()} for name, path in artifacts.items()},
        "scientific_caveat": "The compact sweep and serious run use bounded candidate sets and short games to control full-graph runtime; do not treat them as a full legal-move opening-diverse rating until expanded.",
    }
    output = root / "results/final_experiment_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# MaleCNS recurrent-depth chess experiment

Status: **{status}**

The frozen reference checkpoint uses the rate dynamics at recurrent depth 16. The low-Elo opponent scale comes from the measured calibration in `results/low_elo_calibration`, with Minic Level 0 and Gaia Skill Level 1 recorded as engine settings rather than guessed ratings.

## Current evidence

- Serious-run records: {len(serious_games)} of 400 expected; W/D/L = {wins}/{draws}/{losses}.
- Median Fly move latency: {median_latency:.3f} seconds; recurrent passes: {recurrent_passes}.
- Depth sweep rows: {len(depth_rows)} across depths 1, 2, 4, 8, 16, 32, and 64.
- Control sweep rows: {len(control_rows)} across the original, topology-shuffled, sign-shuffled, and recurrent-attenuated graphs.
- Every recorded game includes PGN, exact calibrated opponent setting, Fly move latency, recurrent-pass count, and candidate-score margin.

## Interpretation boundary

The current local runs use four tactical candidates and two-ply games in the long-run throughput allocation. The compact sweep uses the same bound and four-ply paired games. These artifacts verify the end-to-end real-connectome path and telemetry, but they are not a substitute for an opening-diverse, all-legal-move, long-game rating run. Expand those bounds before making a strong chess-strength or connectome-specific depth claim.

Key artifacts:

- `results/low_elo_calibration/low_elo_calibration.csv`
- `data/teacher_dataset_v1.csv`
- `data/chess_activations_depth16_v1.npz`
- `results/training/v1_depth16_rate_large/readout_checkpoint.npz`
- `results/depth_sweep/metrics.csv` and `results/depth_sweep/rating_vs_depth.png`
- `results/control_sweep/metrics.csv` and `results/control_sweep/control_depth_curves.png`
- `{args.serious_run}/games.csv` and `{args.serious_run}/games.pgn`
"""
    (root / "results/final_experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "serious_games": len(serious_games)}, indent=2))


if __name__ == "__main__":
    main()
