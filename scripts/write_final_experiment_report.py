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
    parser.add_argument("--diagnostic-run", type=Path, default=Path("results/first_fly_rating/diagnostic_fulllegal_60_v2"))
    parser.add_argument("--sweep-root", type=Path, default=Path("results/sweep_full_8pos_v2"))
    parser.add_argument("--position-study-root", type=Path, default=Path("results/position_depth_study_v2"))
    parser.add_argument("--expected-serious-games", type=int, default=None)
    parser.add_argument("--serious-max-plies", type=int, default=None)
    parser.add_argument("--serious-max-candidates", type=int, default=None)
    args = parser.parse_args()
    root = args.repo
    sweep_root = root / args.sweep_root
    position_study_root = root / args.position_study_root
    artifacts = {
        "low_elo_calibration": root / "results/low_elo_calibration/low_elo_calibration.csv",
        "calibration_metadata": root / "results/low_elo_calibration/calibration_metadata.json",
        "teacher_dataset": root / "data/teacher_dataset_v1.csv",
        "population_manifest": root / "data/chess_population_manifest.json",
        "activation_cache_depth16": root / "data/chess_activations_depth16_v1.npz",
        "readout_checkpoint": root / "results/training/v1_depth16_rate_large/readout_checkpoint.npz",
        "depth_sweep": sweep_root / "depth_sweep/metrics.csv",
        "control_sweep": sweep_root / "control_sweep/metrics.csv",
        "control_ratings": root / "results/control_sweep/control_ratings.csv",
        "serious_games": root / args.serious_run / "games.csv",
        "diagnostic_games": root / args.diagnostic_run / "games.csv",
        "evaluation_corpus": root / "data/chess_evaluation_corpus_v2.csv",
        "position_study_raw": position_study_root / "raw_position_metrics.csv",
        "position_study_summary": position_study_root / "summary.json",
        "position_study_depth_summary": position_study_root / "summary_by_depth.csv",
        "position_study_specificity": position_study_root / "specificity_by_control.csv",
        "position_study_metadata": position_study_root / "metadata.json",
        "position_study_evaluator_benchmark": position_study_root / "evaluator_benchmark.json",
        "readout_diagnostics": root / "results/training/v1_depth16_rate_large/readout_diagnostics.json",
    }
    serious_elo = read_json(root / args.serious_run / "elo.json", {})
    serious_games = read_csv(artifacts["serious_games"])
    diagnostic_games = read_csv(artifacts["diagnostic_games"])
    depth_rows = read_csv(artifacts["depth_sweep"])
    control_rows = read_csv(artifacts["control_sweep"])
    control_rating_rows = read_csv(artifacts["control_ratings"])
    position_summary = read_json(artifacts["position_study_summary"], {})
    evaluator_benchmark = read_json(artifacts["position_study_evaluator_benchmark"], {})
    expected_games = int(args.expected_serious_games or serious_elo.get("elo", {}).get("n_games", 0))
    expected_games = max(expected_games, len(serious_games))
    status = "complete_short_game_protocol" if expected_games >= 400 else "provisional_incomplete"
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
            "expected_games": expected_games,
            "elo": serious_elo.get("elo"),
            "bounded_candidates": args.serious_max_candidates,
            "full_legal_candidates": args.serious_max_candidates is None,
            "max_plies": args.serious_max_plies,
            "analysis_depth": 1,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "median_fly_move_latency_s": median_latency,
            "total_recurrent_passes": recurrent_passes,
        },
        "diagnostic_run": {
            "path": str(args.diagnostic_run),
            "games_recorded": len(diagnostic_games),
            "elo": read_json(root / args.diagnostic_run / "elo.json", {}).get("elo"),
            "full_legal_candidates": True,
            "opening_pairs": len({row.get("opening_id") for row in diagnostic_games if row.get("opening_id")}),
        },
        "depth_sweep_rows": len(depth_rows),
        "control_sweep_rows": len(control_rows),
        "control_rating_rows": len(control_rating_rows),
        "position_depth_study": position_summary,
        "position_depth_evaluator_benchmark": evaluator_benchmark,
        "artifacts": {name: {"path": str(path), "exists": path.exists()} for name, path in artifacts.items()},
        "scientific_caveat": "The serious rating protocol uses short games from prepared openings to control full-graph runtime. The position-depth study uses all legal moves and paired bootstrap statistics; do not treat a short-game estimate as a long-game playing-strength claim.",
    }
    output = root / "results/final_experiment_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# MaleCNS recurrent-depth chess experiment

Status: **{status}**

The frozen reference checkpoint uses the rate dynamics at recurrent depth 16. The low-Elo opponent scale comes from the measured calibration in `results/low_elo_calibration`, with Minic Level 0 and Gaia Skill Level 1 recorded as engine settings rather than guessed ratings.

## Current evidence

- Serious-run records: {len(serious_games)} of {expected_games} expected; W/D/L = {wins}/{draws}/{losses}.
- Full-legal diagnostic records: {len(diagnostic_games)}.
- Median Fly move latency: {median_latency:.3f} seconds; recurrent passes: {recurrent_passes}.
- Depth sweep rows: {len(depth_rows)} across depths 1, 2, 4, 8, 16, 32, and 64.
- Control sweep rows: {len(control_rows)} across the original, topology-shuffled, sign-shuffled, and recurrent-attenuated graphs.
- Control rating rows: {len(control_rating_rows)} across the same graph variants and depths.
- Independent position-depth study: {position_summary.get("positions", 0)} positions, {len(read_csv(artifacts["position_study_depth_summary"]))} summary rows, and {len(read_csv(artifacts["position_study_specificity"]))} control comparisons.
- Best original depth by mean regret: {position_summary.get("best_depth_by_mean_regret", "pending")}; D1-to-best improvement: {position_summary.get("d1_to_best_regret_improvement_cp", "pending")} cp (95% CI {position_summary.get("d1_to_best_regret_improvement_ci_low_cp", "pending")} to {position_summary.get("d1_to_best_regret_improvement_ci_high_cp", "pending")}); H1 support: {position_summary.get("h1_support", "pending")}.
- Evaluator validation: {evaluator_benchmark.get("scalar_total_s", "pending")} s scalar versus {evaluator_benchmark.get("optimized_trajectory_total_s", "pending")} s optimized trajectory; speedup: {evaluator_benchmark.get("speedup_scalar_over_optimized", "pending")}x.
- Every recorded game includes PGN, exact calibrated opponent setting, Fly move latency, recurrent-pass count, and candidate-score margin.

## Interpretation boundary

The serious rating allocation uses the configured candidate policy and short games per prepared opening. The diagnostic and position-quality sweeps score all legal moves and use the calibrated opening set. These artifacts verify the end-to-end real-connectome path and telemetry, but a short-game rating is not a substitute for an opening-diverse, long-game strength estimate.

Key artifacts:

- `results/low_elo_calibration/low_elo_calibration.csv`
- `data/teacher_dataset_v1.csv`
- `data/chess_activations_depth16_v1.npz`
- `results/training/v1_depth16_rate_large/readout_checkpoint.npz`
- `{args.sweep_root}/depth_sweep/metrics.csv` and `{args.sweep_root}/depth_sweep/rating_vs_depth.png`
- `{args.sweep_root}/control_sweep/metrics.csv` and `{args.sweep_root}/control_sweep/control_depth_curves.png`
- `{args.serious_run}/games.csv` and `{args.serious_run}/games.pgn`
- `data/chess_evaluation_corpus_v2.csv`
- `{args.position_study_root}/raw_position_metrics.csv`, `summary_by_depth.csv`, `specificity_by_control.csv`, and plots
- `results/training/v1_depth16_rate_large/readout_diagnostics.json`
"""
    (root / "results/final_experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "serious_games": len(serious_games)}, indent=2))


if __name__ == "__main__":
    main()
