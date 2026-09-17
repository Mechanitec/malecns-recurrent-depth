"""Finalize the Plan 3 report after primary and sequential mate evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = (
    "plastic_edge_audit.csv", "plastic_edge_metadata.json", "training_log.csv",
    "stage_metrics.csv", "retention_metrics.csv", "movement_validation.csv",
    "endgame_validation.csv", "tactical_validation.csv", "mate_validation.csv",
    "mate_sequence_validation.csv", "biological_deviation.csv",
    "reward_signal_distribution.csv", "depth_metrics.csv", "run_metadata.json",
    "final_summary.md",
)
PLOTS = (
    "learning_curve.png", "curriculum_retention.png", "regret_distribution_by_stage.png",
    "blunder_rate_by_stage.png", "mate_solve_rate.png",
    "biological_deviation_vs_performance.png", "weight_ratio_distribution.png",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/neuromodulated_curriculum_v1"))
    parser.add_argument("--curriculum-dir", type=Path, default=Path("data/curriculum_v1"))
    args = parser.parse_args()
    metadata_path = args.output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mate_metadata = json.loads((args.output / "mate_sequence_metadata.json").read_text(encoding="utf-8"))
    stage_metrics = pd.read_csv(args.output / "stage_metrics.csv")
    biological = pd.read_csv(args.output / "biological_deviation.csv")

    curriculum = {}
    for stage in ("movement", "endgames", "tactics", "mates"):
        frame = pd.read_csv(args.curriculum_dir / f"{stage}.csv")
        curriculum[stage] = {
            "rows": int(len(frame)),
            "positions": int(frame["position_id"].nunique()),
            "train_positions": int(frame.loc[frame["split"].astype(str) == "train", "position_id"].nunique()),
            "validation_positions": int(frame.loc[frame["split"].astype(str) == "validation", "position_id"].nunique()),
        }
    files_present = {name: (args.output / name).is_file() for name in REQUIRED + PLOTS}
    metadata.update({
        "status": "complete",
        "curriculum": curriculum,
        "mate_sequence_evaluation": mate_metadata,
        "required_artifacts_present": all(files_present[name] for name in REQUIRED),
        "plot_artifacts_present": all(files_present[name] for name in PLOTS),
        "artifact_presence": files_present,
        "finalization": "primary_run_plus_sequential_mate_evaluation",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Plan 3 neuromodulated curriculum results",
        "",
        "The primary run used the real MaleCNS graph, D8 training, localized existing KC-to-MBON plasticity, and a decoder frozen before plasticity.",
        "",
        f"- Training updates: `{metadata.get('training_positions')}`; elapsed: `{metadata.get('elapsed_s'):.1f} s`.",
        f"- Plastic edges: `{metadata.get('plastic_edge_count')}`; topology preserved: `{bool(int(biological['topology_preserved'].all()))}`; signs preserved: `{bool(int(biological['sign_preserved'].all()))}`.",
        f"- Sequential mate evaluation: `{mate_metadata.get('positions_evaluated')}` positive-mate validation positions; solve rate `{mate_metadata.get('aggregate_solve_rate')}`.",
        "- The mate result is descriptive, not a claim of general chess strength; the available positive mate validation set is small.",
        "",
        "## Stage metrics",
        "",
        "```text\n" + stage_metrics.to_string(index=False) + "\n```",
        "",
        "## Curriculum coverage",
        "",
        "```text\n" + pd.DataFrame.from_dict(curriculum, orient="index").to_string() + "\n```",
        "",
        "## Artifact verification",
        "",
        f"- Required artifacts present: `{metadata['required_artifacts_present']}`.",
        f"- Plot artifacts present: `{metadata['plot_artifacts_present']}`.",
    ]
    (args.output / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "required_artifacts_present": metadata["required_artifacts_present"], "plot_artifacts_present": metadata["plot_artifacts_present"]}, indent=2))


if __name__ == "__main__":
    main()
