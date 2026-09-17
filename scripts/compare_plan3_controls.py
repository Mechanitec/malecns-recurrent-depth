"""Compare completed Plan 3 primary and control result directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_result(name: str, path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path / "stage_metrics.csv")
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    columns = [
        "stage", "top1_accuracy_d8", "top3_accuracy_d8", "median_regret_cp_d8",
        "mean_regret_cp_d8", "blunder_rate_gt1000cp_d8", "mate_solve_rate_d8",
    ]
    result = metrics[columns].copy()
    result.insert(0, "condition", name)
    result["training_positions"] = int(metadata.get("training_positions", 0))
    result["elapsed_s"] = float(metadata.get("elapsed_s", float("nan")))
    result["plastic_edge_count"] = int(metadata.get("plastic_edge_count", 0))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=Path("results/neuromodulated_curriculum_v1"))
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/plan3_control_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.concat([
        load_result("reward_aversive", args.primary),
        load_result("frozen", args.frozen),
        load_result("shuffled", args.shuffled),
    ], ignore_index=True)
    table.to_csv(args.output / "control_comparison.csv", index=False)
    lines = [
        "# Plan 3 control comparison",
        "",
        "All conditions use the same real MaleCNS graph, exact KC-to-MBON edge scope, frozen decoder procedure, and curriculum limits unless noted by the run metadata.",
        "",
        "```text",
        table.to_string(index=False),
        "```",
        "",
        "Interpret the reward/aversive condition against both frozen and shuffled-signal controls; these are exploratory held-out curriculum metrics, not a chess Elo estimate.",
    ]
    (args.output / "control_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(table), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
