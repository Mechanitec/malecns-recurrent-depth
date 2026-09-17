"""Compare completed Plan 3 primary and control result directories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


MATCHED_METADATA_FIELDS = (
    "decoder_sha256",
    "graph_structure_hash",
    "curriculum_metadata_sha256",
    "input_manifest_sha256",
    "readout_manifest_sha256",
    "plastic_post_manifest_sha256",
    "plastic_edge_hash",
    "seed",
    "depth_train",
    "training_positions",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))


def load_result(name: str, path: Path, metadata: dict[str, object]) -> pd.DataFrame:
    metrics = pd.read_csv(path / "stage_metrics.csv")
    columns = [
        "stage",
        "top1_accuracy_d8",
        "top3_accuracy_d8",
        "pairwise_ranking_accuracy_d8",
        "median_regret_cp_d8",
        "trimmed_mean_regret_cp_d8",
        "mean_regret_cp_d8",
        "blunder_rate_gt500cp_d8",
        "blunder_rate_gt1000cp_d8",
        "mate_solve_rate_d8",
    ]
    missing = [column for column in columns if column not in metrics.columns]
    if missing:
        raise ValueError(f"{path} is missing required corrected metrics: {missing}")
    result = metrics[columns].copy()
    result.insert(0, "condition", name)
    result["training_positions"] = int(metadata.get("training_positions", 0))
    result["elapsed_s"] = float(metadata.get("elapsed_s", float("nan")))
    result["plastic_edge_count"] = int(metadata.get("plastic_edge_count", 0))
    result["decoder_sha256"] = str(metadata.get("decoder_sha256", ""))
    return result


def require_matched_controls(
    primary_dir: Path,
    primary: dict[str, object],
    frozen: dict[str, object],
    shuffled: dict[str, object],
) -> None:
    for field in MATCHED_METADATA_FIELDS:
        values = {
            "primary": primary.get(field),
            "frozen": frozen.get(field),
            "shuffled": shuffled.get(field),
        }
        if len(set(values.values())) != 1:
            raise ValueError(f"Plan 3 controls are not matched for {field}: {values}")

    if primary.get("control") != "reward_aversive":
        raise ValueError("--primary must be a reward_aversive run")
    if frozen.get("control") != "frozen":
        raise ValueError("--frozen must be a frozen run")
    if shuffled.get("control") != "shuffled":
        raise ValueError("--shuffled must be a shuffled run")

    primary_training_log = primary_dir / "training_log.csv"
    expected_reference_hash = sha256_file(primary_training_log)
    actual_reference_hash = shuffled.get("shuffled_signal_reference_sha256")
    if actual_reference_hash != expected_reference_hash:
        raise ValueError(
            "shuffled control did not use this primary training log as its exact signal reference: "
            f"expected={expected_reference_hash} actual={actual_reference_hash}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=Path("results/neuromodulated_curriculum_v1"))
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/plan3_control_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    primary_metadata = load_metadata(args.primary)
    frozen_metadata = load_metadata(args.frozen)
    shuffled_metadata = load_metadata(args.shuffled)
    require_matched_controls(
        args.primary,
        primary_metadata,
        frozen_metadata,
        shuffled_metadata,
    )

    table = pd.concat([
        load_result("reward_aversive", args.primary, primary_metadata),
        load_result("frozen", args.frozen, frozen_metadata),
        load_result("shuffled", args.shuffled, shuffled_metadata),
    ], ignore_index=True)
    table.to_csv(args.output / "control_comparison.csv", index=False)

    lines = [
        "# Plan 3 control comparison",
        "",
        "Comparability checks passed: graph/manifests, decoder hash, seed, depth, plastic-edge scope, curriculum hash, and update count are identical across conditions. The shuffled condition also references the exact primary `training_log.csv` and applies the same per-stage centered-advantage multiset in permuted order.",
        "",
        "```text",
        table.to_string(index=False),
        "```",
        "",
        "These are held-out curriculum metrics, not a chess Elo estimate.",
    ]
    (args.output / "control_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "comparability_checks": "passed",
        "rows": len(table),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
