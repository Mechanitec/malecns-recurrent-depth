"""Merge control-ensemble members and calculate original-graph percentiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONTROL_VARIANTS = ("degree_preserving_edge_swap_v2", "transmitter_sign_shuffle")
DEPTHS = (2, 8, 16, 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/mechanistic_discovery_v1/control_ensemble"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/control_ensemble"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    files = sorted(args.input.glob("*.csv"))
    files = [path for path in files if path.name != "raw_ensemble.csv"]
    if not files:
        raise FileNotFoundError(f"no control member CSVs in {args.input}")
    frames = [pd.read_csv(path) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    expected = {"original": 1, **{variant: 10 for variant in CONTROL_VARIANTS}}
    counts = raw.groupby("variant")["seed"].nunique().to_dict()
    missing = {variant: expected[variant] - int(counts.get(variant, 0)) for variant in expected if counts.get(variant, 0) < expected[variant]}
    if missing and not args.allow_incomplete:
        raise ValueError(f"ensemble is incomplete: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output / "raw_ensemble.csv", index=False)
    summary = (
        raw.groupby(["variant", "seed", "depth"], as_index=False)
        .agg(
            positions=("position_id", "nunique"),
            mean_regret_cp=("regret_cp", "mean"),
            teacher_agreement=("teacher_best_agreement", "mean"),
            mean_fly_rank_of_teacher_best=("fly_rank_of_teacher_best", "mean"),
            readout_effective_rank=("readout_effective_rank", "mean"),
            candidate_dispersion=("candidate_dispersion", "mean"),
            saturation_fraction=("saturation_fraction_mean", "mean"),
            mean_latency_s=("latency_s", "mean"),
        )
    )
    summary.to_csv(args.output / "summary_by_member.csv", index=False)
    control_summary = summary[summary["variant"].isin(CONTROL_VARIANTS)]
    original_summary = summary[summary["variant"] == "original"]
    percentile_rows = []
    metrics = {
        "mean_regret_cp": "lower_is_better",
        "teacher_agreement": "higher_is_better",
        "readout_effective_rank": "higher_is_better",
        "candidate_dispersion": "higher_is_better",
        "saturation_fraction": "lower_is_better",
    }
    for depth in DEPTHS:
        for variant in CONTROL_VARIANTS:
            controls = control_summary[(control_summary["variant"] == variant) & (control_summary["depth"] == depth)]
            original = original_summary[original_summary["depth"] == depth]
            if controls.empty or original.empty:
                continue
            for metric, direction in metrics.items():
                value = float(original.iloc[0][metric])
                control_values = controls[metric].astype(float)
                percentile = float((control_values <= value).mean() * 100.0)
                if direction == "higher_is_better":
                    percentile = float((control_values >= value).mean() * 100.0)
                percentile_rows.append({
                    "depth": depth, "control_family": variant, "metric": metric,
                    "original_value": value, "control_mean": float(control_values.mean()),
                    "control_min": float(control_values.min()), "control_max": float(control_values.max()),
                    "original_percentile_better": percentile, "n_controls": len(control_values),
                })
    pd.DataFrame(percentile_rows).to_csv(args.output / "original_percentiles.csv", index=False)
    metadata = {
        "status": "complete" if not missing else "incomplete",
        "member_files": len(files), "variant_seed_counts": counts,
        "missing_members": missing, "depths": list(DEPTHS), "positions_per_member": 64,
        "percentile_definition": "fraction of matched controls at least as good as original, with metric direction applied",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
