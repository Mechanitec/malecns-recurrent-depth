"""Merge independent factorial workers and write compact summaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-positions", type=int, default=256)
    args = parser.parse_args()
    inputs = sorted(args.directory.glob("raw_gain_*.csv"))
    if not inputs:
        raise FileNotFoundError("no factorial worker CSVs found")
    frames = [pd.read_csv(path) for path in inputs if path.stat().st_size > 0]
    if not frames:
        raise ValueError("factorial worker CSVs contain no rows")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["position_id", "gain_scale", "input_mode", "depth"])
    expected_cells = args.expected_positions * 10 * 2 * 7
    if len(frame) != expected_cells:
        raise ValueError(f"merged factorial has {len(frame)} cells; expected {expected_cells}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    numeric = [
        "regret_cp", "teacher_best_agreement", "score_margin", "score_dispersion",
        "delta_mean", "recurrent_sensory_ratio_mean", "candidate_dispersion",
        "latency_s", "recurrent_passes",
    ]
    summary = (
        frame.groupby(["gain_scale", "input_mode", "depth"], as_index=False)[numeric]
        .mean(numeric_only=True)
        .sort_values(["gain_scale", "input_mode", "depth"])
    )
    summary.to_csv(args.output.with_name("summary_by_condition.csv"), index=False)
    split_summary = (
        frame.groupby(["source_split", "gain_scale", "input_mode", "depth"], as_index=False)[numeric]
        .mean(numeric_only=True)
    )
    split_summary.to_csv(args.output.with_name("summary_by_source_split.csv"), index=False)
    print(f"merged_rows={len(frame)} conditions={frame[['gain_scale','input_mode','depth']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
