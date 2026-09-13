"""Consolidate already-computed depth/control CSVs into plots and metadata."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    args = parser.parse_args()
    depth_dir = args.root / "depth_sweep"
    control_dir = args.root / "control_sweep"
    depth = rows(depth_dir / "metrics.csv")
    controls = rows(control_dir / "metrics.csv")

    plt.figure(figsize=(7, 4))
    x = [int(row["depth"]) for row in depth]
    y = [float(row["rating"]) for row in depth]
    low = [float(row["rating_ci_low"]) for row in depth]
    high = [float(row["rating_ci_high"]) for row in depth]
    plt.errorbar(x, y, yerr=[[a - b for a, b in zip(y, low)], [a - b for a, b in zip(high, y)]], marker="o")
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Compact diagnostic rating")
    plt.tight_layout()
    plt.savefig(depth_dir / "rating_vs_depth.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    variants = sorted({row["variant"] for row in controls})
    for variant in variants:
        selected = [row for row in controls if row["variant"] == variant]
        plt.plot(
            [int(row["depth"]) for row in selected],
            [float(row["teacher_agreement"]) for row in selected],
            marker="o",
            label=variant,
        )
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Teacher-best agreement")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(control_dir / "control_depth_curves.png", dpi=140)
    plt.close()

    metadata = {
        "checkpoint": str(args.checkpoint),
        "teacher_dataset": str(args.teacher_dataset),
        "depth_rows": len(depth),
        "control_rows": len(controls),
        "variants": variants,
        "note": "Compact bounded sweep; ratings use four-ply paired games and position metrics use the sampled test set.",
    }
    (depth_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (control_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
