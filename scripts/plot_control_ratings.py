"""Plot the completed control rating table with 95% intervals."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    variants = sorted({row["variant"] for row in rows})
    plt.figure(figsize=(9, 5))
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        x = [int(row["depth"]) for row in selected]
        y = [float(row["rating"]) for row in selected]
        low = [float(row["rating_ci_low"]) for row in selected]
        high = [float(row["rating_ci_high"]) for row in selected]
        plt.plot(x, y, marker="o", label=variant)
        plt.fill_between(x, low, high, alpha=0.12)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Compact diagnostic rating")
    plt.legend(fontsize=7)
    plt.tight_layout()
    output = args.table.with_name("control_rating_vs_depth.png")
    plt.savefig(output, dpi=140)
    plt.close()
    print(output)


if __name__ == "__main__":
    main()
