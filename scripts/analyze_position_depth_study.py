"""Analyze paired effects from the independent position-depth study."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
VARIANTS = (
    "original",
    "degree_preserving_topology_shuffle",
    "transmitter_sign_shuffle",
    "recurrent_edges_attenuated_0.05",
)


def read_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["variant"], row["position_id"], int(row["depth"]))
            if key in rows:
                raise ValueError(f"duplicate result row: {key}")
            rows[key] = row
    if not rows:
        raise ValueError("study output is empty")
    return rows


def numeric(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", "None", "nan", "NaN"}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def bootstrap(values: list[float], seed: int, samples: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def paired_values(data, variant: str, depth: int, field: str) -> tuple[list[str], list[float]]:
    positions = sorted({position for current_variant, position, current_depth in data if current_variant == variant and current_depth == 1})
    values = []
    kept = []
    for position in positions:
        first = data.get((variant, position, 1))
        current = data.get((variant, position, depth))
        if first is None or current is None:
            continue
        first_value = numeric(first, field)
        current_value = numeric(current, field)
        if first_value is None or current_value is None:
            continue
        kept.append(position)
        values.append(first_value - current_value)
    return kept, values


def summary_rows(data) -> list[dict[str, object]]:
    output = []
    for variant in VARIANTS:
        for depth in DEPTHS:
            matching = [row for (current_variant, _, current_depth), row in data.items() if current_variant == variant and current_depth == depth]
            regrets = [numeric(row, "regret_cp") for row in matching]
            agreements = [numeric(row, "teacher_best_agreement") for row in matching]
            top3 = [numeric(row, "top3_teacher_agreement") for row in matching]
            latency = [numeric(row, "latency_s") for row in matching]
            output.append({
                "variant": variant,
                "depth": depth,
                "positions": len(matching),
                "mean_regret_cp": mean(regrets),
                "median_regret_cp": float(np.median([x for x in regrets if x is not None])) if any(x is not None for x in regrets) else None,
                "teacher_best_agreement": mean(agreements),
                "top3_teacher_agreement": mean(top3),
                "mean_latency_s": mean(latency),
                "paired_regret_improvement_cp": mean(paired_values(data, variant, depth, "regret_cp")[1]) if depth > 1 else None,
                "paired_agreement_change": mean(paired_values(data, variant, depth, "teacher_best_agreement")[1]) if depth > 1 else None,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/position_depth_study_v2/raw_position_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/position_depth_study_v2"))
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = read_rows(args.input)
    positions = sorted({position for _, position, _ in data})
    summaries = summary_rows(data)
    write_csv(args.output / "summary_by_depth.csv", summaries)

    control_rows = []
    for control in VARIANTS[1:]:
        for depth in DEPTHS[1:]:
            original_positions, original = paired_values(data, "original", depth, "regret_cp")
            control_positions, control_values = paired_values(data, control, depth, "regret_cp")
            original_map = dict(zip(original_positions, original))
            control_map = dict(zip(control_positions, control_values))
            paired_positions = sorted(set(original_map) & set(control_map))
            did = [original_map[position] - control_map[position] for position in paired_positions]
            low, high = bootstrap(did, args.seed + depth)
            control_rows.append({
                "control_variant": control,
                "depth": depth,
                "positions": len(did),
                "mean_control_did_cp": mean(did),
                "bootstrap_ci_low_cp": low,
                "bootstrap_ci_high_cp": high,
            })
    write_csv(args.output / "specificity_by_control.csv", control_rows)

    original_summary = [row for row in summaries if row["variant"] == "original"]
    best_row = min(
        (row for row in original_summary if row["mean_regret_cp"] is not None),
        key=lambda row: float(row["mean_regret_cp"]),
    )
    best_depth = int(best_row["depth"])
    _, primary_effect = paired_values(data, "original", best_depth, "regret_cp")
    effect_low, effect_high = bootstrap(primary_effect, args.seed + 1000)
    best_control = min(
        (row for row in control_rows if row["depth"] == best_depth and row["mean_control_did_cp"] is not None),
        key=lambda row: float(row["mean_control_did_cp"]),
        default=None,
    )
    summary = {
        "positions": len(positions),
        "variants": list(VARIANTS),
        "depths": list(DEPTHS),
        "bootstrap_samples": 2000,
        "bootstrap_seed": args.seed,
        "best_depth_by_mean_regret": best_depth,
        "d1_to_best_regret_improvement_cp": mean(primary_effect),
        "d1_to_best_regret_improvement_ci_low_cp": effect_low,
        "d1_to_best_regret_improvement_ci_high_cp": effect_high,
        "h1_support": bool(effect_low is not None and effect_low > 0),
        "best_control": best_control,
        "specificity_support": bool(best_control and best_control["bootstrap_ci_low_cp"] is not None and best_control["bootstrap_ci_low_cp"] > 0),
        "interpretation": "Position-level paired regret and control difference-in-differences. This is not an Elo estimate.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    original = [row for row in summaries if row["variant"] == "original"]
    x = list(DEPTHS)
    plt.figure(figsize=(7, 4))
    for variant in VARIANTS:
        rows = [row for row in summaries if row["variant"] == variant]
        plt.plot(x, [row["mean_regret_cp"] for row in rows], marker="o", label=variant)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Mean teacher regret (cp)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output / "regret_vs_depth.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    improvements = [paired_values(data, "original", depth, "regret_cp")[1] for depth in DEPTHS[1:]]
    plt.boxplot(improvements, tick_labels=[str(depth) for depth in DEPTHS[1:]], showfliers=False)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Depth")
    plt.ylabel("D1 regret minus D regret (cp)")
    plt.tight_layout()
    plt.savefig(args.output / "improvement_distributions.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    for variant in VARIANTS:
        rows = [row for row in summaries if row["variant"] == variant]
        plt.plot(x, [row["teacher_best_agreement"] for row in rows], marker="o", label=variant)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Teacher-best agreement")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output / "agreement_vs_depth.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    for control in VARIANTS[1:]:
        rows = [row for row in control_rows if row["control_variant"] == control]
        plt.plot([row["depth"] for row in rows], [row["mean_control_did_cp"] for row in rows], marker="o", label=control)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Recurrent depth")
    plt.ylabel("Original minus control improvement (cp)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output / "control_difference_in_differences.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 4))
    for variant in VARIANTS:
        rows = [row for row in summaries if row["variant"] == variant]
        plt.plot([row["mean_latency_s"] for row in rows], [row["mean_regret_cp"] for row in rows], marker="o", label=variant)
    plt.xlabel("Cumulative trajectory latency (s)")
    plt.ylabel("Mean teacher regret (cp)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output / "latency_compute_quality.png", dpi=140)
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
