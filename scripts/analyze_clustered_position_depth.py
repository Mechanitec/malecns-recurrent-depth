"""Cluster-aware reanalysis of the frozen recurrent-depth study."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_position_depth_study import DEPTHS, VARIANTS, numeric


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


def grouped_values(values_by_cluster: dict[str, list[float]]) -> list[float]:
    return [value for values in values_by_cluster.values() for value in values]


def ordinary_bootstrap(values: list[float], rng: np.random.Generator, samples: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def cluster_bootstrap(
    values_by_cluster: dict[str, list[float]], rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    clusters = tuple(values_by_cluster)
    if not clusters:
        return None, None
    if len(clusters) == 1:
        value = float(np.mean(values_by_cluster[clusters[0]]))
        return value, value
    means = []
    for _ in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        values = [value for cluster in selected for value in values_by_cluster[str(cluster)]]
        means.append(float(np.mean(values)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def values_by_source(
    data: dict[tuple[str, str, int], dict[str, str]], variant: str, depth: int, field: str
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for (current_variant, _, current_depth), row in data.items():
        if current_variant != variant or current_depth != depth:
            continue
        value = numeric(row, field)
        if value is not None:
            grouped[row["source_game_id"]].append(value)
    return dict(grouped)


def paired_by_source(
    data: dict[tuple[str, str, int], dict[str, str]], variant: str, depth: int, field: str
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    positions = sorted({position for current_variant, position, current_depth in data if current_variant == variant and current_depth == 1})
    for position in positions:
        first = data.get((variant, position, 1))
        current = data.get((variant, position, depth))
        if first is None or current is None:
            continue
        first_value = numeric(first, field)
        current_value = numeric(current, field)
        if first_value is None or current_value is None:
            continue
        grouped[first["source_game_id"]].append(first_value - current_value)
    return dict(grouped)


def did_by_source(
    data: dict[tuple[str, str, int], dict[str, str]], control: str, depth: int
) -> dict[str, list[float]]:
    original = paired_by_source(data, "original", depth, "regret_cp")
    control_values = paired_by_source(data, control, depth, "regret_cp")
    original_map = {}
    control_map = {}
    for source, values in original.items():
        for value in values:
            original_map.setdefault(source, []).append(value)
    for source, values in control_values.items():
        for value in values:
            control_map.setdefault(source, []).append(value)
    return {
        source: [left - right for left, right in zip(original_map[source], control_map[source])]
        for source in sorted(set(original_map) & set(control_map))
        if len(original_map[source]) == len(control_map[source])
    }


def selection_bootstrap(
    data: dict[tuple[str, str, int], dict[str, str]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, dict[int, int]]:
    source_groups = sorted({row["source_game_id"] for row in data.values()})
    selected_effects = []
    selected_depths: Counter[int] = Counter()
    for _ in range(samples):
        selected = rng.choice(source_groups, size=len(source_groups), replace=True)
        selected_set = list(selected)
        means = {}
        effects = {}
        for depth in DEPTHS:
            regrets = []
            d1 = []
            current = []
            for source in selected_set:
                positions = sorted({position for variant, position, current_depth in data if variant == "original" and current_depth == 1})
                for position in positions:
                    first = data.get(("original", position, 1))
                    row = data.get(("original", position, depth))
                    if first is None or row is None or first["source_game_id"] != source:
                        continue
                    first_value = numeric(first, "regret_cp")
                    current_value = numeric(row, "regret_cp")
                    if first_value is not None and current_value is not None:
                        d1.append(first_value)
                        current.append(current_value)
                        regrets.append(current_value)
            means[depth] = float(np.mean(regrets))
            effects[depth] = float(np.mean(np.asarray(d1) - np.asarray(current)))
        best_depth = min(means, key=means.get)
        selected_depths[int(best_depth)] += 1
        selected_effects.append(effects[best_depth])
    return float(np.quantile(selected_effects, 0.025)), float(np.quantile(selected_effects, 0.975)), dict(selected_depths)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/position_depth_study_v2/raw_position_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/statistics"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = read_rows(args.input)
    summary_rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for depth in DEPTHS:
            grouped = values_by_source(data, variant, depth, "regret_cp")
            values = grouped_values(grouped)
            rng = np.random.default_rng(args.seed + variant_index * 100 + depth)
            ordinary_low, ordinary_high = ordinary_bootstrap(values, rng, args.samples)
            cluster_low, cluster_high = cluster_bootstrap(grouped, rng, args.samples)
            paired = paired_by_source(data, variant, depth, "regret_cp") if depth > 1 else {}
            paired_values = grouped_values(paired)
            paired_rng = np.random.default_rng(args.seed + 10_000 + depth)
            paired_low, paired_high = ordinary_bootstrap(paired_values, paired_rng, args.samples) if paired_values else (None, None)
            paired_cluster_low, paired_cluster_high = cluster_bootstrap(paired, paired_rng, args.samples) if paired else (None, None)
            summary_rows.append({
                "variant": variant,
                "depth": depth,
                "source_groups": len(grouped),
                "positions": len(values),
                "mean_regret_cp": float(np.mean(values)) if values else None,
                "ordinary_ci_low_cp": ordinary_low,
                "ordinary_ci_high_cp": ordinary_high,
                "cluster_ci_low_cp": cluster_low,
                "cluster_ci_high_cp": cluster_high,
                "paired_improvement_cp": float(np.mean(paired_values)) if paired_values else None,
                "paired_ordinary_ci_low_cp": paired_low,
                "paired_ordinary_ci_high_cp": paired_high,
                "paired_cluster_ci_low_cp": paired_cluster_low,
                "paired_cluster_ci_high_cp": paired_cluster_high,
            })
    write_csv(args.output / "clustered_summary_by_depth.csv", summary_rows)

    specificity_rows = []
    for control in VARIANTS[1:]:
        for depth in DEPTHS[1:]:
            grouped = did_by_source(data, control, depth)
            values = grouped_values(grouped)
            rng = np.random.default_rng(args.seed + 20_000 + depth)
            ordinary_low, ordinary_high = ordinary_bootstrap(values, rng, args.samples)
            cluster_low, cluster_high = cluster_bootstrap(grouped, rng, args.samples)
            specificity_rows.append({
                "control_variant": control,
                "depth": depth,
                "source_groups": len(grouped),
                "positions": len(values),
                "mean_did_cp": float(np.mean(values)) if values else None,
                "ordinary_ci_low_cp": ordinary_low,
                "ordinary_ci_high_cp": ordinary_high,
                "cluster_ci_low_cp": cluster_low,
                "cluster_ci_high_cp": cluster_high,
            })
    write_csv(args.output / "clustered_specificity_by_control.csv", specificity_rows)

    original = [row for row in summary_rows if row["variant"] == "original"]
    best = min(original, key=lambda row: float(row["mean_regret_cp"]))
    rng = np.random.default_rng(args.seed + 30_000)
    selected_low, selected_high, depth_frequency = selection_bootstrap(data, rng, args.samples)
    selection = {
        "best_observed_depth": int(best["depth"]),
        "best_observed_mean_regret_cp": best["mean_regret_cp"],
        "best_observed_paired_improvement_cp": best["paired_improvement_cp"],
        "best_observed_nominal_position_ci_cp": [best["paired_ordinary_ci_low_cp"], best["paired_ordinary_ci_high_cp"]],
        "best_observed_cluster_ci_cp": [best["paired_cluster_ci_low_cp"], best["paired_cluster_ci_high_cp"]],
        "selection_aware_bootstrap_ci_cp": [selected_low, selected_high],
        "selected_depth_frequency": {str(depth): count for depth, count in sorted(depth_frequency.items())},
        "interpretation": "D2 was selected after inspecting the same corpus; selection-aware values are exploratory and not confirmatory.",
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
    }
    (args.output / "selection_aware_summary.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps({
        "input": str(args.input),
        "source_group_field": "source_game_id",
        "source_groups": len({row["source_game_id"] for row in data.values()}),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "uncertainty_unit": "source_game_id cluster",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_rows": len(summary_rows), "specificity_rows": len(specificity_rows), "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
