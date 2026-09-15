"""Tune internal-signal stopping policies on dev_screen and confirm on dev_confirm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEPTHS = (1, 2, 4, 8, 16, 32, 64)


def _select_depth(group: pd.DataFrame, policy: str, delta_limit: float, margin_limit: float) -> int:
    rows = group.sort_values("depth")
    if policy == "fixed_d64":
        return 64
    for _, row in rows.iterrows():
        depth = int(row["depth"])
        if depth < 2:
            continue
        stable = int(row["winner_switch_from_previous"]) == 0
        low_delta = float(row["delta_mean"]) <= delta_limit
        high_margin = float(row["score_margin"]) >= margin_limit
        if policy == "stable_delta" and stable and low_delta:
            return depth
        if policy == "margin_and_delta" and stable and low_delta and high_margin:
            return depth
    return 64


def _evaluate(raw: pd.DataFrame, split: str, condition: tuple[str, str], policy: str, delta_limit: float, margin_limit: float) -> dict[str, object]:
    subset = raw[(raw["source_split"] == split) & (raw["gain_scale"] == condition[0]) & (raw["input_mode"] == condition[1])]
    chosen = []
    for position_id, group in subset.groupby("position_id", sort=False):
        depth = _select_depth(group, policy, delta_limit, margin_limit)
        row = group[group["depth"] == depth].iloc[0]
        chosen.append({"position_id": position_id, "depth": depth, "regret_cp": float(row["regret_cp"]), "predicted_move": row["predicted_move"]})
    result = pd.DataFrame(chosen)
    return {
        "positions": int(len(result)), "mean_regret_cp": float(result["regret_cp"].mean()),
        "teacher_agreement": float((result["regret_cp"] == 0).mean()),
        "mean_depth": float(result["depth"].mean()), "p90_depth": float(result["depth"].quantile(0.9)),
        "max_depth_fraction": float((result["depth"] == 64).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/adaptive_stopping"))
    args = parser.parse_args()
    raw = pd.read_csv(args.raw)
    raw["gain_scale"] = raw["gain_scale"].map(lambda value: f"{float(value):.2f}")
    raw["depth"] = raw["depth"].astype(int)
    screen = raw[raw["source_split"] == "dev_screen"]
    regime_row = screen.groupby(["gain_scale", "input_mode", "depth"], as_index=False)["regret_cp"].mean().sort_values("regret_cp").iloc[0]
    conditions = [("1.00", "clamped_sensory"), (str(regime_row["gain_scale"]), str(regime_row["input_mode"]))]
    rows = []
    for gain, mode in dict.fromkeys(conditions):
        subset = screen[(screen["gain_scale"] == gain) & (screen["input_mode"] == mode)]
        delta_values = subset["delta_mean"].dropna().to_numpy()
        margin_values = subset["score_margin"].dropna().to_numpy()
        delta_limits = sorted(set([float(np.quantile(delta_values, q)) for q in (0.25, 0.5, 0.75)] + [0.01, 0.05, 0.1, 0.5]))
        margin_limits = sorted(set([0.0] + [float(np.quantile(margin_values, q)) for q in (0.25, 0.5, 0.75)]))
        for policy in ("fixed_d64", "stable_delta", "margin_and_delta"):
            for delta_limit in delta_limits if policy != "fixed_d64" else [0.0]:
                for margin_limit in margin_limits if policy == "margin_and_delta" else [0.0]:
                    screen_metrics = _evaluate(raw, "dev_screen", (gain, mode), policy, delta_limit, margin_limit)
                    rows.append({
                        "gain_scale": gain, "input_mode": mode, "policy": policy,
                        "delta_limit": delta_limit, "margin_limit": margin_limit,
                        **{f"screen_{key}": value for key, value in screen_metrics.items()},
                    })
    results = pd.DataFrame(rows)
    # The small compute penalty makes the selection explicit while keeping move quality primary.
    results["screen_objective"] = results["screen_mean_regret_cp"] + 1.0 * results["screen_mean_depth"] / 64.0
    selected_rows = []
    for condition, group in results.groupby(["gain_scale", "input_mode"], sort=False):
        chosen = group.sort_values(["screen_objective", "screen_mean_depth", "screen_mean_regret_cp"]).iloc[0]
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows)
    confirmations = []
    for _, row in selected.iterrows():
        metrics = _evaluate(
            raw, "dev_confirm", (str(row["gain_scale"]), str(row["input_mode"])),
            str(row["policy"]), float(row["delta_limit"]), float(row["margin_limit"]),
        )
        confirmations.append({**row.to_dict(), **{f"confirm_{key}": value for key, value in metrics.items()}})
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "screen_policy_grid.csv", index=False)
    pd.DataFrame(confirmations).to_csv(args.output / "confirm_selected_policies.csv", index=False)
    metadata = {
        "status": "complete", "raw_factorial": str(args.raw), "policy_uses_teacher_at_inference": False,
        "selection": "dev_screen minimum mean regret plus 1.0 cp times depth/64; confirm held-out source groups",
        "selected_dev_screen_regime": {"gain_scale": str(regime_row["gain_scale"]), "input_mode": str(regime_row["input_mode"]), "depth": int(regime_row["depth"])},
        "policies": ["fixed_d64", "stable_delta", "margin_and_delta"],
        "outputs": ["screen_policy_grid.csv", "confirm_selected_policies.csv"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
