"""Create data-backed summary plots from a completed Plan 3 run."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def generate(output: Path) -> None:
    stage_order = ["movement", "endgames", "tactics", "mates"]
    metrics = pd.read_csv(output / "stage_metrics.csv")
    biological = pd.read_csv(output / "biological_deviation.csv")

    from PIL import Image, ImageDraw

    def chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, *, line: bool = False) -> None:
        width, height = 960, 520
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = 90, 60, 900, 440
        draw.text((left, 20), title, fill="black")
        draw.line((left, top, left, bottom), fill="black", width=2); draw.line((left, bottom, right, bottom), fill="black", width=2)
        vmax = max(1.0, max((abs(float(value)) for value in values), default=1.0))
        points = []
        for index, (label, value) in enumerate(zip(labels, values)):
            x = left + (right - left) * (index + 0.5) / max(1, len(labels))
            y = bottom - (bottom - top) * max(0.0, float(value)) / vmax
            draw.text((int(x - 25), bottom + 10), label[:12], fill="black")
            if line:
                points.append((int(x), int(y))); draw.ellipse((int(x - 5), int(y - 5), int(x + 5), int(y + 5)), fill="#1f77b4")
            else:
                draw.rectangle((int(x - 35), int(y), int(x + 35), bottom), fill="#1f77b4")
            draw.text((int(x - 22), int(y - 22),), f"{float(value):.2f}", fill="black")
        if line and len(points) > 1:
            draw.line(points, fill="#1f77b4", width=3)
        draw.text((10, 250), ylabel, fill="black")
        image.save(path)

    chart(output / "curriculum_retention.png", "Curriculum retention", metrics["stage"].tolist(), metrics["top1_accuracy_d8"].tolist(), "top-1", line=True)

    blunders = []
    for stage in stage_order:
        path = output / {"movement": "movement_validation.csv", "endgames": "endgame_validation.csv", "tactics": "tactical_validation.csv", "mates": "mate_validation.csv"}[stage]
        frame = pd.read_csv(path)
        frame = frame.loc[frame["depth"] == 8]
        blunders.append(float((frame["regret_cp"] > 0).mean()) if not frame.empty else 0.0)
    chart(output / "blunder_rate_by_stage.png", "D8 blunder rate by stage", stage_order, blunders, "blunder rate")

    chart(output / "mate_solve_rate.png", "D8 mate solve rate", metrics["stage"].tolist(), metrics["mate_solve_rate_d8"].fillna(0.0).tolist(), "solve rate")

    joined = metrics.merge(biological[["stage", "rms_log_ratio"]], on="stage", how="left")
    labels = joined["stage"].tolist()
    values = (joined["top1_accuracy_d8"] / joined["rms_log_ratio"].clip(lower=1e-6)).clip(upper=1.0).tolist()
    chart(output / "biological_deviation_vs_performance.png", "Biological deviation vs performance", labels, values, "top1 / deviation")

    depth = pd.read_csv(output / "depth_metrics.csv")
    depth_values = depth.groupby("depth")["top1_correct"].mean()
    chart(output / "learning_curve.png", "Validation performance by recurrent depth", [str(x) for x in depth_values.index], depth_values.tolist(), "top-1", line=True)

    training = pd.read_csv(output / "training_log.csv")
    regret_values = training.groupby("stage")["regret_cp"].mean().reindex(stage_order).fillna(0.0)
    chart(output / "regret_distribution_by_stage.png", "Mean training regret by stage", stage_order, regret_values.tolist(), "mean regret cp")

    checkpoint = output / "checkpoint_after_mates.npz"
    import numpy as np
    with np.load(checkpoint, allow_pickle=False) as data:
        ratios = np.exp(data["log_ratios"])
    bins = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    counts = []
    labels = []
    for low, high in zip(bins[:-1], bins[1:]):
        counts.append(int(((ratios >= low) & (ratios < high)).sum()) if high < 2.0 else int(((ratios >= low) & (ratios <= high)).sum()))
        labels.append(f"{low:.2f}-{high:.2f}")
    chart(output / "weight_ratio_distribution.png", "Final plastic weight ratio distribution", labels, counts, "edge count")
    print(f"updated plots in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)
