"""Analyze cross-depth transfer for biologically prioritized readout populations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import _geometry_metrics, _metrics, _standardize, fit_probe


PRIORITY = (
    "readout_fan_shaped_body",
    "readout_hdelta_family",
    "readout_pfn",
    "readout_pfr",
    "readout_pfl",
    "readout_mbon",
    "readout_smp",
    "readout_cre",
    "readout_sip",
    "readout_central_complex",
    "readout_lal",
    "readout_baseline_a",
)


def _cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    gram_left = left @ left.T
    gram_right = right @ right.T
    numerator = float(np.sum(gram_left * gram_right))
    denominator = float(np.sqrt(np.sum(gram_left * gram_left) * np.sum(gram_right * gram_right)))
    return numerator / denominator if denominator > 0 else 0.0


def _winner_switch(scores: list[np.ndarray]) -> float:
    if len(scores) < 2:
        return float("nan")
    switches = []
    for previous, current in zip(scores, scores[1:]):
        switches.append(float(np.mean(np.asarray(previous) != np.asarray(current))))
    return float(np.mean(switches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--cache", type=Path, default=Path("results/population_study/_activation_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    positions = load_positions(args.corpus)
    corpus = pd.read_csv(args.corpus)
    corpus["position_id"] = corpus["position_id"].astype(str)
    selected_ids = {str(position["position_id"]) for position in positions}
    frame = corpus[corpus["position_id"].isin(selected_ids)].copy()
    frame["source_split"] = frame["position_id"].map({str(p["position_id"]): str(p["source_split"]) for p in positions})
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    train_mask = frame["source_split"].eq("dev_screen").to_numpy()
    validation_mask = frame["source_split"].eq("dev_confirm").to_numpy()
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)

    manifest_data = {}
    for path in sorted(args.manifests.glob("readout_*.json")):
        manifest_data[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    names = [name for name in PRIORITY if name in manifest_data]
    if not names:
        raise FileNotFoundError("no priority readout manifests found")
    union = np.asarray(sorted({index for payload in manifest_data.values() for index in payload["indices"]}), dtype=np.int64)
    union_positions = {int(index): offset for offset, index in enumerate(union)}
    memmaps = {}
    for depth in DEPTHS:
        path = args.cache / f"union_depth_{depth}.float32"
        if not path.exists():
            raise FileNotFoundError(f"missing activation cache: {path}")
        memmaps[depth] = np.memmap(path, mode="r", dtype=np.float32, shape=(len(frame), len(union)))

    transfer_rows = []
    information_rows = []
    for name in names:
        payload = manifest_data[name]
        columns = np.asarray([union_positions[int(index)] for index in payload["indices"]], dtype=np.int64)
        standardized = {}
        validation_by_depth = {}
        for depth in DEPTHS:
            raw = np.asarray(memmaps[depth][:, columns])
            standardized[depth], _ = _standardize(raw[train_mask], raw[train_mask])
            validation = _standardize(raw[train_mask], raw[validation_mask])[1]
            validation_by_depth[depth] = validation
            self_depth_weights = fit_probe(standardized[depth], train_frame)
            train_metrics = _metrics(standardized[depth], train_frame, self_depth_weights)
            self_depth_metrics = _metrics(validation, validation_frame, self_depth_weights)
            geometry = _geometry_metrics(validation, validation_frame)
            depth_index = DEPTHS.index(depth)
            previous_depth = DEPTHS[depth_index - 1] if depth_index else None
            information_rows.append({
                "population_id": name, "anatomical_family": payload["anatomical_family"],
                "depth": depth, "population_size": payload["actual_count"],
                **geometry,
                "cka_to_depth_1": None,
                **{f"validation_{key}": value for key, value in self_depth_metrics.items()},
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "train_validation_pair_accuracy_gap": train_metrics["pair_accuracy"] - self_depth_metrics["pair_accuracy"],
                "train_validation_top1_accuracy_gap": train_metrics["top1_accuracy"] - self_depth_metrics["top1_accuracy"],
                "train_validation_regret_gap_cp": self_depth_metrics["mean_teacher_regret_cp"] - train_metrics["mean_teacher_regret_cp"],
                "state_norm_mean": float(np.linalg.norm(validation, axis=1).mean()),
                "delta_norm_mean": float(np.linalg.norm(validation - validation_by_depth[previous_depth], axis=1).mean()) if previous_depth is not None else 0.0,
                "saturation_fraction": float(np.mean(np.abs(validation) >= 3.0)),
            })
        for depth in DEPTHS:
            info_index = (info_index for info_index, row in enumerate(information_rows) if row["population_id"] == name and row["depth"] == depth)
            index = next(info_index)
            information_rows[index]["cka_to_depth_1"] = _cka(validation_by_depth[depth], validation_by_depth[DEPTHS[0]])
        for train_depth in DEPTHS:
            weights = fit_probe(standardized[train_depth], train_frame)
            test_winners = []
            for test_depth in DEPTHS:
                raw = np.asarray(memmaps[test_depth][:, columns])
                _, test_x = _standardize(raw[train_mask], raw[validation_mask])
                metrics = _metrics(test_x, validation_frame, weights)
                scores = test_x @ weights
                winners = []
                for _, group in validation_frame.groupby("position_id", sort=False):
                    indices = group.index.to_numpy(dtype=np.int64)
                    winners.append(int(indices[np.argmax(scores[indices])]))
                test_winners.append(np.asarray(winners, dtype=np.int64))
                transfer_rows.append({
                    "population_id": name, "anatomical_family": payload["anatomical_family"],
                    "population_size": payload["actual_count"], "train_depth": train_depth,
                    "test_depth": test_depth, **{f"validation_{key}": value for key, value in metrics.items()},
                })
            switches = _winner_switch(test_winners)
            for row in transfer_rows[-len(DEPTHS):]:
                row["winner_switch_frequency_across_test_depth"] = switches

    info = pd.DataFrame(information_rows)
    for name in names:
        subset = info[info["population_id"] == name].sort_values("depth")
        baseline = subset["effective_rank"].iloc[0]
        info.loc[subset.index, "effective_rank_relative_to_depth_1"] = subset["effective_rank"] / max(float(baseline), 1e-6)
    info.to_csv(args.output / "region_information_by_depth.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(args.output / "depth_probe_transfer.csv", index=False)
    metadata = {
        "status": "complete", "corpus": str(args.corpus), "candidate_rows": len(frame),
        "populations": names, "depths": list(DEPTHS), "train_split": "dev_screen",
        "validation_split": "dev_confirm", "standardization": "training rows only",
        "outputs": ["depth_probe_transfer.csv", "region_information_by_depth.csv"],
    }
    (args.output / "depth_probe_transfer_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
