"""Fit Phase 3 readout probes from a merged activation cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import PROBE_OPTIMIZER, _geometry_metrics, _metrics, _standardize, fit_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--cache", type=Path, default=Path("results/population_study/_activation_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    positions = load_positions(args.corpus)
    frame = pd.read_csv(args.corpus)
    frame["position_id"] = frame["position_id"].astype(str)
    frame = frame[frame["position_id"].isin({str(p["position_id"]) for p in positions})].sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    source = {str(p["position_id"]): str(p["source_split"]) for p in positions}
    frame["source_split"] = frame["position_id"].map(source)
    train_mask = frame["source_split"].eq("dev_screen").to_numpy()
    validation_mask = frame["source_split"].eq("dev_confirm").to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("development corpus must contain both source splits")
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    manifests = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.manifests.glob("readout_*.json")}
    union = np.asarray(sorted({index for payload in manifests.values() for index in payload["indices"]}), dtype=np.int64)
    positions_by_index = {int(index): offset for offset, index in enumerate(union)}
    memmaps = {depth: np.memmap(args.cache / f"union_depth_{depth}.float32", mode="r", dtype=np.float32, shape=(len(frame), len(union))) for depth in DEPTHS}
    result_rows = []
    geometry_rows = []
    for name, payload in sorted(manifests.items()):
        indices = np.asarray(payload["indices"], dtype=np.int64)
        columns = np.asarray([positions_by_index[int(index)] for index in indices], dtype=np.int64)
        for depth in DEPTHS:
            features = np.asarray(memmaps[depth][:, columns])
            train_x, _ = _standardize(features[train_mask], features[train_mask])
            _, validation_x = _standardize(features[train_mask], features[validation_mask])
            weights = fit_probe(train_x, train_frame)
            train_metrics = _metrics(train_x, train_frame, weights)
            metrics = _metrics(validation_x, validation_frame, weights)
            geometry = _geometry_metrics(validation_x, validation_frame)
            result_rows.append({"population_id": name, "anatomical_family": payload["anatomical_family"], "population_role": payload["role"], "population_size": payload["actual_count"], "depth": depth, **{f"validation_{key}": value for key, value in metrics.items()}, **{f"train_{key}": value for key, value in train_metrics.items()}, "train_validation_pair_accuracy_gap": train_metrics["pair_accuracy"] - metrics["pair_accuracy"], "train_validation_top1_accuracy_gap": train_metrics["top1_accuracy"] - metrics["top1_accuracy"], "train_validation_regret_gap_cp": metrics["mean_teacher_regret_cp"] - train_metrics["mean_teacher_regret_cp"], **geometry, "mean_train_feature_std": float(np.mean(np.std(train_x, axis=0))), "mean_validation_score_margin_proxy": float(np.std(validation_x @ weights))})
            geometry_rows.append({"population_id": name, "depth": depth, **geometry, "candidate_dependent_fraction": float(np.mean(np.std(validation_x, axis=0) > 1e-6))})
    pd.DataFrame(result_rows).to_csv(args.output / "brain_region_decoding_atlas.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(args.output / "region_information_by_depth.csv", index=False)
    metadata = {"status": "complete", "positions": len(positions), "candidate_rows": len(frame), "readout_populations": len(manifests), "union_readout_neurons": len(union), "depths": list(DEPTHS), "input_population": "Population Baseline A", "probe": f"train-only standardized {PROBE_OPTIMIZER}", "source": "merged activation shards"}
    (args.output / "brain_region_decoding_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
