"""Create deterministic, auditable input/readout manifests from the atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=Path("results/population_study/malecns_region_atlas.json"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--readout-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/populations_v2"))
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--input-count", type=int, default=1024)
    parser.add_argument("--readout-count", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifests").mkdir(parents=True, exist_ok=True)
    annotations = pd.read_feather(args.annotations)
    if "status" in annotations.columns:
        annotations = annotations.loc[annotations["status"].astype(str).str.lower() == "traced"].copy()
    neurotransmitters = pd.read_feather(args.neurotransmitters)
    nt_body_column = next((column for column in ("bodyId", "body", "body_id") if column in neurotransmitters.columns), None)
    if nt_body_column is None:
        raise KeyError("neurotransmitter table has no body ID column")
    annotation_body_column = next((column for column in ("bodyId", "body", "body_id") if column in annotations.columns), None)
    if annotation_body_column is None:
        raise KeyError("annotation table has no body ID column")
    annotations = annotations.loc[annotations[annotation_body_column].isin(neurotransmitters[nt_body_column])].reset_index(drop=True)
    if annotation_body_column != "bodyId":
        annotations = annotations.rename(columns={annotation_body_column: "bodyId"})
    with args.atlas.open(encoding="utf-8") as handle:
        atlas = json.load(handle)
    graph_neurons = int(atlas["graph"]["neurons"])
    rng = np.random.default_rng(args.seed)
    family_indices = {name: np.asarray(value["graph_indices"], dtype=np.int64) for name, value in atlas["families"].items()}
    superclass = annotations["superclass"].fillna("").astype(str).str.lower().to_numpy()
    all_indices = np.arange(graph_neurons, dtype=np.int64)
    central_mask = ~np.isin(superclass, ["visual_projection", "visual_centrifugal", "ol_sensory", "vnc_sensory", "cb_sensory", "sensory_ascending", "sensory_descending"])
    baseline_sensory = np.load(args.sensory_indices).astype(np.int64)
    baseline_readout = np.load(args.readout_indices).astype(np.int64)
    definitions: list[tuple[str, str, np.ndarray, int, int, str]] = []
    input_families = {
        "MB_Kenyon_cells", "SMP", "CRE", "SIP", "fan_shaped_body", "PFN",
        "PFR", "PFL", "central_complex", "LAL", "descending", "visual_projection",
    }
    for family, indices in family_indices.items():
        definitions.append((f"readout_{family.lower()}", "readout", indices, args.readout_count, 7001, f"atlas family: {family}"))
        if family in input_families:
            definitions.append((f"input_{family.lower()}", "input", indices, args.input_count, 17001, f"atlas family input: {family}"))
    definitions.extend([
        ("readout_baseline_a", "readout", baseline_readout, len(baseline_readout), 7, "existing Population Baseline A indices"),
        ("input_baseline_a", "input", baseline_sensory, len(baseline_sensory), 7, "existing Population Baseline A indices"),
        ("readout_random_whole_brain", "readout", all_indices, args.readout_count, args.seed + 1, "uniform whole-brain random matched-size control"),
        ("readout_random_central_brain", "readout", np.flatnonzero(central_mask), args.readout_count, args.seed + 2, "uniform central-brain random matched-size control"),
        ("input_random_whole_brain", "input", all_indices, args.input_count, args.seed + 3, "uniform whole-brain random matched-size control"),
        ("input_random_central_brain", "input", np.flatnonzero(central_mask), args.input_count, args.seed + 4, "uniform central-brain random matched-size control"),
    ])
    manifests = []
    for population_id, role, candidates, requested, seed, method in definitions:
        candidates = np.unique(np.asarray(candidates, dtype=np.int64))
        count = min(int(requested), len(candidates))
        if count == 0:
            selected = np.empty(0, dtype=np.int64)
        elif population_id.endswith("baseline_a"):
            selected = np.sort(candidates)
        else:
            selected = np.sort(np.random.default_rng(seed).choice(candidates, count, replace=False))
        body_ids = annotations.iloc[selected]["bodyId"].astype(np.int64).tolist()
        manifest = {
            "population_id": population_id, "role": role, "selection_method": method,
            "anatomical_family": population_id.removeprefix(f"{role}_"), "seed": int(seed),
            "requested_count": int(requested), "actual_count": int(len(selected)),
            "body_ids": body_ids, "indices": selected.tolist(),
            "annotation_filters": method, "graph_metric_filters": None,
            "source_data_hashes": {"atlas": sha256_file(args.atlas), "annotations": sha256_file(args.annotations)},
            "selection_code_commit": "working-tree",
        }
        path = args.output / "manifests" / f"{population_id}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append({"population_id": population_id, "role": role, "actual_count": len(selected), "path": str(path)})
    summary = {
        "status": "complete", "graph_neurons": graph_neurons, "seed": args.seed,
        "manifest_count": len(manifests), "manifests": manifests,
        "baseline_manifest_preserved": True,
    }
    (args.output / "manifest_index.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
