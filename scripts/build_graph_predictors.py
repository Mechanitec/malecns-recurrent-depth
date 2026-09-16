"""Relate interface quality to simple MaleCNS graph predictors."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from malecns_rd.malecns import load_malecns_feather


def _distances(matrix: object, sources: np.ndarray) -> np.ndarray:
    distance = np.full(matrix.shape[0], -1, dtype=np.int32)  # type: ignore[attr-defined]
    queue: deque[int] = deque()
    for source in np.unique(sources):
        if 0 <= int(source) < len(distance):
            distance[int(source)] = 0
            queue.append(int(source))
    while queue:
        node = queue.popleft()
        start, end = matrix.indptr[node], matrix.indptr[node + 1]  # type: ignore[attr-defined]
        for neighbor in matrix.indices[start:end]:  # type: ignore[attr-defined]
            neighbor = int(neighbor)
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--readout-indices", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, default=Path("results/population_study/malecns_region_atlas.json"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph, annotations = load_malecns_feather(args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=3)
    sensory = np.load(args.sensory_indices).astype(np.int64)
    baseline_readout = np.load(args.readout_indices).astype(np.int64)
    forward = _distances(graph.weights.tocsc(), sensory)
    reverse = _distances(graph.weights.tocsr(), baseline_readout)
    scc_count, scc_labels = connected_components(graph.weights, directed=True, connection="strong", return_labels=True)
    scc_size = np.bincount(scc_labels)
    in_degree = np.asarray(graph.weights.getnnz(axis=1)).ravel()
    out_degree = np.asarray(graph.weights.getnnz(axis=0)).ravel()
    in_strength = np.asarray(np.abs(graph.weights).sum(axis=1)).ravel()
    out_strength = np.asarray(np.abs(graph.weights).sum(axis=0)).ravel()
    manifests = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.manifests.glob("*.json")}
    rows = []
    for name, payload in manifests.items():
        indices = np.asarray(payload["indices"], dtype=np.int64)
        if payload.get("role") not in {"input", "readout"} or not len(indices):
            continue
        rows.append({
            "population_id": name, "role": payload["role"], "anatomical_family": payload["anatomical_family"], "population_size": len(indices),
            "mean_in_degree": float(np.mean(in_degree[indices])), "mean_out_degree": float(np.mean(out_degree[indices])),
            "mean_in_strength": float(np.mean(in_strength[indices])), "mean_out_strength": float(np.mean(out_strength[indices])),
            "reachable_from_baseline_sensory_fraction": float(np.mean(forward[indices] >= 0)),
            "median_distance_from_baseline_sensory": float(np.median(forward[indices][forward[indices] >= 0])) if np.any(forward[indices] >= 0) else None,
            "reachable_to_baseline_readout_fraction": float(np.mean(reverse[indices] >= 0)),
            "median_distance_to_baseline_readout": float(np.median(reverse[indices][reverse[indices] >= 0])) if np.any(reverse[indices] >= 0) else None,
            "scc_member_fraction": float(np.mean(scc_size[scc_labels[indices]] > 1)), "scc_count": int(len(np.unique(scc_labels[indices]))),
        })
    output = pd.DataFrame(rows)
    atlas_path = args.atlas
    if atlas_path.exists():
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        output["atlas_mapping_confidence"] = output["anatomical_family"].map({name: value["confidence"] for name, value in atlas["families"].items()})
    for source_name in ("brain_region_decoding_atlas.csv", "input_population_results.csv"):
        path = args.output / source_name
        if not path.exists():
            continue
        evidence = pd.read_csv(path)
        if source_name.startswith("brain"):
            evidence = evidence.rename(columns={"population_id": "population_id"})
            evidence = evidence[evidence["depth"].isin([1, 8])][["population_id", "depth", "validation_mean_teacher_regret_cp"]]
        else:
            evidence = evidence[evidence["depth"].isin([1, 8])].groupby("readout_population_id", as_index=False)["validation_mean_teacher_regret_cp"].mean().rename(columns={"readout_population_id": "population_id"})
            evidence["depth"] = "input-study"
        output = output.merge(evidence, on="population_id", how="left")
    output.to_csv(args.output / "graph_predictors.csv", index=False)
    metadata = {"status": "complete", "graph_neurons": graph.n_neurons, "graph_edges": graph.n_edges, "scc_count": int(scc_count), "outputs": ["graph_predictors.csv"], "modeling": "descriptive development-set association; no inferential claim"}
    (args.output / "graph_predictors_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
