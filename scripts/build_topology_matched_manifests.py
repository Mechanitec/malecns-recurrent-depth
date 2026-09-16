"""Create deterministic topology-matched controls for prioritized families."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from malecns_rd.malecns import load_malecns_feather


def _distance(matrix: object, sources: np.ndarray) -> np.ndarray:
    distances = np.full(matrix.shape[0], -1, dtype=np.int32)  # type: ignore[attr-defined]
    queue: deque[int] = deque()
    for source in np.unique(sources):
        distances[int(source)] = 0
        queue.append(int(source))
    while queue:
        node = queue.popleft()
        for neighbor in matrix.indices[matrix.indptr[node]:matrix.indptr[node + 1]]:  # type: ignore[attr-defined]
            neighbor = int(neighbor)
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--readout-indices", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, default=Path("results/population_study/malecns_region_atlas.json"))
    parser.add_argument("--output", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--seed", type=int, default=23001)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph, _ = load_malecns_feather(args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=3)
    atlas = json.loads(args.atlas.read_text(encoding="utf-8"))
    sensory = np.load(args.sensory_indices).astype(np.int64)
    baseline_readout = np.load(args.readout_indices).astype(np.int64)
    forward = _distance(graph.weights.tocsc(), sensory)
    reverse = _distance(graph.weights.tocsr(), baseline_readout)
    in_degree = np.asarray(graph.weights.getnnz(axis=1)).ravel()
    out_degree = np.asarray(graph.weights.getnnz(axis=0)).ravel()
    scc_count, scc_labels = connected_components(graph.weights, directed=True, connection="strong", return_labels=True)
    scc_size = np.bincount(scc_labels)
    candidates = np.arange(graph.n_neurons, dtype=np.int64)
    rows = []
    for family, payload in atlas["families"].items():
        target = np.asarray(payload["graph_indices"], dtype=np.int64)
        if len(target) < 8:
            continue
        count = min(args.count, len(target))
        target_profile = np.asarray([np.median(in_degree[target]), np.median(out_degree[target]), np.median(forward[target][forward[target] >= 0]) if np.any(forward[target] >= 0) else 0.0, np.median(reverse[target][reverse[target] >= 0]) if np.any(reverse[target] >= 0) else 0.0, np.mean(scc_size[scc_labels[target]] > 1)], dtype=np.float64)
        values = np.column_stack([in_degree, out_degree, np.maximum(forward, 0), np.maximum(reverse, 0), (scc_size[scc_labels] > 1).astype(np.float64)])
        scale = np.maximum(np.std(values, axis=0), 1.0)
        distance = np.linalg.norm((values - target_profile) / scale, axis=1)
        excluded = set(target.tolist())
        pool = np.asarray([index for index in candidates if int(index) not in excluded], dtype=np.int64)
        nearest = pool[np.argsort(distance[pool], kind="stable")[:min(len(pool), count * 8)]]
        selected = np.sort(np.random.default_rng(args.seed + sum(ord(char) for char in family)).choice(nearest, count, replace=False))
        name = f"topology_readout_{family.lower()}"
        manifest = {"population_id": name, "role": "readout", "selection_method": "nearest standardized graph-profile with deterministic random tie-break", "anatomical_family": f"topology_matched_{family}", "seed": args.seed, "requested_count": args.count, "actual_count": len(selected), "body_ids": [], "indices": selected.tolist(), "target_family": family, "target_profile": target_profile.tolist(), "source_data_hashes": {"atlas": _hash(args.atlas), "annotations": _hash(args.annotations)}, "selection_code_commit": "working-tree"}
        (args.output / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        rows.append({"family": family, "manifest": name, "target_count": len(target), "control_count": len(selected), "target_profile": target_profile.tolist()})
    (args.output.parent / "topology_matched_index.json").write_text(json.dumps({"status": "complete", "graph_neurons": graph.n_neurons, "scc_count": int(scc_count), "families": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "families": len(rows), "scc_count": int(scc_count)}, indent=2))


if __name__ == "__main__":
    main()
