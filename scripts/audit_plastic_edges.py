"""Audit the predeclared MaleCNS KC-like to MBON edge set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.malecns import load_malecns_feather
from malecns_rd.neuromodulated_plasticity import select_kc_mbon_edges


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--kc-manifest", type=Path, required=True)
    parser.add_argument("--mbon-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/neuromodulated_curriculum_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    graph, annotations = load_malecns_feather(
        args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=3
    )
    kc = manifest(args.kc_manifest)
    mbon = manifest(args.mbon_manifest)
    state = select_kc_mbon_edges(graph, np.asarray(kc["indices"], dtype=np.int64), np.asarray(mbon["indices"], dtype=np.int64))
    rows = []
    in_degree = np.diff(graph.weights.tocsc().indptr)
    out_degree = np.diff(graph.weights.tocsr().indptr)
    for offset, post, pre, weight in zip(state.edge_offsets, state.post_indices, state.pre_indices, state.original_weights):
        rows.append({
            "edge_offset": int(offset),
            "pre_index": int(pre),
            "post_index": int(post),
            "pre_body_id": int(graph.body_ids[pre]),
            "post_body_id": int(graph.body_ids[post]),
            "original_weight": float(weight),
            "original_sign": int(np.sign(weight)),
            "pre_in_kc_manifest": True,
            "post_in_mbon_manifest": True,
            "pre_out_degree": int(out_degree[pre]),
            "post_in_degree": int(in_degree[post]),
        })
    pd.DataFrame(rows).to_csv(args.output / "plastic_edge_audit.csv", index=False)
    metadata = {
        "status": "complete",
        "graph_neurons": graph.n_neurons,
        "graph_edges": graph.n_edges,
        "plastic_edge_count": state.edge_count,
        "plastic_edge_hash": state.edge_hash,
        "kc_manifest": str(args.kc_manifest),
        "kc_manifest_sha256": sha256_file(args.kc_manifest),
        "mbon_manifest": str(args.mbon_manifest),
        "mbon_manifest_sha256": sha256_file(args.mbon_manifest),
        "source_data_hashes": {str(path): sha256_file(path) for path in (args.annotations, args.neurotransmitters, args.weights)},
        "annotation_columns": [str(column) for column in annotations.columns],
        "selection_rule": "existing graph edges with pre in input_mb_kenyon_cells and post in readout_mbon",
        "sign_policy": "original signs are preserved",
        "topology_policy": "no edges are created or deleted",
        "outputs": ["plastic_edge_audit.csv", "plastic_edge_metadata.json"],
    }
    (args.output / "plastic_edge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
