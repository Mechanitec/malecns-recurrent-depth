"""Build a literature-prioritized atlas from the actual MaleCNS annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from malecns_rd.malecns import load_malecns_feather


FAMILY_RULES: dict[str, dict[str, object]] = {
    "MB_Kenyon_cells": {"pattern": r"kenyon|mushroom.?body|(?:^|[^a-z])kc[a-z0-9'_-]*", "confidence": "high", "note": "class/type labels containing Kenyon or KC"},
    "MBON": {"pattern": r"mbon", "confidence": "high", "note": "class/type labels containing MBON"},
    "DAN_modulatory_MB": {"pattern": r"dan|dopaminergic", "confidence": "high", "note": "class/type labels containing DAN or dopaminergic"},
    "SMP": {"pattern": r"(?:^|[^a-z])smp[0-9a-z'_-]*", "confidence": "medium", "note": "cell labels containing SMP"},
    "CRE": {"pattern": r"(?:^|[^a-z])cre[0-9a-z'_-]*", "confidence": "medium", "note": "cell labels containing CRE"},
    "SIP": {"pattern": r"(?:^|[^a-z])sip[0-9a-z'_-]*", "confidence": "medium", "note": "cell labels containing SIP"},
    "fan_shaped_body": {"pattern": r"fan.?shaped|(?:^|[^a-z])fb(?:[0-9a-z'_-]*|$)", "confidence": "medium", "note": "cell labels containing FB/fan-shaped body"},
    "hDelta_family": {"pattern": r"hdelta|h_delta", "confidence": "high", "note": "cell labels containing hDelta"},
    "FC": {"pattern": r"(?:^|[^a-z])fc[0-9a-z'_-]*", "confidence": "medium", "note": "cell labels containing FC"},
    "PFN": {"pattern": r"pfn", "confidence": "high", "note": "cell labels containing PFN"},
    "PFR": {"pattern": r"pfr", "confidence": "high", "note": "cell labels containing PFR"},
    "PFL": {"pattern": r"pfl", "confidence": "high", "note": "cell labels containing PFL"},
    "EPG": {"pattern": r"epg", "confidence": "high", "note": "cell labels containing EPG"},
    "Delta7": {"pattern": r"delta.?7|delta7", "confidence": "high", "note": "cell labels containing Delta7"},
    "PB_EB_associated": {"pattern": r"protocerebral.?bridge|(?:^|[^a-z])pb(?:[0-9a-z'_-]*|$)|ellipsoid.?body|(?:^|[^a-z])eb(?:[0-9a-z'_-]*|$)", "confidence": "medium", "note": "cell labels containing PB/EB/protocerebral bridge"},
    "central_complex": {"pattern": r"(?:^|[^a-z])cx(?:[0-9a-z'_-]*|$)|central.?complex", "confidence": "high", "note": "class/type labels containing CX or central complex"},
    "LAL": {"pattern": r"lal", "confidence": "medium", "note": "cell labels containing LAL"},
    "descending": {"pattern": r"descending", "confidence": "high", "note": "superclass labels containing descending"},
    "VNC_motor_efferent": {"pattern": r"vnc_motor|vnc_efferent|efferent", "confidence": "high", "note": "superclass labels containing motor/efferent"},
    "visual_projection": {"pattern": r"visual_projection|visual_centrifugal", "confidence": "high", "note": "superclass labels containing visual projection"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _body_id_column(frame: pd.DataFrame) -> str:
    for name in ("bodyId", "body", "body_id"):
        if name in frame.columns:
            return name
    raise KeyError(f"no body ID column in {list(frame.columns)}")


def _nt_labels(path: Path) -> dict[int, str]:
    nt = pd.read_feather(path)
    body_col = _body_id_column(nt)
    label = next((c for c in ("consensus_nt", "predicted_nt", "nt", "neurotransmitter", "celltype_nt") if c in nt.columns), None)
    if label is None:
        prob_cols = [c for c in nt.columns if c.lower() in {"acetylcholine", "ach", "gaba", "glutamate", "histamine", "dopamine", "serotonin", "octopamine"}]
        if not prob_cols:
            return {}
        values = nt[prob_cols].to_numpy(dtype=np.float32)
        labels = np.asarray(prob_cols, dtype=object)[np.argmax(values, axis=1)]
    else:
        labels = nt[label].fillna("unknown").astype(str).str.lower().to_numpy()
    return {int(body): str(value) for body, value in zip(nt[body_col], labels)}


def _multi_source_bfs(adjacency: object, sources: np.ndarray) -> np.ndarray:
    matrix = adjacency
    indptr = matrix.indptr  # type: ignore[attr-defined]
    indices = matrix.indices  # type: ignore[attr-defined]
    distances = np.full(matrix.shape[0], -1, dtype=np.int32)  # type: ignore[attr-defined]
    queue: deque[int] = deque()
    for source in np.unique(np.asarray(sources, dtype=np.int64)):
        if 0 <= source < len(distances):
            distances[source] = 0
            queue.append(int(source))
    while queue:
        node = queue.popleft()
        start, end = int(indptr[node]), int(indptr[node + 1])
        for neighbor in indices[start:end]:
            neighbor = int(neighbor)
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "max": None}
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)), "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--readout-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--min-synapses", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph, annotations = load_malecns_feather(
        args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=args.min_synapses
    )
    body_col = _body_id_column(annotations)
    searchable = ("class", "subclass", "superclass", "type", "flywireType", "hemibrainType", "mancType", "group", "instance")
    text = annotations[[c for c in searchable if c in annotations.columns]].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    labels: dict[str, np.ndarray] = {}
    for family, rule in FAMILY_RULES.items():
        labels[family] = text.str.contains(str(rule["pattern"]), regex=True, na=False).to_numpy()

    sensory_indices = np.load(args.sensory_indices).astype(np.int64)
    readout_indices = np.load(args.readout_indices).astype(np.int64)
    outgoing = graph.weights.tocsc()
    incoming = graph.weights.tocsr()
    forward_distances = _multi_source_bfs(outgoing, sensory_indices)
    reverse_distances = _multi_source_bfs(incoming, readout_indices)
    scc_count, scc_labels = connected_components(graph.weights, directed=True, connection="strong", return_labels=True)
    scc_sizes = np.bincount(scc_labels)
    reciprocal = graph.weights.astype(bool).multiply(graph.weights.T.astype(bool)).getnnz(axis=0) > 0
    in_degree = np.asarray(graph.weights.getnnz(axis=1)).ravel()
    out_degree = np.asarray(graph.weights.getnnz(axis=0)).ravel()
    in_strength = np.asarray(np.abs(graph.weights).sum(axis=1)).ravel()
    out_strength = np.asarray(np.abs(graph.weights).sum(axis=0)).ravel()
    nt_map = _nt_labels(args.neurotransmitters)
    body_ids = annotations[body_col].astype(np.int64).to_numpy()
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for family, mask in labels.items():
        indices = np.flatnonzero(mask)
        body_family = body_ids[indices]
        nt_counts = pd.Series([nt_map.get(int(body), "unknown") for body in body_family]).value_counts().to_dict()
        overlaps = {other: int(np.count_nonzero(mask & other_mask)) for other, other_mask in labels.items() if other != family and np.any(mask & other_mask)}
        family_scc = scc_labels[indices]
        details[family] = {
            "pattern": FAMILY_RULES[family]["pattern"], "confidence": FAMILY_RULES[family]["confidence"],
            "mapping_note": FAMILY_RULES[family]["note"], "count": int(len(indices)), "body_ids": body_family.tolist(),
            "graph_indices": indices.tolist(), "neurotransmitter_counts": nt_counts, "overlap_counts": overlaps,
        }
        rows.append({
            "family": family, "annotation_filter": str(FAMILY_RULES[family]["pattern"]),
            "mapping_confidence": str(FAMILY_RULES[family]["confidence"]), "mapping_note": str(FAMILY_RULES[family]["note"]),
            "neuron_count": len(indices), "body_ids": ";".join(str(value) for value in body_family),
            "in_degree_mean": float(np.mean(in_degree[indices])) if len(indices) else None,
            "in_degree_median": float(np.median(in_degree[indices])) if len(indices) else None,
            "out_degree_mean": float(np.mean(out_degree[indices])) if len(indices) else None,
            "out_degree_median": float(np.median(out_degree[indices])) if len(indices) else None,
            "weighted_in_strength_mean": float(np.mean(in_strength[indices])) if len(indices) else None,
            "weighted_out_strength_mean": float(np.mean(out_strength[indices])) if len(indices) else None,
            "reciprocal_fraction": float(np.mean(reciprocal[indices])) if len(indices) else None,
            "scc_member_fraction": float(np.mean(scc_sizes[family_scc] > 1)) if len(indices) else None,
            "scc_count": int(len(np.unique(family_scc))) if len(indices) else 0,
            "input_to_family_distance_median": float(np.median(forward_distances[indices][forward_distances[indices] >= 0])) if np.any(forward_distances[indices] >= 0) else None,
            "family_to_readout_distance_median": float(np.median(reverse_distances[indices][reverse_distances[indices] >= 0])) if np.any(reverse_distances[indices] >= 0) else None,
            "neurotransmitter_counts": json.dumps(nt_counts, sort_keys=True),
            "overlap_counts": json.dumps(overlaps, sort_keys=True),
        })
    pd.DataFrame(rows).sort_values("family").to_csv(args.output / "malecns_region_atlas.csv", index=False)
    payload = {
        "status": "complete", "graph": {"neurons": graph.n_neurons, "edges": graph.n_edges, "scc_count": int(scc_count)},
        "inputs": {"annotations": {"path": str(args.annotations), "sha256": sha256_file(args.annotations)}, "neurotransmitters": {"path": str(args.neurotransmitters), "sha256": sha256_file(args.neurotransmitters)}, "weights": {"path": str(args.weights), "sha256": sha256_file(args.weights)}},
        "baseline_population": {"sensory_count": int(len(sensory_indices)), "readout_count": int(len(readout_indices)), "sensory_indices": sensory_indices.tolist(), "readout_indices": readout_indices.tolist()},
        "annotation_columns_searched": [c for c in searchable if c in annotations.columns], "families": details,
    }
    (args.output / "malecns_region_atlas.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "families": {key: value["count"] for key, value in details.items()}, "scc_count": scc_count}, indent=2))


if __name__ == "__main__":
    main()
