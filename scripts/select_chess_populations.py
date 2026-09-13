"""Select reproducible sensory and motor/readout populations from MaleCNS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from malecns_rd.malecns import load_malecns_feather


SENSORY_SUPERCLASSES = (
    "visual_projection",
    "visual_centrifugal",
    "ol_sensory",
    "vnc_sensory",
    "cb_sensory",
    "sensory_ascending",
    "sensory_descending",
)
READOUT_SUPERCLASSES = (
    "descending_neuron",
    "vnc_motor",
    "cb_motor",
    "vnc_efferent",
    "efferent_ascending",
    "efferent_descending",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_populations(
    annotations: Path,
    neurotransmitters: Path,
    weights: Path,
    *,
    sensory_output: Path,
    readout_output: Path,
    manifest_output: Path,
    sensory_count: int = 1024,
    readout_count: int = 256,
    min_synapses: int = 3,
    seed: int = 7,
) -> dict[str, object]:
    graph, ann = load_malecns_feather(
        annotations,
        neurotransmitters,
        weights,
        traced_only=True,
        min_synapses=min_synapses,
    )
    superclass = ann["superclass"].fillna("").astype(str).str.lower()
    graph_index = np.arange(graph.n_neurons, dtype=np.int64)
    outgoing = np.asarray(graph.weights.getnnz(axis=0)).ravel() > 0
    incoming = np.asarray(graph.weights.getnnz(axis=1)).ravel() > 0

    sensory_candidates = graph_index[superclass.isin(SENSORY_SUPERCLASSES).to_numpy() & outgoing]
    readout_candidates = graph_index[superclass.isin(READOUT_SUPERCLASSES).to_numpy() & incoming]
    if len(sensory_candidates) < sensory_count:
        raise ValueError(
            f"only {len(sensory_candidates)} usable sensory neurons; "
            f"cannot select {sensory_count}"
        )
    if len(readout_candidates) < readout_count:
        raise ValueError(
            f"only {len(readout_candidates)} usable readout neurons; "
            f"cannot select {readout_count}"
        )

    rng = np.random.default_rng(seed)
    sensory_indices = np.sort(rng.choice(sensory_candidates, sensory_count, replace=False))
    readout_indices = np.sort(rng.choice(readout_candidates, readout_count, replace=False))
    sensory_output.parent.mkdir(parents=True, exist_ok=True)
    readout_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(sensory_output, sensory_indices)
    np.save(readout_output, readout_indices)

    manifest = {
        "seed": int(seed),
        "min_synapses": int(min_synapses),
        "graph_neurons": int(graph.n_neurons),
        "graph_edges": int(graph.n_edges),
        "sensory_classes": list(SENSORY_SUPERCLASSES),
        "readout_classes": list(READOUT_SUPERCLASSES),
        "sensory_candidates": int(len(sensory_candidates)),
        "readout_candidates": int(len(readout_candidates)),
        "sensory_count": int(len(sensory_indices)),
        "readout_count": int(len(readout_indices)),
        "sensory_output": str(sensory_output),
        "readout_output": str(readout_output),
        "selection_strategy": "seeded_uniform_without_replacement_sorted_indices",
        "input_files": {
            "annotations": {"path": str(annotations), "sha256": _sha256(annotations)},
            "neurotransmitters": {"path": str(neurotransmitters), "sha256": _sha256(neurotransmitters)},
            "weights": {"path": str(weights), "sha256": _sha256(weights)},
        },
        "output_sha256": {
            "sensory_indices": _sha256(sensory_output.with_suffix(sensory_output.suffix + ".npy") if sensory_output.suffix != ".npy" else sensory_output),
            "readout_indices": _sha256(readout_output.with_suffix(readout_output.suffix + ".npy") if readout_output.suffix != ".npy" else readout_output),
        },
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-output", type=Path, required=True)
    parser.add_argument("--readout-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--sensory-count", type=int, default=1024)
    parser.add_argument("--readout-count", type=int, default=256)
    parser.add_argument("--min-synapses", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    manifest = select_populations(
        args.annotations,
        args.neurotransmitters,
        args.weights,
        sensory_output=args.sensory_output,
        readout_output=args.readout_output,
        manifest_output=args.manifest_output,
        sensory_count=args.sensory_count,
        readout_count=args.readout_count,
        min_synapses=args.min_synapses,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
