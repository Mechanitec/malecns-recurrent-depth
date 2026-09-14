"""Audit graph-control invariants before using specificity claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.graph_controls import degree_preserving_edge_swap_v2, graph_invariants
from run_position_depth_study import graph_variant


def vector_summary(values: list[float] | list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
    }


def compact_invariants(graph) -> dict[str, object]:
    raw = graph_invariants(graph)
    return {
        "n_neurons": raw["n_neurons"],
        "n_edges": raw["n_edges"],
        "weight_sum": raw["weight_sum"],
        "absolute_weight_sum": raw["absolute_weight_sum"],
        "self_loops": raw["self_loops"],
        "in_degree": vector_summary(raw["in_degree"]),
        "out_degree": vector_summary(raw["out_degree"]),
        "weighted_in_strength": vector_summary(raw["weighted_in_strength"]),
        "weighted_out_strength": vector_summary(raw["weighted_out_strength"]),
        "weak_components": int(connected_components(graph.weights, directed=True, connection="weak", return_labels=False)),
    }


def compare_arrays(left: dict[str, object], right: dict[str, object], key: str) -> bool:
    return np.array_equal(np.asarray(left[key]), np.asarray(right[key]))


def old_shuffle_collision_count(graph, seed: int) -> int:
    weights = graph.weights.tocoo(copy=True)
    rng = np.random.default_rng(seed)
    shuffled_dst = weights.row.copy()
    rng.shuffle(shuffled_dst)
    coordinates = set(zip(weights.col.tolist(), shuffled_dst.tolist()))
    return int(len(weights.data) - len(coordinates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--connectome-weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1/graph_control_audit/audit.json"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--swaps", type=int, default=100_000)
    args = parser.parse_args()
    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
        min_synapses=3,
    )
    original = loaded.agent.engine.graph
    old = graph_variant(original, "degree_preserving_topology_shuffle", args.seed)
    corrected = degree_preserving_edge_swap_v2(original, args.seed, swaps=args.swaps)
    original_raw = graph_invariants(original)
    old_raw = graph_invariants(old)
    corrected_raw = graph_invariants(corrected)
    payload = {
        "seed": args.seed,
        "requested_swaps": args.swaps,
        "corrected_control": "degree_preserving_edge_swap_v2",
        "original": compact_invariants(original),
        "current_shuffle": compact_invariants(old),
        "corrected_shuffle": compact_invariants(corrected),
        "current_shuffle_invariants": {
            "in_degree_sequence_exact": compare_arrays(original_raw, old_raw, "in_degree"),
            "out_degree_sequence_exact": compare_arrays(original_raw, old_raw, "out_degree"),
            "weighted_in_strength_sequence_exact": compare_arrays(original_raw, old_raw, "weighted_in_strength"),
            "weighted_out_strength_sequence_exact": compare_arrays(original_raw, old_raw, "weighted_out_strength"),
            "duplicate_coordinate_collisions": old_shuffle_collision_count(original, args.seed),
        },
        "corrected_shuffle_invariants": {
            "in_degree_sequence_exact": compare_arrays(original_raw, corrected_raw, "in_degree"),
            "out_degree_sequence_exact": compare_arrays(original_raw, corrected_raw, "out_degree"),
            "weighted_in_strength_sequence_exact": compare_arrays(original_raw, corrected_raw, "weighted_in_strength"),
            "weighted_out_strength_sequence_exact": compare_arrays(original_raw, corrected_raw, "weighted_out_strength"),
            "duplicate_coordinate_collisions": int(corrected_raw["n_edges"] - len(corrected.weights.data)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
