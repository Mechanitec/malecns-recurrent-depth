"""Audit the current seeded sensory/readout interface against MaleCNS graph structure."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.malecns import load_malecns_feather
from run_mechanistic_factorial import DEPTHS, load_positions


def _bfs(matrix, sources: np.ndarray) -> np.ndarray:
    distances = np.full(matrix.shape[0], -1, dtype=np.int32)
    queue: deque[int] = deque()
    for source in np.unique(np.asarray(sources, dtype=np.int64)):
        if 0 <= source < len(distances):
            distances[source] = 0
            queue.append(int(source))
    while queue:
        node = queue.popleft()
        for neighbor in matrix.indices[matrix.indptr[node]:matrix.indptr[node + 1]]:
            neighbor = int(neighbor)
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _pair_distance(features: np.ndarray, groups: list[np.ndarray]) -> float:
    values = []
    for indices in groups:
        x = features[indices].astype(np.float64)
        if len(x) > 1:
            values.append(float(np.mean(np.linalg.norm(x[:, None] - x[None, :], axis=2))))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--readout-indices", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--atlas", type=Path, default=Path("results/population_study/malecns_region_atlas.json"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--activation-positions", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph, annotations = load_malecns_feather(args.annotations, args.neurotransmitters, args.weights, traced_only=True, min_synapses=3)
    sensory = np.load(args.sensory_indices).astype(np.int64)
    readout = np.load(args.readout_indices).astype(np.int64)
    with args.atlas.open(encoding="utf-8") as handle:
        atlas = json.load(handle)
    family_indices = {name: np.asarray(value["graph_indices"], dtype=np.int64) for name, value in atlas["families"].items()}
    forward = _bfs(graph.weights.tocsc(), sensory)
    reverse = _bfs(graph.weights.tocsr(), readout)
    binary = graph.weights.astype(bool)
    reciprocal = binary.multiply(binary.T).getnnz(axis=0) > 0
    in_degree = np.asarray(graph.weights.getnnz(axis=1)).ravel()
    out_degree = np.asarray(graph.weights.getnnz(axis=0)).ravel()
    in_strength = np.asarray(np.abs(graph.weights).sum(axis=1)).ravel()
    out_strength = np.asarray(np.abs(graph.weights).sum(axis=0)).ravel()
    role = np.full(graph.n_neurons, "other", dtype=object)
    role[sensory] = "baseline_sensory"
    role[readout] = "baseline_readout"
    rows = []
    for index in np.unique(np.concatenate([sensory, readout])):
        index = int(index)
        row = annotations.iloc[index]
        rows.append({
            "graph_index": index, "body_id": int(row["bodyId"]), "role": role[index],
            "superclass": str(row.get("superclass", "")), "class": str(row.get("class", "")),
            "subclass": str(row.get("subclass", "")), "type": str(row.get("type", "")),
            "flywire_type": str(row.get("flywireType", "")), "hemibrain_type": str(row.get("hemibrainType", "")),
            "manc_type": str(row.get("mancType", "")), "in_degree": int(in_degree[index]),
            "out_degree": int(out_degree[index]), "weighted_in_strength": float(in_strength[index]),
            "weighted_out_strength": float(out_strength[index]), "reciprocal_edge": int(reciprocal[index]),
            "input_reach_depth": int(forward[index]), "output_reverse_distance": int(reverse[index]),
        })
    audit = pd.DataFrame(rows).sort_values(["role", "graph_index"])
    audit.to_csv(args.output / "baseline_population_audit.csv", index=False)
    family_distance = {}
    for family, indices in family_indices.items():
        input_values = forward[indices]
        output_values = reverse[indices]
        family_distance[family] = {
            "count": int(len(indices)),
            "baseline_sensory_to_family": {"reachable_fraction": float(np.mean(input_values >= 0)), "median": float(np.median(input_values[input_values >= 0])) if np.any(input_values >= 0) else None},
            "family_to_baseline_readout": {"reachable_fraction": float(np.mean(output_values >= 0)), "median": float(np.median(output_values[output_values >= 0])) if np.any(output_values >= 0) else None},
        }

    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    if not isinstance(loaded.agent.engine, RecurrentDepthEngine):
        raise RuntimeError("baseline activation audit requires the rate checkpoint")
    positions = load_positions(args.development_corpus)[:args.activation_positions]
    groups = []
    sensory_blocks = []
    for position in positions:
        moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
        groups.append(np.arange(sum(len(x) for x in groups), sum(len(x) for x in groups) + len(moves), dtype=np.int64))
        sensory_blocks.append(np.column_stack([loaded.agent.projector.project(encode_board_move(position["board"], move)) for move in moves]))
    trajectory = loaded.agent.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=True)
    activation_rows = []
    for depth in DEPTHS:
        state = trajectory.snapshots[depth]
        readout_state = state[readout].T
        readout_std = np.std(readout_state, axis=0)
        activation_rows.append({
            "depth": depth, "activation_variance_mean": float(np.mean(np.var(readout_state, axis=0))),
            "candidate_discriminability_mean_l2": _pair_distance(readout_state, groups),
            "nontrivial_candidate_dependent_fraction": float(np.mean(readout_std > 1e-6)),
            "saturation_fraction_mean": float(np.mean(trajectory.observations[depth]["saturation_fraction"])),
            "state_norm_mean": float(np.mean(trajectory.observations[depth]["state_norm"])),
            "delta_mean": float(np.mean(trajectory.observations[depth]["delta"])),
            "latency_s": float(trajectory.latency_s[depth]),
        })
    activation_summary = pd.DataFrame(activation_rows)
    activation_summary.to_csv(args.output / "baseline_activation_by_depth.csv", index=False)
    summary = {
        "status": "complete", "population_id": "Population Baseline A", "sensory_count": int(len(sensory)), "readout_count": int(len(readout)),
        "graph_neurons": graph.n_neurons, "graph_edges": graph.n_edges, "family_distances": family_distance,
        "activation_positions": len(positions), "activation_candidate_rows": int(sum(len(group) for group in groups)),
        "outputs": ["baseline_population_audit.csv", "baseline_activation_by_depth.csv", "baseline_population_summary.json"],
    }
    (args.output / "baseline_population_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
