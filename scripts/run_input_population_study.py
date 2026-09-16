"""Evaluate biologically motivated symbolic-feature entry populations."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_features import HashedSensoryProjector, encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from run_mechanistic_factorial import DEPTHS, load_positions
from run_probe_transfer import _geometry, _metrics, _standardize, fit_probe


INPUTS = (
    "input_mb_kenyon_cells", "input_fan_shaped_body", "input_smp", "input_cre",
    "input_sip", "input_pfn", "input_pfl", "input_central_complex",
    "input_baseline_a", "input_random_whole_brain", "input_random_central_brain",
)
READOUTS = (
    "readout_fan_shaped_body", "readout_pfn", "readout_pfl",
    "readout_mbon", "readout_central_complex", "readout_baseline_a",
)


def _bfs_reach_depths(matrix: object, sources: np.ndarray, depths: tuple[int, ...]) -> dict[int, float]:
    distances = np.full(matrix.shape[0], -1, dtype=np.int32)  # type: ignore[attr-defined]
    queue: deque[int] = deque()
    for source in np.unique(sources):
        if 0 <= int(source) < len(distances):
            distances[int(source)] = 0
            queue.append(int(source))
    while queue:
        node = queue.popleft()
        if distances[node] >= max(depths):
            continue
        start, end = matrix.indptr[node], matrix.indptr[node + 1]  # type: ignore[attr-defined]
        for neighbor in matrix.indices[start:end]:  # type: ignore[attr-defined]
            neighbor = int(neighbor)
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return {depth: float(np.mean((distances >= 0) & (distances <= depth))) for depth in depths}


def _frame(corpus_path: Path, positions: list[dict[str, object]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(corpus_path)
    frame["position_id"] = frame["position_id"].astype(str)
    ids = {str(position["position_id"]) for position in positions}
    frame = frame[frame["position_id"].isin(ids)].copy()
    frame["source_split"] = frame["position_id"].map({str(p["position_id"]): str(p["source_split"]) for p in positions})
    frame = frame.sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    return frame, frame["source_split"].eq("dev_screen").to_numpy(), frame["source_split"].eq("dev_confirm").to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study"))
    parser.add_argument("--limit-positions", type=int, default=256)
    parser.add_argument("--batch-positions", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    positions = load_positions(args.corpus)[:args.limit_positions]
    frame, train_mask, validation_mask = _frame(args.corpus, positions)
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    manifest_data = {}
    for path in sorted(args.manifests.glob("*.json")):
        manifest_data[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    input_names = [name for name in INPUTS if name in manifest_data]
    readout_names = [name for name in READOUTS if name in manifest_data]
    if not input_names or not readout_names:
        raise FileNotFoundError("input or readout manifests are incomplete")
    readout_union = np.asarray(sorted({index for name in readout_names for index in manifest_data[name]["indices"]}), dtype=np.int64)
    union_positions = {int(index): offset for offset, index in enumerate(readout_union)}
    loaded = load_fly_agent_from_checkpoint(args.checkpoint, annotations_path=args.annotations, neurotransmitters_path=args.neurotransmitters, connectome_weights_path=args.weights, sensory_indices_path=args.sensory_indices)
    base = loaded.agent
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("input population study requires the rate checkpoint")
    graph = base.engine.graph
    row_count = len(frame)
    rows_by_key = {(str(row.position_id), str(row.move_uci)): index for index, row in frame.iterrows()}
    projector_settings = {"fanout": base.projector.fanout, "seed": base.projector.seed, "amplitude": base.projector.amplitude}
    result_rows = []
    reachability = {}
    for input_name in input_names:
        input_indices = np.asarray(manifest_data[input_name]["indices"], dtype=np.int64)
        projector = HashedSensoryProjector(graph.n_neurons, input_indices, **projector_settings)
        reach = _bfs_reach_depths(graph.weights.tocsc(), input_indices, DEPTHS)
        for depth, value in reach.items():
            reachability[(input_name, depth)] = value
        cache_dir = args.output / "_input_activation_cache" / input_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        memmaps = {depth: np.memmap(cache_dir / f"depth_{depth}.float32", mode="w+", dtype=np.float32, shape=(row_count, len(readout_union))) for depth in DEPTHS}
        for start in range(0, len(positions), args.batch_positions):
            batch = positions[start:start + args.batch_positions]
            sensory_blocks = []
            batch_keys = []
            for position in batch:
                moves = sorted(position["board"].legal_moves, key=lambda move: move.uci())
                sensory_blocks.append(np.column_stack([projector.project(encode_board_move(position["board"], move)) for move in moves]))
                batch_keys.extend((str(position["position_id"]), move.uci()) for move in moves)
            trajectory = base.engine.run_batch_trajectory(np.concatenate(sensory_blocks, axis=1), depths=DEPTHS, clamp_sensory=True, observe=False)
            rows = np.asarray([rows_by_key[key] for key in batch_keys], dtype=np.int64)
            for depth in DEPTHS:
                memmaps[depth][rows, :] = trajectory.snapshots[depth][readout_union, :].T
            print(f"{input_name}: trajectory batch {min(start + args.batch_positions, len(positions))}/{len(positions)}", flush=True)
        for memmap in memmaps.values():
            memmap.flush()
        for readout_name in readout_names:
            payload = manifest_data[readout_name]
            columns = np.asarray([union_positions[int(index)] for index in payload["indices"]], dtype=np.int64)
            for depth in DEPTHS:
                features = np.asarray(memmaps[depth][:, columns])
                train_x, _ = _standardize(features[train_mask], features[train_mask])
                _, validation_x = _standardize(features[train_mask], features[validation_mask])
                weights = fit_probe(train_x, train_frame)
                metrics = _metrics(validation_x, validation_frame, weights)
                distance, rank_value = _geometry(validation_x, validation_frame)
                result_rows.append({
                    "input_population_id": input_name, "input_anatomical_family": manifest_data[input_name]["anatomical_family"],
                    "input_population_size": manifest_data[input_name]["actual_count"], "readout_population_id": readout_name,
                    "readout_anatomical_family": payload["anatomical_family"], "readout_population_size": payload["actual_count"],
                    "depth": depth, "input_reachable_fraction": reachability[(input_name, depth)],
                    "within_position_candidate_distance": distance, "effective_rank": rank_value,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                })
        for path in cache_dir.glob("depth_*.float32"):
            path.unlink()
        cache_dir.rmdir()
    pd.DataFrame(result_rows).to_csv(args.output / "input_population_results.csv", index=False)
    metadata = {"status": "complete", "positions": len(positions), "candidate_rows": row_count, "inputs": input_names, "readouts": readout_names, "depths": list(DEPTHS), "projector": projector_settings, "outputs": ["input_population_results.csv"]}
    (args.output / "input_population_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
