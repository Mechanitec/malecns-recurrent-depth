"""Merge contiguous Phase 3 activation shards into one deterministic cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mechanistic_factorial import DEPTHS, load_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    parser.add_argument("--shards", type=Path, default=Path("results/population_study/shards"))
    parser.add_argument("--output", type=Path, default=Path("results/population_study/_activation_cache"))
    parser.add_argument("--limit-positions", type=int, default=256)
    args = parser.parse_args()
    positions = load_positions(args.corpus)[:args.limit_positions]
    frame = pd.read_csv(args.corpus)
    frame["position_id"] = frame["position_id"].astype(str)
    frame = frame[frame["position_id"].isin({str(p["position_id"]) for p in positions})].sort_values(["position_id", "move_uci"]).reset_index(drop=True)
    manifest_paths = sorted(args.manifests.glob("readout_*.json"))
    union = np.asarray(sorted({index for path in manifest_paths for index in json.loads(path.read_text(encoding="utf-8"))["indices"]}), dtype=np.int64)
    shard_metadata = []
    for path in sorted(args.shards.glob("shard_*/activation_shard_metadata.json")):
        shard_metadata.append(json.loads(path.read_text(encoding="utf-8")))
    shard_metadata.sort(key=lambda item: item["position_start"])
    if not shard_metadata:
        raise FileNotFoundError(f"no shard metadata in {args.shards}")
    expected_start = 0
    expected_rows = 0
    for metadata in shard_metadata:
        if metadata["position_start"] != expected_start:
            raise ValueError("shards are not contiguous")
        expected_start = metadata["position_end"]
        expected_rows += int(metadata["candidate_rows"])
    if expected_start != len(positions) or expected_rows != len(frame):
        raise ValueError(f"shards cover {expected_start} positions and {expected_rows} rows; expected {len(positions)} and {len(frame)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for depth in DEPTHS:
        target = np.memmap(args.output / f"union_depth_{depth}.float32", mode="w+", dtype=np.float32, shape=(len(frame), len(union)))
        row_cursor = 0
        for metadata in shard_metadata:
            source = args.shards / f"shard_{metadata['position_start']:03d}" / "_activation_cache" / f"union_depth_{depth}.float32"
            local = np.memmap(source, mode="r", dtype=np.float32, shape=(int(metadata["candidate_rows"]), len(union)))
            for start in range(0, len(local), 1024):
                stop = min(start + 1024, len(local))
                target[row_cursor + start:row_cursor + stop] = local[start:stop]
            row_cursor += len(local)
            del local
        target.flush()
        del target
    metadata = {"status": "complete", "positions": len(positions), "candidate_rows": len(frame), "union_readout_neurons": len(union), "depths": list(DEPTHS), "shards": shard_metadata}
    (args.output.parent / "activation_cache_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
