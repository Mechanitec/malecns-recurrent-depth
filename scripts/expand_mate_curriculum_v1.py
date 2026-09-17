"""Add Stockfish principal-variation lessons to a labelled mate curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_chess_curriculum_v1 import expand_mate_principal_variations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-dir", type=Path, default=Path("data/curriculum_v1"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=20_000)
    args = parser.parse_args()
    path = args.curriculum_dir / "mates.csv"
    frame = pd.read_csv(path)
    expanded, metadata = expand_mate_principal_variations(frame, args.stockfish, args.nodes)
    expanded.to_csv(path, index=False)
    metadata_path = args.curriculum_dir / "metadata.json"
    current = json.loads(metadata_path.read_text(encoding="utf-8"))
    current["stages"]["mates"] = int(len(expanded))
    current["mate_pv_expansion"] = metadata
    current["mate_pv_stockfish"] = str(args.stockfish)
    current["mate_pv_nodes"] = args.nodes
    metadata_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
