"""Fill synthetic curriculum labels with deterministic Stockfish analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from malecns_rd.chess_teacher import StockfishTeacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-dir", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--refresh", action="store_true", help="relabel all endgame and mate candidates")
    args = parser.parse_args()

    stages = ["movement", "endgames", "tactics", "mates"]
    frames = {stage: pd.read_csv(args.curriculum_dir / f"{stage}.csv") for stage in stages}
    missing_fens = set()
    for frame in frames.values():
        missing_fens.update(frame.loc[frame["teacher_cp"].isna(), "fen"].astype(str))
        if "mate_distance" in frame.columns:
            missing_fens.update(frame.loc[(frame["stage"] == "mates") & frame["mate_distance"].isna(), "fen"].astype(str))
    if args.refresh:
        for stage in ("endgames", "mates"):
            missing_fens.update(frames[stage]["fen"].astype(str))
    labels: dict[tuple[str, str], tuple[float, float, int, int | None]] = {}
    with StockfishTeacher(str(args.stockfish), nodes=args.nodes, threads=args.threads) as teacher:
        teacher_metadata = teacher.metadata()
        import chess
        candidate_moves_by_fen: dict[str, set[object]] = {}
        for frame in frames.values():
            for _, row in frame.loc[frame["fen"].astype(str).isin(missing_fens)].iterrows():
                candidate_moves_by_fen.setdefault(str(row["fen"]), set()).add(chess.Move.from_uci(str(row["move_uci"])))
        for fen in sorted(missing_fens):
            board = chess.Board(fen)
            for candidate in teacher.evaluate_position(board, position_id=fen, candidate_moves=candidate_moves_by_fen.get(fen)):
                labels[(fen, candidate.move_uci)] = (
                    candidate.teacher_cp, candidate.teacher_target,
                    candidate.is_best, candidate.mate_distance,
                )

    for stage, frame in frames.items():
        for index, row in frame.iterrows():
            key = (str(row["fen"]), str(row["move_uci"]))
            if pd.isna(row["teacher_cp"]) and key in labels:
                cp, target, is_best, mate_distance = labels[key]
                frame.at[index, "teacher_cp"] = cp
                frame.at[index, "teacher_target"] = target
                frame.at[index, "is_best"] = is_best
                frame.at[index, "mate_distance"] = mate_distance
            elif str(stage) == "mates" and key in labels:
                frame.at[index, "mate_distance"] = labels[key][3]
        frame.to_csv(args.curriculum_dir / f"{stage}.csv", index=False)

    metadata_path = args.curriculum_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "endgame_labels": "stockfish",
        "stockfish": teacher_metadata,
        "labelled_fen_count": len(missing_fens),
        "labelled_files_sha256": {
            stage: hashlib.sha256((args.curriculum_dir / f"{stage}.csv").read_bytes()).hexdigest()
            for stage in stages
        },
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "labelled_fen_count": len(missing_fens), "stockfish": teacher_metadata}, indent=2))


if __name__ == "__main__":
    main()
