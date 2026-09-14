"""Create a deterministic development-only all-legal chess corpus."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from malecns_rd.chess_teacher import StockfishTeacher


CORPUS_COLUMNS = (
    "position_id", "source_game_id", "phase", "ply", "fen", "side_to_move",
    "move_uci", "teacher_cp", "teacher_target", "is_best", "legal_move_count",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_values(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[column] for row in csv.DictReader(handle)}


def iter_positions(*, source_groups: int, offsets: tuple[int, ...], seed: int):
    import chess

    for source_index in range(source_groups):
        source_id = f"dev_source_{source_index:03d}"
        for offset in offsets:
            # Each source/offset is generated from the initial position so the
            # four observations per source remain independent and the target
            # grid cannot lose positions because a longer continuation ended.
            board = None
            for retry in range(100):
                candidate = chess.Board()
                rng = random.Random(seed + source_index * 1_000_003 + offset * 10_007 + retry)
                while candidate.ply() < offset:
                    if candidate.is_game_over(claim_draw=True):
                        break
                    legal = sorted(candidate.legal_moves, key=lambda move: move.uci())
                    if not legal:
                        break
                    quiet = [move for move in legal if not candidate.is_capture(move)]
                    choices = quiet if quiet else legal
                    candidate.push(choices[rng.randrange(len(choices))])
                if not candidate.is_game_over(claim_draw=True) and candidate.legal_moves.count() >= 2:
                    board = candidate
                    break
            if board is None:
                raise RuntimeError(f"could not generate a legal nonterminal position: {source_id} offset {offset}")
            phase = "early" if offset <= 8 else "middle" if offset <= 16 else "late"
            yield {
                "position_id": f"{source_id}_offset_{offset:02d}",
                "source_game_id": source_id,
                "phase": phase,
                "ply": board.ply(),
                "fen": board.fen(),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, type=Path)
    parser.add_argument("--teacher-dataset", required=True, type=Path)
    parser.add_argument("--old-evaluation-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-groups", type=int, default=64)
    parser.add_argument("--offsets", default="4,12,20,28")
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--target-scale-cp", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.source_groups < 1:
        parser.error("--source-groups must be positive")
    offsets = tuple(sorted({int(value) for value in args.offsets.split(",") if value.strip()}))
    if not offsets or offsets[0] < 0:
        parser.error("--offsets must contain non-negative integers")
    fitting_fens = existing_values(args.teacher_dataset, "fen")
    consumed_fens = existing_values(args.old_evaluation_corpus, "fen")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_values(output, "position_id") if args.resume else set()
    corpus_fens = existing_values(output, "fen") if args.resume else set()
    mode = "a" if args.resume and output.exists() else "w"
    with StockfishTeacher(
        str(args.stockfish), nodes=args.nodes, threads=args.threads,
        hash_mb=args.hash_mb, target_scale_cp=args.target_scale_cp,
    ) as metadata_teacher:
        chess = metadata_teacher._chess
        teacher_metadata = metadata_teacher.metadata()
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        if mode == "w":
            writer.writeheader()
        for item in iter_positions(source_groups=args.source_groups, offsets=offsets, seed=args.seed):
            if item["position_id"] in completed:
                continue
            if item["fen"] in fitting_fens or item["fen"] in consumed_fens:
                raise RuntimeError(f"development position overlaps consumed data: {item['fen']}")
            if item["fen"] in corpus_fens:
                raise RuntimeError(f"duplicate development FEN: {item['fen']}")
            board = chess.Board(item["fen"])
            candidates = []
            for attempt in range(3):
                with StockfishTeacher(
                    str(args.stockfish), nodes=args.nodes, threads=args.threads,
                    hash_mb=args.hash_mb, target_scale_cp=args.target_scale_cp,
                ) as position_teacher:
                    candidates = position_teacher.evaluate_position(board, position_id=item["position_id"])
                if candidates:
                    break
                time.sleep(0.1 * (attempt + 1))
            if not candidates:
                raise RuntimeError(f"Stockfish returned no candidates for {item['position_id']}")
            for candidate in candidates:
                row = asdict(candidate)
                row.update({key: item[key] for key in ("source_game_id", "phase", "ply")})
                writer.writerow({key: row.get(key, "") for key in CORPUS_COLUMNS})
            handle.flush()
            completed.add(item["position_id"])
            corpus_fens.add(item["fen"])
    with output.open(newline="", encoding="utf-8") as stream:
        candidate_rows = sum(1 for _ in csv.DictReader(stream))
    expected_positions = args.source_groups * len(offsets)
    if len(completed) != expected_positions:
        raise RuntimeError(
            f"development corpus has {len(completed)} positions; expected {expected_positions}"
        )
    metadata = {
        "status": "development_only",
        "positions": len(completed),
        "candidate_rows": candidate_rows,
        "source_groups": args.source_groups,
        "expected_positions": expected_positions,
        "offsets_after_source_start": list(offsets),
        "source_seed": args.seed,
        "source_policy": "independent seeded random legal continuations from the standard initial position",
        "phases": ["early", "middle", "late"],
        "teacher_dataset": str(args.teacher_dataset),
        "teacher_dataset_sha256": sha256_file(args.teacher_dataset),
        "old_evaluation_corpus": str(args.old_evaluation_corpus),
        "old_evaluation_corpus_sha256": sha256_file(args.old_evaluation_corpus),
        "stockfish": teacher_metadata,
        "columns": list(CORPUS_COLUMNS),
        "corpus_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"positions": len(completed), "candidate_rows": candidate_rows, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
