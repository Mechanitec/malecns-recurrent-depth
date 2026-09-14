"""Freeze an independent, Stockfish-labelled evaluation corpus."""
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
    "position_id",
    "source_game_id",
    "phase",
    "ply",
    "fen",
    "side_to_move",
    "move_uci",
    "teacher_cp",
    "teacher_target",
    "is_best",
    "legal_move_count",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_openings(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("opening_fens", payload) if isinstance(payload, dict) else payload
    return [
        {"opening_id": str(item["opening_pair"]), "fen": str(item["fen"])}
        for item in entries
    ]


def existing_values(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[column] for row in csv.DictReader(handle)}


def iter_positions(openings, *, offsets: tuple[int, ...], seed: int):
    import chess

    for opening_index, opening in enumerate(openings):
        board = chess.Board(opening["fen"])
        opening_ply = board.ply()
        rng = random.Random(seed + opening_index)
        for offset in offsets:
            while board.ply() - opening_ply < offset:
                if board.is_game_over(claim_draw=True):
                    break
                legal = sorted(board.legal_moves, key=lambda move: move.uci())
                if not legal:
                    break
                quiet = [move for move in legal if not board.is_capture(move)]
                choices = quiet if quiet else legal
                board.push(choices[rng.randrange(len(choices))])
            if board.is_game_over(claim_draw=True) or len(list(board.legal_moves)) < 2:
                continue
            phase = "early" if offset <= 8 else "middle" if offset <= 16 else "late"
            yield {
                "position_id": f"{opening['opening_id']}_offset_{offset:02d}",
                "source_game_id": opening["opening_id"],
                "phase": phase,
                "ply": board.ply(),
                "fen": board.fen(),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, type=Path)
    parser.add_argument("--opening-metadata", required=True, type=Path)
    parser.add_argument("--teacher-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--target-scale-cp", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--offsets", default="0,4,8,12,16,20,24")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    offsets = tuple(sorted({int(value) for value in args.offsets.split(",") if value.strip()}))
    if not offsets or offsets[0] < 0:
        parser.error("--offsets must contain non-negative integers")
    openings = load_openings(args.opening_metadata)
    fitting_fens = existing_values(args.teacher_dataset, "fen")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and output.exists():
        output.unlink()
    completed = existing_values(output, "position_id") if args.resume else set()
    corpus_fens = existing_values(output, "fen") if args.resume else set()
    mode = "a" if args.resume and output.exists() else "w"
    # Keep source selection independent from Stockfish. This makes the corpus
    # reproducible even when the engine's search state varies between games.
    with StockfishTeacher(
        str(args.stockfish),
        nodes=args.nodes,
        threads=args.threads,
        hash_mb=args.hash_mb,
        target_scale_cp=args.target_scale_cp,
    ) as metadata_teacher:
        chess = metadata_teacher._chess
        teacher_metadata = metadata_teacher.metadata()
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        if mode == "w":
            writer.writeheader()
        for item in iter_positions(openings, offsets=offsets, seed=args.seed):
            if item["fen"] in fitting_fens:
                raise RuntimeError(f"evaluation position overlaps fitting dataset: {item['fen']}")
            if item["position_id"] in completed:
                continue
            if item["fen"] in corpus_fens:
                raise RuntimeError(f"duplicate evaluation FEN: {item['fen']}")
            board = chess.Board(item["fen"])
            candidates = []
            for attempt in range(3):
                with StockfishTeacher(
                    str(args.stockfish),
                    nodes=args.nodes,
                    threads=args.threads,
                    hash_mb=args.hash_mb,
                    target_scale_cp=args.target_scale_cp,
                ) as position_teacher:
                    candidates = position_teacher.evaluate_position(
                        board, position_id=item["position_id"]
                    )
                if candidates:
                    break
                time.sleep(0.1 * (attempt + 1))
            if not candidates:
                raise RuntimeError(
                    f"Stockfish returned no candidates for {item['position_id']} after 3 attempts"
                )
            for candidate in candidates:
                row = asdict(candidate)
                row.update({key: item[key] for key in ("source_game_id", "phase", "ply")})
                writer.writerow({key: row.get(key, "") for key in CORPUS_COLUMNS})
            handle.flush()
            completed.add(item["position_id"])
            corpus_fens.add(item["fen"])

        with output.open(newline="", encoding="utf-8") as stream:
            candidate_rows = sum(1 for _ in csv.DictReader(stream))
        metadata = {
            "status": "evaluation/test_only",
            "positions": len(completed),
            "candidate_rows": candidate_rows,
            "opening_count": len(openings),
            "offsets_after_opening": list(offsets),
            "source_seed": args.seed,
            "source_policy": "seeded RNG over sorted non-captures, falling back to captures",
            "phases": ["early", "middle", "late"],
            "source": "deterministic legal continuations from calibrated opening FENs",
            "opening_metadata": str(args.opening_metadata),
            "opening_metadata_sha256": sha256_file(args.opening_metadata),
            "teacher_dataset": str(args.teacher_dataset),
            "teacher_dataset_sha256": sha256_file(args.teacher_dataset),
            "stockfish": teacher_metadata,
            "columns": list(CORPUS_COLUMNS),
            "corpus_sha256": sha256_file(output),
        }
        output.with_suffix(output.suffix + ".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"positions": len(completed), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
