"""Generate candidate-move labels from full-strength Stockfish."""
from __future__ import annotations

import argparse
from pathlib import Path

from malecns_rd.chess_teacher import (
    StockfishTeacher,
    iter_generated_fens,
    iter_fens_from_pgn,
    iter_fens_from_text,
    write_teacher_metadata,
    write_teacher_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish executable")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fen-file", type=Path, help="Text file with one FEN per line")
    source.add_argument("--pgn", type=Path, help="PGN file sampled into training positions")
    source.add_argument("--generated-positions", type=int, help="Deterministically generate this many positions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--target-scale-cp", type=float, default=400.0)
    parser.add_argument("--skip-opening-plies", type=int, default=8)
    parser.add_argument("--sample-every-plies", type=int, default=4)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--position-seed", type=int, default=101)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.fen_file:
        fens = iter_fens_from_text(args.fen_file)
    elif args.pgn:
        fens = iter_fens_from_pgn(
            args.pgn,
            skip_opening_plies=args.skip_opening_plies,
            sample_every_plies=args.sample_every_plies,
            max_positions=args.max_positions,
        )
    else:
        fens = iter_generated_fens(args.generated_positions, seed=args.position_seed)

    with StockfishTeacher(
        args.stockfish,
        nodes=args.nodes,
        threads=args.threads,
        hash_mb=args.hash_mb,
        target_scale_cp=args.target_scale_cp,
    ) as teacher:
        positions, rows = write_teacher_dataset(
            args.output,
            fens,
            teacher,
            max_positions=args.max_positions,
            resume=args.resume,
        )
        write_teacher_metadata(
            args.output,
            positions=positions,
            rows=rows,
            metadata={
                "stockfish": str(args.stockfish),
                "nodes": args.nodes,
                "threads": args.threads,
                "hash_mb": args.hash_mb,
                "target_scale_cp": args.target_scale_cp,
                "position_seed": args.position_seed,
                "teacher_engine": teacher.metadata(),
            },
        )
    print(f"wrote {rows} candidate rows from {positions} positions to {args.output}")


if __name__ == "__main__":
    main()
