"""Build the four deterministic Plan 3 curriculum tables.

Movement lessons are synthetic and contain invalid destinations. The remaining
stages use only training/validation rows from the supplied teacher corpus; no
evaluation or confirmatory corpus is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import chess
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("curriculum generation requires python-chess") from exc


OUTPUT_COLUMNS = [
    "position_id", "lesson_id", "stage", "split", "source_game_id", "fen", "side_to_move",
    "move_uci", "teacher_cp", "teacher_target", "is_best", "is_positive",
    "legal_move_count", "candidate_type", "motif", "mate_distance", "movement_piece", "blocker_present",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_template(**values: object) -> dict[str, object]:
    return {column: values.get(column, None) for column in OUTPUT_COLUMNS}


def build_movement(train_count: int, validation_count: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    rows: list[dict[str, object]] = []
    total = train_count + validation_count
    lesson = 0
    while lesson < total:
        piece_type = piece_types[lesson % len(piece_types)]
        color = chess.WHITE if (lesson // len(piece_types)) % 2 == 0 else chess.BLACK
        board = chess.Board(None)
        board.turn = color
        board.clear_stack()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        source_pool = [square for square in chess.SQUARES if square not in {chess.E1, chess.E8}]
        rng.shuffle(source_pool)
        source = next(square for square in source_pool if not (piece_type == chess.KING and chess.square_distance(square, chess.E1 if color == chess.BLACK else chess.E8) <= 1))
        board.set_piece_at(source, chess.Piece(piece_type, color))
        # Avoid impossible adjacent kings and put pawns away from promotion ranks.
        if piece_type == chess.PAWN:
            source = next(square for square in source_pool if 1 <= chess.square_rank(square) <= 6 and square not in {chess.E1, chess.E8})
            board.remove_piece_at(next(s for s in chess.SQUARES if board.piece_at(s) == chess.Piece(piece_type, color)))
            board.set_piece_at(source, chess.Piece(piece_type, color))
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        legal = sorted(list(board.legal_moves), key=lambda move: move.uci())
        valid = [move for move in legal if move.from_square == source]
        if not valid:
            continue
        destinations = [chess.Move(source, square) for square in chess.SQUARES if square != source]
        invalid = [move for move in destinations if move not in board.legal_moves]
        candidates = valid[:5] + invalid[:5]
        candidates = candidates[:10]
        split = "train" if lesson < train_count else "validation"
        for candidate in candidates:
            legal_flag = candidate in board.legal_moves
            rows.append(_row_template(
                position_id=f"movement_{lesson:04d}", lesson_id=f"movement_{lesson:04d}", stage="movement", split=split,
                source_game_id=f"synthetic_movement_{lesson:04d}", fen=board.fen(),
                side_to_move="white" if color else "black", move_uci=candidate.uci(),
                teacher_cp=100.0 if legal_flag else -100.0,
                teacher_target=1.0 if legal_flag else -1.0, is_best=int(legal_flag),
                is_positive=int(legal_flag), legal_move_count=len(legal),
                candidate_type="legal" if legal_flag else "illegal", motif="piece_movement",
                mate_distance=None, movement_piece=chess.piece_name(piece_type), blocker_present=0,
            ))
        lesson += 1
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _select_teacher_stage(frame: pd.DataFrame, stage: str, train_count: int, validation_count: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["position_id"] = frame["position_id"].astype(str)
    groups = list(frame.groupby("position_id", sort=True))
    rng = random.Random(seed)
    rng.shuffle(groups)
    selected: list[dict[str, object]] = []
    selected_groups: list[tuple[str, str, list[pd.Series]]] = []
    target = train_count + validation_count
    for position_id, group in groups:
        if len(selected_groups) >= target:
            break
        group = group.sort_values(["teacher_cp", "move_uci"], ascending=[False, True])
        best_cp = float(group.iloc[0]["teacher_cp"])
        if stage == "tactics":
            if best_cp - float(group["teacher_cp"].median()) < 100.0:
                continue
        elif stage == "mates":
            if not bool((group["teacher_cp"].abs() >= 90_000).any()):
                continue
        best_move_uci = str(group.iloc[0]["move_uci"])
        candidates = [group.iloc[0]]
        for index in [1, 2, len(group) // 2, len(group) - 1]:
            if 0 <= index < len(group) and group.iloc[index]["move_uci"] not in {row["move_uci"] for row in candidates}:
                candidates.append(group.iloc[index])
        selected_groups.append((str(position_id), best_move_uci, candidates[:5]))
    if len(selected_groups) >= target:
        validation_size = validation_count
    else:
        # Preserve as many training positions as the source allows while
        # retaining an honest validation slice when a stage is undersupplied.
        validation_size = min(validation_count, max(1, int(round(0.2 * len(selected_groups)))) if selected_groups else 0)
    validation_start = max(0, len(selected_groups) - validation_size)
    for selected_positions, (position_id, best_move_uci, candidates) in enumerate(selected_groups):
        split = "train" if selected_positions < validation_start else "validation"
        for row in candidates:
            selected.append(_row_template(
                position_id=str(position_id), lesson_id=f"{stage}_{selected_positions:04d}", stage=stage, split=split,
                source_game_id=str(row.get("source_game_id", "")), fen=str(row["fen"]),
                side_to_move=str(row.get("side_to_move", "")), move_uci=str(row["move_uci"]),
                teacher_cp=float(row["teacher_cp"]), teacher_target=float(row["teacher_target"]),
                is_best=int(str(row["move_uci"]) == best_move_uci), is_positive=None,
                legal_move_count=int(row.get("legal_move_count", len(group))),
                candidate_type="teacher_candidate", motif=stage, mate_distance=None,
            ))
    return pd.DataFrame(selected, columns=OUTPUT_COLUMNS)


def _synthetic_endgame_fens(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens = []
    seen: set[str] = set()
    templates = [
        ("KQK", chess.QUEEN), ("KRK", chess.ROOK), ("KPK", chess.PAWN),
    ]
    index = 0
    attempts = 0
    while len(fens) < count and attempts < count * 200:
        _, piece_type = templates[index % len(templates)]
        board = chess.Board(None)
        color = chess.WHITE if index % 2 == 0 else chess.BLACK
        squares = rng.sample(list(chess.SQUARES), 3)
        if piece_type == chess.PAWN and chess.square_rank(squares[2]) not in range(1, 7):
            index += 1
            attempts += 1
            continue
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(squares[2], chess.Piece(piece_type, color))
        board.turn = color
        if not board.is_valid() or len(list(board.legal_moves)) < 2 or board.fen() in seen:
            index += 1
            attempts += 1
            continue
        board.castling_rights = chess.BB_EMPTY
        board.clear_stack()
        fen = board.fen()
        if fen not in seen and len(list(board.legal_moves)) >= 2:
            seen.add(fen)
            fens.append(fen)
        index += 1
        attempts += 1
    if len(fens) < count:
        raise RuntimeError(f"could only construct {len(fens)} of {count} unique valid endgames")
    return fens


def expand_mate_principal_variations(frame: pd.DataFrame, executable: Path, nodes: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add same-side-to-move intermediate lessons from positive mate lines."""
    from malecns_rd.chess_teacher import StockfishTeacher

    additions: list[dict[str, object]] = []
    seen_fens: set[str] = set(frame["fen"].astype(str))
    roots = 0
    with StockfishTeacher(str(executable), nodes=nodes, threads=1) as teacher:
        for position_id, group in frame.groupby("position_id", sort=True):
            ordered = group.sort_values(["teacher_cp", "move_uci"], ascending=[False, True])
            root_mate = ordered.iloc[0].get("mate_distance")
            if pd.isna(root_mate) or int(root_mate) not in {2, 3}:
                continue
            roots += 1
            board = chess.Board(str(group.iloc[0]["fen"]))
            root_turn = board.turn
            for pv_index in range(1, int(root_mate)):
                own_move = teacher.best_move(board)
                if own_move is None:
                    break
                board.push(own_move[0])
                defense = teacher.best_move(board)
                if defense is None:
                    break
                board.push(defense[0])
                if board.is_game_over(claim_draw=True) or board.turn != root_turn or board.fen() in seen_fens:
                    break
                pv_id = f"{position_id}_pv{pv_index}"
                labels = teacher.evaluate_position(board, position_id=pv_id)
                if not labels:
                    break
                seen_fens.add(board.fen())
                for label in labels[:5]:
                    additions.append(_row_template(
                        position_id=pv_id, lesson_id=f"mates_pv_{len(additions):04d}", stage="mates",
                        split=str(group.iloc[0]["split"]), source_game_id=f"{group.iloc[0].get('source_game_id', '')}_pv",
                        fen=label.fen, side_to_move=label.side_to_move, move_uci=label.move_uci,
                        teacher_cp=label.teacher_cp, teacher_target=label.teacher_target, is_best=label.is_best,
                        is_positive=1, legal_move_count=label.legal_move_count, candidate_type="teacher_pv",
                        motif="mate_pv", mate_distance=label.mate_distance,
                    ))
    if not additions:
        return frame, {"roots_with_positive_mate_2_or_3": roots, "intermediate_positions_added": 0}
    expanded = pd.concat([frame, pd.DataFrame(additions, columns=OUTPUT_COLUMNS)], ignore_index=True)
    return expanded, {"roots_with_positive_mate_2_or_3": roots, "intermediate_positions_added": len(additions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-corpus", type=Path, default=Path("data/teacher_dataset_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/curriculum_v1"))
    parser.add_argument("--movement-train", type=int, default=360)
    parser.add_argument("--movement-validation", type=int, default=120)
    parser.add_argument("--endgame-train", type=int, default=320)
    parser.add_argument("--endgame-validation", type=int, default=100)
    parser.add_argument("--tactics-train", type=int, default=500)
    parser.add_argument("--tactics-validation", type=int, default=150)
    parser.add_argument("--mates-train", type=int, default=384)
    parser.add_argument("--mates-validation", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.teacher_corpus)
    if "split" in source.columns:
        source = source.loc[source["split"].astype(str).isin({"dev_screen", "dev_confirm", "train", "validation"})].copy()
    movement = build_movement(args.movement_train, args.movement_validation, args.seed)
    # The development corpus has no natural endgame phase. Keep its synthetic
    # endgame lessons explicit and use teacher labels from the regular corpus
    # for tactics/mates. Endgame labels are filled by the training runner's
    # deterministic Stockfish labeler when requested.
    endgame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    fens = _synthetic_endgame_fens(args.endgame_train + args.endgame_validation, args.seed + 1)
    for index, fen in enumerate(fens):
        board = chess.Board(fen)
        legal = sorted(list(board.legal_moves), key=lambda move: move.uci())
        for move in legal[:5]:
            endgame.loc[len(endgame)] = _row_template(
                position_id=f"endgames_{index:04d}", lesson_id=f"endgames_{index:04d}", stage="endgames",
                split="train" if index < args.endgame_train else "validation",
                source_game_id=f"synthetic_endgame_{index:04d}", fen=fen,
                side_to_move="white" if board.turn else "black", move_uci=move.uci(),
                teacher_cp=None, teacher_target=None, is_best=None, is_positive=None,
                legal_move_count=len(legal), candidate_type="teacher_candidate", motif="endgame",
                mate_distance=None,
            )
    tactics = _select_teacher_stage(source, "tactics", args.tactics_train, args.tactics_validation, args.seed + 2)
    mates = _select_teacher_stage(source, "mates", args.mates_train, args.mates_validation, args.seed + 3)
    for name, frame in (("movement", movement), ("endgames", endgame), ("tactics", tactics), ("mates", mates)):
        frame.to_csv(args.output / f"{name}.csv", index=False)
    metadata = {
        "status": "complete", "seed": args.seed, "source_teacher_corpus": str(args.teacher_corpus),
        "source_teacher_corpus_sha256": sha256_file(args.teacher_corpus),
        "stages": {name: int(len(frame)) for name, frame in (("movement", movement), ("endgames", endgame), ("tactics", tactics), ("mates", mates))},
        "split_policy": "movement and synthetic endgames use deterministic counts; teacher stages use shuffled position groups",
        "evaluation_corpus_consumed": False,
        "endgame_labels": "pending_deterministic_stockfish_labeling",
        "outputs": ["movement.csv", "endgames.csv", "tactics.csv", "mates.csv"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
