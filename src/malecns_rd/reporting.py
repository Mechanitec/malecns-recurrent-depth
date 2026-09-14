"""Derive report metadata from persisted chess game artifacts."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def derive_tournament_artifact_metadata(
    rows: Iterable[Mapping[str, object]], *, reference_depth: int
) -> dict[str, object]:
    """Derive game-policy metadata from raw game rows and embedded PGNs.

    The game CSV does not store the CLI configuration that created it.  The
    persisted PGN and telemetry do store enough evidence to determine the
    observed ply cap, termination mix, and whether the Fly scored every legal
    move.  Candidate evaluations are inferred from recurrent passes divided by
    the fixed checkpoint depth.
    """
    rows = list(rows)
    if reference_depth < 1:
        raise ValueError("reference_depth must be positive")
    if not rows:
        return {
            "games": 0,
            "max_plies": None,
            "max_plies_source": "no_rows",
            "termination_counts": {},
            "full_legal_candidates": None,
            "bounded_candidates": None,
            "candidate_evaluations_per_fly_move_max": None,
            "recurrent_passes_per_candidate_depth": 0,
        }

    try:
        import chess
        import chess.pgn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("report metadata derivation requires python-chess") from exc

    termination_counts = Counter(str(row.get("termination", "")) for row in rows)
    observed_plies = [int(row["plies"]) for row in rows]
    capped_plies = [
        int(row["plies"])
        for row in rows
        if str(row.get("termination", "")) == "MAX_PLIES_ADJUDICATION"
    ]
    candidate_evaluations = 0
    legal_candidates = 0
    candidate_counts_per_move: list[float] = []
    recurrent_passes = 0

    for row in rows:
        pgn_text = str(row.get("pgn", ""))
        game = chess.pgn.read_game(__import__("io").StringIO(pgn_text))
        if game is None:
            raise ValueError("game row contains an unreadable PGN")
        board = game.board()
        fly_is_white = str(row.get("fly_color", "white")) == "white"
        fly_turn_legal_counts: list[int] = []
        for move in game.mainline_moves():
            if board.turn == (chess.WHITE if fly_is_white else chess.BLACK):
                fly_turn_legal_counts.append(board.legal_moves.count())
            board.push(move)
        passes = int(row.get("total_recurrent_passes", 0) or 0)
        fly_moves = int(row.get("fly_move_count", 0) or 0)
        if fly_moves != len(fly_turn_legal_counts):
            raise ValueError("raw game telemetry does not match PGN Fly move count")
        inferred_candidates = passes / reference_depth
        candidate_evaluations += int(round(inferred_candidates))
        legal_candidates += sum(fly_turn_legal_counts)
        recurrent_passes += passes
        if fly_moves:
            candidate_counts_per_move.append(inferred_candidates / fly_moves)

    bounded = candidate_evaluations < legal_candidates
    return {
        "games": len(rows),
        "max_plies": max(capped_plies) if capped_plies else max(observed_plies),
        "max_plies_source": "MAX_PLIES_ADJUDICATION observed" if capped_plies else "observed maximum",
        "termination_counts": dict(sorted(termination_counts.items())),
        "full_legal_candidates": not bounded,
        "bounded_candidates": bounded,
        "candidate_evaluations": candidate_evaluations,
        "legal_candidates_from_pgn": legal_candidates,
        "candidate_evaluations_per_fly_move_max": max(candidate_counts_per_move, default=None),
        "recurrent_passes": recurrent_passes,
        "recurrent_passes_per_candidate_depth": recurrent_passes // reference_depth,
        "reference_depth": reference_depth,
    }
