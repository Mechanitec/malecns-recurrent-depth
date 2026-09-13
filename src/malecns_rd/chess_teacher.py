from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import math
from pathlib import Path
from typing import Iterable, Iterator


TEACHER_COLUMNS = ("position_id", "fen", "move_uci", "teacher_cp", "teacher_target", "is_best")


def _require_chess():
    try:
        import chess  # type: ignore
        import chess.engine  # type: ignore
        import chess.pgn  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Chess teacher support requires: pip install -e '.[chess]'") from exc
    return chess


def centipawn_to_target(cp: float, *, scale_cp: float = 400.0) -> float:
    """Compress teacher centipawns to a bounded training target in [-1, 1]."""
    if scale_cp <= 0.0:
        raise ValueError("scale_cp must be > 0")
    return float(math.tanh(float(cp) / scale_cp))


@dataclass(frozen=True)
class TeacherCandidate:
    position_id: str
    fen: str
    move_uci: str
    teacher_cp: float
    teacher_target: float
    is_best: int


class StockfishTeacher:
    """Full-strength Stockfish candidate evaluator used only to label training data."""

    def __init__(
        self,
        executable: str,
        *,
        nodes: int = 20_000,
        threads: int = 1,
        hash_mb: int = 128,
        mate_score_cp: int = 100_000,
        target_scale_cp: float = 400.0,
    ) -> None:
        if nodes < 1:
            raise ValueError("nodes must be >= 1")
        chess = _require_chess()
        self._chess = chess
        self.nodes = int(nodes)
        self.mate_score_cp = int(mate_score_cp)
        self.target_scale_cp = float(target_scale_cp)
        self._engine = chess.engine.SimpleEngine.popen_uci(executable)
        settings: dict[str, object] = {}
        if "Threads" in self._engine.options:
            settings["Threads"] = int(threads)
        if "Hash" in self._engine.options:
            settings["Hash"] = int(hash_mb)
        if settings:
            self._engine.configure(settings)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _score_info(self, board, info) -> tuple[str, float] | None:
        pv = info.get("pv") or []
        if not pv:
            return None
        move = pv[0]
        score = info.get("score")
        if score is None:
            return None
        cp = score.pov(board.turn).score(mate_score=self.mate_score_cp)
        if cp is None:
            return None
        return move.uci(), float(cp)

    def evaluate_position(self, board, *, position_id: str) -> list[TeacherCandidate]:
        legal = sorted(list(board.legal_moves), key=lambda m: m.uci())
        if len(legal) < 2:
            return []
        limit = self._chess.engine.Limit(nodes=self.nodes)
        infos = self._engine.analyse(board, limit, multipv=len(legal), root_moves=legal)
        if isinstance(infos, dict):
            infos = [infos]
        scores: dict[str, float] = {}
        for info in infos:
            parsed = self._score_info(board, info)
            if parsed is not None:
                scores[parsed[0]] = parsed[1]

        # Some engines/builds can return fewer MultiPV entries than requested.
        # Fill any gaps with a root-move-constrained evaluation.
        for move in legal:
            uci = move.uci()
            if uci in scores:
                continue
            info = self._engine.analyse(board, limit, root_moves=[move])
            parsed = self._score_info(board, info)
            if parsed is not None:
                scores[uci] = parsed[1]

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < 2:
            return []
        best_uci = ranked[0][0]
        fen = board.fen()
        return [
            TeacherCandidate(
                position_id=str(position_id),
                fen=fen,
                move_uci=uci,
                teacher_cp=cp,
                teacher_target=centipawn_to_target(cp, scale_cp=self.target_scale_cp),
                is_best=int(uci == best_uci),
            )
            for uci, cp in ranked
        ]


def iter_fens_from_text(path: str | Path) -> Iterator[str]:
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            yield value


def iter_fens_from_pgn(
    path: str | Path,
    *,
    skip_opening_plies: int = 8,
    sample_every_plies: int = 4,
    max_positions: int | None = None,
) -> Iterator[str]:
    if skip_opening_plies < 0 or sample_every_plies < 1:
        raise ValueError("invalid PGN sampling parameters")
    chess = _require_chess()
    emitted = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if ply <= skip_opening_plies:
                    continue
                if (ply - skip_opening_plies) % sample_every_plies != 0:
                    continue
                if board.is_game_over(claim_draw=True):
                    continue
                yield board.fen()
                emitted += 1
                if max_positions is not None and emitted >= max_positions:
                    return


def write_teacher_dataset(
    path: str | Path,
    fens: Iterable[str],
    teacher: StockfishTeacher,
    *,
    max_positions: int | None = None,
) -> tuple[int, int]:
    """Evaluate positions and write candidate labels. Returns positions, rows."""
    chess = _require_chess()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions = rows = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TEACHER_COLUMNS))
        writer.writeheader()
        for raw_fen in fens:
            if max_positions is not None and positions >= max_positions:
                break
            board = chess.Board(raw_fen)
            candidates = teacher.evaluate_position(board, position_id=str(positions))
            if not candidates:
                continue
            for candidate in candidates:
                writer.writerow(asdict(candidate))
                rows += 1
            positions += 1
    return positions, rows
