from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Iterator


TEACHER_COLUMNS = (
    "position_id", "source_game_id", "split", "fen", "side_to_move",
    "move_uci", "teacher_cp", "teacher_target", "is_best", "legal_move_count",
)


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
    source_game_id: str = ""
    split: str = "train"
    side_to_move: str = ""
    legal_move_count: int = 0
    mate_distance: int | None = None


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
        self.threads = int(threads)
        self.hash_mb = int(hash_mb)
        self.mate_score_cp = int(mate_score_cp)
        self.target_scale_cp = float(target_scale_cp)
        self.executable = str(executable)
        self._engine = chess.engine.SimpleEngine.popen_uci(executable)
        settings: dict[str, object] = {}
        if "Threads" in self._engine.options:
            settings["Threads"] = int(threads)
        if "Hash" in self._engine.options:
            settings["Hash"] = int(hash_mb)
        if settings:
            self._engine.configure(settings)

    def metadata(self) -> dict[str, object]:
        digest = hashlib.sha256(Path(self.executable).read_bytes()).hexdigest()
        return {
            "engine": dict(self._engine.id),
            "executable": self.executable,
            "executable_sha256": digest,
            "nodes": self.nodes,
            "threads": self.threads,
            "hash_mb": self.hash_mb,
            "mate_score_cp": self.mate_score_cp,
            "target_scale_cp": self.target_scale_cp,
        }

    def close(self) -> None:
        self._engine.quit()

    def best_move(self, board) -> tuple[object, float, int | None] | None:
        """Return the engine's best move, score, and mate distance for a board."""
        if board.is_game_over(claim_draw=True):
            return None
        info = self._engine.analyse(board, self._chess.engine.Limit(nodes=self.nodes))
        pv = info.get("pv") or []
        score = info.get("score")
        if not pv or score is None:
            return None
        pov_score = score.pov(board.turn)
        cp = pov_score.score(mate_score=self.mate_score_cp)
        if cp is None:
            return None
        return pv[0], float(cp), pov_score.mate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _score_info(self, board, info) -> tuple[str, float, int | None] | None:
        pv = info.get("pv") or []
        if not pv:
            return None
        move = pv[0]
        score = info.get("score")
        if score is None:
            return None
        pov_score = score.pov(board.turn)
        cp = pov_score.score(mate_score=self.mate_score_cp)
        if cp is None:
            return None
        return move.uci(), float(cp), pov_score.mate()

    def evaluate_position(self, board, *, position_id: str, candidate_moves=None) -> list[TeacherCandidate]:
        all_legal = sorted(list(board.legal_moves), key=lambda m: m.uci())
        legal = all_legal if candidate_moves is None else sorted(list(candidate_moves), key=lambda m: m.uci())
        if any(move not in board.legal_moves for move in legal):
            raise ValueError("candidate_moves must be legal in the supplied board")
        if len(legal) < 2:
            return []
        limit = self._chess.engine.Limit(nodes=self.nodes)
        infos = self._engine.analyse(board, limit, multipv=len(legal), root_moves=legal)
        if isinstance(infos, dict):
            infos = [infos]
        scores: dict[str, tuple[float, int | None]] = {}
        for info in infos:
            parsed = self._score_info(board, info)
            if parsed is not None:
                scores[parsed[0]] = (parsed[1], parsed[2])

        # Some engines/builds can return fewer MultiPV entries than requested.
        # Fill any gaps with a root-move-constrained evaluation.
        for move in legal:
            uci = move.uci()
            if uci in scores:
                continue
            info = self._engine.analyse(board, limit, root_moves=[move])
            parsed = self._score_info(board, info)
            if parsed is not None:
                scores[uci] = (parsed[1], parsed[2])

        ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
        if len(ranked) < 2:
            return []
        best_uci = ranked[0][0]
        fen = board.fen()
        return [
            TeacherCandidate(
                position_id=str(position_id),
                fen=fen,
                move_uci=uci,
                teacher_cp=cp[0],
                teacher_target=centipawn_to_target(cp[0], scale_cp=self.target_scale_cp),
                is_best=int(uci == best_uci),
                side_to_move="white" if board.turn else "black",
                legal_move_count=len(all_legal),
                mate_distance=cp[1],
            )
            for uci, cp in ranked
        ]


def iter_fens_from_text(path: str | Path) -> Iterator[str]:
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            yield value


def iter_generated_fens(count: int, *, seed: int = 101, plies: int = 12) -> Iterator[str]:
    """Generate reproducible, diverse positions for a pilot teacher corpus."""
    if count < 1 or plies < 1:
        raise ValueError("count and plies must be >= 1")
    chess = _require_chess()
    rng = random.Random(seed)
    seen: set[str] = set()
    while len(seen) < count:
        board = chess.Board()
        for _ in range(plies + rng.randrange(0, 8)):
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            if not legal:
                break
            board.push(rng.choice(legal))
            if board.is_game_over(claim_draw=True):
                break
        if board.is_game_over(claim_draw=True) or board.fen() in seen:
            continue
        seen.add(board.fen())
        yield board.fen()


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
    resume: bool = False,
) -> tuple[int, int]:
    """Evaluate positions and write candidate labels. Returns positions, rows."""
    chess = _require_chess()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions = rows = 0
    completed_fens: set[str] = set()
    if resume and output.exists():
        with output.open(newline="", encoding="utf-8") as existing:
            completed_fens = {str(row["fen"]) for row in csv.DictReader(existing)}
    mode = "a" if resume and output.exists() else "w"
    with output.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TEACHER_COLUMNS))
        if mode == "w":
            writer.writeheader()
        for source_index, raw_fen in enumerate(fens):
            if max_positions is not None and source_index >= max_positions:
                break
            board = chess.Board(raw_fen)
            position_id = str(source_index)
            if str(raw_fen) in completed_fens:
                positions = source_index + 1
                continue
            source_game_id = f"generated_{source_index // 50:04d}"
            split_bucket = int.from_bytes(hashlib.blake2b(source_game_id.encode(), digest_size=2).digest(), "little") % 10
            split = "train" if split_bucket < 8 else "validation" if split_bucket == 8 else "test"
            candidates = teacher.evaluate_position(board, position_id=position_id)
            if not candidates:
                positions = source_index + 1
                continue
            for candidate in candidates:
                row = asdict(candidate)
                row.update({"source_game_id": source_game_id, "split": split})
                writer.writerow(row)
                rows += 1
            positions = source_index + 1
            completed_fens.add(str(raw_fen))
    return positions, rows


def write_teacher_metadata(path: str | Path, *, positions: int, rows: int, metadata: dict[str, object]) -> None:
    output = Path(path)
    total_rows = 0
    position_ids: set[str] = set()
    with output.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            position_ids.add(str(row["position_id"]))
    payload = {
        **metadata,
        "positions": len(position_ids) if position_ids else int(positions),
        "candidate_rows": total_rows if total_rows else int(rows),
        "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "columns": list(TEACHER_COLUMNS),
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
