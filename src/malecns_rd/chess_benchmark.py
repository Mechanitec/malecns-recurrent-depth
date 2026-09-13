from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Iterable

from .elo import EloEstimate, GameObservation, estimate_elo


def _require_chess():
    try:
        import chess  # type: ignore
        import chess.engine  # type: ignore
        import chess.pgn  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Chess benchmark requires python-chess: pip install -e '.[chess]'"
        ) from exc
    return chess


@dataclass(frozen=True)
class StockfishConfig:
    executable: str
    elo: int
    move_time_s: float = 0.05
    threads: int = 1
    hash_mb: int = 64


@dataclass(frozen=True)
class ChessGameRecord:
    game_index: int
    opponent_elo: int
    fly_color: str
    result: str
    fly_score: float
    plies: int
    termination: str
    final_fen: str


@dataclass(frozen=True)
class TournamentResult:
    games: tuple[ChessGameRecord, ...]
    elo: EloEstimate


def _option_bounds(option) -> tuple[int | None, int | None]:
    return getattr(option, "min", None), getattr(option, "max", None)


class StockfishOpponent:
    """UCI Stockfish opponent configured through UCI_LimitStrength/UCI_Elo.

    The requested integer is a *nominal calibrated target*, not a guarantee that
    realized strength is exact under every time control/hardware setup. Runtime
    option bounds are inspected so the harness follows the installed Stockfish
    build rather than hard-coding one release's range.
    """

    def __init__(self, config: StockfishConfig) -> None:
        chess = _require_chess()
        self.config = config
        self._engine = chess.engine.SimpleEngine.popen_uci(config.executable)
        options = self._engine.options
        if "UCI_LimitStrength" not in options or "UCI_Elo" not in options:
            self._engine.quit()
            raise RuntimeError("Stockfish build does not expose UCI_LimitStrength/UCI_Elo")
        lo, hi = _option_bounds(options["UCI_Elo"])
        if lo is not None and config.elo < lo:
            self._engine.quit()
            raise ValueError(f"requested Elo {config.elo} is below engine minimum {lo}")
        if hi is not None and config.elo > hi:
            self._engine.quit()
            raise ValueError(f"requested Elo {config.elo} is above engine maximum {hi}")
        self._engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": int(config.elo),
            "Threads": int(config.threads),
            "Hash": int(config.hash_mb),
            "Ponder": False,
        })
        self._limit = chess.engine.Limit(time=float(config.move_time_s))

    @property
    def elo(self) -> int:
        return self.config.elo

    def choose_move(self, board):
        return self._engine.play(board, self._limit).move

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _fly_score_from_result(result: str, fly_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if fly_is_white else 0.0
    if result == "0-1":
        return 0.0 if fly_is_white else 1.0
    raise ValueError(f"unsupported result {result!r}")


def play_one_game(
    fly_agent,
    stockfish: StockfishOpponent,
    *,
    fly_is_white: bool,
    game_index: int = 0,
    start_fen: str | None = None,
    max_plies: int = 600,
) -> ChessGameRecord:
    chess = _require_chess()
    board = chess.Board(start_fen) if start_fen else chess.Board()

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fly_turn = board.turn == (chess.WHITE if fly_is_white else chess.BLACK)
        move = fly_agent.choose_move(board) if fly_turn else stockfish.choose_move(board)
        if move not in board.legal_moves:
            raise RuntimeError(f"agent returned illegal move: {move}")
        board.push(move)

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None
        result = outcome.result()
        termination = outcome.termination.name
    else:
        result = "1/2-1/2"
        termination = "MAX_PLIES_ADJUDICATION"

    return ChessGameRecord(
        game_index=game_index,
        opponent_elo=stockfish.elo,
        fly_color="white" if fly_is_white else "black",
        result=result,
        fly_score=_fly_score_from_result(result, fly_is_white),
        plies=board.ply(),
        termination=termination,
        final_fen=board.fen(),
    )


def run_elo_tournament(
    fly_agent,
    *,
    stockfish_executable: str,
    opponent_elos: Iterable[int],
    games_per_elo: int,
    move_time_s: float = 0.05,
    max_plies: int = 600,
) -> TournamentResult:
    if games_per_elo < 2:
        raise ValueError("games_per_elo must be >= 2 for color balancing")
    records: list[ChessGameRecord] = []
    game_index = 0
    for opponent_elo in opponent_elos:
        config = StockfishConfig(stockfish_executable, int(opponent_elo), move_time_s)
        with StockfishOpponent(config) as opponent:
            for local_index in range(games_per_elo):
                fly_is_white = (local_index % 2 == 0)
                records.append(play_one_game(
                    fly_agent,
                    opponent,
                    fly_is_white=fly_is_white,
                    game_index=game_index,
                    max_plies=max_plies,
                ))
                game_index += 1

    observations = [GameObservation(r.opponent_elo, r.fly_score) for r in records]
    return TournamentResult(tuple(records), estimate_elo(observations))


def save_tournament(result: TournamentResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    games_path = output / "games.csv"
    with games_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result.games[0]).keys()) if result.games else [])
        if result.games:
            writer.writeheader()
            for game in result.games:
                writer.writerow(asdict(game))
    (output / "elo.json").write_text(
        json.dumps(asdict(result.elo), indent=2), encoding="utf-8"
    )
