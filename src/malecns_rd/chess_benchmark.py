from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import subprocess
from pathlib import Path
from typing import Iterable

from .elo import EloEstimate, GameObservation, estimate_elo


# Alfil's published UCI_Elo ladder. We intentionally require an exact level
# instead of silently rounding a requested rating to the nearest supported one.
ALFIL_NOMINAL_ELOS: tuple[int, ...] = tuple(range(0, 3001, 200))


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
class AlfilConfig:
    executable: str
    elo: int
    move_time_s: float = 0.05


@dataclass(frozen=True)
class ChessGameRecord:
    game_index: int
    opponent_engine: str
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
    stockfish_floor: int


def _option_bounds(option) -> tuple[int | None, int | None]:
    return getattr(option, "min", None), getattr(option, "max", None)


def _option_values(option) -> tuple[int, ...]:
    raw = getattr(option, "var", None) or ()
    values: list[int] = []
    for value in raw:
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(values)


def inspect_uci_elo(executable: str) -> tuple[int | None, int | None, tuple[int, ...]]:
    """Return advertised UCI_Elo bounds/choices for a local UCI engine."""
    chess = _require_chess()
    engine = chess.engine.SimpleEngine.popen_uci(executable)
    try:
        options = engine.options
        if "UCI_LimitStrength" not in options or "UCI_Elo" not in options:
            raise RuntimeError(
                f"engine {executable!r} does not expose UCI_LimitStrength/UCI_Elo"
            )
        option = options["UCI_Elo"]
        lo, hi = _option_bounds(option)
        return lo, hi, _option_values(option)
    finally:
        engine.quit()


def detect_stockfish_floor(executable: str) -> int:
    lo, _, values = inspect_uci_elo(executable)
    if lo is not None:
        return int(lo)
    if values:
        return min(values)
    raise RuntimeError("could not determine Stockfish UCI_Elo minimum")


def select_opponent_engine(
    elo: int,
    *,
    stockfish_floor: int,
    alfil_elos: Iterable[int] = ALFIL_NOMINAL_ELOS,
) -> str:
    """Route ratings below Stockfish's floor to Alfil.

    Alfil's nominal ladder is discrete. Unsupported sub-Stockfish values fail
    explicitly so the benchmark never pretends that an uncalibrated rounded
    value is the requested Elo.
    """
    elo = int(elo)
    if elo >= int(stockfish_floor):
        return "stockfish"
    supported = tuple(int(x) for x in alfil_elos)
    if elo not in supported:
        raise ValueError(
            f"Elo {elo} is below Stockfish floor {stockfish_floor} but is not an "
            f"Alfil nominal level. Supported low levels: {supported}"
        )
    return "alfil"


class _UciEloOpponent:
    engine_name = "uci"

    def __init__(self, executable: str, elo: int, move_time_s: float) -> None:
        chess = _require_chess()
        self._elo = int(elo)
        self._engine = chess.engine.SimpleEngine.popen_uci(executable)
        options = self._engine.options
        if "UCI_LimitStrength" not in options or "UCI_Elo" not in options:
            self._engine.quit()
            raise RuntimeError(
                f"{self.engine_name} does not expose UCI_LimitStrength/UCI_Elo"
            )
        option = options["UCI_Elo"]
        lo, hi = _option_bounds(option)
        values = _option_values(option)
        if values and self._elo not in values:
            self._engine.quit()
            raise ValueError(
                f"requested Elo {self._elo} is not advertised by {self.engine_name}; "
                f"choices={values}"
            )
        if lo is not None and self._elo < lo:
            self._engine.quit()
            raise ValueError(
                f"requested Elo {self._elo} is below {self.engine_name} minimum {lo}"
            )
        if hi is not None and self._elo > hi:
            self._engine.quit()
            raise ValueError(
                f"requested Elo {self._elo} is above {self.engine_name} maximum {hi}"
            )
        self._limit = chess.engine.Limit(time=float(move_time_s))

    @property
    def elo(self) -> int:
        return self._elo

    @property
    def name(self) -> str:
        return self.engine_name

    def choose_move(self, board):
        return self._engine.play(board, self._limit).move

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class StockfishOpponent(_UciEloOpponent):
    """Stockfish opponent using its runtime-advertised UCI_Elo range."""

    engine_name = "stockfish"

    def __init__(self, config: StockfishConfig) -> None:
        super().__init__(config.executable, config.elo, config.move_time_s)
        options = self._engine.options
        settings: dict[str, object] = {
            "UCI_LimitStrength": True,
            "UCI_Elo": int(config.elo),
        }
        if "Threads" in options:
            settings["Threads"] = int(config.threads)
        if "Hash" in options:
            settings["Hash"] = int(config.hash_mb)
        self._engine.configure(settings)


class AlfilOpponent:
    """Alfil opponent with tolerance for its non-standard ``ponder none`` output."""

    engine_name = "alfil"

    def __init__(self, config: AlfilConfig) -> None:
        chess = _require_chess()
        if int(config.elo) not in ALFIL_NOMINAL_ELOS:
            raise ValueError(
                f"Alfil nominal Elo must be one of {ALFIL_NOMINAL_ELOS}; got {config.elo}"
            )
        lo, hi, values = inspect_uci_elo(config.executable)
        if values and int(config.elo) not in values:
            raise ValueError(
                f"requested Elo {config.elo} is not advertised by Alfil; "
                f"choices={values}"
            )
        if lo is not None and int(config.elo) < lo:
            raise ValueError(
                f"requested Elo {config.elo} is below Alfil minimum {lo}"
            )
        if hi is not None and int(config.elo) > hi:
            raise ValueError(
                f"requested Elo {config.elo} is above Alfil maximum {hi}"
            )

        self._chess = chess
        self._elo = int(config.elo)
        self._limit_ms = max(1, round(float(config.move_time_s) * 1000))
        self._process = subprocess.Popen(
            [config.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        try:
            self._send("uci")
            self._wait_for("uciok")
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {self._elo}")
            self._send("isready")
            self._wait_for("readyok")
        except Exception:
            self.close()
            raise

    @property
    def elo(self) -> int:
        return self._elo

    @property
    def name(self) -> str:
        return self.engine_name

    def _send(self, command: str) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Alfil process stdin is unavailable")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _wait_for(self, expected: str) -> None:
        if self._process.stdout is None:
            raise RuntimeError("Alfil process stdout is unavailable")
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Alfil exited before returning {expected}; "
                    f"exit_code={self._process.poll()}"
                )
            if line.strip() == expected:
                return

    def choose_move(self, board):
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {self._limit_ms}")
        if self._process.stdout is None:
            raise RuntimeError("Alfil process stdout is unavailable")
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Alfil exited while choosing a move; "
                    f"exit_code={self._process.poll()}"
                )
            parts = line.strip().split()
            if parts and parts[0].lower() == "bestmove":
                if len(parts) < 2 or parts[1].lower() in {"0000", "none"}:
                    raise RuntimeError(f"Alfil returned no legal move: {line.strip()}")
                return self._chess.Move.from_uci(parts[1])

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._send("quit")
            self._process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()

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
    opponent,
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
        move = fly_agent.choose_move(board) if fly_turn else opponent.choose_move(board)
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
        opponent_engine=opponent.name,
        opponent_elo=opponent.elo,
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
    alfil_executable: str | None = None,
    stockfish_floor: int | None = None,
    move_time_s: float = 0.05,
    max_plies: int = 600,
) -> TournamentResult:
    """Run a hybrid Elo ladder: Alfil below Stockfish, Stockfish above it."""
    if games_per_elo < 2:
        raise ValueError("games_per_elo must be >= 2 for color balancing")
    if stockfish_floor is None:
        stockfish_floor = detect_stockfish_floor(stockfish_executable)

    ratings = [int(x) for x in opponent_elos]
    routed = [
        select_opponent_engine(x, stockfish_floor=stockfish_floor)
        for x in ratings
    ]
    if "alfil" in routed and not alfil_executable:
        raise ValueError(
            "alfil_executable is required when opponent_elos include ratings "
            f"below Stockfish floor {stockfish_floor}"
        )

    records: list[ChessGameRecord] = []
    game_index = 0
    for opponent_elo, engine_name in zip(ratings, routed):
        if engine_name == "alfil":
            config = AlfilConfig(str(alfil_executable), opponent_elo, move_time_s)
            opponent_cm = AlfilOpponent(config)
        else:
            config = StockfishConfig(stockfish_executable, opponent_elo, move_time_s)
            opponent_cm = StockfishOpponent(config)

        with opponent_cm as opponent:
            for local_index in range(games_per_elo):
                fly_is_white = local_index % 2 == 0
                records.append(play_one_game(
                    fly_agent,
                    opponent,
                    fly_is_white=fly_is_white,
                    game_index=game_index,
                    max_plies=max_plies,
                ))
                game_index += 1

    observations = [GameObservation(r.opponent_elo, r.fly_score) for r in records]
    return TournamentResult(
        tuple(records),
        estimate_elo(observations),
        stockfish_floor=int(stockfish_floor),
    )


def save_tournament(result: TournamentResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    games_path = output / "games.csv"
    with games_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(result.games[0]).keys()) if result.games else [],
        )
        if result.games:
            writer.writeheader()
            for game in result.games:
                writer.writerow(asdict(game))
    summary = {
        "elo": asdict(result.elo),
        "stockfish_floor": result.stockfish_floor,
    }
    (output / "elo.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
