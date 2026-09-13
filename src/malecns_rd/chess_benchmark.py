from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from .elo import EloEstimate, GameObservation, estimate_elo
from .live_state import (
    CandidateScore,
    LiveBenchmarkState,
    outcome_counts,
    read_live_state,
    write_live_state,
)


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
        # Do not configure Ponder explicitly: python-chess manages it internally.
        self._engine.configure(settings)


class AlfilOpponent(_UciEloOpponent):
    """Alfil opponent for the low-rating portion of the benchmark ladder."""

    engine_name = "alfil"

    def __init__(self, config: AlfilConfig) -> None:
        if int(config.elo) not in ALFIL_NOMINAL_ELOS:
            raise ValueError(
                f"Alfil nominal Elo must be one of {ALFIL_NOMINAL_ELOS}; got {config.elo}"
            )
        super().__init__(config.executable, config.elo, config.move_time_s)
        self._engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": int(config.elo),
        })


def _fly_score_from_result(result: str, fly_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if fly_is_white else 0.0
    if result == "0-1":
        return 0.0 if fly_is_white else 1.0
    raise ValueError(f"unsupported result {result!r}")


def _write_live(path: str | Path | None, **changes) -> None:
    if path is None:
        return
    current = read_live_state(path) or LiveBenchmarkState()
    # Force a fresh timestamp on every emitted snapshot.
    changes["updated_at_utc"] = None
    write_live_state(path, replace(current, **changes))


def _candidate_snapshot(fly_agent, limit: int = 12) -> tuple[int | None, tuple[CandidateScore, ...]]:
    decision = getattr(fly_agent, "last_decision", None)
    if not isinstance(decision, dict):
        return getattr(fly_agent, "depth", None), ()
    depth = decision.get("depth", getattr(fly_agent, "depth", None))
    try:
        depth_value = int(depth) if depth is not None else None
    except (TypeError, ValueError):
        depth_value = None

    candidates: list[CandidateScore] = []
    for item in decision.get("candidates", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            move = str(item["move"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(score):
            candidates.append(CandidateScore(move, score))
        if len(candidates) >= limit:
            break
    return depth_value, tuple(candidates)


def play_one_game(
    fly_agent,
    opponent,
    *,
    fly_is_white: bool,
    game_index: int = 0,
    start_fen: str | None = None,
    max_plies: int = 600,
    live_state_path: str | Path | None = None,
    games_total: int = 0,
    run_id: str | None = None,
) -> ChessGameRecord:
    chess = _require_chess()
    board = chess.Board(start_fen) if start_fen else chess.Board()
    fly_color = "white" if fly_is_white else "black"

    _write_live(
        live_state_path,
        status="running",
        message=f"Game {game_index + 1} in progress",
        run_id=run_id,
        game_index=game_index,
        games_total=games_total,
        opponent_engine=opponent.name,
        opponent_elo=opponent.elo,
        fly_color=fly_color,
        ply=board.ply(),
        fen=board.fen(),
        last_move=None,
        last_actor=None,
        candidate_scores=(),
        recurrent_depth=getattr(fly_agent, "depth", None),
    )

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fly_turn = board.turn == (chess.WHITE if fly_is_white else chess.BLACK)
        if fly_turn:
            move = fly_agent.choose_move(board)
            actor = "fly"
            recurrent_depth, candidates = _candidate_snapshot(fly_agent)
        else:
            move = opponent.choose_move(board)
            actor = opponent.name
            recurrent_depth = getattr(fly_agent, "depth", None)
            candidates = ()
        if move not in board.legal_moves:
            raise RuntimeError(f"agent returned illegal move: {move}")
        move_uci = move.uci()
        board.push(move)
        _write_live(
            live_state_path,
            status="running",
            message=f"Game {game_index + 1} in progress",
            run_id=run_id,
            game_index=game_index,
            games_total=games_total,
            opponent_engine=opponent.name,
            opponent_elo=opponent.elo,
            fly_color=fly_color,
            ply=board.ply(),
            fen=board.fen(),
            last_move=move_uci,
            last_actor=actor,
            recurrent_depth=recurrent_depth,
            candidate_scores=candidates,
        )

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
        fly_color=fly_color,
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
    live_state_path: str | Path | None = None,
    run_id: str | None = None,
) -> TournamentResult:
    """Run a hybrid Elo ladder and optionally publish live JSON snapshots."""
    if games_per_elo < 2:
        raise ValueError("games_per_elo must be >= 2 for color balancing")
    if stockfish_floor is None:
        stockfish_floor = detect_stockfish_floor(stockfish_executable)

    ratings = [int(x) for x in opponent_elos]
    if not ratings:
        raise ValueError("opponent_elos must contain at least one rating")
    routed = [
        select_opponent_engine(x, stockfish_floor=stockfish_floor)
        for x in ratings
    ]
    if "alfil" in routed and not alfil_executable:
        raise ValueError(
            "alfil_executable is required when opponent_elos include ratings "
            f"below Stockfish floor {stockfish_floor}"
        )

    total_games = len(ratings) * games_per_elo
    resolved_run_id = run_id
    if resolved_run_id is None and live_state_path is not None:
        resolved_run_id = Path(live_state_path).parent.name

    _write_live(
        live_state_path,
        status="running",
        message="Tournament starting",
        run_id=resolved_run_id,
        game_index=0,
        games_completed=0,
        games_total=total_games,
        rolling_elo=None,
        rolling_ci_low=None,
        rolling_ci_high=None,
        rolling_censored=None,
        wins=0,
        draws=0,
        losses=0,
        candidate_scores=(),
    )

    records: list[ChessGameRecord] = []
    game_index = 0
    try:
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
                    record = play_one_game(
                        fly_agent,
                        opponent,
                        fly_is_white=fly_is_white,
                        game_index=game_index,
                        max_plies=max_plies,
                        live_state_path=live_state_path,
                        games_total=total_games,
                        run_id=resolved_run_id,
                    )
                    records.append(record)
                    game_index += 1

                    observations = [
                        GameObservation(r.opponent_elo, r.fly_score) for r in records
                    ]
                    rolling = estimate_elo(observations)
                    wins, draws, losses = outcome_counts(r.fly_score for r in records)
                    _write_live(
                        live_state_path,
                        status="running",
                        message=f"{len(records)} / {total_games} games completed",
                        games_completed=len(records),
                        games_total=total_games,
                        rolling_elo=rolling.rating,
                        rolling_ci_low=rolling.ci95_low,
                        rolling_ci_high=rolling.ci95_high,
                        rolling_censored=rolling.censored,
                        wins=wins,
                        draws=draws,
                        losses=losses,
                    )
    except KeyboardInterrupt:
        _write_live(
            live_state_path,
            status="stopped",
            message="Tournament interrupted",
            games_completed=len(records),
        )
        raise
    except Exception as exc:
        _write_live(
            live_state_path,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
            games_completed=len(records),
        )
        raise

    observations = [GameObservation(r.opponent_elo, r.fly_score) for r in records]
    final_elo = estimate_elo(observations)
    wins, draws, losses = outcome_counts(r.fly_score for r in records)
    _write_live(
        live_state_path,
        status="completed",
        message="Tournament completed",
        games_completed=len(records),
        games_total=total_games,
        rolling_elo=final_elo.rating,
        rolling_ci_low=final_elo.ci95_low,
        rolling_ci_high=final_elo.ci95_high,
        rolling_censored=final_elo.censored,
        wins=wins,
        draws=draws,
        losses=losses,
    )
    return TournamentResult(
        tuple(records),
        final_elo,
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
