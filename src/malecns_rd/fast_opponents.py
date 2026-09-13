from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chess_benchmark import (
    ChessGameRecord,
    StockfishConfig,
    StockfishOpponent,
    TournamentResult,
    _require_chess,
    _write_live,
    detect_stockfish_floor,
    play_one_game,
)
from .elo import GameObservation, estimate_elo
from .live_state import outcome_counts
from .position_analysis import (
    AnalysisConfig,
    StockfishPositionEvaluator,
    play_one_game_with_analysis,
)


# Gaia 4's published low-strength ladder. These are estimates rather than an
# absolute physical Elo scale; we preserve the exact published anchor labels.
GAIA_RATING_TO_LEVEL: dict[int, int] = {
    580: 1,
    700: 2,
    820: 3,
    940: 4,
    1060: 5,
    1180: 6,
    1300: 7,
}
FAST_LOW_ELOS: tuple[int, ...] = (0, *GAIA_RATING_TO_LEVEL.keys())


@dataclass(frozen=True)
class MinicConfig:
    executable: str
    level: int = 0
    nominal_elo: int = 0
    search_depth: int = 64
    threads: int = 1
    hash_mb: int = 16

    def __post_init__(self) -> None:
        if not 0 <= self.level <= 100:
            raise ValueError("Minic level must be in [0, 100]")
        if self.search_depth < 1:
            raise ValueError("search_depth must be >= 1")


@dataclass(frozen=True)
class GaiaConfig:
    executable: str
    rating: int
    search_depth: int = 64
    threads: int = 1
    hash_mb: int = 16

    def __post_init__(self) -> None:
        if self.rating not in GAIA_RATING_TO_LEVEL:
            raise ValueError(
                f"Gaia low-rating anchor must be one of {tuple(GAIA_RATING_TO_LEVEL)}; "
                f"got {self.rating}"
            )
        if self.search_depth < 1:
            raise ValueError("search_depth must be >= 1")


class _DepthBoundedUciOpponent:
    engine_name = "uci"

    def __init__(
        self,
        executable: str,
        *,
        nominal_elo: int,
        search_depth: int,
        settings: dict[str, object],
    ) -> None:
        chess = _require_chess()
        self._chess = chess
        self._elo = int(nominal_elo)
        self._engine = chess.engine.SimpleEngine.popen_uci(executable)
        options = self._engine.options
        safe_settings = {k: v for k, v in settings.items() if k in options}
        if safe_settings:
            self._engine.configure(safe_settings)
        # Low-strength Gaia/Minic levels have their own internal search cap.
        # A high outer depth avoids wall-clock dependence while letting those
        # internal caps decide when to stop.
        self._limit = chess.engine.Limit(depth=int(search_depth))

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


class MinicOpponent(_DepthBoundedUciOpponent):
    """Very-low-strength Minic opponent.

    Level 0 is Minic's built-in random mover and returns without a normal
    search, making it suitable as the near-zero benchmark anchor.
    """

    engine_name = "minic"

    def __init__(self, config: MinicConfig) -> None:
        super().__init__(
            config.executable,
            nominal_elo=config.nominal_elo,
            search_depth=config.search_depth,
            settings={
                "Level": int(config.level),
                "Threads": int(config.threads),
                "Hash": int(config.hash_mb),
                "nodesBasedLevel": True,
            },
        )
        if "Level" not in self._engine.options:
            self.close()
            raise RuntimeError("Minic executable does not expose UCI option 'Level'")


class GaiaOpponent(_DepthBoundedUciOpponent):
    """Gaia 4 low-strength opponent using its published level/rating anchors."""

    engine_name = "gaia"

    def __init__(self, config: GaiaConfig) -> None:
        level = GAIA_RATING_TO_LEVEL[int(config.rating)]
        super().__init__(
            config.executable,
            nominal_elo=config.rating,
            search_depth=config.search_depth,
            settings={
                "Skill Level": int(level),
                "OwnBook": False,
                "Threads": int(config.threads),
                "Hash": int(config.hash_mb),
            },
        )
        if "Skill Level" not in self._engine.options:
            self.close()
            raise RuntimeError("Gaia executable does not expose UCI option 'Skill Level'")


def select_fast_opponent(elo: int, *, stockfish_floor: int) -> str:
    """Select the high-throughput opponent for one nominal rating.

    0 is the Minic random-mover anchor. Gaia supplies published anchors from
    580 through 1300. Stockfish handles its runtime-advertised range above that.
    Ratings in the uncalibrated gap are rejected rather than invented.
    """
    elo = int(elo)
    if elo >= int(stockfish_floor):
        return "stockfish"
    if elo == 0:
        return "minic"
    if elo in GAIA_RATING_TO_LEVEL:
        return "gaia"
    raise ValueError(
        f"No calibrated fast opponent for Elo {elo}. Supported below Stockfish "
        f"floor {stockfish_floor}: {FAST_LOW_ELOS}. Intermediate Minic levels "
        "must be calibrated before they are assigned Elo labels."
    )


def _make_fast_opponent(
    engine_name: str,
    opponent_elo: int,
    *,
    stockfish_executable: str,
    minic_executable: str | None,
    gaia_executable: str | None,
    move_time_s: float,
):
    if engine_name == "minic":
        if not minic_executable:
            raise ValueError("minic_executable is required for the Elo 0 anchor")
        return MinicOpponent(MinicConfig(str(minic_executable), level=0, nominal_elo=0))
    if engine_name == "gaia":
        if not gaia_executable:
            raise ValueError("gaia_executable is required for Gaia low-rating anchors")
        return GaiaOpponent(GaiaConfig(str(gaia_executable), rating=int(opponent_elo)))
    return StockfishOpponent(
        StockfishConfig(stockfish_executable, int(opponent_elo), move_time_s)
    )


def _validate_fast_ladder(
    ratings: Iterable[int],
    *,
    stockfish_floor: int,
    minic_executable: str | None,
    gaia_executable: str | None,
) -> list[tuple[int, str]]:
    routed = [(int(x), select_fast_opponent(int(x), stockfish_floor=stockfish_floor)) for x in ratings]
    if any(name == "minic" for _, name in routed) and not minic_executable:
        raise ValueError("minic_executable is required because the ladder includes Elo 0")
    if any(name == "gaia" for _, name in routed) and not gaia_executable:
        raise ValueError("gaia_executable is required because the ladder includes Gaia ratings")
    return routed


def _rolling_update(records: list[ChessGameRecord]):
    observations = [GameObservation(r.opponent_elo, r.fly_score) for r in records]
    rolling = estimate_elo(observations)
    wins, draws, losses = outcome_counts(r.fly_score for r in records)
    return rolling, wins, draws, losses


def run_fast_elo_tournament(
    fly_agent,
    *,
    stockfish_executable: str,
    opponent_elos: Iterable[int],
    games_per_elo: int,
    minic_executable: str | None = None,
    gaia_executable: str | None = None,
    stockfish_floor: int | None = None,
    move_time_s: float = 0.05,
    max_plies: int = 600,
    live_state_path: str | Path | None = None,
    run_id: str | None = None,
) -> TournamentResult:
    if games_per_elo < 2:
        raise ValueError("games_per_elo must be >= 2 for color balancing")
    if stockfish_floor is None:
        stockfish_floor = detect_stockfish_floor(stockfish_executable)
    ratings = [int(x) for x in opponent_elos]
    if not ratings:
        raise ValueError("opponent_elos must contain at least one rating")
    routed = _validate_fast_ladder(
        ratings,
        stockfish_floor=int(stockfish_floor),
        minic_executable=minic_executable,
        gaia_executable=gaia_executable,
    )

    total_games = len(ratings) * games_per_elo
    resolved_run_id = run_id or (
        Path(live_state_path).parent.name if live_state_path is not None else None
    )
    _write_live(
        live_state_path,
        status="running",
        message="Fast tournament starting",
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
        for opponent_elo, engine_name in routed:
            opponent_cm = _make_fast_opponent(
                engine_name,
                opponent_elo,
                stockfish_executable=stockfish_executable,
                minic_executable=minic_executable,
                gaia_executable=gaia_executable,
                move_time_s=move_time_s,
            )
            with opponent_cm as opponent:
                for local_index in range(games_per_elo):
                    record = play_one_game(
                        fly_agent,
                        opponent,
                        fly_is_white=local_index % 2 == 0,
                        game_index=game_index,
                        max_plies=max_plies,
                        live_state_path=live_state_path,
                        games_total=total_games,
                        run_id=resolved_run_id,
                    )
                    records.append(record)
                    game_index += 1
                    rolling, wins, draws, losses = _rolling_update(records)
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

    final_elo, wins, draws, losses = _rolling_update(records)
    _write_live(
        live_state_path,
        status="completed",
        message="Fast tournament completed",
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
    return TournamentResult(tuple(records), final_elo, int(stockfish_floor))


def run_fast_elo_tournament_with_analysis(
    fly_agent,
    *,
    stockfish_executable: str,
    opponent_elos: Iterable[int],
    games_per_elo: int,
    analysis_config: AnalysisConfig,
    minic_executable: str | None = None,
    gaia_executable: str | None = None,
    stockfish_floor: int | None = None,
    move_time_s: float = 0.05,
    max_plies: int = 600,
    live_state_path: str | Path | None = None,
    evaluation_state_path: str | Path | None = None,
    evaluation_history_path: str | Path | None = None,
    run_id: str | None = None,
) -> TournamentResult:
    if games_per_elo < 2:
        raise ValueError("games_per_elo must be >= 2 for color balancing")
    if stockfish_floor is None:
        stockfish_floor = detect_stockfish_floor(stockfish_executable)
    ratings = [int(x) for x in opponent_elos]
    if not ratings:
        raise ValueError("opponent_elos must contain at least one rating")
    routed = _validate_fast_ladder(
        ratings,
        stockfish_floor=int(stockfish_floor),
        minic_executable=minic_executable,
        gaia_executable=gaia_executable,
    )

    for artifact in (evaluation_state_path, evaluation_history_path):
        if artifact is not None:
            Path(artifact).unlink(missing_ok=True)

    total_games = len(ratings) * games_per_elo
    resolved_run_id = run_id or (
        Path(live_state_path).parent.name if live_state_path is not None else None
    )
    _write_live(
        live_state_path,
        status="running",
        message="Fast tournament starting with independent Stockfish analysis",
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
        with StockfishPositionEvaluator(analysis_config) as evaluator:
            for opponent_elo, engine_name in routed:
                opponent_cm = _make_fast_opponent(
                    engine_name,
                    opponent_elo,
                    stockfish_executable=stockfish_executable,
                    minic_executable=minic_executable,
                    gaia_executable=gaia_executable,
                    move_time_s=move_time_s,
                )
                with opponent_cm as opponent:
                    for local_index in range(games_per_elo):
                        record = play_one_game_with_analysis(
                            fly_agent,
                            opponent,
                            evaluator,
                            fly_is_white=local_index % 2 == 0,
                            game_index=game_index,
                            max_plies=max_plies,
                            live_state_path=live_state_path,
                            evaluation_state_path=evaluation_state_path,
                            evaluation_history_path=evaluation_history_path,
                            games_total=total_games,
                            run_id=resolved_run_id,
                        )
                        records.append(record)
                        game_index += 1
                        rolling, wins, draws, losses = _rolling_update(records)
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

    final_elo, wins, draws, losses = _rolling_update(records)
    _write_live(
        live_state_path,
        status="completed",
        message="Fast tournament completed",
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
    return TournamentResult(tuple(records), final_elo, int(stockfish_floor))
