from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Iterable

from .chess_benchmark import (
    AlfilConfig,
    AlfilOpponent,
    ChessGameRecord,
    StockfishConfig,
    StockfishOpponent,
    TournamentResult,
    _candidate_snapshot,
    _fly_score_from_result,
    _require_chess,
    _write_live,
    detect_stockfish_floor,
    select_opponent_engine,
)
from .elo import GameObservation, estimate_elo
from .live_state import outcome_counts


@dataclass(frozen=True)
class AnalysisConfig:
    """Independent full-strength Stockfish position-analysis settings.

    A fixed search depth is preferred over wall-clock time for reproducibility.
    The analysis engine is always a separate process from the playing opponent.
    """

    executable: str
    depth: int = 18
    threads: int = 1
    hash_mb: int = 128

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("analysis depth must be >= 1")
        if self.threads < 1:
            raise ValueError("analysis threads must be >= 1")
        if self.hash_mb < 1:
            raise ValueError("analysis hash_mb must be >= 1")


@dataclass(frozen=True)
class PositionEvaluation:
    game_index: int
    ply: int
    fen: str
    fly_color: str
    last_move: str | None
    last_actor: str | None
    white_cp: int | None
    white_mate: int | None
    fly_cp: int | None
    fly_mate: int | None
    analysis_depth: int
    analyzed_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_to_advantage_fraction(cp: int | float | None, mate: int | None = None) -> float:
    """Map an evaluation to a [0, 1] display bar.

    This is deliberately a visualization transform, NOT a win-probability model.
    0.5 is equal, values above 0.5 favor the evaluated side. Mate saturates.
    """
    if mate is not None:
        if mate > 0:
            return 1.0
        if mate < 0:
            return 0.0
        return 0.5
    if cp is None:
        return 0.5
    return float(0.5 + 0.5 * math.tanh(float(cp) / 400.0))


def human_eval(cp: int | None, mate: int | None, *, side: str = "Fly") -> str:
    if mate is not None:
        if mate > 0:
            return f"{side} has mate in {mate}"
        if mate < 0:
            return f"{side} is getting mated in {abs(mate)}"
        return "Forced mate"
    if cp is None:
        return "Evaluation unavailable"
    pawns = cp / 100.0
    if abs(pawns) < 0.20:
        return f"Approximately equal ({pawns:+.2f})"
    if pawns > 0:
        return f"{side} better ({pawns:+.2f})"
    return f"Opponent better ({pawns:+.2f} from {side.lower()} perspective)"


def write_position_evaluation(path: str | Path, evaluation: PositionEvaluation) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(asdict(evaluation), indent=2), encoding="utf-8")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def append_evaluation_history(path: str | Path, evaluation: PositionEvaluation) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(evaluation)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class StockfishPositionEvaluator:
    """Full-strength Stockfish observer that never chooses a game move."""

    def __init__(self, config: AnalysisConfig) -> None:
        chess = _require_chess()
        self.config = config
        self._chess = chess
        self._engine = chess.engine.SimpleEngine.popen_uci(config.executable)
        settings: dict[str, object] = {}
        options = self._engine.options
        if "UCI_LimitStrength" in options:
            settings["UCI_LimitStrength"] = False
        if "Threads" in options:
            settings["Threads"] = int(config.threads)
        if "Hash" in options:
            settings["Hash"] = int(config.hash_mb)
        if settings:
            self._engine.configure(settings)
        self._limit = chess.engine.Limit(depth=int(config.depth))

    def evaluate(
        self,
        board,
        *,
        game_index: int,
        fly_is_white: bool,
        last_move: str | None,
        last_actor: str | None,
    ) -> PositionEvaluation:
        info = self._engine.analyse(board, self._limit)
        score = info.get("score")
        if score is None:
            white_cp = None
            white_mate = None
        else:
            white_score = score.pov(self._chess.WHITE)
            white_mate = white_score.mate()
            raw_cp = white_score.score(mate_score=100000)
            white_cp = int(raw_cp) if raw_cp is not None else None

        fly_color = "white" if fly_is_white else "black"
        if fly_is_white:
            fly_cp = white_cp
            fly_mate = white_mate
        else:
            fly_cp = -white_cp if white_cp is not None else None
            fly_mate = -white_mate if white_mate is not None else None

        return PositionEvaluation(
            game_index=int(game_index),
            ply=int(board.ply()),
            fen=board.fen(),
            fly_color=fly_color,
            last_move=last_move,
            last_actor=last_actor,
            white_cp=white_cp,
            white_mate=white_mate,
            fly_cp=fly_cp,
            fly_mate=fly_mate,
            analysis_depth=int(self.config.depth),
            analyzed_at_utc=utc_now_iso(),
        )

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _evaluate_and_publish(
    evaluator: StockfishPositionEvaluator,
    board,
    *,
    game_index: int,
    fly_is_white: bool,
    last_move: str | None,
    last_actor: str | None,
    state_path: str | Path | None,
    history_path: str | Path | None,
) -> PositionEvaluation:
    evaluation = evaluator.evaluate(
        board,
        game_index=game_index,
        fly_is_white=fly_is_white,
        last_move=last_move,
        last_actor=last_actor,
    )
    if state_path is not None:
        write_position_evaluation(state_path, evaluation)
    if history_path is not None:
        append_evaluation_history(history_path, evaluation)
    return evaluation


def play_one_game_with_analysis(
    fly_agent,
    opponent,
    evaluator: StockfishPositionEvaluator,
    *,
    fly_is_white: bool,
    game_index: int = 0,
    start_fen: str | None = None,
    max_plies: int = 600,
    live_state_path: str | Path | None = None,
    evaluation_state_path: str | Path | None = None,
    evaluation_history_path: str | Path | None = None,
    games_total: int = 0,
    run_id: str | None = None,
    requested_elo: float | None = None,
) -> ChessGameRecord:
    chess = _require_chess()
    board = chess.Board(start_fen) if start_fen else chess.Board()
    fly_color = "white" if fly_is_white else "black"
    calibrated_elo = float(getattr(opponent, "calibrated_elo", opponent.elo))
    opponent_setting = str(getattr(opponent, "setting", ""))
    fly_move_latencies: list[float] = []
    recurrent_passes = 0
    score_margins: list[float] = []

    _write_live(
        live_state_path,
        status="running",
        message=f"Game {game_index + 1} in progress",
        run_id=run_id,
        game_index=game_index,
        games_total=games_total,
        opponent_engine=opponent.name,
        opponent_elo=calibrated_elo,
        fly_color=fly_color,
        ply=board.ply(),
        fen=board.fen(),
        last_move=None,
        last_actor=None,
        candidate_scores=(),
        recurrent_depth=getattr(fly_agent, "depth", None),
    )
    _evaluate_and_publish(
        evaluator,
        board,
        game_index=game_index,
        fly_is_white=fly_is_white,
        last_move=None,
        last_actor=None,
        state_path=evaluation_state_path,
        history_path=evaluation_history_path,
    )

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fly_turn = board.turn == (chess.WHITE if fly_is_white else chess.BLACK)
        if fly_turn:
            started = time.perf_counter()
            move = fly_agent.choose_move(board)
            fly_move_latencies.append(time.perf_counter() - started)
            actor = "fly"
            recurrent_depth, candidates = _candidate_snapshot(fly_agent)
            decision = getattr(fly_agent, "last_decision", {})
            if isinstance(decision, dict):
                recurrent_passes += int(decision.get("recurrent_passes", 0) or 0)
                margin = decision.get("score_margin")
                if margin is not None:
                    score_margins.append(float(margin))
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
            opponent_elo=calibrated_elo,
            fly_color=fly_color,
            ply=board.ply(),
            fen=board.fen(),
            last_move=move_uci,
            last_actor=actor,
            recurrent_depth=recurrent_depth,
            candidate_scores=candidates,
        )
        _evaluate_and_publish(
            evaluator,
            board,
            game_index=game_index,
            fly_is_white=fly_is_white,
            last_move=move_uci,
            last_actor=actor,
            state_path=evaluation_state_path,
            history_path=evaluation_history_path,
        )

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None
        result = outcome.result()
        termination = outcome.termination.name
    else:
        result = "1/2-1/2"
        termination = "MAX_PLIES_ADJUDICATION"

    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = result
    game.headers["White"] = "MaleCNS-RD" if fly_is_white else opponent.name
    game.headers["Black"] = opponent.name if fly_is_white else "MaleCNS-RD"
    game.headers["OpponentElo"] = str(calibrated_elo)

    return ChessGameRecord(
        game_index=game_index,
        opponent_engine=opponent.name,
        opponent_elo=calibrated_elo,
        fly_color=fly_color,
        result=result,
        fly_score=_fly_score_from_result(result, fly_is_white),
        plies=board.ply(),
        termination=termination,
        final_fen=board.fen(),
        opponent_setting=opponent_setting,
        opponent_requested_elo=requested_elo,
        opponent_calibrated_elo=calibrated_elo,
        fly_move_count=len(fly_move_latencies),
        median_fly_move_latency_s=(statistics.median(fly_move_latencies) if fly_move_latencies else None),
        total_recurrent_passes=recurrent_passes,
        mean_candidate_score_margin=(statistics.fmean(score_margins) if score_margins else None),
        pgn=str(game),
    )


def run_elo_tournament_with_analysis(
    fly_agent,
    *,
    stockfish_executable: str,
    opponent_elos: Iterable[int],
    games_per_elo: int,
    analysis_config: AnalysisConfig,
    alfil_executable: str | None = None,
    stockfish_floor: int | None = None,
    move_time_s: float = 0.05,
    max_plies: int = 600,
    live_state_path: str | Path | None = None,
    evaluation_state_path: str | Path | None = None,
    evaluation_history_path: str | Path | None = None,
    run_id: str | None = None,
) -> TournamentResult:
    """Run the usual Elo ladder with an independent Stockfish observer."""
    if games_per_elo < 2:
        raise ValueError("games_per_elo must be >= 2 for color balancing")
    if stockfish_floor is None:
        stockfish_floor = detect_stockfish_floor(stockfish_executable)

    ratings = [int(x) for x in opponent_elos]
    if not ratings:
        raise ValueError("opponent_elos must contain at least one rating")
    routed = [select_opponent_engine(x, stockfish_floor=stockfish_floor) for x in ratings]
    if "alfil" in routed and not alfil_executable:
        raise ValueError(
            "alfil_executable is required when opponent_elos include ratings "
            f"below Stockfish floor {stockfish_floor}"
        )

    # Each output directory represents one benchmark run. Start a fresh
    # evaluation trace instead of silently mixing data from an older rerun.
    for artifact in (evaluation_state_path, evaluation_history_path):
        if artifact is not None:
            Path(artifact).unlink(missing_ok=True)

    total_games = len(ratings) * games_per_elo
    resolved_run_id = run_id
    if resolved_run_id is None and live_state_path is not None:
        resolved_run_id = Path(live_state_path).parent.name

    _write_live(
        live_state_path,
        status="running",
        message="Tournament starting with independent Stockfish position analysis",
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
            for opponent_elo, engine_name in zip(ratings, routed):
                if engine_name == "alfil":
                    opponent_cm = AlfilOpponent(
                        AlfilConfig(str(alfil_executable), opponent_elo, move_time_s)
                    )
                else:
                    opponent_cm = StockfishOpponent(
                        StockfishConfig(stockfish_executable, opponent_elo, move_time_s)
                    )

                with opponent_cm as opponent:
                    for local_index in range(games_per_elo):
                        fly_is_white = local_index % 2 == 0
                        record = play_one_game_with_analysis(
                            fly_agent,
                            opponent,
                            evaluator,
                            fly_is_white=fly_is_white,
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
    return TournamentResult(tuple(records), final_elo, stockfish_floor=int(stockfish_floor))
