"""Local UCI engine smoke tests and throughput measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Iterable

from .chess_benchmark import detect_stockfish_floor, _require_chess


SMOKE_FENS: tuple[str, ...] = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2N1BN2/PPP2PPP/R2QKB1R w KQ - 0 6",
    "r2q1rk1/ppp1bppp/2n1pn2/3p4/3P4/1PN1PN2/PB3PPP/R2QKB1R w KQ - 2 8",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2N1BN2/PPP2PPP/R2QKB1R b KQ - 1 6",
    "r3r1k1/ppp1qppp/2n1b3/3pP3/3P4/2N1B3/PPPQ1PPP/2RR2K1 w - - 2 14",
    "2r2rk1/1bqnbppp/p3p3/pp1pP3/3P4/1PN1BN2/PB3PPP/2RQR1K1 w - - 4 16",
    "r1b2rk1/ppqnbppp/2p1p3/3pP3/3P1P2/2N1BN2/PPPQ2PP/2RR2K1 b - - 0 15",
    "r2q1rk1/1b1nbppp/p3p3/pp1pP3/3P1P2/1PN1BN2/PB1NQ1PP/2RR2K1 w - - 2 16",
    "2rqr1k1/pp1nbppp/2p1p3/3pP3/3P1P2/1PN1BN2/PB1NQ1PP/2RR2K1 b - - 5 18",
)


@dataclass(frozen=True)
class SmokeConfiguration:
    engine: str
    setting: str
    executable: str
    options: dict[str, object]
    advertised_elo: int | None = None


@dataclass(frozen=True)
class SmokeGame:
    engine: str
    setting: str
    game_index: int
    opening_index: int
    engine_color: str
    status: str
    plies: int
    engine_moves: int
    elapsed_s: float
    median_move_latency_s: float | None
    error: str = ""


def executable_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_configurations(
    *,
    stockfish: str | Path,
    minic: str | Path,
    gaia: str | Path,
    stockfish_floor: int | None = None,
    minic_levels: Iterable[int] = (0, 1, 5, 10, 15, 20, 25, 30),
    gaia_levels: Iterable[int] = range(1, 8),
) -> list[SmokeConfiguration]:
    """Return the fixed weak-engine smoke matrix plus Stockfish's floor."""
    if stockfish_floor is None:
        stockfish_floor = detect_stockfish_floor(str(stockfish))
    configs: list[SmokeConfiguration] = []
    for level in minic_levels:
        configs.append(
            SmokeConfiguration(
                engine="minic",
                setting=f"Level={int(level)}",
                executable=str(minic),
                options={"Level": int(level), "Threads": 1, "Hash": 16, "nodesBasedLevel": True},
            )
        )
    for level in gaia_levels:
        configs.append(
            SmokeConfiguration(
                engine="gaia",
                setting=f"Skill Level={int(level)}",
                executable=str(gaia),
                options={"Skill Level": int(level), "Threads": 1, "Hash": 16, "OwnBook": False},
            )
        )
    configs.append(
        SmokeConfiguration(
            engine="stockfish",
            setting=f"UCI_Elo={int(stockfish_floor)}",
            executable=str(stockfish),
            options={"UCI_LimitStrength": True, "UCI_Elo": int(stockfish_floor), "Threads": 1, "Hash": 16},
            advertised_elo=int(stockfish_floor),
        )
    )
    return configs


class _TimedUciEngine:
    def __init__(self, config: SmokeConfiguration, *, move_time_s: float, timeout_s: float) -> None:
        chess = _require_chess()
        self._chess = chess
        self._engine = chess.engine.SimpleEngine.popen_uci(config.executable, timeout=timeout_s)
        try:
            self._engine.timeout = float(timeout_s)
            safe_options = {key: value for key, value in config.options.items() if key in self._engine.options}
            if safe_options:
                self._engine.configure(safe_options)
        except Exception:
            self._engine.quit()
            raise
        self._limit = chess.engine.Limit(time=float(move_time_s))
        self.engine_id = dict(self._engine.id)

    def choose_move(self, board):
        result = self._engine.play(board, self._limit)
        if result.move is None or result.move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal move: {result.move}")
        return result.move

    def close(self) -> None:
        self._engine.quit()


def _play_smoke_game(
    engine: _TimedUciEngine,
    *,
    engine_name: str,
    setting: str,
    game_index: int,
    opening_index: int,
    engine_color: str,
    move_time_s: float,
    max_plies: int,
    game_timeout_s: float,
    seed: int,
) -> SmokeGame:
    chess = _require_chess()
    import random

    board = chess.Board(SMOKE_FENS[opening_index % len(SMOKE_FENS)])
    rng = random.Random(seed + game_index)
    engine_is_white = engine_color == "white"
    latencies: list[float] = []
    started = time.perf_counter()
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if time.perf_counter() - started > game_timeout_s:
                raise TimeoutError(f"game exceeded {game_timeout_s:.3f}s")
            engine_turn = board.turn == (chess.WHITE if engine_is_white else chess.BLACK)
            if engine_turn:
                move_started = time.perf_counter()
                move = engine.choose_move(board)
                latencies.append(time.perf_counter() - move_started)
            else:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = rng.choice(legal_moves)
            board.push(move)
        elapsed = time.perf_counter() - started
        return SmokeGame(
            engine_name,
            setting,
            game_index,
            opening_index,
            engine_color,
            "complete",
            board.ply(),
            len(latencies),
            elapsed,
            statistics.median(latencies) if latencies else None,
        )
    except Exception as exc:
        return SmokeGame(
            engine_name,
            setting,
            game_index,
            opening_index,
            engine_color,
            "error",
            board.ply(),
            len(latencies),
            time.perf_counter() - started,
            statistics.median(latencies) if latencies else None,
            f"{type(exc).__name__}: {exc}",
        )


def run_engine_smoke(
    configurations: Iterable[SmokeConfiguration],
    *,
    output_dir: str | Path,
    games_per_config: int = 10,
    move_time_s: float = 0.02,
    max_plies: int = 120,
    game_timeout_s: float = 15.0,
    startup_timeout_s: float = 10.0,
    seed: int = 7,
) -> list[dict[str, object]]:
    """Run complete games and write incremental game and summary artifacts."""
    if games_per_config < 1:
        raise ValueError("games_per_config must be >= 1")
    configurations = list(configurations)
    if not configurations:
        raise ValueError("configurations must contain at least one engine")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    games_path = root / "engine_smoke_games.csv"
    summary_path = root / "engine_throughput.csv"
    metadata_path = root / "engine_smoke_metadata.json"
    games_path.unlink(missing_ok=True)
    summary: list[dict[str, object]] = []
    all_games: list[SmokeGame] = []
    game_fields = list(asdict(SmokeGame("", "", 0, 0, "", "", 0, 0, 0.0, None)).keys())
    with games_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=game_fields)
        writer.writeheader()
        for config in configurations:
            config_games: list[SmokeGame] = []
            engine: _TimedUciEngine | None = None
            engine_id: dict[str, str] = {}
            restarts = 0
            for game_index in range(games_per_config):
                if engine is None:
                    try:
                        engine = _TimedUciEngine(config, move_time_s=move_time_s, timeout_s=startup_timeout_s)
                        engine_id = engine.engine_id
                    except Exception as exc:
                        game = SmokeGame(config.engine, config.setting, game_index, game_index % len(SMOKE_FENS), "white" if game_index % 2 == 0 else "black", "error", 0, 0, 0.0, None, f"startup {type(exc).__name__}: {exc}")
                        writer.writerow(asdict(game))
                        stream.flush()
                        config_games.append(game)
                        restarts += 1
                        continue
                game = _play_smoke_game(
                    engine,
                    engine_name=config.engine,
                    setting=config.setting,
                    game_index=game_index,
                    opening_index=game_index % len(SMOKE_FENS),
                    engine_color="white" if game_index % 2 == 0 else "black",
                    move_time_s=move_time_s,
                    max_plies=max_plies,
                    game_timeout_s=game_timeout_s,
                    seed=seed,
                )
                writer.writerow(asdict(game))
                stream.flush()
                config_games.append(game)
                if game.status == "error":
                    engine.close()
                    engine = None
                    restarts += 1
            if engine is not None:
                engine.close()
            all_games.extend(config_games)
            completed = [game for game in config_games if game.status == "complete"]
            latencies = [game.median_move_latency_s for game in completed if game.median_move_latency_s is not None]
            total_elapsed = sum(game.elapsed_s for game in completed)
            row = {
                "engine": config.engine,
                "setting": config.setting,
                "advertised_elo": config.advertised_elo,
                "executable": config.executable,
                "executable_sha256": executable_sha256(config.executable),
                "engine_name": engine_id.get("name", ""),
                "engine_author": engine_id.get("author", ""),
                "games_requested": games_per_config,
                "games_completed": len(completed),
                "games_failed": len(config_games) - len(completed),
                "restarts": restarts,
                "median_move_latency_s": statistics.median(latencies) if latencies else None,
                "moves_per_second": sum(game.engine_moves for game in completed) / sum(game.elapsed_s for game in completed) if total_elapsed else None,
                "games_per_hour": len(completed) * 3600.0 / total_elapsed if total_elapsed else None,
                "options": json.dumps(config.options, sort_keys=True),
            }
            summary.append(row)
            with summary_path.open("w", newline="", encoding="utf-8") as summary_stream:
                writer_summary = csv.DictWriter(summary_stream, fieldnames=list(row.keys()))
                writer_summary.writeheader()
                writer_summary.writerows(summary)

    metadata = {
        "seed": int(seed),
        "smoke_fens": list(SMOKE_FENS),
        "games_per_config": int(games_per_config),
        "move_time_s": float(move_time_s),
        "max_plies": int(max_plies),
        "game_timeout_s": float(game_timeout_s),
        "startup_timeout_s": float(startup_timeout_s),
        "configurations": [asdict(config) for config in configurations],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary
