"""Joint low-strength engine calibration on a common Elo scale."""
from __future__ import annotations

from dataclasses import dataclass
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import minimize

from .chess_benchmark import _require_chess
from .engine_smoke import SmokeConfiguration, _TimedUciEngine, executable_sha256


ELO_SCALE = np.log(10.0) / 400.0
GAIA_ANCHOR = ("gaia", "Skill Level=1")
MINIC_LEVELS = tuple(range(31))
GAIA_LEVELS = tuple(range(1, 8))


@dataclass(frozen=True)
class CalibrationGame:
    game_id: str
    matchup_id: str
    opening_pair: str
    opening_fen: str
    game_in_pair: int
    white_engine: str
    white_setting: str
    black_engine: str
    black_setting: str
    result: str
    white_score: float
    plies: int
    elapsed_s: float
    status: str = "complete"
    error: str = ""


@dataclass(frozen=True)
class CalibrationResult:
    rows: tuple[dict[str, object], ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CalibratedRating:
    engine: str
    setting: str
    calibrated_elo: float
    mcr0: float


def participant_id(engine: str, setting: str) -> str:
    return f"{engine}:{setting}"


def load_calibration_table(path: str | Path) -> list[CalibratedRating]:
    """Load measured calibration rows for benchmark opponent selection."""
    rows: list[CalibratedRating] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                calibrated = float(row["raw_calibrated_elo"])
                mcr0 = float(row["mcr0"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid calibration row: {row}") from exc
            rows.append(CalibratedRating(str(row["engine"]), str(row["setting"]), calibrated, mcr0))
    if not rows:
        raise ValueError(f"calibration table {path} is empty")
    return rows


def make_opening_fens(count: int = 20, *, seed: int = 71, plies: int = 8) -> list[tuple[str, str]]:
    """Create a deterministic suite of ordinary mid-opening positions."""
    chess = _require_chess()
    rng = random.Random(seed)
    openings: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index in range(count):
        board = chess.Board()
        for _ in range(plies):
            moves = sorted(board.legal_moves, key=lambda move: move.uci())
            board.push(rng.choice(moves))
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        openings.append((f"opening_{index:03d}", fen))
    if len(openings) != count:
        raise RuntimeError(f"could only create {len(openings)} unique openings out of {count}")
    return openings


def _score_for_white(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5
    raise ValueError(f"unsupported chess result {result!r}")


def _row_game(row: Mapping[str, object]) -> tuple[str, str, float, str]:
    white = participant_id(str(row["white_engine"]), str(row["white_setting"]))
    black = participant_id(str(row["black_engine"]), str(row["black_setting"]))
    result = str(row["result"])
    return white, black, _score_for_white(result), str(row.get("opening_pair", ""))


def _components(participants: list[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    parent = {name: name for name in participants}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)
    groups: dict[str, list[str]] = {}
    for name in participants:
        groups.setdefault(find(name), []).append(name)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def _fit_component(
    participants: list[str],
    observations: list[tuple[str, str, float]],
    *,
    anchor: str,
    anchor_rating: float,
) -> tuple[dict[str, float], float, bool]:
    if anchor not in participants:
        return {}, float("nan"), False
    ordered = [name for name in participants if name != anchor]
    index = {name: position for position, name in enumerate(ordered)}

    def ratings(values: np.ndarray) -> dict[str, float]:
        result = {anchor: float(anchor_rating)}
        result.update({name: float(values[position]) for name, position in index.items()})
        return result

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        current = ratings(values)
        nll = 0.0
        gradient = np.zeros_like(values)
        aggregated = Counter(observations)
        for (left, right, score), weight in aggregated.items():
            p = 1.0 / (1.0 + np.exp(np.clip(-ELO_SCALE * (current[left] - current[right]), -60.0, 60.0)))
            p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
            nll -= weight * (score * np.log(p) + (1.0 - score) * np.log1p(-p))
            delta = weight * ELO_SCALE * (p - score)
            if left != anchor:
                gradient[index[left]] += delta
            if right != anchor:
                gradient[index[right]] -= delta
        return float(nll), gradient

    initial = np.full(len(ordered), float(anchor_rating))
    result = minimize(
        lambda values: objective(values),
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(anchor_rating - 5000.0, anchor_rating + 5000.0)] * len(ordered),
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7},
    )
    return ratings(result.x), float(result.fun), bool(result.success)


def _pava(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight_sum = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / weight_sum, weight_sum, left[2] + right[2]])
    output: list[float] = []
    for value, _, size in blocks:
        output.extend([value] * int(size))
    return output


def fit_calibration(
    games: Iterable[Mapping[str, object]],
    *,
    anchor: str = participant_id(*GAIA_ANCHOR),
    anchor_rating: float = 580.0,
    bootstrap_samples: int = 200,
    seed: int = 7,
) -> CalibrationResult:
    """Fit all connected participants and bootstrap CIs by opening pair."""
    complete = [row for row in games if str(row.get("status", "complete")) == "complete"]
    if not complete:
        raise ValueError("at least one complete calibration game is required")
    participants = sorted({name for row in complete for name in _row_game(row)[:2]})
    observations = [_row_game(row)[:3] for row in complete]
    edges = [(left, right) for left, right, _, _ in (_row_game(row) for row in complete)]
    components = _components(participants, edges)
    anchor_component = next((component for component in components if anchor in component), [])
    ratings, log_likelihood, fit_success = _fit_component(
        anchor_component,
        [(left, right, score) for left, right, score in observations if left in anchor_component and right in anchor_component],
        anchor=anchor,
        anchor_rating=anchor_rating,
    )

    counts: dict[str, dict[str, int]] = {name: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for name in participants}
    for row in complete:
        white, black, score, _ = _row_game(row)
        for name, own_score in ((white, score), (black, 1.0 - score)):
            counts[name]["games"] += 1
            if own_score == 1.0:
                counts[name]["wins"] += 1
            elif own_score == 0.5:
                counts[name]["draws"] += 1
            else:
                counts[name]["losses"] += 1

    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in complete:
        groups.setdefault(_row_game(row)[3], []).append(row)
    rng = random.Random(seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in participants}
    group_values = list(groups.values())
    for _ in range(max(0, int(bootstrap_samples))):
        sampled = [row for _ in range(len(group_values)) for row in rng.choice(group_values)]
        sample_observations = [_row_game(row)[:3] for row in sampled]
        sample_ratings, _, sample_success = _fit_component(
            anchor_component,
            [(left, right, score) for left, right, score in sample_observations if left in anchor_component and right in anchor_component],
            anchor=anchor,
            anchor_rating=anchor_rating,
        )
        if not sample_success:
            continue
        for name, value in sample_ratings.items():
            bootstrap[name].append(value)

    def interval(name: str) -> tuple[float | None, float | None]:
        values = bootstrap[name]
        if len(values) < 2:
            return None, None
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    monotonic = dict(ratings)
    minic = [name for name in participants if name.startswith("minic: ")]
    # The participant IDs use `minic:Level=...`; sort by the numeric level.
    minic = [name for name in participants if name.startswith("minic:")]
    minic.sort(key=lambda name: int(name.split("=", 1)[1]))
    if minic and all(name in ratings for name in minic):
        fitted = _pava([ratings[name] for name in minic], [counts[name]["games"] for name in minic])
        monotonic.update(dict(zip(minic, fitted)))

    rows: list[dict[str, object]] = []
    for name in participants:
        engine, setting = name.split(":", 1)
        low, high = interval(name)
        rating = ratings.get(name)
        rows.append({
            "engine": engine,
            "setting": setting,
            "raw_calibrated_elo": rating,
            "monotonic_calibrated_elo": monotonic.get(name),
            "mcr0": (rating - ratings["minic:Level=0"]) if rating is not None and "minic:Level=0" in ratings else None,
            "ci95_low": low,
            "ci95_high": high,
            **counts[name],
        })

    pair_stats: list[dict[str, object]] = []
    for pair, pair_rows in sorted(groups.items()):
        scores = [_row_game(row)[2] for row in pair_rows]
        pair_stats.append({
            "opening_pair": pair,
            "games": len(pair_rows),
            "score_white_mean": float(np.mean(scores)),
            "saturated": bool(np.mean(scores) < 0.05 or np.mean(scores) > 0.95),
        })
    diagnostics = {
        "anchor": anchor,
        "anchor_rating": float(anchor_rating),
        "participants": participants,
        "components": components,
        "disconnected_from_anchor": [component for component in components if anchor not in component],
        "complete_games": len(complete),
        "bootstrap_samples_requested": int(bootstrap_samples),
        "bootstrap_samples_used": min((len(values) for values in bootstrap.values()), default=0),
        "fit_success": fit_success,
        "log_likelihood": -log_likelihood,
        "saturated_links": pair_stats,
    }
    return CalibrationResult(tuple(rows), diagnostics)


def write_calibration_outputs(result: CalibrationResult, output_dir: str | Path, *, metadata: Mapping[str, object]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "low_elo_calibration.csv"
    rows = list(result.rows)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (root / "calibration_diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n", encoding="utf-8")
    (root / "calibration_metadata.json").write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")


def _participant_configuration(engine: str, setting: str, minic_path: str, gaia_path: str) -> SmokeConfiguration:
    if engine == "minic":
        level = int(setting.split("=", 1)[1])
        return SmokeConfiguration(engine, setting, minic_path, {"Level": level, "Threads": 1, "Hash": 16, "nodesBasedLevel": True})
    level = int(setting.split("=", 1)[1])
    return SmokeConfiguration(engine, setting, gaia_path, {"Skill Level": level, "Threads": 1, "Hash": 16, "OwnBook": False})


def calibration_schedule() -> list[tuple[str, str, str, str]]:
    pairs: list[tuple[str, str, str, str]] = []
    for level in range(30):
        pairs.append(("minic", f"Level={level}", "minic", f"Level={level + 1}"))
    for left, right in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
        pairs.append(("minic", f"Level={left}", "minic", f"Level={right}"))
    for level in (20, 22, 24, 26, 28, 30):
        pairs.append(("gaia", "Skill Level=1", "minic", f"Level={level}"))
    return pairs


def _play_calibration_game(left_engine, right_engine, fen: str, *, left_is_white: bool, max_plies: int, game_timeout_s: float):
    chess = _require_chess()
    board = chess.Board(fen)
    started = time.perf_counter()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if time.perf_counter() - started > game_timeout_s:
            raise TimeoutError(f"game exceeded {game_timeout_s:.3f}s")
        left_turn = board.turn == (chess.WHITE if left_is_white else chess.BLACK)
        move = left_engine.choose_move(board) if left_turn else right_engine.choose_move(board)
        if move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal move: {move}")
        board.push(move)
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None
        result = outcome.result()
    else:
        result = "1/2-1/2"
    return result, board.ply(), time.perf_counter() - started


def run_calibration_experiment(
    *,
    minic_path: str | Path,
    gaia_path: str | Path,
    output_dir: str | Path,
    opening_pairs: int = 20,
    games_per_pair: int = 2,
    move_time_s: float = 0.01,
    max_plies: int = 120,
    game_timeout_s: float = 15.0,
    opening_seed: int = 71,
    seed: int = 7,
) -> tuple[list[CalibrationGame], dict[str, object]]:
    """Run paired color-swapped games and append each result immediately."""
    if games_per_pair != 2:
        raise ValueError("calibration requires exactly two color-swapped games per opening pair")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    games_path = root / "games.csv"
    openings = make_opening_fens(opening_pairs, seed=opening_seed)
    fieldnames = list(CalibrationGame.__dataclass_fields__.keys())
    existing_keys: set[tuple[str, int, int]] = set()
    if games_path.exists():
        with games_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") == "complete":
                    existing_keys.add((
                        row.get("matchup_id", ""),
                        int(row.get("opening_pair_index", -1)),
                        int(row.get("game_in_pair", -1)),
                    ))
    write_header = not games_path.exists() or games_path.stat().st_size == 0
    all_games: list[CalibrationGame] = []
    with games_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames + ["opening_pair_index"])
        if write_header:
            writer.writeheader()
        for matchup_index, (left_engine_name, left_setting, right_engine_name, right_setting) in enumerate(calibration_schedule()):
            matchup_id = f"matchup_{matchup_index:03d}"
            left_config = _participant_configuration(left_engine_name, left_setting, str(minic_path), str(gaia_path))
            right_config = _participant_configuration(right_engine_name, right_setting, str(minic_path), str(gaia_path))
            left_engine = right_engine = None
            try:
                for opening_index, (opening_id, fen) in enumerate(openings):
                    if all((matchup_id, opening_index, game_in_pair) in existing_keys for game_in_pair in (0, 1)):
                        continue
                    try:
                        if left_engine is None:
                            left_engine = _TimedUciEngine(left_config, move_time_s=move_time_s, timeout_s=game_timeout_s)
                        if right_engine is None:
                            right_engine = _TimedUciEngine(right_config, move_time_s=move_time_s, timeout_s=game_timeout_s)
                        for game_in_pair, left_is_white in enumerate((True, False)):
                            if (matchup_id, opening_index, game_in_pair) in existing_keys:
                                continue
                            game_id = f"{matchup_id}_{opening_id}_{game_in_pair}"
                            result, plies_done, elapsed = _play_calibration_game(
                                left_engine, right_engine, fen, left_is_white=left_is_white,
                                max_plies=max_plies, game_timeout_s=game_timeout_s,
                            )
                            white_engine = left_engine_name if left_is_white else right_engine_name
                            white_setting = left_setting if left_is_white else right_setting
                            black_engine = right_engine_name if left_is_white else left_engine_name
                            black_setting = right_setting if left_is_white else left_setting
                            game = CalibrationGame(
                                game_id, matchup_id, opening_id, fen, game_in_pair,
                                white_engine, white_setting, black_engine, black_setting,
                                result, _score_for_white(result), plies_done, elapsed,
                            )
                            row = {**game.__dict__, "opening_pair_index": opening_index}
                            writer.writerow(row)
                            stream.flush()
                            all_games.append(game)
                            existing_keys.add((matchup_id, opening_index, game_in_pair))
                    except Exception as exc:
                        game = CalibrationGame(
                            f"{matchup_id}_{opening_id}_error", matchup_id, opening_id, fen, -1,
                            left_engine_name, left_setting, right_engine_name, right_setting,
                            "", 0.5, 0, 0.0, "error", f"{type(exc).__name__}: {exc}",
                        )
                        writer.writerow({**game.__dict__, "opening_pair_index": opening_index})
                        stream.flush()
                        all_games.append(game)
                        for engine in (left_engine, right_engine):
                            if engine is not None:
                                engine.close()
                        left_engine = right_engine = None
            finally:
                for engine in (left_engine, right_engine):
                    if engine is not None:
                        engine.close()

    metadata = {
        "opening_seed": int(opening_seed),
        "opening_pairs": int(opening_pairs),
        "games_per_pair": int(games_per_pair),
        "move_time_s": float(move_time_s),
        "max_plies": int(max_plies),
        "game_timeout_s": float(game_timeout_s),
        "seed": int(seed),
        "schedule": [list(item) for item in calibration_schedule()],
        "opening_fens": [{"opening_pair": pair, "fen": fen} for pair, fen in openings],
        "executables": {
            "minic": {"path": str(minic_path), "sha256": executable_sha256(minic_path)},
            "gaia": {"path": str(gaia_path), "sha256": executable_sha256(gaia_path)},
        },
    }
    return all_games, metadata
