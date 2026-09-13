from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    path: Path
    elo: float
    ci95_low: float
    ci95_high: float
    n_games: int
    censored: str | None
    stockfish_floor: int | None
    score: float


BENCHMARK_GAME_COLUMNS = {
    "game_index",
    "opponent_engine",
    "opponent_elo",
    "fly_color",
    "result",
    "fly_score",
    "plies",
    "termination",
    "final_fen",
}

TRAINING_COLUMNS = {
    "step",
    "epoch",
    "loss",
    "validation_loss",
    "teacher_agreement",
    "candidate_accuracy",
    "elo",
    "elo_ci_low",
    "elo_ci_high",
    "recurrent_depth",
    "learning_rate",
    "checkpoint",
}


def _float_or(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def discover_benchmark_runs(results_root: str | Path) -> list[BenchmarkRun]:
    """Discover saved chess benchmark runs under ``results_root``.

    A run directory is recognized by the presence of both ``games.csv`` and
    ``elo.json``. The function accepts both the current nested summary format
    (``{"elo": {...}, "stockfish_floor": ...}``) and the older flat Elo JSON.
    """
    root = Path(results_root)
    if not root.exists():
        return []

    runs: list[BenchmarkRun] = []
    for elo_path in root.rglob("elo.json"):
        run_dir = elo_path.parent
        games_path = run_dir / "games.csv"
        if not games_path.exists():
            continue
        try:
            summary = json.loads(elo_path.read_text(encoding="utf-8"))
            games = pd.read_csv(games_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

        elo_payload = summary.get("elo", summary)
        estimate = _float_or(
            elo_payload.get("rating", elo_payload.get("elo", elo_payload.get("estimate", 0.0))),
            0.0,
        )
        low = _float_or(
            elo_payload.get("ci95_low", elo_payload.get("lower", estimate)),
            estimate,
        )
        high = _float_or(
            elo_payload.get("ci95_high", elo_payload.get("upper", estimate)),
            estimate,
        )
        censored = elo_payload.get("censored")
        stockfish_floor = _int_or(summary.get("stockfish_floor"))
        score = float(games["fly_score"].mean()) if "fly_score" in games and len(games) else 0.0

        try:
            run_id = str(run_dir.relative_to(root)) or run_dir.name
        except ValueError:
            run_id = run_dir.name

        runs.append(
            BenchmarkRun(
                run_id=run_id,
                path=run_dir,
                elo=estimate,
                ci95_low=low,
                ci95_high=high,
                n_games=int(len(games)),
                censored=str(censored) if censored not in (None, "", "none") else None,
                stockfish_floor=stockfish_floor,
                score=score,
            )
        )

    runs.sort(key=lambda r: (r.path.stat().st_mtime if r.path.exists() else 0.0, r.run_id))
    return runs


def load_games(run: BenchmarkRun | str | Path) -> pd.DataFrame:
    run_dir = run.path if isinstance(run, BenchmarkRun) else Path(run)
    df = pd.read_csv(run_dir / "games.csv")
    missing = BENCHMARK_GAME_COLUMNS - set(df.columns)
    if missing:
        # Older runs may predate opponent_engine; retain compatibility.
        if missing == {"opponent_engine"}:
            df = df.copy()
            df["opponent_engine"] = "stockfish"
        else:
            raise ValueError(f"games.csv missing required columns: {sorted(missing)}")
    return df


def benchmark_history_frame(runs: Iterable[BenchmarkRun]) -> pd.DataFrame:
    rows = [
        {
            "run_id": r.run_id,
            "elo": r.elo,
            "ci95_low": r.ci95_low,
            "ci95_high": r.ci95_high,
            "games": r.n_games,
            "score": r.score,
            "censored": r.censored or "",
        }
        for r in runs
    ]
    return pd.DataFrame(rows)


def score_by_opponent(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["opponent_engine", "opponent_elo", "games", "score"])
    grouped = (
        games.groupby(["opponent_engine", "opponent_elo"], as_index=False)
        .agg(games=("fly_score", "size"), score=("fly_score", "mean"))
        .sort_values(["opponent_elo", "opponent_engine"])
    )
    return grouped


def result_distribution(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["outcome", "games"])

    def outcome(score: float) -> str:
        if score >= 0.75:
            return "Win"
        if score <= 0.25:
            return "Loss"
        return "Draw"

    values = games["fly_score"].map(outcome).value_counts()
    return pd.DataFrame(
        {
            "outcome": ["Win", "Draw", "Loss"],
            "games": [int(values.get(x, 0)) for x in ["Win", "Draw", "Loss"]],
        }
    )


def load_training_history(path: str | Path) -> pd.DataFrame:
    """Load a training-history CSV and normalize optional columns.

    ``step`` is preferred; diagnostic exports may provide ``epoch`` instead.
    All other dashboard fields are optional so the logger can grow with the
    training code without breaking old experiments.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=sorted(TRAINING_COLUMNS))
    df = pd.read_csv(p)
    if "step" not in df.columns and "epoch" in df.columns:
        df["step"] = df["epoch"]
    if "step" not in df.columns:
        raise ValueError("training history must contain a 'step' or 'epoch' column")
    df = df.sort_values("step").reset_index(drop=True)
    for col in sorted(TRAINING_COLUMNS - set(df.columns)):
        df[col] = pd.NA
    return df


def discover_training_histories(results_root: str | Path) -> list[Path]:
    root = Path(results_root)
    if not root.exists():
        return []
    paths = list(root.rglob("training_history.csv"))
    paths.extend(root.rglob("training*.csv"))
    return sorted(
        set(paths),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, str(p)),
    )
