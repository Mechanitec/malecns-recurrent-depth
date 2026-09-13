import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.dashboard_data import (
    benchmark_history_frame,
    discover_benchmark_runs,
    load_games,
    load_training_history,
    result_distribution,
    score_by_opponent,
)
from malecns_rd.training_log import TrainingMetric, append_training_metric


def _write_run(tmp_path: Path) -> Path:
    run = tmp_path / "chess" / "run_001"
    run.mkdir(parents=True)
    games = pd.DataFrame([
        {"game_index": 0, "opponent_engine": "alfil", "opponent_elo": 200, "fly_color": "white", "result": "1-0", "fly_score": 1.0, "plies": 31, "termination": "CHECKMATE", "final_fen": "fen0"},
        {"game_index": 1, "opponent_engine": "alfil", "opponent_elo": 200, "fly_color": "black", "result": "1/2-1/2", "fly_score": 0.5, "plies": 80, "termination": "STALEMATE", "final_fen": "fen1"},
        {"game_index": 2, "opponent_engine": "stockfish", "opponent_elo": 1320, "fly_color": "white", "result": "0-1", "fly_score": 0.0, "plies": 44, "termination": "CHECKMATE", "final_fen": "fen2"},
    ])
    games.to_csv(run / "games.csv", index=False)
    (run / "elo.json").write_text(json.dumps({
        "elo": {"rating": 310.0, "ci95_low": 250.0, "ci95_high": 370.0, "censored": None},
        "stockfish_floor": 1320,
    }), encoding="utf-8")
    return run


def test_dashboard_discovers_and_summarizes_benchmark(tmp_path):
    _write_run(tmp_path)
    runs = discover_benchmark_runs(tmp_path)
    assert len(runs) == 1
    run = runs[0]
    assert run.elo == 310.0
    assert run.n_games == 3
    assert run.stockfish_floor == 1320
    assert abs(run.score - 0.5) < 1e-9
    hist = benchmark_history_frame(runs)
    assert hist.iloc[0]["elo"] == 310.0


def test_dashboard_groups_opponents_and_results(tmp_path):
    run_dir = _write_run(tmp_path)
    games = load_games(run_dir)
    grouped = score_by_opponent(games)
    alfil = grouped[(grouped.opponent_engine == "alfil") & (grouped.opponent_elo == 200)].iloc[0]
    assert alfil.games == 2
    assert abs(alfil.score - 0.75) < 1e-9
    dist = result_distribution(games).set_index("outcome")["games"].to_dict()
    assert dist == {"Win": 1, "Draw": 1, "Loss": 1}


def test_training_log_round_trip(tmp_path):
    path = tmp_path / "training_history.csv"
    append_training_metric(path, TrainingMetric(step=1, loss=1.2, elo=50, recurrent_depth=4))
    append_training_metric(path, TrainingMetric(step=2, loss=0.9, elo=80, recurrent_depth=8))
    df = load_training_history(path)
    assert list(df.step) == [1, 2]
    assert list(df.elo) == [50.0, 80.0]
    assert "teacher_agreement" in df.columns


def test_training_diagnostics_use_epoch_as_step(tmp_path):
    path = tmp_path / "training_diagnostics.csv"
    pd.DataFrame([
        {"epoch": 2, "loss": 0.8, "validation_loss": 0.9},
        {"epoch": 1, "loss": 1.0, "validation_loss": 1.1},
    ]).to_csv(path, index=False)

    df = load_training_history(path)

    assert list(df.step) == [1, 2]
    assert list(df.epoch) == [1, 2]
    assert list(df.loss) == [1.0, 0.8]
