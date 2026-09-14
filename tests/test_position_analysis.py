from malecns_rd.position_analysis import (
    AnalysisConfig,
    cp_to_advantage_fraction,
    human_eval,
)
from malecns_rd.chess_benchmark import play_one_game


class _FirstLegalAgent:
    name = "test-agent"
    depth = 1
    last_decision = {"depth": 1, "recurrent_passes": 1, "candidates": []}

    def choose_move(self, board):
        return next(iter(board.legal_moves))


class _FirstLegalOpponent:
    name = "test-opponent"
    elo = 100
    calibrated_elo = 100.0
    setting = "test"

    def choose_move(self, board):
        return next(iter(board.legal_moves))


def test_advantage_fraction_is_centered_and_monotonic():
    assert cp_to_advantage_fraction(0) == 0.5
    assert cp_to_advantage_fraction(200) > 0.5
    assert cp_to_advantage_fraction(-200) < 0.5
    assert cp_to_advantage_fraction(1000) > cp_to_advantage_fraction(200)


def test_mate_saturates_display_bar():
    assert cp_to_advantage_fraction(None, mate=3) == 1.0
    assert cp_to_advantage_fraction(None, mate=-2) == 0.0


def test_human_eval_labels_fly_perspective():
    assert "Fly better" in human_eval(125, None, side="Fly")
    assert "Opponent better" in human_eval(-125, None, side="Fly")
    assert human_eval(None, 4, side="Fly") == "Fly has mate in 4"


def test_analysis_config_validates_reproducible_settings():
    cfg = AnalysisConfig("stockfish", depth=18, threads=1, hash_mb=128)
    assert cfg.depth == 18

    for kwargs in (
        {"depth": 0},
        {"threads": 0},
        {"hash_mb": 0},
    ):
        try:
            AnalysisConfig("stockfish", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_max_plies_is_relative_to_opening_fen():
    import chess

    board = chess.Board()
    board.push_uci("e2e4")
    record = play_one_game(
        _FirstLegalAgent(),
        _FirstLegalOpponent(),
        fly_is_white=False,
        start_fen=board.fen(),
        max_plies=2,
    )
    assert record.plies == board.ply() + 2
    assert record.fly_move_count == 1
    assert record.start_fen == board.fen()
