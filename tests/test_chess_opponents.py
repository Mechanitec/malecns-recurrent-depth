import pytest

from malecns_rd.chess_benchmark import (
    ALFIL_NOMINAL_ELOS,
    select_opponent_engine,
)


def test_alfil_nominal_ladder():
    assert ALFIL_NOMINAL_ELOS[0] == 0
    assert ALFIL_NOMINAL_ELOS[-1] == 3000
    assert 1200 in ALFIL_NOMINAL_ELOS
    assert 1300 not in ALFIL_NOMINAL_ELOS


def test_routes_below_stockfish_floor_to_alfil():
    assert select_opponent_engine(0, stockfish_floor=1320) == "alfil"
    assert select_opponent_engine(1200, stockfish_floor=1320) == "alfil"


def test_routes_at_stockfish_floor_to_stockfish():
    assert select_opponent_engine(1320, stockfish_floor=1320) == "stockfish"
    assert select_opponent_engine(1600, stockfish_floor=1320) == "stockfish"


def test_rejects_unsupported_gap_rating():
    with pytest.raises(ValueError, match="not an Alfil nominal level"):
        select_opponent_engine(1300, stockfish_floor=1320)
