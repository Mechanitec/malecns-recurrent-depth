import pytest

from malecns_rd.fast_opponents import (
    FAST_LOW_ELOS,
    GAIA_RATING_TO_LEVEL,
    GaiaConfig,
    MinicConfig,
    select_fast_opponent,
)


def test_fast_ladder_routes_known_anchors():
    floor = 1320
    assert select_fast_opponent(0, stockfish_floor=floor) == "minic"
    for elo in GAIA_RATING_TO_LEVEL:
        assert select_fast_opponent(elo, stockfish_floor=floor) == "gaia"
    assert select_fast_opponent(1320, stockfish_floor=floor) == "stockfish"
    assert select_fast_opponent(1800, stockfish_floor=floor) == "stockfish"


def test_uncalibrated_gap_is_rejected():
    with pytest.raises(ValueError, match="No calibrated fast opponent"):
        select_fast_opponent(400, stockfish_floor=1320)


def test_fast_config_validation():
    assert FAST_LOW_ELOS[0] == 0
    assert GaiaConfig("gaia", 580).rating == 580
    assert MinicConfig("minic", level=0).level == 0
    with pytest.raises(ValueError):
        GaiaConfig("gaia", 600)
    with pytest.raises(ValueError):
        MinicConfig("minic", level=101)
