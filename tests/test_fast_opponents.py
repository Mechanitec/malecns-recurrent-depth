import pytest

from malecns_rd.fast_opponents import (
    FAST_LOW_ELOS,
    GAIA_RATING_TO_LEVEL,
    GaiaConfig,
    MinicConfig,
    select_fast_opponent,
)
from malecns_rd.engine_smoke import default_configurations
from malecns_rd.low_elo_calibration import CalibratedRating
from malecns_rd.fast_opponents import _validate_fast_ladder


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


def test_engine_smoke_matrix_contains_requested_levels(tmp_path):
    configs = default_configurations(
        stockfish=tmp_path / "stockfish.exe",
        minic=tmp_path / "minic.exe",
        gaia=tmp_path / "gaia.exe",
        stockfish_floor=1320,
    )
    assert len(configs) == 16
    assert [config.setting for config in configs[:8]] == [
        "Level=0", "Level=1", "Level=5", "Level=10",
        "Level=15", "Level=20", "Level=25", "Level=30",
    ]
    assert configs[-1].setting == "UCI_Elo=1320"
    assert all("Ponder" not in config.options for config in configs)


def test_measured_calibration_routes_to_nearest_actual_setting():
    table = [
        CalibratedRating("minic", "Level=0", 310.1, 0.0),
        CalibratedRating("minic", "Level=10", 327.5, 17.4),
        CalibratedRating("gaia", "Skill Level=1", 580.0, 269.9),
    ]
    routed = _validate_fast_ladder(
        [0, 400, 580, 1320],
        stockfish_floor=1320,
        minic_executable="minic",
        gaia_executable="gaia",
        calibration_rows=table,
    )
    assert routed[0][1:] == ("minic", table[0])
    assert routed[1][1:] == ("minic", table[1])
    assert routed[2][1:] == ("gaia", table[2])
    assert routed[3] == (1320, "stockfish", None)
