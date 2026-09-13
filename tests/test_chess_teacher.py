import math
import pytest

from malecns_rd.chess_teacher import centipawn_to_target


def test_centipawn_target_is_bounded_and_monotonic():
    vals = [centipawn_to_target(x) for x in (-2000, -400, 0, 400, 2000)]
    assert vals == sorted(vals)
    assert vals[2] == 0.0
    assert all(-1.0 <= x <= 1.0 for x in vals)
    assert vals[1] == pytest.approx(-math.tanh(1.0))
    assert vals[3] == pytest.approx(math.tanh(1.0))


def test_centipawn_scale_must_be_positive():
    with pytest.raises(ValueError):
        centipawn_to_target(10, scale_cp=0)
