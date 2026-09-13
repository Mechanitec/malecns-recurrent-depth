import numpy as np
import pandas as pd
from malecns_rd.low_elo_calibration import (
    calibrate,
    default_calibration_matchups,
    fit_bradley_terry,
    pava_monotonic,
    schedule_frame,
    simulate_calibration_games,
)


def test_schedule_is_color_balanced_and_connected():
    s = schedule_frame(openings=3, max_minic_level=6)
    assert len(s) % 2 == 0
    for _, g in s.groupby("pair_id"):
        assert len(g) == 2
        assert g.iloc[0].white == g.iloc[1].black
        assert g.iloc[0].black == g.iloc[1].white
    edges = default_calibration_matchups(max_minic_level=6)
    assert ("gaia_580", "minic_6") in edges or ("minic_6", "gaia_580") in edges


def test_pava_monotonic():
    raw = [0, 20, 18, 40, 39, 60]
    got = pava_monotonic(raw)
    assert np.all(np.diff(got) >= -1e-12)
    assert np.isclose(got[1], got[2])


def test_fit_recovers_known_order_and_anchor_scale():
    schedule = schedule_frame(openings=80, max_minic_level=6)
    true = {f"minic_{i}": 580.0 * i / 6 for i in range(7)}
    true["gaia_580"] = 580.0
    games = simulate_calibration_games(schedule, true, seed=42, draw_band=0.08)
    fit = calibrate(games, bootstrap_samples=12, bootstrap_seed=3)
    r = fit.ratings.set_index("engine")
    assert r.loc["minic_0", "mcr"] == 0.0
    assert r.loc["gaia_580", "mcr"] == 580.0
    seq = r.loc[[f"minic_{i}" for i in range(7)], "mcr"].to_numpy()
    assert np.all(np.diff(seq) >= -1e-8)
    assert abs(r.loc["minic_3", "mcr"] - 290.0) < 90
    assert (r["bootstrap_samples"] >= 10).all()


def test_disconnected_fit_rejected():
    games = pd.DataFrame({
        "white": ["a", "c"], "black": ["b", "d"], "white_score": [1, 1]
    })
    try:
        fit_bradley_terry(games)
    except ValueError as e:
        assert "connected" in str(e)
    else:
        raise AssertionError("expected disconnected graph rejection")
