import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.elo import GameObservation, estimate_elo, expected_score
from malecns_rd.low_elo_calibration import fit_calibration, make_opening_fens


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == 0.5


def test_elo_recovers_equal_rating_from_equal_score():
    games = [GameObservation(1600, s) for s in ([1.0, 0.0] * 100)]
    estimate = estimate_elo(games)
    assert abs(estimate.rating - 1600.0) < 1e-6
    assert estimate.ci95_low < 1600 < estimate.ci95_high


def test_elo_combines_multiple_opponent_ratings():
    true_rating = 1700.0
    # Use deterministic fractional observations at the theoretical expectations.
    # This tests the estimating equation directly without Monte Carlo noise.
    games = [
        GameObservation(1400, expected_score(true_rating, 1400)),
        GameObservation(1600, expected_score(true_rating, 1600)),
        GameObservation(1800, expected_score(true_rating, 1800)),
        GameObservation(2000, expected_score(true_rating, 2000)),
    ] * 50
    estimate = estimate_elo(games)
    assert math.isclose(estimate.rating, true_rating, abs_tol=1e-6)


def test_elo_reports_censoring_for_all_wins():
    estimate = estimate_elo([GameObservation(1320, 1.0)] * 10, upper_bound=3000)
    assert estimate.censored == "above"


def test_calibration_fixes_gaia_anchor_and_bootstraps_by_opening_pair():
    rows = []
    for opening_index in range(20):
        pair = f"opening_{opening_index:03d}"
        for game_index, result in enumerate(("1-0", "0-1")):
            rows.append({
                "opening_pair": pair,
                "white_engine": "gaia" if game_index == 0 else "minic",
                "white_setting": "Skill Level=1" if game_index == 0 else "Level=0",
                "black_engine": "minic" if game_index == 0 else "gaia",
                "black_setting": "Level=0" if game_index == 0 else "Skill Level=1",
                "result": result,
                "status": "complete",
            })
    result = fit_calibration(rows, bootstrap_samples=40, seed=3)
    by_name = {(row["engine"], row["setting"]): row for row in result.rows}
    assert by_name[("gaia", "Skill Level=1")]["raw_calibrated_elo"] == 580.0
    assert by_name[("minic", "Level=0")]["mcr0"] == 0.0
    assert by_name[("minic", "Level=0")]["ci95_low"] is not None
    assert result.diagnostics["disconnected_from_anchor"] == []


def test_calibration_openings_are_reproducible():
    assert make_opening_fens(5, seed=9) == make_opening_fens(5, seed=9)
