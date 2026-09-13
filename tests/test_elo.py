import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.elo import GameObservation, estimate_elo, expected_score


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
