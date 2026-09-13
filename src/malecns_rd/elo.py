from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


_ELO_SCALE = math.log(10.0) / 400.0


@dataclass(frozen=True)
class GameObservation:
    """One game result against a calibrated opponent.

    score is from the fly's perspective: 1.0 win, 0.5 draw, 0.0 loss.
    """

    opponent_elo: float
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(frozen=True)
class EloEstimate:
    rating: float
    standard_error: float | None
    ci95_low: float | None
    ci95_high: float | None
    n_games: int
    total_score: float
    censored: str | None = None


def expected_score(player_elo: float, opponent_elo: float) -> float:
    """Classical Elo expected score for a single opponent."""
    x = (opponent_elo - player_elo) * _ELO_SCALE
    if x > 40.0:
        return 0.0
    if x < -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def estimate_elo(
    games: Iterable[GameObservation],
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 4000.0,
    iterations: int = 100,
) -> EloEstimate:
    """Estimate one rating against opponents whose Elo values are treated as known.

    The MLE solves sum(score_i - E_i(R)) = 0. Draws contribute a half point.
    The reported interval is the local asymptotic 95% interval from observed
    Fisher information. It is intentionally labelled as an approximation:
    paired openings and repeated games can be correlated in real tournaments.
    """
    obs = list(games)
    if not obs:
        raise ValueError("at least one game is required")
    if not lower_bound < upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")

    def score_equation(rating: float) -> float:
        return sum(g.score - expected_score(rating, g.opponent_elo) for g in obs)

    lo_val = score_equation(lower_bound)
    hi_val = score_equation(upper_bound)
    total = sum(g.score for g in obs)

    if lo_val <= 0.0:
        return EloEstimate(lower_bound, None, None, None, len(obs), total, "below")
    if hi_val >= 0.0:
        return EloEstimate(upper_bound, None, None, None, len(obs), total, "above")

    lo, hi = float(lower_bound), float(upper_bound)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if score_equation(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    rating = (lo + hi) / 2.0

    info = 0.0
    for game in obs:
        p = expected_score(rating, game.opponent_elo)
        info += (_ELO_SCALE ** 2) * p * (1.0 - p)
    if info <= 0.0:
        return EloEstimate(rating, None, None, None, len(obs), total)
    se = 1.0 / math.sqrt(info)
    z = 1.959963984540054
    return EloEstimate(
        rating=rating,
        standard_error=se,
        ci95_low=rating - z * se,
        ci95_high=rating + z * se,
        n_games=len(obs),
        total_score=total,
    )
