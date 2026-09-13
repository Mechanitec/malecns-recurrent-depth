from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOG10 = math.log(10.0)
ELO_LOGIT_SCALE = 400.0 / LOG10


@dataclass(frozen=True)
class CalibrationFit:
    ratings: pd.DataFrame
    raw_strength: dict[str, float]
    anchor_low: str
    anchor_high: str
    anchor_high_rating: float


def default_calibration_matchups(
    *,
    max_minic_level: int = 30,
    gaia_anchor: str = "gaia_580",
) -> list[tuple[str, str]]:
    """Connected, information-efficient matchup graph for Minic 0..N + Gaia580."""
    if max_minic_level < 1:
        raise ValueError("max_minic_level must be >= 1")
    edges: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a == b:
            return
        edges.add((a, b) if a < b else (b, a))

    for level in range(max_minic_level):
        add(f"minic_{level}", f"minic_{level + 1}")
    for level in range(0, max_minic_level, 5):
        hi = min(level + 5, max_minic_level)
        if hi > level:
            add(f"minic_{level}", f"minic_{hi}")
    starts = sorted(set([max(0, max_minic_level - d) for d in (10, 8, 6, 4, 2, 0)]))
    for level in starts:
        add(f"minic_{level}", gaia_anchor)
    return sorted(edges)


def schedule_frame(
    *,
    openings: int = 40,
    max_minic_level: int = 30,
    gaia_anchor: str = "gaia_580",
) -> pd.DataFrame:
    """Two colour-swapped games per opening per matchup."""
    if openings < 1:
        raise ValueError("openings must be >= 1")
    rows: list[dict[str, object]] = []
    game_index = 0
    for matchup_index, (a, b) in enumerate(
        default_calibration_matchups(max_minic_level=max_minic_level, gaia_anchor=gaia_anchor)
    ):
        for opening in range(openings):
            pair_id = f"m{matchup_index:03d}-o{opening:03d}"
            rows.append({
                "game_index": game_index,
                "matchup_id": matchup_index,
                "opening_id": opening,
                "pair_id": pair_id,
                "white": a,
                "black": b,
            })
            game_index += 1
            rows.append({
                "game_index": game_index,
                "matchup_id": matchup_index,
                "opening_id": opening,
                "pair_id": pair_id,
                "white": b,
                "black": a,
            })
            game_index += 1
    return pd.DataFrame(rows)


def _prepare_games(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    required = {"white", "black", "white_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games missing columns: {sorted(missing)}")
    frame = games.copy().reset_index(drop=True)
    frame["white"] = frame["white"].astype(str)
    frame["black"] = frame["black"].astype(str)
    frame["white_score"] = pd.to_numeric(frame["white_score"], errors="raise").astype(float)
    if not frame["white_score"].between(0.0, 1.0).all():
        raise ValueError("white_score must be in [0, 1]")
    if (frame["white"] == frame["black"]).any():
        raise ValueError("an engine cannot play itself in one row")
    names = sorted(set(frame["white"]) | set(frame["black"]))
    if len(names) < 2:
        raise ValueError("need at least two engines")
    return frame, names, {name: i for i, name in enumerate(names)}


def _is_connected(frame: pd.DataFrame, names: Sequence[str]) -> bool:
    adj = {n: set() for n in names}
    for w, b in zip(frame["white"], frame["black"]):
        adj[w].add(b)
        adj[b].add(w)
    seen = set()
    stack = [names[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur] - seen)
    return len(seen) == len(names)


def fit_bradley_terry(
    games: pd.DataFrame,
    *,
    l2: float = 1e-5,
    fit_white_advantage: bool = True,
) -> tuple[dict[str, float], float]:
    """Fit latent logit strengths using W/D/L scores as fractional outcomes."""
    if l2 < 0:
        raise ValueError("l2 must be >= 0")
    frame, names, index = _prepare_games(games)
    if not _is_connected(frame, names):
        raise ValueError("match graph must be connected")

    wi = frame["white"].map(index).to_numpy(dtype=np.int64)
    bi = frame["black"].map(index).to_numpy(dtype=np.int64)
    y = frame["white_score"].to_numpy(dtype=np.float64)
    n = len(names)
    p = n + (1 if fit_white_advantage else 0)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float]:
        s = x[:n] - np.mean(x[:n])
        w = float(x[n]) if fit_white_advantage else 0.0
        return s, w

    def objective_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        s, white_adv = unpack(x)
        z = s[wi] - s[bi] + white_adv
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
        if l2:
            loss += 0.5 * l2 * float(np.dot(s, s))
        prob = np.empty_like(z)
        pos = z >= 0
        prob[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        prob[~pos] = ez / (1.0 + ez)
        residual = prob - y
        gs = np.zeros(n, dtype=np.float64)
        np.add.at(gs, wi, residual)
        np.add.at(gs, bi, -residual)
        if l2:
            gs += l2 * s
        gs -= np.mean(gs)
        if fit_white_advantage:
            grad = np.concatenate([gs, [float(np.sum(residual))]])
        else:
            grad = gs
        return loss, grad

    x0 = np.zeros(p, dtype=np.float64)
    res = minimize(
        lambda x: objective_grad(x)[0],
        x0,
        jac=lambda x: objective_grad(x)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not res.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {res.message}")
    strengths, white_adv = unpack(res.x)
    return {name: float(strengths[index[name]]) for name in names}, float(white_adv)


def anchor_strengths(
    strengths: dict[str, float],
    *,
    low_anchor: str = "minic_0",
    high_anchor: str = "gaia_580",
    high_rating: float = 580.0,
) -> dict[str, float]:
    if low_anchor not in strengths or high_anchor not in strengths:
        raise ValueError("both anchors must occur in fitted strengths")
    lo = float(strengths[low_anchor])
    hi = float(strengths[high_anchor])
    gap = hi - lo
    if gap <= 1e-9:
        raise ValueError("high anchor must fit stronger than low anchor")
    return {name: high_rating * (float(value) - lo) / gap for name, value in strengths.items()}


def pava_monotonic(values: Sequence[float], weights: Sequence[float] | None = None) -> np.ndarray:
    """Weighted nondecreasing isotonic regression via pooled adjacent violators."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("values must be a non-empty vector")
    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape or np.any(w <= 0):
            raise ValueError("weights must be positive and match values")

    blocks: list[list[float | int]] = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([i, i, float(wi), float(yi)])
        while len(blocks) >= 2 and float(blocks[-2][3]) > float(blocks[-1][3]):
            b2 = blocks.pop()
            b1 = blocks.pop()
            ww = float(b1[2]) + float(b2[2])
            mean = (float(b1[3]) * float(b1[2]) + float(b2[3]) * float(b2[2])) / ww
            blocks.append([int(b1[0]), int(b2[1]), ww, mean])
    out = np.empty(len(y), dtype=np.float64)
    for start, end, _, mean in blocks:
        out[int(start): int(end) + 1] = float(mean)
    return out


def _fit_anchored(
    games: pd.DataFrame,
    low_anchor: str,
    high_anchor: str,
    high_rating: float,
) -> tuple[dict[str, float], float]:
    strengths, white_adv = fit_bradley_terry(games)
    return anchor_strengths(
        strengths,
        low_anchor=low_anchor,
        high_anchor=high_anchor,
        high_rating=high_rating,
    ), white_adv


def calibrate(
    games: pd.DataFrame,
    *,
    low_anchor: str = "minic_0",
    high_anchor: str = "gaia_580",
    high_rating: float = 580.0,
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 7,
    isotonic_minic: bool = True,
) -> CalibrationFit:
    frame, names, _ = _prepare_games(games)
    anchored, white_adv = _fit_anchored(frame, low_anchor, high_anchor, high_rating)

    boot: dict[str, list[float]] = {name: [] for name in names}
    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        if "pair_id" in frame.columns:
            cluster_keys = frame["pair_id"].astype(str)
        else:
            cluster_keys = pd.Series([f"g{i}" for i in range(len(frame))])
        unique = cluster_keys.drop_duplicates().to_numpy()
        cluster_rows = {key: np.flatnonzero(cluster_keys.to_numpy() == key) for key in unique}

        successful = 0
        attempts = 0
        max_attempts = max(bootstrap_samples * 4, bootstrap_samples + 20)
        while successful < bootstrap_samples and attempts < max_attempts:
            attempts += 1
            sampled = rng.choice(unique, size=len(unique), replace=True)
            pieces: list[pd.DataFrame] = []
            for j, key in enumerate(sampled):
                part = frame.iloc[cluster_rows[key]].copy()
                part["_boot_cluster"] = j
                pieces.append(part)
            sample = pd.concat(pieces, ignore_index=True)
            try:
                vals, _ = _fit_anchored(sample, low_anchor, high_anchor, high_rating)
            except (ValueError, RuntimeError):
                continue
            for name in names:
                if name in vals and math.isfinite(vals[name]):
                    boot[name].append(vals[name])
            successful += 1

    rows: list[dict[str, object]] = []
    for name in names:
        samples = np.asarray(boot[name], dtype=np.float64)
        if len(samples):
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
            se = float(np.std(samples, ddof=1)) if len(samples) > 1 else float("nan")
        else:
            ci_low = ci_high = se = float("nan")
        rows.append({
            "engine": name,
            "raw_mcr": float(anchored[name]),
            "mcr": float(anchored[name]),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "bootstrap_se": se,
            "bootstrap_samples": int(len(samples)),
            "is_anchor": name in {low_anchor, high_anchor},
        })
    ratings = pd.DataFrame(rows)

    if isotonic_minic:
        minic = ratings[ratings["engine"].str.match(r"^minic_\d+$")].copy()
        if not minic.empty:
            minic["level"] = minic["engine"].str.split("_").str[-1].astype(int)
            minic = minic.sort_values("level")
            se = minic["bootstrap_se"].to_numpy(dtype=float)
            weights = np.ones_like(se, dtype=float)
            valid_se = np.isfinite(se) & (se > 1e-6)
            weights[valid_se] = 1.0 / np.square(se[valid_se])
            if int(minic.iloc[0]["level"]) == 0:
                weights[0] = max(weights.max(initial=1.0) * 1e6, 1e6)
            iso = pava_monotonic(minic["raw_mcr"].to_numpy(dtype=float), weights)
            if int(minic.iloc[0]["level"]) == 0:
                iso -= iso[0]
            mapping = dict(zip(minic["engine"], iso))
            ratings["mcr"] = ratings.apply(
                lambda r: float(mapping.get(r["engine"], r["raw_mcr"])), axis=1
            )

    ratings.loc[ratings["engine"] == low_anchor, ["raw_mcr", "mcr"]] = 0.0
    ratings.loc[ratings["engine"] == high_anchor, ["raw_mcr", "mcr"]] = float(high_rating)
    ratings = ratings.sort_values(
        "engine",
        key=lambda s: s.map(
            lambda x: (0, int(x.split("_")[1])) if x.startswith("minic_") else (1, x)
        ),
    ).reset_index(drop=True)

    strengths, _ = fit_bradley_terry(frame)
    ratings.attrs["white_advantage_logit"] = white_adv
    ratings.attrs["white_advantage_elo"] = white_adv * ELO_LOGIT_SCALE
    return CalibrationFit(ratings, strengths, low_anchor, high_anchor, high_rating)


def simulate_calibration_games(
    schedule: pd.DataFrame,
    true_ratings: dict[str, float],
    *,
    seed: int = 1,
    draw_band: float = 0.12,
    white_advantage_elo: float = 35.0,
) -> pd.DataFrame:
    """Synthetic validator only; not used for empirical calibration."""
    rng = np.random.default_rng(seed)
    out = schedule.copy()
    scores: list[float] = []
    for w, b in zip(out["white"], out["black"]):
        rw, rb = true_ratings[str(w)], true_ratings[str(b)]
        z = (rw - rb + white_advantage_elo) / ELO_LOGIT_SCALE
        p = 1.0 / (1.0 + math.exp(-z))
        u = rng.random()
        p_draw = draw_band * (1.0 - abs(2.0 * p - 1.0))
        p_win = p * (1.0 - p_draw)
        if u < p_win:
            scores.append(1.0)
        elif u < p_win + p_draw:
            scores.append(0.5)
        else:
            scores.append(0.0)
    out["white_score"] = scores
    return out
