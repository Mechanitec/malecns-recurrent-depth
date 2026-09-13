from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CandidateScore:
    move: str
    score: float


@dataclass(frozen=True)
class LiveBenchmarkState:
    """Small file-backed snapshot consumed by the dashboard.

    The benchmark/training process is the writer and Streamlit is a read-only
    consumer. Atomic replacement keeps readers from observing partial JSON.
    """

    status: str = "idle"
    run_id: str | None = None
    game_index: int | None = None
    games_completed: int = 0
    games_total: int = 0
    opponent_engine: str | None = None
    opponent_elo: int | None = None
    fly_color: str | None = None
    ply: int = 0
    fen: str | None = None
    last_move: str | None = None
    last_actor: str | None = None
    recurrent_depth: int | None = None
    candidate_scores: tuple[CandidateScore, ...] = ()
    rolling_elo: float | None = None
    rolling_ci_low: float | None = None
    rolling_ci_high: float | None = None
    rolling_censored: str | None = None
    wins: int = 0
    draws: int = 0
    losses: int = 0
    updated_at_utc: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_live_state(path: str | Path, state: LiveBenchmarkState) -> None:
    """Atomically replace ``path`` with one JSON snapshot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    payload["updated_at_utc"] = state.updated_at_utc or utc_now_iso()

    tmp = output.with_name(output.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def read_live_state(path: str | Path) -> LiveBenchmarkState | None:
    """Read one snapshot; return ``None`` if no usable state exists yet."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    raw_candidates = payload.pop("candidate_scores", []) or []
    candidates: list[CandidateScore] = []
    for item in raw_candidates:
        try:
            candidates.append(CandidateScore(str(item["move"]), float(item["score"])))
        except (KeyError, TypeError, ValueError):
            continue

    fields = LiveBenchmarkState.__dataclass_fields__
    clean = {k: v for k, v in payload.items() if k in fields and k != "candidate_scores"}
    try:
        return LiveBenchmarkState(candidate_scores=tuple(candidates), **clean)
    except (TypeError, ValueError):
        return None


def outcome_counts(scores: Iterable[float]) -> tuple[int, int, int]:
    wins = draws = losses = 0
    for raw in scores:
        score = float(raw)
        if score >= 0.75:
            wins += 1
        elif score <= 0.25:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses
