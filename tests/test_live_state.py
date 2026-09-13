from pathlib import Path

from malecns_rd.live_state import (
    CandidateScore,
    LiveBenchmarkState,
    outcome_counts,
    read_live_state,
    write_live_state,
)


def test_live_state_roundtrip(tmp_path: Path):
    path = tmp_path / "live_state.json"
    state = LiveBenchmarkState(
        status="running",
        run_id="depth16",
        game_index=3,
        games_completed=3,
        games_total=20,
        opponent_engine="alfil",
        opponent_elo=600,
        fly_color="white",
        ply=17,
        fen="8/8/8/8/8/8/8/8 w - - 0 1",
        last_move="e2e4",
        last_actor="fly",
        recurrent_depth=16,
        candidate_scores=(
            CandidateScore("e2e4", 1.25),
            CandidateScore("d2d4", 0.75),
        ),
        rolling_elo=480.0,
        rolling_ci_low=420.0,
        rolling_ci_high=540.0,
        wins=1,
        draws=1,
        losses=1,
    )
    write_live_state(path, state)
    loaded = read_live_state(path)

    assert loaded is not None
    assert loaded.status == "running"
    assert loaded.opponent_engine == "alfil"
    assert loaded.candidate_scores[0].move == "e2e4"
    assert loaded.candidate_scores[0].score == 1.25
    assert loaded.updated_at_utc is not None


def test_live_state_bad_json_is_tolerated(tmp_path: Path):
    path = tmp_path / "live_state.json"
    path.write_text("{not complete", encoding="utf-8")
    assert read_live_state(path) is None


def test_outcome_counts():
    assert outcome_counts([1.0, 0.5, 0.0, 1.0]) == (2, 1, 1)
