from malecns_rd.position_analysis import (
    AnalysisConfig,
    cp_to_advantage_fraction,
    human_eval,
)


def test_advantage_fraction_is_centered_and_monotonic():
    assert cp_to_advantage_fraction(0) == 0.5
    assert cp_to_advantage_fraction(200) > 0.5
    assert cp_to_advantage_fraction(-200) < 0.5
    assert cp_to_advantage_fraction(1000) > cp_to_advantage_fraction(200)


def test_mate_saturates_display_bar():
    assert cp_to_advantage_fraction(None, mate=3) == 1.0
    assert cp_to_advantage_fraction(None, mate=-2) == 0.0


def test_human_eval_labels_fly_perspective():
    assert "Fly better" in human_eval(125, None, side="Fly")
    assert "Opponent better" in human_eval(-125, None, side="Fly")
    assert human_eval(None, 4, side="Fly") == "Fly has mate in 4"


def test_analysis_config_validates_reproducible_settings():
    cfg = AnalysisConfig("stockfish", depth=18, threads=1, hash_mb=128)
    assert cfg.depth == 18

    for kwargs in (
        {"depth": 0},
        {"threads": 0},
        {"hash_mb": 0},
    ):
        try:
            AnalysisConfig("stockfish", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
