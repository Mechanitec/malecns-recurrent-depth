from malecns_rd.reporting import derive_tournament_artifact_metadata


def test_raw_game_metadata_detects_short_bounded_run():
    rows = [
        {
            "plies": "2",
            "termination": "MAX_PLIES_ADJUDICATION",
            "fly_color": "white",
            "fly_move_count": "1",
            "total_recurrent_passes": "64",
            "pgn": '[Result "1/2-1/2"]\n\n1. e4 e5 1/2-1/2',
        }
    ]

    metadata = derive_tournament_artifact_metadata(rows, reference_depth=16)

    assert metadata["max_plies"] == 2
    assert metadata["full_legal_candidates"] is False
    assert metadata["bounded_candidates"] is True
    assert metadata["candidate_evaluations_per_fly_move_max"] == 4
    assert metadata["termination_counts"] == {"MAX_PLIES_ADJUDICATION": 1}
