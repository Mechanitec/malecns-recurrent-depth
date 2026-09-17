import pandas as pd

from scripts.build_chess_curriculum_v1 import _select_teacher_stage


def test_teacher_stage_best_flags_are_scoped_to_each_position():
    rows = []
    for position_id, scores in (("p1", (300, 100, -50)), ("p2", (250, 25, -75))):
        for index, score in enumerate(scores):
            rows.append({
                "position_id": position_id,
                "fen": "8/8/8/8/8/8/4K3/4k2R w - - 0 1",
                "move_uci": f"h1h{index + 2}",
                "teacher_cp": score,
                "teacher_target": score / 400,
                "source_game_id": position_id,
                "side_to_move": "white",
                "legal_move_count": 3,
            })
    selected = _select_teacher_stage(pd.DataFrame(rows), "tactics", 2, 0, 7)
    assert selected.groupby("position_id")["is_best"].sum().to_dict() == {"p1": 1, "p2": 1}
