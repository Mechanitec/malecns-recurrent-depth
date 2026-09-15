"""Characterize where recurrent depth helps or harms development positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASELINE_GAIN = "1.00"
BASELINE_MODE = "clamped_sensory"
DEPTHS = (1, 2, 4, 8, 16, 32, 64)


def _move_features(board: chess.Board, uci: str) -> dict[str, int]:
    move = chess.Move.from_uci(uci)
    return {
        "teacher_best_is_capture": int(board.is_capture(move)),
        "teacher_best_is_check": int(board.gives_check(move)),
        "teacher_best_is_promotion": int(move.promotion is not None),
        "teacher_best_is_castling": int(board.is_castling(move)),
    }


def _material_balance(board: chess.Board) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    return sum(
        (1 if piece.color == chess.WHITE else -1) * values[piece.piece_type]
        for piece in board.piece_map().values()
    )


def _rank_correlation(x: pd.Series, y: pd.Series) -> float | None:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return None
    value = spearmanr(x[valid], y[valid]).statistic
    return float(value) if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/chess_development_corpus_v1.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanistic_discovery_v1/position_discovery"),
    )
    args = parser.parse_args()
    raw = pd.read_csv(args.raw)
    required = {"position_id", "source_split", "gain_scale", "input_mode", "depth", "regret_cp"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw factorial is missing columns: {sorted(missing)}")
    raw["gain_scale"] = raw["gain_scale"].map(lambda value: f"{float(value):.2f}")
    raw["input_mode"] = raw["input_mode"].astype(str)
    raw["depth"] = raw["depth"].astype(int)
    args.output.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    position_rows = (
        corpus.sort_values(["position_id", "teacher_cp", "move_uci"], ascending=[True, False, True])
        .groupby("position_id", as_index=False)
        .first()
    )
    best_second = corpus.groupby("position_id")["teacher_cp"].apply(
        lambda values: float(values.nlargest(2).iloc[0] - values.nlargest(2).iloc[1])
        if len(values) >= 2 else np.nan
    ).rename("teacher_best_second_margin_cp")
    position_rows = position_rows.merge(best_second, on="position_id", how="left")
    legal_counts = corpus.groupby("position_id")["move_uci"].nunique().to_dict()
    position_rows["legal_move_count"] = position_rows["position_id"].map(legal_counts)
    position_rows["board"] = position_rows["fen"].map(chess.Board)
    position_rows["side_to_move"] = position_rows["board"].map(lambda b: "white" if b.turn else "black")
    position_rows["material_balance_white_cp"] = position_rows["board"].map(_material_balance)
    move_flags = position_rows.apply(
        lambda row: _move_features(row["board"], str(row["move_uci"])), axis=1, result_type="expand"
    )
    position_rows = pd.concat([position_rows.drop(columns=["board"]), move_flags], axis=1)

    baseline = raw[(raw["gain_scale"] == BASELINE_GAIN) & (raw["input_mode"] == BASELINE_MODE)].copy()
    if baseline.empty:
        raise ValueError("baseline factorial condition is absent")
    wide = baseline.pivot(index="position_id", columns="depth")
    descriptors = position_rows.set_index("position_id").copy()
    for depth in (1, 2, 4, 8, 16, 64):
        for column in (
            "regret_cp", "teacher_best_agreement", "score_margin", "state_norm_mean",
            "delta_mean", "saturation_fraction_mean", "readout_effective_rank",
            "candidate_dispersion", "recurrent_sensory_ratio_mean", "fly_rank_of_teacher_best",
        ):
            if (column, depth) in wide:
                descriptors[f"d{depth}_{column}"] = wide[(column, depth)]
    move_wide = baseline.pivot(index="position_id", columns="depth", values="predicted_move")
    descriptors["leader_switches_across_depth"] = move_wide.apply(
        lambda row: int(sum(a != b for a, b in zip(row.dropna().to_numpy()[:-1], row.dropna().to_numpy()[1:]))),
        axis=1,
    )
    descriptors["d1_to_d2_improvement_cp"] = descriptors["d1_regret_cp"] - descriptors["d2_regret_cp"]
    descriptors["d1_to_d8_improvement_cp"] = descriptors["d1_regret_cp"] - descriptors["d8_regret_cp"]
    descriptors["d1_to_d2_state_delta_change"] = descriptors["d2_delta_mean"] - descriptors["d1_delta_mean"]
    descriptors["d1_to_d4_state_delta_change"] = descriptors["d4_delta_mean"] - descriptors["d1_delta_mean"]

    screen = raw[raw["source_split"] == "dev_screen"]
    regime = (
        screen.groupby(["gain_scale", "input_mode", "depth"], as_index=False)["regret_cp"]
        .mean()
        .sort_values(["regret_cp", "gain_scale", "input_mode", "depth"])
        .iloc[0]
    )
    best_condition = raw[
        (raw["gain_scale"] == regime["gain_scale"])
        & (raw["input_mode"] == regime["input_mode"])
    ]
    best_depth = int(regime["depth"])
    best_position = best_condition[best_condition["depth"] == best_depth].set_index("position_id")
    descriptors["best_regime_gain_scale"] = str(regime["gain_scale"])
    descriptors["best_regime_input_mode"] = str(regime["input_mode"])
    descriptors["best_regime_depth"] = best_depth
    descriptors["best_regime_regret_cp"] = best_position["regret_cp"]
    descriptors["d1_to_best_regime_improvement_cp"] = (
        descriptors["d1_regret_cp"] - descriptors["best_regime_regret_cp"]
    )
    descriptors = descriptors.reset_index().drop(columns=["move_uci", "teacher_cp", "teacher_target", "is_best"], errors="ignore")
    descriptors.to_csv(args.output / "position_descriptors.csv", index=False)

    analysis_columns = [
        "teacher_best_second_margin_cp", "legal_move_count", "ply", "material_balance_white_cp",
        "teacher_best_is_capture", "teacher_best_is_check", "teacher_best_is_promotion",
        "teacher_best_is_castling", "d1_state_norm_mean", "d2_state_norm_mean", "d4_state_norm_mean",
        "d1_candidate_dispersion", "d2_candidate_dispersion", "d4_candidate_dispersion",
        "leader_switches_across_depth", "d1_to_d2_state_delta_change", "d1_to_d4_state_delta_change",
    ]
    outcomes = ["d1_to_d2_improvement_cp", "d1_to_d8_improvement_cp", "d1_to_best_regime_improvement_cp"]
    correlation_rows = []
    for outcome in outcomes:
        for descriptor in analysis_columns:
            correlation_rows.append({
                "outcome": outcome,
                "descriptor": descriptor,
                "spearman_rho": _rank_correlation(descriptors[descriptor], descriptors[outcome]),
                "n": int((descriptors[descriptor].notna() & descriptors[outcome].notna()).sum()),
            })
    pd.DataFrame(correlation_rows).to_csv(args.output / "descriptor_correlations.csv", index=False)

    ranked = descriptors[[
        "position_id", "source_game_id", "phase", "ply", "fen", "d1_regret_cp", "d2_regret_cp",
        "d8_regret_cp", "best_regime_regret_cp", "d1_to_d2_improvement_cp", "d1_to_d8_improvement_cp",
        "d1_to_best_regime_improvement_cp", "leader_switches_across_depth", "d4_saturation_fraction_mean",
        "d4_candidate_dispersion", "teacher_best_second_margin_cp",
    ]].copy()
    ranked.sort_values("d1_to_best_regime_improvement_cp", ascending=False).head(25).to_csv(
        args.output / "most_improved_positions.csv", index=False
    )
    ranked.sort_values("d1_to_best_regime_improvement_cp", ascending=True).head(25).to_csv(
        args.output / "most_harmed_positions.csv", index=False
    )
    metadata = {
        "status": "complete",
        "scope": "exploratory position-level analysis",
        "raw_factorial": str(args.raw),
        "positions": int(len(descriptors)),
        "baseline": {"gain_scale": BASELINE_GAIN, "input_mode": BASELINE_MODE},
        "selected_dev_screen_regime": {
            "gain_scale": str(regime["gain_scale"]),
            "input_mode": str(regime["input_mode"]),
            "depth": best_depth,
            "mean_regret_cp": float(regime["regret_cp"]),
        },
        "outputs": [
            "position_descriptors.csv", "descriptor_correlations.csv",
            "most_improved_positions.csv", "most_harmed_positions.csv",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
