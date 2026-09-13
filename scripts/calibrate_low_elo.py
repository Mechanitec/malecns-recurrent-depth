"""Fit the Minic low-level ladder onto MaleCNS Chess Rating (MCR)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from malecns_rd.low_elo_calibration import calibrate, schedule_frame


def _parse_label(label: str) -> tuple[str, int]:
    kind, _, raw = str(label).lower().partition("_")
    if kind not in {"minic", "gaia"} or not raw:
        raise ValueError(f"unsupported calibration engine label: {label!r}")
    value = int(raw)
    if kind == "minic" and not 0 <= value <= 100:
        raise ValueError("Minic level must be in [0, 100]")
    if kind == "gaia" and value != 580:
        raise ValueError("calibration currently uses only Gaia 580 as the high anchor")
    return kind, value


def _opponent(label: str, minic: str, gaia: str, search_depth: int):
    from malecns_rd.fast_opponents import GaiaConfig, GaiaOpponent, MinicConfig, MinicOpponent

    kind, value = _parse_label(label)
    if kind == "minic":
        return MinicOpponent(
            MinicConfig(minic, level=value, nominal_elo=0, search_depth=search_depth)
        )
    return GaiaOpponent(GaiaConfig(gaia, rating=580, search_depth=search_depth))


def _play_game(white_engine, black_engine, fen: str, max_plies: int):
    from malecns_rd.chess_benchmark import _require_chess

    chess = _require_chess()
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = engine.choose_move(board)
        if move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal move: {move}")
        board.push(move)
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None
        result, termination = outcome.result(), outcome.termination.name
    else:
        result, termination = "1/2-1/2", "MAX_PLIES_ADJUDICATION"
    white_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[result]
    return result, white_score, termination, board.ply(), board.fen()


def _run_schedule(args) -> pd.DataFrame:
    from malecns_rd.chess_benchmark import _require_chess

    chess = _require_chess()
    schedule = pd.read_csv(args.schedule).sort_values(["matchup_id", "game_index"])
    required = {"game_index", "matchup_id", "opening_id", "pair_id", "white", "black"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing columns: {sorted(missing)}")

    if args.openings_fen:
        openings = [
            line.strip()
            for line in args.openings_fen.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        openings = [chess.STARTING_FEN]
    if not openings:
        raise ValueError("opening FEN file is empty")
    for fen in openings:
        chess.Board(fen)

    previous = pd.DataFrame()
    completed: set[int] = set()
    if args.output.exists() and not args.no_resume:
        previous = pd.read_csv(args.output)
        completed = set(previous["game_index"].astype(int))
    new_rows: list[dict[str, object]] = []

    for _, group in schedule.groupby("matchup_id", sort=False):
        labels = sorted(set(group["white"].astype(str)) | set(group["black"].astype(str)))
        if len(labels) != 2:
            raise ValueError("each matchup must contain exactly two engines")
        pending = group[~group["game_index"].astype(int).isin(completed)]
        if pending.empty:
            continue
        a = _opponent(labels[0], args.minic, args.gaia, args.search_depth)
        try:
            b = _opponent(labels[1], args.minic, args.gaia, args.search_depth)
            try:
                engines = {labels[0]: a, labels[1]: b}
                for _, row in pending.iterrows():
                    fen = openings[int(row["opening_id"]) % len(openings)]
                    result, score, termination, plies, final_fen = _play_game(
                        engines[str(row["white"])],
                        engines[str(row["black"])],
                        fen,
                        args.max_plies,
                    )
                    record = dict(row)
                    record.update(
                        start_fen=fen,
                        result=result,
                        white_score=score,
                        termination=termination,
                        plies=plies,
                        final_fen=final_fen,
                    )
                    new_rows.append(record)
                    completed.add(int(row["game_index"]))
                    parts = ([previous] if not previous.empty else []) + (
                        [pd.DataFrame(new_rows)] if new_rows else []
                    )
                    merged = pd.concat(parts, ignore_index=True).drop_duplicates(
                        "game_index", keep="last"
                    )
                    merged = merged.sort_values("game_index")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
                    merged.to_csv(tmp, index=False)
                    tmp.replace(args.output)
            finally:
                b.close()
        finally:
            a.close()

    parts = ([previous] if not previous.empty else []) + (
        [pd.DataFrame(new_rows)] if new_rows else []
    )
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates("game_index", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("schedule", help="Create the color-balanced match schedule")
    make.add_argument("--openings", type=int, default=40)
    make.add_argument("--max-minic-level", type=int, default=30)
    make.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="Execute a Minic/Gaia calibration schedule")
    run.add_argument("--schedule", type=Path, required=True)
    run.add_argument("--minic", required=True, help="Path to Minic executable")
    run.add_argument("--gaia", required=True, help="Path to Gaia executable")
    run.add_argument(
        "--openings-fen", type=Path, default=None, help="Text file containing one FEN per line"
    )
    run.add_argument("--search-depth", type=int, default=64)
    run.add_argument("--max-plies", type=int, default=500)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--no-resume", action="store_true")

    fit = sub.add_parser("fit", help="Fit a calibration from completed games")
    fit.add_argument("--games", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--bootstrap", type=int, default=500)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--low-anchor", default="minic_0")
    fit.add_argument("--high-anchor", default="gaia_580")
    fit.add_argument("--high-rating", type=float, default=580.0)
    args = parser.parse_args()

    if args.command == "schedule":
        frame = schedule_frame(openings=args.openings, max_minic_level=args.max_minic_level)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"matchups: {frame['matchup_id'].nunique()}")
        print(f"opening pairs: {frame['pair_id'].nunique()}")
        print(f"games: {len(frame)}")
        print(f"saved: {args.output}")
        return

    if args.command == "run":
        games = _run_schedule(args)
        schedule = pd.read_csv(args.schedule)
        print(f"completed games: {len(games)} / {len(schedule)}")
        print(f"saved: {args.output}")
        return

    games = pd.read_csv(args.games)
    result = calibrate(
        games,
        low_anchor=args.low_anchor,
        high_anchor=args.high_anchor,
        high_rating=args.high_rating,
        bootstrap_samples=args.bootstrap,
        bootstrap_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.ratings.to_csv(args.output, index=False)
    meta = {
        "low_anchor": args.low_anchor,
        "high_anchor": args.high_anchor,
        "high_rating": args.high_rating,
        "games": int(len(games)),
        "opening_pairs": int(games["pair_id"].nunique()) if "pair_id" in games else None,
        "white_advantage_elo": result.ratings.attrs.get("white_advantage_elo"),
        "bootstrap_samples_requested": args.bootstrap,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(result.ratings.to_string(index=False))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
