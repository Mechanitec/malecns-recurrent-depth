from __future__ import annotations

from html import escape
from pathlib import Path
import sys
import json

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.dashboard_data import (  # noqa: E402
    benchmark_history_frame,
    discover_benchmark_runs,
    discover_training_histories,
    load_games,
    load_training_history,
    result_distribution,
    score_by_opponent,
)
from malecns_rd.live_state import read_live_state  # noqa: E402

st.set_page_config(page_title="MaleCNS-RD Chess Lab", page_icon="♟", layout="wide")

st.title("MaleCNS-RD Chess Lab")
st.caption("Visual benchmark, live-game, and training dashboard for the recurrent-depth fly chess project.")

results_root = Path(st.sidebar.text_input("Results directory", str(ROOT / "results"))).expanduser()
st.sidebar.caption("Benchmark runs are discovered from games.csv + elo.json; live runs use live_state.json.")
auto_refresh = st.sidebar.toggle("Auto-refresh live view", value=True)
refresh_seconds = st.sidebar.slider("Live refresh interval (s)", 1, 10, 2)

benchmark_runs = discover_benchmark_runs(results_root)
training_paths = discover_training_histories(results_root)

live_tab, benchmark_tab, training_tab, experiment_tab = st.tabs(
    ["Live", "Benchmark", "Training", "Experiment"]
)


PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


def _fen_board_html(fen: str, orientation: str = "white") -> str:
    try:
        ranks = fen.split()[0].split("/")
        board: list[list[str]] = []
        for rank in ranks:
            cells: list[str] = []
            for ch in rank:
                if ch.isdigit():
                    cells.extend([""] * int(ch))
                else:
                    cells.append(PIECES.get(ch, ""))
            if len(cells) != 8:
                raise ValueError
            board.append(cells)
        if len(board) != 8:
            raise ValueError
    except (IndexError, ValueError):
        return "<div>Invalid FEN</div>"

    if orientation == "black":
        board = [list(reversed(row)) for row in reversed(board)]

    cells_html: list[str] = []
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            light = (r + c) % 2 == 0
            bg = "#e8e8e8" if light else "#777777"
            fg = "#111111"
            cells_html.append(
                f'<div style="display:flex;align-items:center;justify-content:center;'
                f'background:{bg};color:{fg};font-size:clamp(22px,4vw,48px);'
                f'aspect-ratio:1/1;line-height:1">{escape(piece)}</div>'
            )
    return (
        '<div style="display:grid;grid-template-columns:repeat(8,1fr);'
        'width:min(100%,560px);border:1px solid #777">'
        + "".join(cells_html)
        + "</div>"
    )


def _live_state_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        root.rglob("live_state.json"),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, str(p)),
    )


def _elo_text(value: float | None, censored: str | None) -> str:
    if value is None:
        return "—"
    prefix = ""
    if censored == "below":
        prefix = "< "
    elif censored == "above":
        prefix = "> "
    return f"{prefix}{value:.0f}"


with live_tab:
    st.caption("The benchmark process writes an atomic JSON snapshot after each move; this panel only reads it.")

    @st.fragment(run_every=f"{refresh_seconds}s" if auto_refresh else None)
    def render_live() -> None:
        paths = _live_state_paths(results_root)
        if not paths:
            st.info(
                "No live_state.json found yet. Start scripts/run_chess_benchmark.py; "
                "it writes <output>/live_state.json automatically."
            )
            return

        labels = [str(p.relative_to(results_root)) for p in paths]
        selected = st.selectbox("Live run", labels, index=len(labels) - 1, key="live-run-select")
        state_path = paths[labels.index(selected)]
        state = read_live_state(state_path)
        if state is None:
            st.warning("Live state is not readable yet; waiting for the next atomic update.")
            return

        status_text = state.status.upper()
        if state.status == "completed":
            st.success(f"{status_text}: {state.message or ''}")
        elif state.status == "error":
            st.error(f"{status_text}: {state.message or ''}")
        elif state.status == "stopped":
            st.warning(f"{status_text}: {state.message or ''}")
        else:
            st.info(f"{status_text}: {state.message or ''}")

        progress = (
            state.games_completed / state.games_total
            if state.games_total > 0
            else 0.0
        )
        st.progress(min(max(progress, 0.0), 1.0))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Rolling Elo", _elo_text(state.rolling_elo, state.rolling_censored))
        if state.rolling_ci_low is not None and state.rolling_ci_high is not None:
            ci = f"{state.rolling_ci_low:.0f} – {state.rolling_ci_high:.0f}"
        else:
            ci = "—"
        c2.metric("95% interval", ci)
        c3.metric("Games", f"{state.games_completed}/{state.games_total}")
        c4.metric("W / D / L", f"{state.wins} / {state.draws} / {state.losses}")
        opponent_value = state.opponent_calibrated_elo if state.opponent_calibrated_elo is not None else state.opponent_elo
        opp = "—" if opponent_value is None else f"{state.opponent_engine} {opponent_value:.1f}"
        if state.opponent_setting:
            opp = f"{opp} · {state.opponent_setting}"
        c5.metric("Opponent", opp)
        c6.metric("Recurrent depth", state.recurrent_depth if state.recurrent_depth is not None else "—")

        left, right = st.columns([1.05, 1.0])
        with left:
            st.subheader("Current position")
            if state.fen:
                orientation = "black" if state.fly_color == "black" else "white"
                st.markdown(_fen_board_html(state.fen, orientation), unsafe_allow_html=True)
                details = []
                if state.fly_color:
                    details.append(f"Fly: **{state.fly_color}**")
                if state.ply:
                    details.append(f"Ply: **{state.ply}**")
                if state.last_move:
                    details.append(f"Last move: **{state.last_move}**")
                if state.last_actor:
                    details.append(f"Actor: **{state.last_actor}**")
                st.markdown(" · ".join(details))
                st.caption(state.fen)
            else:
                st.info("Waiting for the first board position.")

        with right:
            st.subheader("Fly candidate moves")
            if state.candidate_scores:
                candidates = pd.DataFrame(
                    [{"move": x.move, "score": x.score} for x in state.candidate_scores]
                )
                chart = (
                    alt.Chart(candidates)
                    .mark_bar()
                    .encode(
                        x=alt.X("score:Q", title="Neural candidate score"),
                        y=alt.Y("move:N", sort="-x", title=None),
                        tooltip=["move", alt.Tooltip("score:Q", format=".5f")],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Candidate scores appear immediately after a FlyCandidateMoveAgent move.")

            st.subheader("Latest training")
            live_training_paths = discover_training_histories(results_root)
            if live_training_paths:
                latest_path = live_training_paths[-1]
                history = load_training_history(latest_path)
                if not history.empty:
                    last = history.iloc[-1]
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Step", int(last["step"]))
                    t2.metric(
                        "Loss",
                        f"{float(last['loss']):.4f}" if pd.notna(last["loss"]) else "—",
                    )
                    t3.metric(
                        "Training Elo",
                        f"{float(last['elo']):.0f}" if pd.notna(last["elo"]) else "—",
                    )
                    st.caption(str(latest_path.relative_to(results_root)))
            else:
                st.caption("No training_history.csv found yet.")

        if state.updated_at_utc:
            st.caption(f"Last live update: {state.updated_at_utc} UTC")

    render_live()


with benchmark_tab:
    if not benchmark_runs:
        st.info("No saved chess benchmark runs found yet. Run scripts/run_chess_benchmark.py and save output under results/.")
    else:
        labels = [r.run_id for r in benchmark_runs]
        selected_label = st.selectbox("Benchmark run", labels, index=len(labels) - 1)
        run = benchmark_runs[labels.index(selected_label)]
        games = load_games(run)

        elo_text = f"{run.elo:.0f}"
        if run.censored:
            elo_text = f"{run.censored} {run.elo:.0f}"
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Estimated Elo", elo_text)
        c2.metric("95% interval", f"{run.ci95_low:.0f} – {run.ci95_high:.0f}")
        c3.metric("Games", f"{run.n_games}")
        c4.metric("Score", f"{100 * run.score:.1f}%")
        c5.metric("Stockfish floor", f"{run.stockfish_floor if run.stockfish_floor is not None else '—'}")

        left, right = st.columns([1.45, 1.0])
        with left:
            st.subheader("Score by opponent")
            by_opp = score_by_opponent(games)
            if not by_opp.empty:
                chart = (
                    alt.Chart(by_opp)
                    .mark_bar()
                    .encode(
                        x=alt.X("opponent_elo:O", title="Opponent nominal Elo"),
                        y=alt.Y("score:Q", title="Fly score", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("opponent_engine:N", title="Engine"),
                        tooltip=["opponent_engine", "opponent_elo", "games", alt.Tooltip("score:Q", format=".1%")],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
        with right:
            st.subheader("Results")
            dist = result_distribution(games)
            chart = (
                alt.Chart(dist)
                .mark_bar()
                .encode(
                    x=alt.X("outcome:N", sort=["Win", "Draw", "Loss"], title=None),
                    y=alt.Y("games:Q", title="Games"),
                    tooltip=["outcome", "games"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

        history = benchmark_history_frame(benchmark_runs)
        if len(history) >= 2:
            st.subheader("Elo progression across benchmark runs")
            history = history.reset_index(names="run_number")
            line = (
                alt.Chart(history)
                .mark_line(point=True)
                .encode(
                    x=alt.X("run_number:Q", title="Run"),
                    y=alt.Y("elo:Q", title="Estimated Elo"),
                    tooltip=["run_id", "elo", "ci95_low", "ci95_high", "games", alt.Tooltip("score:Q", format=".1%")],
                )
            )
            band = (
                alt.Chart(history)
                .mark_area(opacity=0.18)
                .encode(x="run_number:Q", y="ci95_low:Q", y2="ci95_high:Q")
            )
            st.altair_chart(band + line, use_container_width=True)

        st.subheader("Game log")
        filters = st.columns(3)
        engine_values = ["All"] + sorted(str(x) for x in games["opponent_engine"].dropna().unique())
        selected_engine = filters[0].selectbox("Engine", engine_values)
        color_values = ["All"] + sorted(str(x) for x in games["fly_color"].dropna().unique())
        selected_color = filters[1].selectbox("Fly color", color_values)
        result_values = ["All"] + sorted(str(x) for x in games["result"].dropna().unique())
        selected_result = filters[2].selectbox("Result", result_values)
        shown = games.copy()
        if selected_engine != "All":
            shown = shown[shown["opponent_engine"] == selected_engine]
        if selected_color != "All":
            shown = shown[shown["fly_color"] == selected_color]
        if selected_result != "All":
            shown = shown[shown["result"] == selected_result]
        st.dataframe(shown, use_container_width=True, hide_index=True)

with training_tab:
    if not training_paths:
        st.info(
            "No training history found. Training code can append dashboard-compatible rows with "
            "malecns_rd.training_log.append_training_metric(...)."
        )
        st.code(
            "from malecns_rd.training_log import TrainingMetric, append_training_metric\n"
            "append_training_metric('results/run_001/training_history.csv', TrainingMetric(\n"
            "    step=100, loss=0.81, teacher_agreement=0.42, elo=240, recurrent_depth=16\n"
            "))",
            language="python",
        )
    else:
        labels = [str(p.relative_to(results_root)) if p.is_relative_to(results_root) else str(p) for p in training_paths]
        selected = st.selectbox("Training history", labels, index=len(labels) - 1)
        training_path = training_paths[labels.index(selected)]
        history = load_training_history(training_path)

        last = history.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Step", int(last["step"]))
        c2.metric("Loss", f"{float(last['loss']):.4f}" if pd.notna(last["loss"]) else "—")
        c3.metric("Teacher agreement", f"{100 * float(last['teacher_agreement']):.1f}%" if pd.notna(last["teacher_agreement"]) else "—")
        c4.metric("Candidate accuracy", f"{100 * float(last['candidate_accuracy']):.1f}%" if pd.notna(last["candidate_accuracy"]) else "—")
        c5.metric("Checkpoint Elo", f"{float(last['elo']):.0f}" if pd.notna(last["elo"]) else "—")

        metric_cols = [c for c in ["loss", "validation_loss"] if history[c].notna().any()]
        if metric_cols:
            st.subheader("Loss")
            melted = history[["step"] + metric_cols].melt("step", var_name="metric", value_name="value").dropna()
            st.altair_chart(
                alt.Chart(melted).mark_line().encode(
                    x=alt.X("step:Q", title="Training step"),
                    y=alt.Y("value:Q", title="Loss"),
                    color=alt.Color("metric:N", title=None),
                    tooltip=["step", "metric", "value"],
                ),
                use_container_width=True,
            )

        quality_cols = [c for c in ["teacher_agreement", "candidate_accuracy"] if history[c].notna().any()]
        if quality_cols:
            st.subheader("Move-quality learning")
            melted = history[["step"] + quality_cols].melt("step", var_name="metric", value_name="value").dropna()
            st.altair_chart(
                alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X("step:Q", title="Training step"),
                    y=alt.Y("value:Q", title="Fraction", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", title=None),
                    tooltip=["step", "metric", alt.Tooltip("value:Q", format=".1%")],
                ),
                use_container_width=True,
            )

        if history["elo"].notna().any():
            st.subheader("Chess Elo during training")
            elo_df = history[history["elo"].notna()].copy()
            line = alt.Chart(elo_df).mark_line(point=True).encode(
                x=alt.X("step:Q", title="Training step"),
                y=alt.Y("elo:Q", title="Estimated Elo"),
                tooltip=["step", "elo", "elo_ci_low", "elo_ci_high", "checkpoint", "recurrent_depth"],
            )
            if elo_df["elo_ci_low"].notna().any() and elo_df["elo_ci_high"].notna().any():
                band = alt.Chart(elo_df).mark_area(opacity=0.18).encode(
                    x="step:Q", y="elo_ci_low:Q", y2="elo_ci_high:Q"
                )
                st.altair_chart(band + line, use_container_width=True)
            else:
                st.altair_chart(line, use_container_width=True)

        st.subheader("Training log")
        st.dataframe(history, use_container_width=True, hide_index=True)

with experiment_tab:
    summary_path = results_root / "final_experiment_summary.json"
    if summary_path.exists():
        try:
            st.subheader("Experiment summary")
            st.json(json.loads(summary_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            st.warning("Final experiment summary is not readable yet.")

    depth_path = results_root / "depth_sweep" / "metrics.csv"
    control_path = results_root / "control_sweep" / "metrics.csv"
    if depth_path.exists():
        depth_frame = pd.read_csv(depth_path)
        st.subheader("Frozen-checkpoint depth sweep")
        st.altair_chart(
            alt.Chart(depth_frame).mark_line(point=True).encode(
                x=alt.X("depth:Q", title="Recurrent depth"),
                y=alt.Y("rating:Q", title="Diagnostic rating"),
                tooltip=["depth", "rating", "rating_ci_low", "rating_ci_high", "latency_s", "recurrent_passes"],
            ),
            use_container_width=True,
        )
        st.dataframe(depth_frame, use_container_width=True, hide_index=True)
    if control_path.exists():
        control_frame = pd.read_csv(control_path)
        st.subheader("Connectome controls")
        st.altair_chart(
            alt.Chart(control_frame).mark_line(point=True).encode(
                x=alt.X("depth:Q", title="Recurrent depth"),
                y=alt.Y("teacher_agreement:Q", title="Teacher-best agreement", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("variant:N", title="Graph variant"),
                tooltip=["variant", "depth", "teacher_agreement", "quality_loss_cp", "latency_s"],
            ),
            use_container_width=True,
        )
        st.dataframe(control_frame, use_container_width=True, hide_index=True)

    st.subheader("Benchmark design")
    st.markdown(
        "- **Measured Minic/Gaia** settings are used below the detected Stockfish floor; legacy Alfil remains optional.\n"
        "- **Stockfish** is used at and above its runtime-reported minimum.\n"
        "- The **Live** tab reads an atomic `live_state.json` written after each move/game.\n"
        "- Each saved game stores opponent engine, nominal Elo, color, result, plies, termination, and final FEN.\n"
        "- Recurrent-depth experiments should freeze graph, adapter, readout, training checkpoint, opening suite, and engine settings."
    )
    st.subheader("Recommended training dashboard fields")
    st.dataframe(
        pd.DataFrame(
            [
                ["loss", "Optimization objective"],
                ["validation_loss", "Held-out objective"],
                ["teacher_agreement", "Agreement with teacher engine move"],
                ["candidate_accuracy", "Correct candidate ranking rate"],
                ["elo", "Periodic measured chess strength"],
                ["recurrent_depth", "Depth used by checkpoint"],
                ["learning_rate", "Optimizer learning rate"],
                ["checkpoint", "Saved model/checkpoint identifier"],
            ],
            columns=["Field", "Meaning"],
        ),
        use_container_width=True,
        hide_index=True,
    )
