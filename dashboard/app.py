from __future__ import annotations

from pathlib import Path
import sys

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

st.set_page_config(page_title="MaleCNS-RD Chess Lab", page_icon="♟", layout="wide")

st.title("MaleCNS-RD Chess Lab")
st.caption("Visual benchmark and training dashboard for the recurrent-depth fly chess project.")

results_root = Path(st.sidebar.text_input("Results directory", str(ROOT / "results"))).expanduser()
st.sidebar.caption("The dashboard auto-discovers benchmark runs containing games.csv + elo.json.")

benchmark_runs = discover_benchmark_runs(results_root)
training_paths = discover_training_histories(results_root)

benchmark_tab, training_tab, experiment_tab = st.tabs(["Benchmark", "Training", "Experiment"])

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
    st.subheader("Benchmark design")
    st.markdown(
        "- **Alfil** is used for nominal Elo levels below the detected Stockfish floor.\n"
        "- **Stockfish** is used at and above its runtime-reported minimum.\n"
        "- Each game stores opponent engine, nominal Elo, color, result, plies, termination, and final FEN.\n"
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
