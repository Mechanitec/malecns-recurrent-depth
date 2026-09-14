from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.live_state import read_live_state  # noqa: E402
from malecns_rd.position_analysis import (  # noqa: E402
    cp_to_advantage_fraction,
    human_eval,
)

st.set_page_config(page_title="Live Fly Chess Analysis", page_icon="🪰", layout="wide")
st.title("🪰 Live Fly Chess Analysis")
st.caption(
    "Current game, fly-side board marker, neural candidate scores, and an independent full-strength Stockfish evaluation."
)

results_root = Path(st.sidebar.text_input("Results directory", str(ROOT / "results"))).expanduser()
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (s)", 1, 10, 2)

PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

FLY_SVG = """
<svg width="54" height="42" viewBox="0 0 108 84" xmlns="http://www.w3.org/2000/svg" aria-label="fly">
  <ellipse cx="35" cy="25" rx="25" ry="15" fill="#dfe8ef" stroke="#4b5563" stroke-width="3" transform="rotate(-28 35 25)"/>
  <ellipse cx="73" cy="25" rx="25" ry="15" fill="#dfe8ef" stroke="#4b5563" stroke-width="3" transform="rotate(28 73 25)"/>
  <ellipse cx="54" cy="48" rx="17" ry="25" fill="#262626"/>
  <ellipse cx="54" cy="29" rx="14" ry="12" fill="#3b3b3b"/>
  <circle cx="46" cy="24" r="5" fill="#b91c1c"/>
  <circle cx="62" cy="24" r="5" fill="#b91c1c"/>
  <path d="M42 45 L20 36 M42 52 L18 57 M43 59 L24 75 M66 45 L88 36 M66 52 L90 57 M65 59 L84 75" stroke="#262626" stroke-width="4" stroke-linecap="round"/>
  <path d="M49 69 L45 80 M59 69 L63 80" stroke="#262626" stroke-width="4" stroke-linecap="round"/>
</svg>
"""


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
            cells_html.append(
                f'<div style="display:flex;align-items:center;justify-content:center;'
                f'background:{bg};color:#111;font-size:clamp(22px,4vw,48px);'
                f'aspect-ratio:1/1;line-height:1">{escape(piece)}</div>'
            )
    return (
        '<div style="display:grid;grid-template-columns:repeat(8,1fr);'
        'width:min(100%,560px);border:1px solid #777">'
        + "".join(cells_html)
        + "</div>"
    )


def _player_banner(name: str, subtitle: str, *, is_fly: bool) -> str:
    icon = FLY_SVG if is_fly else '<div style="font-size:34px">♟</div>'
    return f"""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 10px;margin:5px 0;
                border:1px solid #d1d5db;border-radius:10px;background:#f8fafc;max-width:560px">
      {icon}
      <div>
        <div style="font-weight:700;font-size:18px">{escape(name)}</div>
        <div style="font-size:13px;color:#64748b">{escape(subtitle)}</div>
      </div>
    </div>
    """


def _advantage_bar(fraction: float, label: str) -> str:
    pct = max(0.0, min(100.0, 100.0 * fraction))
    return f"""
    <div style="max-width:560px;margin:8px 0 14px 0">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#64748b">
        <span>Opponent</span><span>{escape(label)}</span><span>Fly</span>
      </div>
      <div style="height:22px;border-radius:11px;overflow:hidden;border:1px solid #94a3b8;
                  background:#1f2937;position:relative">
        <div style="width:{pct:.2f}%;height:100%;background:#f8fafc"></div>
        <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#ef4444"></div>
      </div>
    </div>
    """


def _live_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("live_state.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)


def _read_eval(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.fragment(run_every=f"{refresh_seconds}s" if auto_refresh else None)
def render_live_analysis() -> None:
    live_paths = _live_paths(results_root)
    if not live_paths:
        st.info("No live benchmark found. Start scripts/run_chess_benchmark.py.")
        return

    labels = [str(p.relative_to(results_root)) for p in live_paths]
    selected = st.selectbox("Live run", labels, index=len(labels) - 1)
    live_path = live_paths[labels.index(selected)]
    run_dir = live_path.parent
    state = read_live_state(live_path)
    if state is None:
        st.warning("Waiting for a readable live_state.json snapshot.")
        return

    eval_state = _read_eval(run_dir / "position_eval.json")
    eval_history_path = run_dir / "evaluation_history.csv"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", state.status.upper())
    c2.metric("Games", f"{state.games_completed}/{state.games_total}")
    c3.metric("W / D / L", f"{state.wins} / {state.draws} / {state.losses}")
    opponent_value = state.opponent_calibrated_elo if state.opponent_calibrated_elo is not None else state.opponent_elo
    opponent_detail = f"{opponent_value:.1f}" if opponent_value is not None else "—"
    if state.opponent_setting:
        opponent_detail = f"{opponent_detail} · {state.opponent_setting}"
    c4.metric("Opponent", f"{state.opponent_engine or '—'} {opponent_detail}")
    c5.metric("Depth", state.recurrent_depth if state.recurrent_depth is not None else "—")

    left, right = st.columns([1.12, 0.88])
    with left:
        st.subheader("Board")
        if state.fen:
            orientation = "black" if state.fly_color == "black" else "white"
            opponent_color = "white" if state.fly_color == "black" else "black"
            top_name = f"Opponent — {opponent_color.title()}"
            bottom_name = f"Fly — {state.fly_color.title() if state.fly_color else 'Unknown'}"
            opponent_label = f"{state.opponent_engine or 'engine'} {state.opponent_elo if state.opponent_elo is not None else ''}"
            st.markdown(_player_banner(top_name, opponent_label, is_fly=False), unsafe_allow_html=True)
            st.markdown(_fen_board_html(state.fen, orientation), unsafe_allow_html=True)
            st.markdown(_player_banner(bottom_name, "MaleCNS-RD", is_fly=True), unsafe_allow_html=True)
            detail = []
            if state.ply:
                detail.append(f"Ply **{state.ply}**")
            if state.last_move:
                detail.append(f"Last move **{state.last_move}**")
            if state.last_actor:
                detail.append(f"by **{state.last_actor}**")
            st.markdown(" · ".join(detail))
        else:
            st.info("Waiting for the first board position.")

    with right:
        st.subheader("Independent Stockfish position evaluation")
        if eval_state:
            fly_cp = eval_state.get("fly_cp")
            fly_mate = eval_state.get("fly_mate")
            depth = eval_state.get("analysis_depth")
            label = human_eval(
                int(fly_cp) if fly_cp is not None else None,
                int(fly_mate) if fly_mate is not None else None,
                side="Fly",
            )
            if fly_mate is not None:
                metric = f"Mate {int(fly_mate):+d}"
            elif fly_cp is not None:
                metric = f"{float(fly_cp) / 100.0:+.2f}"
            else:
                metric = "—"
            a1, a2 = st.columns(2)
            a1.metric("Fly-perspective eval", metric)
            a2.metric("Analysis depth", depth if depth is not None else "—")
            st.markdown(
                _advantage_bar(
                    cp_to_advantage_fraction(
                        float(fly_cp) if fly_cp is not None else None,
                        int(fly_mate) if fly_mate is not None else None,
                    ),
                    label,
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "Evaluation is produced by a separate full-strength Stockfish process. "
                "The bar is a display mapping, not a calibrated win probability."
            )
        else:
            st.info("Position evaluation not available yet.")

        st.subheader("Fly candidate moves")
        if state.candidate_scores:
            candidates = pd.DataFrame([{"move": x.move, "score": x.score} for x in state.candidate_scores])
            st.altair_chart(
                alt.Chart(candidates).mark_bar().encode(
                    x=alt.X("score:Q", title="Neural score"),
                    y=alt.Y("move:N", sort="-x", title=None),
                    tooltip=["move", alt.Tooltip("score:Q", format=".5f")],
                ),
                use_container_width=True,
            )
        else:
            st.caption("Candidate scores appear after a MaleCNS-RD move.")

    if eval_history_path.exists() and state.game_index is not None:
        try:
            hist = pd.read_csv(eval_history_path)
        except (OSError, ValueError):
            hist = pd.DataFrame()
        if not hist.empty and "game_index" in hist.columns:
            hist = hist[hist["game_index"] == state.game_index].copy()
            if not hist.empty:
                hist["eval_pawns"] = pd.to_numeric(hist["fly_cp"], errors="coerce") / 100.0
                mate = pd.to_numeric(hist["fly_mate"], errors="coerce")
                hist.loc[mate > 0, "eval_pawns"] = 12.0
                hist.loc[mate < 0, "eval_pawns"] = -12.0
                hist["eval_pawns"] = hist["eval_pawns"].clip(-12, 12)
                st.subheader("Evaluation through the current game")
                chart = alt.Chart(hist).mark_line(point=True).encode(
                    x=alt.X("ply:Q", title="Ply"),
                    y=alt.Y("eval_pawns:Q", title="Fly advantage (pawns; mate clipped at ±12)"),
                    tooltip=["ply", "last_move", "last_actor", "fly_cp", "fly_mate"],
                )
                zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
                st.altair_chart(chart + zero, use_container_width=True)

    if state.updated_at_utc:
        st.caption(f"Live state updated: {state.updated_at_utc} UTC")


render_live_analysis()
