# Live Stockfish position evaluation

The chess benchmark can run a second, independent Stockfish process as an observer while the fly plays Alfil or limited-strength Stockfish.

This analysis process **never chooses a game move**. It evaluates the position after the initial setup and after every move, then writes:

- `position_eval.json` — latest evaluation snapshot;
- `evaluation_history.csv` — per-ply evaluation history;
- `live_state.json` — existing benchmark telemetry.

## Why use a separate engine?

The playing opponent must remain at its requested nominal Elo. Reusing the opponent process for full-strength analysis risks changing its state/options or leaking stronger search into move selection. The observer therefore launches a separate Stockfish process with `UCI_LimitStrength=false`.

A fixed analysis depth is used instead of analysis time so runs are easier to reproduce. The default is depth 18 with one thread.

## Run

```bash
python scripts/run_chess_benchmark.py \
  --stockfish /path/to/stockfish \
  --alfil /path/to/alfil \
  --elos 0,200,400,600,800,1000,1200,1320,1400 \
  --games-per-elo 20 \
  --analysis-depth 18 \
  --output results/fly_depth16
```

Use a different Stockfish binary for analysis if desired:

```bash
--analysis-stockfish /path/to/full-strength-stockfish
```

Disable position evaluation with `--no-position-eval`.

## Evaluation convention

The raw Stockfish score is persisted from White's point of view and also transformed into a fly-perspective score:

- positive `fly_cp`: position favors the fly;
- negative `fly_cp`: position favors the opponent;
- positive `fly_mate`: fly has forced mate;
- negative `fly_mate`: opponent has forced mate.

The dashboard's advantage bar uses a bounded `tanh` display transform. It is **not** a calibrated win probability.

## Dashboard

Streamlit automatically exposes `dashboard/pages/1_Live_Analysis.py` as a live-analysis page. It shows:

- the current board oriented from the fly's side;
- a vector fly icon next to the fly's board side;
- opponent identity and nominal Elo;
- full-strength Stockfish evaluation from the fly's perspective;
- an advantage bar and per-ply evaluation chart;
- latest MaleCNS candidate move scores;
- game and tournament progress.

The fly graphic is inline SVG code stored with the dashboard. No generated image or external asset is required.
