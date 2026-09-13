# Chess benchmark and training dashboard

The project includes a Streamlit dashboard for saved benchmark runs, live chess games, and training histories.

## Install and launch

```bash
pip install -e '.[dashboard]'
streamlit run dashboard/app.py
```

By default the app scans `results/`. The directory can be changed from the sidebar.

## Live tab

The benchmark process writes an atomic `live_state.json` after moves and completed games. The dashboard reads that file independently, so the chess engines and UI do not share process state.

`run_chess_benchmark.py` writes `<output>/live_state.json` by default. The path can be overridden with `--live-state`.

The live view shows:

- tournament status and progress;
- rolling Elo and confidence interval;
- wins/draws/losses;
- current opponent and nominal Elo;
- recurrent depth;
- current chess position, oriented from the fly's side;
- last move and actor;
- the latest fly candidate-move neural scores;
- latest training step/loss/Elo when a training history is present.

The live panel auto-refreshes using Streamlit fragments. Refresh can be disabled or slowed from the sidebar.

Candidate rankings are persisted across the opponent's reply so a human polling the dashboard every few seconds can still inspect the fly's most recent decision.

## Benchmark tab

A completed benchmark run is discovered when one directory contains both:

- `games.csv`
- `elo.json`

The dashboard shows:

- estimated Elo and 95% interval;
- total games and aggregate score;
- detected Stockfish Elo floor;
- score by opponent rating;
- Alfil vs Stockfish split;
- win/draw/loss distribution;
- Elo progression across saved runs;
- a filterable game log.

Older game logs without `opponent_engine` remain readable and are treated as Stockfish-only runs.

## Training tab

Training histories are discovered from files named `training_history.csv` or `training*.csv` under `results/`.

Only `step` is mandatory. Supported fields are:

- `epoch`
- `loss`
- `validation_loss`
- `teacher_agreement`
- `candidate_accuracy`
- `elo`
- `elo_ci_low`
- `elo_ci_high`
- `recurrent_depth`
- `learning_rate`
- `checkpoint`

Training code can append rows with the built-in logger:

```python
from malecns_rd.training_log import TrainingMetric, append_training_metric

append_training_metric(
    "results/run_001/training_history.csv",
    TrainingMetric(
        step=100,
        loss=0.81,
        validation_loss=0.86,
        teacher_agreement=0.42,
        candidate_accuracy=0.39,
        elo=240,
        elo_ci_low=205,
        elo_ci_high=275,
        recurrent_depth=16,
        learning_rate=1e-3,
        checkpoint="step_0100",
    ),
)
```

The training tab plots optimization loss, held-out loss, move-quality metrics, and periodic measured chess Elo. This keeps visual reporting tied to saved experiment artifacts rather than ephemeral UI state.

## Example live benchmark

```bash
python scripts/run_chess_benchmark.py \
  --alfil /path/to/alfil \
  --stockfish /path/to/stockfish \
  --elos 0,200,400,600,800,1000,1200,1320,1400 \
  --games-per-elo 20 \
  --output results/depth16
```

Then run the dashboard in a second terminal. The live tab will discover `results/depth16/live_state.json` automatically.

## Experimental interpretation

The dashboard is a reporting surface, not an optimizer. For recurrent-depth comparisons, keep the graph, chess adapter, readout, training checkpoint, opening suite, opponent binaries, and time controls fixed while changing the tested variable.
