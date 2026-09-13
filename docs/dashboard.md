# Chess benchmark and training dashboard

The project includes a Streamlit dashboard for saved chess benchmark runs and training histories.

## Install and launch

```bash
pip install -e '.[dashboard]'
streamlit run dashboard/app.py
```

By default the app scans `results/`. The directory can be changed from the sidebar.

## Benchmark tab

A benchmark run is discovered when one directory contains both:

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

## Experimental interpretation

The dashboard is a reporting surface, not an optimizer. For recurrent-depth comparisons, keep the graph, chess adapter, readout, training checkpoint, opening suite, opponent binaries, and time controls fixed while changing the tested variable.
