# MaleCNS recurrent-depth research prototype

This repository studies whether a real biological connectome can function as a **shared recurrent computational block** whose useful capability changes with internal depth.

The core computation is:

```text
state[d+1] = F(state[d], sensory, W_connectome)
```

The same graph and same dynamics are reused at every recurrent depth. In the primary frozen-fly experiment, graph, dynamics, sensory mapping, readout population and trained readout checkpoint stay fixed; only recurrent depth changes.

The locked scientific scope is in `Research Goal.md`. The current Codex execution plan is in `plan.md`.

## Implemented system

The repository now includes:

- sparse signed directed `ConnectomeGraph`;
- rate-state and LIF recurrent engines;
- fixed and adaptive recurrent depth;
- MaleCNS v1.0 Feather loader with transmitter-based sign handling;
- deterministic chess board+candidate encoding and sensory projection;
- reproducible sensory/readout population selection;
- Stockfish teacher-data generation;
- activation caching on the real MaleCNS graph;
- linear readout training and reloadable checkpoints;
- full-legal-move `FlyCandidateMoveAgent` inference;
- calibrated low-strength Minic/Gaia opponent support, with Stockfish above its supported floor;
- independent Stockfish position evaluation that never chooses moves for either player;
- live Streamlit board/evaluation dashboard;
- recurrent-depth sweeps and matched graph controls;
- position-quality metrics, short-game diagnostic ratings, PGN/telemetry output and experiment metadata.

Alfil remains legacy/optional support only; it is not the preferred high-throughput weak-opponent path.

## Current scientific status

The full real-connectome path is working. A v1 rate-dynamics checkpoint trained at reference depth 16 has been evaluated at depths:

```text
1, 2, 4, 8, 16, 32, 64
```

On the first 8 held-out all-legal positions, the original MaleCNS graph showed a large D1 -> D2/D8 reduction in teacher centipawn regret, but the sample is too small for a strong conclusion. Matched shuffled/attenuated controls sometimes perform as well as or better than the original graph. Therefore the present evidence supports at most a **candidate generic recurrent-computation effect**, not yet a MaleCNS-specific biological advantage.

See `docs/current_results.md` for the exact numbers and interpretation boundary.

The existing 402-game serious-run artifact is a **short-game protocol**, not a real long-game Elo result. Those games were capped after two plies and unresolved games were adjudicated as draws, so the reported ~410 value must not be interpreted as the chess rating of the fly system.

## Current priority

The highest-value next experiment is a much larger paired held-out position study using the exact same checkpoint and graph controls. `plan.md` instructs Codex to:

1. optimize all-legal position evaluation without changing the mathematics;
2. freeze an independent 128-512 position evaluation corpus;
3. run all depths on original + three graph controls;
4. compute paired bootstrap confidence intervals and difference-in-differences;
5. diagnose the weak v1 readout;
6. optionally train a stronger Stage-0 readout using only train/validation data.

## Dynamics

### Rate-state engine

`RecurrentDepthEngine` performs sparse signed propagation and a recurrent state update using the same graph at every pass.

### LIF engine

`LIFRecurrentDepthEngine` includes membrane decay, signed recurrent spike drive, threshold/reset behavior, optional refractory steps, spike-count readout and adaptive stopping.

For LIF, recurrent depth is also simulated biological integration time, so matched-time / single-pulse controls are required before interpreting deeper LIF runs as more computation rather than simply more time.

## Real MaleCNS data

Expected official files:

```text
data/body-annotations-male-cns-v1.0-minconf-0.5.feather
data/body-neurotransmitters-male-cns-v1.0.feather
data/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Large MaleCNS data files are intentionally local-only and must not be committed.

## Local setup

```bash
pip install -e '.[chess,malecns,dashboard,dev]'
pytest -q
```

Useful entry points include:

```text
scripts/select_chess_populations.py
scripts/generate_teacher_dataset.py
scripts/extract_chess_activations.py
scripts/train_chess_readout.py
scripts/run_chess_benchmark.py
scripts/run_depth_control_sweep.py
scripts/run_control_ratings.py
scripts/calibrate_low_elo.py
```

Run the dashboard with:

```bash
streamlit run dashboard/app.py
```

## Chess architecture

The fly does not use one output neuron per chess move. Every legal move is scored as a candidate:

1. encode board + candidate move;
2. project into a fixed sensory population;
3. run the same MaleCNS-derived recurrent system for depth `D`;
4. read a scalar candidate score from the fixed readout population;
5. choose the legal move with highest score.

This makes the central causal comparison possible:

```text
same checkpoint + same graph + same position
D = 1, 2, 4, 8, 16, 32, 64
```

## Scientific controls

Core controls currently include:

- original signed MaleCNS graph;
- degree-preserving topology shuffle;
- transmitter-sign shuffle;
- recurrent edges strongly attenuated;
- rate versus LIF dynamics where appropriate.

A claim that the biological wiring is important requires the real MaleCNS graph to benefit from recurrence more than matched controls under the same protocol.

## Interpretation boundary

This repository tests computation in a MaleCNS-derived artificial recurrent system. It does **not** establish that a living fruit fly understands chess, that the simplified dynamics reproduce biological cognition, or that recurrent depth is equivalent to human conscious thought.

The term “recurrent thinking” is shorthand for repeated hidden-state computation through the same network.
