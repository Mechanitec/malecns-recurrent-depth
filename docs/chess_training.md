# Chess readout training

The first training stage deliberately keeps the MaleCNS graph and neuron dynamics frozen. Only the final linear readout weights are optimized.

## Objective

For each chess position, Stockfish labels every legal candidate move. The connectome then processes `board + candidate move` using the same frozen projector and recurrent dynamics used during play. Training learns a readout vector `w` such that:

`score(best_teacher_move) > score(other_move)`

The loss is best-vs-rest pairwise logistic ranking loss. This directly matches the fly agent's move-selection rule.

## Pipeline

### 1. Select fixed chess populations

The chess adapter injects board-and-move features into a fixed sensory population
and reads one score from a fixed descending/motor population. Select both from
the annotated MaleCNS graph with a stable seed:

```bash
python scripts/select_chess_populations.py \
  --annotations data/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --neurotransmitters data/body-neurotransmitters-male-cns-v1.0.feather \
  --weights data/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --sensory-output data/chess_sensory_indices.npy \
  --readout-output data/chess_readout_indices.npy \
  --manifest-output data/chess_population_manifest.json \
  --seed 7
```

The selector samples 1,024 connected sensory neurons and 256 connected
descending/motor/efferent neurons by default. Keep these files fixed across
teacher datasets, checkpoints, and recurrent-depth comparisons.

### 2. Build teacher data

From a PGN corpus:

```bash
python scripts/generate_teacher_dataset.py \
  --stockfish /path/to/stockfish \
  --pgn data/games.pgn \
  --output data/chess_teacher.csv \
  --nodes 20000 \
  --max-positions 5000
```

Or supply a text file with one FEN per line using `--fen-file`.

The output contains one row per legal candidate with FEN, UCI move, teacher centipawns, a bounded teacher target, and best-move flag.

### 3. Freeze sensory/readout populations and extract activations

Store graph-index vectors as NumPy `.npy` files, then run:

```bash
python scripts/extract_chess_activations.py \
  --annotations data/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --neurotransmitters data/body-neurotransmitters-male-cns-v1.0.feather \
  --weights data/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --teacher-csv data/chess_teacher.csv \
  --sensory-indices data/chess_sensory_indices.npy \
  --readout-indices data/chess_readout_indices.npy \
  --dynamics lif \
  --depth 16 \
  --projector-seed 7 \
  --output data/chess_activations_depth16.npz
```

The activation cache stores readout-neuron activity once per candidate so readout training can run quickly without repeatedly simulating the full connectome.

### 4. Train the readout

```bash
python scripts/train_chess_readout.py \
  --cache data/chess_activations_depth16.npz \
  --output-dir results/training/depth16 \
  --epochs 100 \
  --learning-rate 0.03
```

Outputs:

- `readout_checkpoint.npz` — readout weights + frozen readout indices + depth/projector metadata;
- `training_history.csv` — dashboard-compatible loss and quality metrics;
- `training_diagnostics.csv` — train/validation pair and top-1 accuracies;
- `training_metrics.json` — final summary.

## Leakage prevention

Train/validation splitting is performed by **position ID**, never by candidate row. All legal moves from one position remain in the same split.

## Depth experiments

There are two scientifically different experiments:

1. **Train separately at each depth.** Measures the best achievable adapter at each depth.
2. **Train once, then change only inference depth.** Stronger causal test of recurrent-depth scaling because the checkpoint is frozen.

The second should be the primary recurrent-depth claim. The first is useful for engineering optimization.

## What remains frozen

For a clean depth comparison keep fixed:

- MaleCNS topology and filtering;
- neurotransmitter signs;
- neuron dynamics;
- sensory population;
- readout population;
- projector seed/fanout/amplitude;
- teacher dataset and split seed;
- trained readout checkpoint (for the primary depth sweep).
