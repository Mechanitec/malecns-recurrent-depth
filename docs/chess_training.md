# Chess readout training

The first training stage deliberately keeps the MaleCNS graph and neural dynamics frozen. Only the final decoder/readout learns.

This is **Stage 0** in the locked research scope: it is treated as an interface that learns how to interpret the frozen recurrent system, not as chess training of the connectome itself.

## Objective

For each chess position, Stockfish labels legal candidate moves. The connectome processes `board + candidate move` using the same frozen projector and recurrent dynamics used during play. Training learns a readout vector `w` so higher-quality teacher moves receive higher scores.

The current objective is best-vs-rest pairwise logistic ranking loss.

## Current v1 result

The v1 depth-16 rate checkpoint is operational and is the frozen baseline used in the first recurrent-depth experiment.

However, its training history shows weak learning:

- pairwise logistic loss remains near `ln(2) ~= 0.6931`;
- validation loss is also close to random-pair baseline;
- validation teacher-best agreement is usually low;
- pairwise candidate accuracy is only modestly above 0.5;
- candidate score margins are very small.

Therefore v1 must be preserved as the baseline checkpoint, but it should not be treated as an optimized chess decoder.

The next step is **diagnosis before more epochs**.

The recorded diagnosis is in `results/training/v1_depth16_rate_large/readout_diagnostics.json`. It covers the 2,000-row activation cache (65 positions) and finds an effective feature rank of about 4.55, no bounded-activation saturation, a 0.00077 near-zero feature fraction, readout weight norm 3.34, and checkpoint gradient norm `1.46e-4`. Recomputed all-cache top-1 agreement is 12.3%; the stored train/validation metrics are 9.6%/23.1% top-1 and 77.0%/58.2% pair accuracy. These results point to a weak or poorly separated fixed-feature decoder, not an activation saturation bug. The independent evaluation corpus is excluded from this diagnosis.

## v1 diagnostics required

Using the existing activation cache, measure:

- feature mean, standard deviation and dynamic range;
- fraction of zero / near-zero features;
- saturation if activations are bounded;
- variance across positions;
- variance across legal candidates within the same position;
- pairwise activation distance within positions;
- effective feature rank / singular-value spectrum on a tractable sample;
- label/pair balance;
- gradient norm at initialization and after training;
- readout weight norm across epochs;
- score-margin distribution;
- train-vs-validation ranking metrics.

The diagnosis should determine whether poor learning is primarily caused by:

- very small numerical feature scale;
- nearly constant readout features;
- insufficient linear separability;
- optimizer/learning-rate behavior;
- insufficient teacher data;
- or a bug in score orientation, labels or splitting.

Any discovered correctness bug must be covered by a regression test before retraining.

## v2 Stage-0 decoder plan

Only proceed after the v1 baseline and frozen evaluation corpus are preserved.

Because pairwise logistic regression on fixed features is convex, v2 should use a numerically stronger fitting procedure rather than simply running Adam for more epochs.

Candidate improvements include:

1. **Train-only feature standardization**
   - compute mean/std from training features only;
   - store transform parameters in the checkpoint;
   - apply the exact same transform during inference.

2. **Robust convex optimization**
   - use L-BFGS / scipy optimization for pairwise logistic + L2, or another well-tested convex solver;
   - verify objective/gradient numerically on a small synthetic problem.

3. **Validation-selected regularization**
   - use a small predeclared L2 grid;
   - select on validation metrics only;
   - never inspect the frozen evaluation corpus for model selection.

4. **Best-checkpoint preservation**
   - save the checkpoint selected by a declared validation criterion rather than the final epoch/iteration.

5. **Numerical provenance**
   - save feature transform, optimizer, convergence status, objective, iteration count, regularization and cache hash in checkpoint metadata.

## v2 acceptance criteria

Compared with v1 on the same validation split, v2 should show a clear improvement in at least one primary decoder metric without a serious regression in the others:

- lower pairwise validation loss;
- higher pairwise validation accuracy;
- higher teacher-best top-1 agreement;
- larger/stabler candidate score margins.

Checkpoint reload must reproduce scores deterministically for fixed test positions.

If v2 does not clearly outperform v1, preserve the negative result and do not tune against the held-out evaluation corpus.

## Pipeline

### 1. Select fixed chess populations

Use the reproducible population selector and keep the resulting files fixed across comparable experiments:

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

### 2. Build teacher data

Teacher generation must use deterministic Stockfish analysis limits and preserve source/game/opening provenance. For serious train/validation/test work, split by source game/opening rather than candidate rows.

All candidate moves from one position must remain in the same split.

### 3. Extract frozen MaleCNS activations

Reference Stage-0 baseline:

```text
rate dynamics
reference depth D=16
fixed sensory population
fixed readout population
fixed projector
fixed MaleCNS topology/dynamics
```

Cache activations once so decoder experiments do not repeatedly simulate the full connectome.

### 4. Train decoder

The existing `scripts/train_chess_readout.py` is the baseline. v2 may extend it or add a new compatible training mode, but old checkpoints/results must remain loadable where practical.

Save at least:

```text
readout_checkpoint.npz
training_history.csv
training_diagnostics.csv
training_metrics.json
```

For v2 also save activation-scale diagnostics and solver/convergence metadata.

## Leakage prevention

Use three distinct roles:

- **train**: fit decoder parameters;
- **validation**: choose regularization/optimizer/checkpoint;
- **evaluation/test-only**: used only after the decoder is frozen to measure recurrent-depth behavior.

The large position-depth study in `plan.md` must not be used to tune v2.

The completed v1 position-depth study used 140 held-out positions and all seven predeclared depths. It found D2 as the lowest mean-regret depth, but the paired improvement over D1 had a 95% bootstrap interval crossing zero. Use this result as the frozen v1 baseline. Any v2 decoder selection must use only the existing train/validation split, followed by a fresh held-out confirmation.

## Two different depth-training questions

There are two scientifically different experiments:

1. **Depth-specific decoder training** — train a separate decoder at each depth. This measures best achievable interface quality at each depth but confounds learning with depth.
2. **Frozen-checkpoint depth sweep** — train once at a reference depth, freeze the checkpoint and vary only inference depth. This is the primary causal recurrent-depth experiment.

The second remains the main claim for Experiment A.

## Future connectome training

The locked research scope later permits controlled learning inside the MaleCNS-derived network (transmitter gains, population physiology, then existing synaptic strengths) while topology remains fixed. That work belongs to later Experiments B-D and must not be mixed into the current Stage-0 baseline without explicit labeling.
