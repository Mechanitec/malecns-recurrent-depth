# Current MaleCNS recurrent-depth chess results

This document summarizes the current evidence on `main`. It is a living results note, not the locked research contract. `Research Goal.md` remains authoritative for scope.

## Current baseline system

- Real MaleCNS-derived graph.
- Rate-state recurrent dynamics.
- Frozen Stage-0 linear readout checkpoint trained at reference depth 16.
- Fixed chess encoder/projector and fixed sensory/readout populations.
- All legal moves scored for the held-out position metrics.
- Recurrent depths tested: `1, 2, 4, 8, 16, 32, 64`.
- Controls: original, degree-preserving topology shuffle, transmitter-sign shuffle, recurrent weights attenuated to 5%.

## Exploratory 8-position depth result

Current original-graph metrics from `results/sweep_full_8pos_v2/depth_sweep/metrics.csv`:

| Depth | Teacher agreement | Teacher regret cp | Stockfish eval loss cp | Mean latency s |
|---:|---:|---:|---:|---:|
| 1 | 0.000 | 756.500 | 685.125 | 1.124 |
| 2 | 0.125 | 345.375 | 446.625 | 2.279 |
| 4 | 0.125 | 354.875 | 453.000 | 4.665 |
| 8 | 0.125 | 316.375 | 463.000 | 9.051 |
| 16 | 0.125 | 316.375 | 463.000 | 16.672 |
| 32 | 0.125 | 493.000 | 545.375 | 33.201 |
| 64 | 0.125 | 519.875 | 520.500 | 69.978 |

Relative to D1, teacher regret improves strongly at D2-D16. The best observed teacher-regret value is 316.375 cp at D8/D16, about 58% lower than D1. Independent Stockfish post-move loss is lowest at D2 in this tiny sample.

This is **exploratory only** because 8 positions give extremely coarse agreement statistics: one position changes agreement by 12.5 percentage points.

## Control comparison

Selected D8 values from `results/sweep_full_8pos_v2/control_sweep/metrics.csv`:

| Variant | Teacher agreement | Teacher regret cp | Stockfish eval loss cp |
|---|---:|---:|---:|
| Original | 0.125 | 316.375 | 463.000 |
| Degree-preserving topology shuffle | 0.250 | 256.125 | 399.250 |
| Transmitter-sign shuffle | 0.125 | 396.750 | 480.875 |
| Recurrent weights x0.05 | 0.125 | 316.375 | 463.000 |

The topology-shuffled graph is better than the original on these D8 metrics, and the strongly attenuated recurrent graph matches the original on teacher regret at D8/D16 in this sample.

Therefore the current data do **not** support a claim that the biological MaleCNS wiring is responsible for the observed recurrent-depth improvement.

The justified interpretation is narrower:

> A potentially useful effect of repeated recurrent computation appears between D1 and moderate depths in the current system, but the 8-position sample is too small and matched controls do not yet show a MaleCNS-specific advantage.

## Rating artifacts: interpretation boundary

The serious short-game artifact contains 402/402 draws with a reported estimate near 409.7. The compact depth/control rating tables similarly return about 445 at every condition.

These are **not valid estimates of normal chess playing strength** because the short protocols cap games after very few plies and unresolved games are adjudicated as draws.

They remain useful for checking tournament plumbing, color balance, logging and opponent routing. They should not be used as evidence for or against recurrent-depth chess strength.

A future serious rating must use long enough games or a validated independent evaluation-based adjudication rule that produces informative outcomes.

## v1 readout training quality

`results/training/v1_depth16_rate_large/training_history.csv` shows weak Stage-0 readout learning:

- pairwise logistic training loss remains near `ln(2) ~= 0.6931`;
- validation loss is similarly close to random-pair baseline;
- teacher-best agreement is generally low;
- candidate pairwise accuracy is only modestly above chance;
- score margins are small.

This does not invalidate the frozen-depth experiment, but it means the current checkpoint is a weak decoder. It should be preserved as v1 for reproducibility while a v2 Stage-0 decoder is developed using train/validation data only.

## Immediate next experiment

The next headline experiment is a much larger paired held-out position study, not another forced-short-game tournament.

Target:

```text
128-256+ unseen positions
x 7 recurrent depths
x 4 graph variants
x all legal moves
```

Primary statistic:

```text
regret_D1 - regret_D
```

with paired bootstrap confidence intervals over positions.

Biological specificity is tested with a paired difference-in-differences between the original graph's D1->D improvement and each control's D1->D improvement.

See `../plan.md` and `experiment_protocol.md` for the execution details.

## Claims currently allowed

It is reasonable to say:

- the real MaleCNS recurrent chess pipeline is operational;
- moderate recurrent depth changed and, on the small sample, often improved move quality relative to D1;
- very deep recurrence (D32/D64) did not continue improving the current original system;
- matched controls prevent a biological-structure claim at present;
- the current decoder is weak and should be improved separately from the frozen v1 result.

It is **not** yet reasonable to say:

- the fly system has a genuine ~410 or ~445 chess Elo;
- more recurrent depth monotonically makes it smarter;
- the biological MaleCNS topology is superior to shuffled topology for chess;
- the current data establish abstract reasoning in a biological fly connectome.
