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

## Plan 2 population-study status

The open Plan 2 work extends the baseline with a MaleCNS-specific interface
study. The verified atlas currently covers 164,620 neurons, 10,227,924 graph
edges and 6,965 strongly connected components. It records the actual MaleCNS
annotation filters used for mushroom-body, central-complex, fan-shaped-body,
FC/PFN/PFR/PFL, LAL/descending and related candidate families.

The baseline audit and brain-region decoding phases are complete on the
development corpus. The cross-depth transfer output contains 588 rows for 12
priority readout populations across 49 train/test depth pairs. The companion
information-flow table contains 84 rows for those populations across the seven
requested depths and includes representation similarity, state geometry,
winner switching, saturation and train-validation gap diagnostics.

Input-population, H-BIO-1 and population-size scaling sweeps are executed as
long-running development analyses. Their final CSV and metadata files are
required before reporting a population-selection result. Population-v2
selection and fresh held-out confirmation must remain downstream of those
development analyses.

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

## Completed 140-position recurrent-depth study

The independent corpus contains 140 legal, nonterminal positions with early, middle and late phases. The frozen v1 checkpoint was evaluated on all legal moves at D1, D2, D4, D8, D16, D32 and D64 for the original graph and three predeclared controls. The study produced 3,920 raw rows: 140 positions x 4 variants x 7 depths. The corpus and checkpoint hashes are recorded in the per-variant metadata and reproducibility manifest.

For the original graph, mean teacher regret was 581.39 cp at D1 and 570.51 cp at D2. D2 was the empirical minimum, with a paired D1-to-D2 improvement of 10.88 cp, but the current position-level 95% paired bootstrap CI was -60.66 to 81.80 cp. Regret worsened at D4 and remained higher through D64. Therefore this run does not support H1.

The control difference-in-differences did not establish MaleCNS-specificity. At D2, the current position-bootstrap estimates are 86.62 cp [19.08, 157.57] for the topology shuffle, 27.73 cp [-35.36, 94.58] for the transmitter-sign shuffle, and 0.00 cp [0.00, 0.00] for recurrent-edge attenuation. These results are not yet sufficient for a biological-specificity claim.

### Important statistical caveat

The 140 positions are not 140 fully independent source games. They are generated as multiple offsets from 20 source openings/trajectories. The current bootstrap resamples positions individually. The next analysis must therefore also bootstrap by `source_game_id` / source opening and treat that cluster-aware interval as the preferred uncertainty estimate. This is especially important before interpreting apparently non-zero control difference-in-differences.

The optimized evaluator was validated against the scalar reference. On one 36-candidate position across D1-D64, independent scalar evaluation took 111.658 s and one batched trajectory with snapshots took 53.740 s, a 2.078x speedup.

## Most interesting current clue: deep recurrence appears destructive under baseline dynamics

The larger study changes the interpretation of the earlier 8-position pilot. The strong pilot improvement did not reproduce. Instead, the original graph shows only a small uncertain D1-to-D2 gain followed by a large deterioration at D4 and deeper.

The mean regret rises much more sharply than the median, which suggests that deep recurrence may be causing catastrophic failures on a subset of positions rather than uniformly degrading every position.

Two controls sharpen the question:

- recurrent edges attenuated to 5% remain near the shallow-depth quality level instead of showing the same large deep-depth collapse;
- a topology-shuffled graph can outperform the original graph at some moderate depths.

This motivates a mechanistic program: measure state norms, saturation, candidate-state separation, effective rank, winner switching, recurrent/sensory drive balance, and probe recoverability across depth. The key question is whether deep recurrence destroys chess information or merely rotates it away from the frozen linear readout.

## Control comparison from the exploratory 8-position run

Selected D8 values from `results/sweep_full_8pos_v2/control_sweep/metrics.csv`:

| Variant | Teacher agreement | Teacher regret cp | Stockfish eval loss cp |
|---|---:|---:|---:|
| Original | 0.125 | 316.375 | 463.000 |
| Degree-preserving topology shuffle | 0.250 | 256.125 | 399.250 |
| Transmitter-sign shuffle | 0.125 | 396.750 | 480.875 |
| Recurrent weights x0.05 | 0.125 | 316.375 | 463.000 |

The topology-shuffled graph is better than the original on these D8 metrics, and the strongly attenuated recurrent graph matches the original on teacher regret at D8/D16 in this tiny sample.

Therefore the current data do **not** support a claim that biological MaleCNS wiring is responsible for a beneficial recurrent-depth effect.

## Control-construction audit needed

The current topology shuffle permutes destination indices and reconstructs a sparse matrix. Because duplicate sparse coordinates may coalesce, the next work must verify that the final shuffled matrix preserves the intended in/out-degree structure after construction. If not, a stricter directed edge-swap control should be added and versioned without deleting the old results.

One shuffled graph is also not an adequate empirical null distribution. The next discovery plan uses ensembles of topology and sign shuffles to locate the original MaleCNS graph within a matched random-control distribution.

## Rating artifacts: interpretation boundary

The historical short-game rating artifacts are diagnostic, not normal chess-strength estimates. They contain forced-short-game adjudication and therefore cannot establish a genuine ~410 or ~445 Elo.

There is also a current report-generation inconsistency around `results/first_fly_rating/serious_400_bounded`: the consolidated summary metadata describes it as full-legal/unbounded, while the raw game artifact contains short-game adjudication behavior. The next work must derive these summary fields from raw game data and add a regression test.

These rating artifacts remain useful for checking tournament plumbing, color balance, logging and opponent routing. They should not be used as evidence for or against recurrent-depth chess strength.

## v1 readout training quality

`results/training/v1_depth16_rate_large/readout_diagnostics.json` shows that the current decoder is weak and the fixed representation is highly compressed:

- 2,000 candidate rows from 65 positions;
- 256 readout features;
- effective feature rank about 4.55;
- mean within-position candidate L2 distance about 0.0043;
- checkpoint validation pair accuracy about 58.2%;
- checkpoint validation top-1 agreement about 23.1%;
- recomputed all-cache top-1 agreement about 12.3%;
- no evidence that simple tanh saturation or all-zero activity alone explains the weakness.

The checkpoint was trained at D16, yet the independent corpus performs best at D2. This is why the next mechanistic experiment includes a **depth-specific probe transfer matrix**. If a fresh linear probe can recover chess information at deep states, the representation may be rotating; if every probe degrades at depth, information is genuinely being lost.

## Immediate next experiment

The next work is a full mechanistic discovery day rather than another short-game tournament or a simple “train longer” pass. The priorities are:

- cluster-aware reanalysis of the completed 140-position study;
- audit/correction of graph controls;
- internal recurrent-dynamics observability;
- a fresh development-only corpus with many more independent source groups;
- recurrent-gain x clamp/single-pulse x depth response surface;
- depth-specific probe transfer matrix and raw-encoder/sensory baselines;
- position-level analysis of when recurrence helps or hurts;
- ensembles of shuffled controls;
- adaptive-depth stopping based only on internal signals.

See `../plan.md` for the execution details.

## Claims currently allowed

It is reasonable to say:

- the real MaleCNS recurrent chess pipeline is operational;
- the early 8-position recurrent benefit did not robustly reproduce in the 140-position study;
- D2 is the empirical minimum-regret depth for the current v1 system, but its improvement over D1 is not established;
- deeper recurrence D4-D64 substantially worsens the current original system;
- matched controls do not support a biological-topology advantage;
- current baseline dynamics may be over-amplifying recurrence, as suggested by the 5%-strength control;
- the current decoder is weak and low-dimensional;
- the mechanism of deep-depth deterioration is now a primary research target.

It is **not** yet reasonable to say:

- the fly system has a genuine ~410 or ~445 chess Elo;
- more recurrent depth monotonically makes it smarter;
- the biological MaleCNS topology is superior to shuffled topology for chess;
- the existing topology-control result is definitive before its invariants and shuffle-seed distribution are checked;
- the current 140 positions provide 140 independent statistical units;
- the current data establish abstract reasoning in a biological fly connectome.
