# Recurrent-depth experiment protocol

## Goal

Test whether additional applications of the same MaleCNS-derived recurrent block improve chess decision quality under controlled conditions, and determine whether any benefit is specific to the biological wiring rather than generic recurrence.

The core frozen-system comparison is:

```text
same graph + same dynamics + same projector + same readout + same position
only recurrent depth changes
```

The locked research scope is defined in `../Research Goal.md`.

## Primary hypotheses

### H1 — recurrent-depth effect

For the original frozen MaleCNS-derived system, one or more depths `D > 1` produce better held-out chess move quality than `D = 1`.

### H1-specificity — biological structure

The improvement from recurrent depth is greater for the original MaleCNS graph than for matched topology/sign/recurrent-strength controls.

A positive H1 result without H1-specificity is still useful, but it must be described as a **generic recurrent-computation effect**, not a connectome-specific result.

## Current evidence motivating the next run

The first all-legal held-out sweep used only 8 positions. On the original graph, teacher regret fell from 756.5 cp at D1 to 345.4 cp at D2 and 316.4 cp at D8/D16. However, controls sometimes matched or exceeded the original graph; for example, the degree-preserving topology shuffle reached 256.1 cp regret at D8. The sample is therefore exploratory only.

The next protocol must scale the same paired comparison to at least 128 unseen positions, preferably 256 or more.

## Fixed factors

For the primary frozen-depth experiment keep fixed:

- MaleCNS input files and preprocessing;
- neuron set;
- graph topology for the original condition;
- edge signs/weights for the original condition;
- rate dynamics parameters;
- sensory and readout populations;
- board/move encoder;
- sensory projector seed/fanout/amplitude;
- clamp mode;
- readout checkpoint;
- evaluation corpus;
- Stockfish teacher/evaluator configuration;
- graph-control seeds;
- numerical dtype where practical.

Only recurrent depth changes inside a given graph condition.

## Required depths

```text
1, 2, 4, 8, 16, 32, 64
```

Additional depths may be added later around an empirical optimum, but these seven predeclared depths must remain visible in the main comparison.

## Evaluation corpus

Use an independent corpus that is never used to fit or select the readout.

Target size:

- minimum: 128 unique positions;
- preferred: 256;
- stretch: 512+ if optimized inference permits.

Requirements:

- no FEN overlap with train/validation fitting data;
- no candidate-row leakage;
- deterministic source/seed;
- legal nonterminal FENs;
- reasonable early/middle/late spread;
- exact source/game/opening ID;
- all legal candidate moves scored by Stockfish;
- deterministic teacher tie-breaking;
- exact engine hash and analysis limit recorded.

Bootstrap/statistical units are **positions**, never candidate rows.

## Primary metric

For position `i` and depth `D`:

```text
regret_i(D) = teacher_best_cp_i - teacher_cp_i(move_selected_by_fly_at_D)
```

Lower is better.

The primary paired recurrent-depth effect is:

```text
Delta_i(D) = regret_i(1) - regret_i(D)
```

Positive values mean deeper recurrence improved the selected move for that position.

Report mean and median `Delta`, 95% paired bootstrap CI over positions, and improved/unchanged/worsened fractions.

## Secondary position metrics

Record at least:

- teacher-best agreement;
- top-3 teacher agreement;
- selected move's teacher rank;
- fly rank assigned to teacher-best move;
- candidate-score margin;
- candidate-score dispersion;
- optional Spearman/Kendall candidate rank correlation;
- recurrent passes;
- wall latency.

Independent post-move Stockfish evaluation is useful but expensive. It may be applied to a fixed predeclared subset while root-move teacher scores remain the primary scalable metric.

## Graph controls

Required matched controls:

1. original signed MaleCNS graph;
2. degree-preserving topology shuffle;
3. transmitter/sign shuffle;
4. recurrent weights attenuated to 5%.

Use a predeclared control seed. Do not reroll controls after seeing results.

For control `C`, the main specificity statistic is the paired difference-in-differences:

```text
Specificity(D,C)
  = [regret_original(1) - regret_original(D)]
    - [regret_C(1) - regret_C(D)]
```

Bootstrap positions to obtain uncertainty.

Interpretation:

- positive original depth effect + similar controls -> generic recurrence;
- original depth effect materially larger than controls -> evidence toward MaleCNS-specific use of recurrence;
- shuffled/control equal or better -> no biological-structure advantage under the tested setup.

## Computational optimization rules

Optimization is allowed only if it preserves the exact tested computation.

Preferred optimizations:

- run once to max depth and snapshot states at requested intermediate depths;
- batch all legal candidates through sparse-matrix x dense-matrix operations for the rate engine.

Every optimized path must be numerically compared with the scalar reference on multiple positions/depths/graph variants. Chosen move and ranking should match within a documented floating-point tie tolerance.

Do not introduce pruning, approximate candidate sets, learned surrogates or altered dynamics merely to make the headline position experiment faster.

## Critical LIF confound

For LIF, deeper recurrence is also more membrane-integration time. Any future LIF headline result must distinguish:

1. biological-time scaling;
2. frozen-observation depth;
3. single-pulse propagation;
4. rate-state recurrence control.

The current large chess position study should use the rate-state engine first because it is the cleaner computational-depth experiment.

## Short-game rating caveat

A game capped after a small number of plies and then adjudicated as `1/2-1/2` does not provide normal chess outcome information. Ratings derived mostly or entirely from such forced draws are diagnostic harness values only.

Do not present the existing ~410/~445 short-protocol values as actual chess Elo.

A future serious chess rating requires sufficiently long games or a predeclared position-adjudication method that converts objective evaluations into outcomes without assigning every unresolved short game a draw.

## Success criteria

Evidence for H1 requires a reproducible paired improvement over D1 on a substantially larger held-out position set, with effect size and uncertainty reported.

Evidence for biological specificity additionally requires the original graph's recurrent-depth benefit to exceed matched controls under the same corpus/checkpoint/procedure.

High-depth deterioration is a valid outcome and should be used to estimate an empirical depth optimum rather than hidden.

## Failure / null criteria

Treat the following plainly as negative or null evidence:

- D > 1 does not improve paired regret on the larger corpus;
- confidence intervals are too broad to distinguish the effect from zero;
- shuffled/attenuated controls improve equally or more;
- apparent gains disappear under all-legal scoring;
- effect depends on one or a few outlier positions;
- optimized evaluator fails scalar-equivalence tests;
- results change materially under trivial numerical perturbations.

## Required artifacts

The next large study should preserve:

```text
results/position_depth_study_v2/raw_position_metrics.csv
results/position_depth_study_v2/summary_by_depth.csv
results/position_depth_study_v2/specificity_by_control.csv
results/position_depth_study_v2/metadata.json
results/position_depth_study_v2/progress.json
results/position_depth_study_v2/*.png
```

Small result tables/plots should be committed. Large caches/data may remain local if hashes and compact summaries are preserved.
