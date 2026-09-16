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

## Completed v1 scaled study

The completed scaled study used 140 legal nonterminal positions derived from 20 source openings with multiple deterministic offsets. It evaluated all legal candidate moves for the original graph and three controls at depths `1, 2, 4, 8, 16, 32, 64`, producing 3,920 raw rows.

For the original graph, mean teacher regret was 581.39 cp at D1 and 570.51 cp at D2. D2 was the empirical minimum, with a paired mean improvement of 10.88 cp. The existing position-level 95% bootstrap interval crosses zero, and mean regret worsens strongly from D4 onward. Therefore the current frozen-v1 result does not support H1.

The existing control results also do not support biological specificity. The next phase is mechanistic discovery rather than another short-game rating run.

## Plan 2 population-study execution record

Plan 2 preserves the v1 checkpoint, seeded baseline interface and consumed
evaluation corpus. The completed atlas and baseline-audit phases use actual
MaleCNS annotations and store reproducible population manifests. The completed
brain-region and information-flow phases use the development corpus, fixed
input populations, depths `1, 2, 4, 8, 16, 32, 64`, and the standardized
linear ranking probe. Their reports include cross-depth transfer, effective
rank, candidate-state separation, representation similarity, winner switching,
state norms, saturation and train-validation gaps.

The input-family, H-BIO-1 and population-size sweeps are development-only
analyses. Do not freeze a population rule or generate the confirmatory corpus
until those outputs have been written and audited. A fresh confirmation must
remain source-diverse and must not be used for population or probe tuning.

## Fixed factors in a causal depth comparison

Within any one frozen causal depth comparison keep fixed:

- MaleCNS input files and preprocessing;
- neuron set;
- graph topology for that condition;
- edge signs/weights for that condition;
- rate dynamics parameters;
- sensory and readout populations;
- board/move encoder;
- sensory projector seed/fanout/amplitude;
- input mode (clamped or single-pulse);
- readout checkpoint;
- position corpus;
- teacher configuration;
- graph/control seed;
- numerical dtype where practical.

Only recurrent depth changes inside a given condition. If another factor such as recurrent gain, leak, clamp mode, or graph variant is being studied, treat it as an explicit factorial condition and do not silently retune it inside the depth comparison.

## Required depths

```text
1, 2, 4, 8, 16, 32, 64
```

Additional depths may be added around an empirical optimum, but these seven should remain visible for direct comparison with the v1 baseline.

## Data roles and contamination control

Use distinct data roles:

```text
training / fitting
validation / development
final evaluation / confirmation
```

Never tune a readout, recurrent gain, clamp mode, adaptive-stopping threshold, population choice, graph-control construction, or other setting on a corpus that is then presented as fresh held-out confirmation.

The completed 140-position corpus is now consumed evaluation data and must not be used to select v2 settings.

A new `development_only` corpus may be used for exploratory mechanism discovery and model selection. Any positive headline result selected there requires a fresh held-out confirmation later.

## Source-group dependence and bootstrap unit

Positions sampled at several offsets from the same source opening/game are correlated. Therefore the statistical unit is not automatically the individual FEN.

Every grouped corpus must retain `source_game_id` or an equivalent source identifier.

For headline uncertainty:

- ordinary position bootstrap may be reported for continuity;
- **cluster bootstrap by `source_game_id` / source opening is preferred whenever multiple positions share a source trajectory**;
- resample whole source groups with replacement and carry all positions from a sampled group together;
- use matched sampled source groups across paired depth and original-vs-control comparisons;
- report both number of positions and number of independent source groups.

If a corpus contains one independently sourced position per game, the position and cluster bootstrap become effectively equivalent.

Candidate rows are never the bootstrap unit.

## Depth selection and selection-aware reporting

If the best depth is chosen after inspecting a dataset, a nominal CI for that selected depth is exploratory and subject to selection bias.

For a confirmatory positive claim, either:

```text
predeclare the target depth before final evaluation
or
select the target depth on development data and confirm it on a fresh held-out corpus
```

For exploratory best-depth summaries, label the selection explicitly and use a max-statistic, nested bootstrap, or another selection-aware analysis when practical.

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

Report mean and median `Delta`, cluster-aware 95% uncertainty where required, and improved/unchanged/worsened fractions.

## Secondary position metrics

Record at least:

- teacher-best agreement;
- top-3 teacher agreement;
- selected move's teacher rank;
- fly rank assigned to teacher-best move;
- candidate-score margin;
- candidate-score dispersion;
- Spearman/Kendall candidate rank correlation where practical;
- recurrent passes;
- wall latency.

Independent post-move Stockfish evaluation is useful but expensive. It may be applied to a fixed predeclared subset while root-move teacher scores remain the primary scalable metric.

The independent evaluator is an observer only and must never influence the Fly's move choice.

## Input protocol: clamped versus single-pulse

Additional recurrent passes can mean different things depending on how sensory evidence is delivered.

Distinguish:

```text
clamped input
    sensory evidence is injected at every recurrent step

single-pulse input
    sensory evidence is delivered at depth 1 only, then internal recurrence evolves without re-injection
```

A benefit or collapse that exists only under clamped input may reflect repeated external driving rather than purely internal computation. Mechanistic studies should compare both modes.

## Recurrent-strength experiments

Current evidence suggests recurrence strength may determine whether deeper computation helps or harms performance.

Treat recurrent gain / recurrent-weight scale as an explicit development factor. Within a fixed gain condition, depth remains the causal variable.

If a gain is selected from development data, freeze it before any fresh held-out confirmation.

Record both:

```text
engine recurrent_gain
graph-weight multiplicative scale
```

so the effective recurrent strength is unambiguous.

## Graph controls

Required control classes include:

1. original signed MaleCNS graph;
2. a validated directed degree-preserving topology shuffle;
3. transmitter/sign shuffle;
4. recurrent-strength attenuation or equivalent gain control.

A strong biological-specificity claim should compare the original graph against **an ensemble of matched shuffled controls**, not one arbitrary seed.

### Control-graph invariant audit

Before using a shuffled graph for biological inference, validate the *final sparse graph after construction/coalescing*.

Record at least:

- neuron count;
- stored nonzero edge count;
- in-degree sequence;
- out-degree sequence;
- weighted in/out-strength summaries;
- self-loop count;
- duplicate/coalesced-edge count if applicable;
- random seed;
- construction algorithm.

If a control is described as degree-preserving, degree preservation must hold for the final graph representation. If a better control is introduced, preserve old artifacts and version the corrected control separately.

For control `C`, the specificity statistic remains:

```text
Specificity(D,C)
  = [regret_original(1) - regret_original(D)]
    - [regret_C(1) - regret_C(D)]
```

Use the same source-group cluster bootstrap for this difference-in-differences when grouped positions are present.

## Recurrent-dynamics observability

Mechanistic studies should collect compact internal summaries without changing inference semantics.

Useful measures include:

- hidden-state L2 norm;
- step-to-step state delta and relative delta;
- recurrent-drive norm;
- sensory-drive norm;
- recurrent/sensory drive ratio;
- fractions of units near activation saturation;
- readout-population norm/variance;
- candidate-state centroid and dispersion;
- pairwise candidate cosine similarity;
- effective rank of candidate representations;
- score dispersion and top1-top2 margin;
- identity of the leading move by depth;
- number and location of leader switches;
- recurrent passes and latency.

Instrumentation must be covered by equivalence tests showing it does not alter candidate scores or rankings beyond documented floating-point tolerance.

## Information preservation versus representation rotation

A frozen readout can fail at deeper depth for at least two distinct reasons:

```text
information destruction
    deeper states no longer contain linearly recoverable task signal

representation rotation
    useful signal remains but moves into directions the frozen readout does not use
```

Distinguish these with depth-specific linear probes trained only on training/validation data.

Use the same position split at every depth. Fit standardization/whitening only on training data. Evaluate every trained probe across every inference depth to obtain a cross-depth transfer matrix.

Compare against non-connectome baselines such as:

- raw chess encoder features;
- sensory-projected features before recurrence;
- MaleCNS features at each depth.

This directly tests whether recurrence adds, preserves, rotates, or destroys linearly usable chess information.

## Teacher labels and teacher-stability checks

Teacher scores must be from the side-to-move perspective and cover every legal move.

Record exact engine executable hash, version, nodes/depth/time limit, threads, hash size, mate handling, and deterministic settings.

For new corpora, run a fixed-subset teacher-stability check at stronger search budgets. Report best-move agreement, rank stability, and score/regret stability between budgets so teacher noise is visible.

## Corpus construction

Headline position studies should use legal, nonterminal, reasonably ordinary chess positions with exact provenance.

Prefer many independent source games/openings over many correlated offsets from a small number of sources.

Record at least:

```text
position_id
source_game_id
phase / ply
FEN
side to move
legal move count
teacher score for every legal move
teacher-best marker
```

Programmatically verify:

- no duplicate FENs;
- no forbidden overlap with fitting/evaluation corpora;
- legal-move set exactly matches teacher-labelled candidates;
- deterministic teacher-best tie breaking;
- score orientation is correct.

## Position-level discovery analyses

Exploratory studies may investigate *when* recurrence helps or hurts using descriptors such as:

- phase / ply;
- legal move count;
- teacher best-vs-second margin;
- D1 regret;
- tactical properties of teacher-best move;
- material balance;
- teacher score spread;
- internal state stability;
- candidate-representation separation/effective rank;
- number of leader switches across depth.

Use grouped summaries, rank correlations, or simple regularized models as appropriate. Cluster/bootstrap by source game and label the analysis exploratory. Do not turn broad pattern mining into a confirmatory claim without fresh data.

## Adaptive test-time depth

Adaptive recurrent depth is a valid research branch if the stopping rule uses only information available internally at inference time.

Candidate signals include:

- leader stability across requested depths;
- score-margin stability;
- state-delta convergence;
- candidate-representation collapse/stability;
- a lightweight rule trained on development-only internal metrics.

Teacher/Stockfish information may be used to train or evaluate a development policy but cannot be consulted by the stopping rule at inference time.

Tune on development data, freeze thresholds/policy, then evaluate on a separate development-confirm split or fresh held-out corpus.

## Computational optimization rules

Optimization is allowed only if it preserves the exact tested computation.

Preferred optimizations:

- run once to max depth and snapshot intermediate requested depths;
- batch all legal candidates through sparse-matrix x dense-matrix operations for the rate engine.

Every optimized path must be numerically compared with the scalar reference across positions, depths, and graph variants. Chosen move and ranking should match within a documented floating-point tie tolerance.

Do not introduce pruning, approximate candidate sets, learned surrogates, or altered dynamics merely to make the scientific position experiment faster.

## Critical LIF confound

For LIF, deeper recurrence is also more membrane-integration time. Any future LIF headline result must distinguish computational depth from simulated biological time using appropriate controls such as single-pulse propagation, matched-time comparisons, or rate-state recurrence.

The current mechanistic discovery program should use the rate-state engine first because it is the cleaner recurrent-compute abstraction.

## Short-game rating caveat

A game capped after a small number of plies and then adjudicated as `1/2-1/2` does not provide normal chess outcome information. Ratings derived mostly or entirely from such forced draws are diagnostic harness values only.

Do not present the historical ~410/~445 short-protocol values as actual chess Elo.

A future serious chess rating requires sufficiently long games or a predeclared validated adjudication method that yields informative outcomes.

## Success and null criteria

Evidence for a recurrent-depth effect requires a reproducible paired improvement over D1 with effect size and uncertainty reported on data not used to tune the tested setting.

Evidence for biological specificity additionally requires the original graph's recurrent-depth benefit to exceed matched control distributions under the same corpus/checkpoint/procedure.

High-depth deterioration is a valid result and should be analyzed mechanistically rather than hidden.

Treat the following plainly as negative or null evidence:

- D > 1 does not improve paired regret;
- uncertainty includes zero or is too broad;
- shuffled/attenuated controls improve equally or more;
- apparent gains disappear under all-legal scoring;
- effects are driven by a few positions/source groups;
- optimized evaluation fails scalar equivalence;
- results are unstable under trivial numerical perturbations;
- the original graph lies well inside the matched shuffle ensemble.

## Reproducibility requirements

For each substantive run record:

- git commit;
- Python/package environment where practical;
- critical input hashes;
- checkpoint hash;
- population hashes;
- graph/control algorithm and seeds;
- dynamics parameters;
- recurrent-depth list;
- clamp/single-pulse mode;
- corpus ID/hash, position count, and source-group count;
- teacher engine hash/options;
- observer engine options if used;
- raw row count and artifact hash;
- test status.

Long experiments must be resumable and flush progress incrementally.

Commit small CSV/JSON summaries, plots, tests, and documentation. Avoid committing engine binaries, large Feather files, massive state dumps, or huge transient caches.

## Immediate next program

The v1 140-position study is the frozen reference. The immediate work is the mechanistic discovery program in `../plan.md`:

- cluster-aware reanalysis;
- graph-control audit;
- recurrent-state observability;
- fresh development corpus;
- recurrent-gain × input-mode × depth sweep;
- depth-specific probe transfer matrix;
- raw-encoder/sensory baselines;
- position-type discovery;
- shuffle ensembles;
- adaptive stopping.

The objective is to learn *why* the current system behaves as it does and identify any regime where recurrent computation produces a robust, interesting effect before committing to another large confirmatory run.
