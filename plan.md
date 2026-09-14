# MaleCNS Recurrent-Depth Chess: Mechanistic Discovery Day Plan

**Execution target:** a genuinely full local workday on the Windows workstation using Codex / Luna High.

**Research scope:** `Research Goal.md` is LOCKED. Do not edit or reinterpret it. The overall research question remains unchanged. This plan changes the *next experimental emphasis*: instead of immediately trying to make the chess decoder stronger, use the completed 140-position result to discover **why shallow recurrence sometimes helps, why deeper recurrence collapses, whether useful information is being destroyed or merely rotated, and whether any recurrent regime produces a genuinely interesting effect.**

The previous “full-day” plan finished in about 1.5 hours. Do not stop after implementing infrastructure. This plan intentionally contains several independent experiments, heavier compute, replicated controls, and mechanistic analyses. Complete all mandatory blocks before declaring the day finished. If the mandatory blocks complete unusually early, continue into the extension ladder at the end rather than stopping.

---

# Starting evidence

The completed frozen-v1 study used 140 held-out positions, 4 graph variants, all legal moves, and recurrent depths `1,2,4,8,16,32,64`.

For the original MaleCNS graph:

| Depth | Mean regret cp | Median regret cp | Teacher-best | Top-3 | Mean latency s |
|---:|---:|---:|---:|---:|---:|
| 1 | 581.39 | 465.0 | 5.7% | 9.3% | 1.64 |
| 2 | 570.51 | 435.5 | 7.1% | 20.0% | 3.29 |
| 4 | 1310.79 | 515.0 | 4.3% | 12.9% | 6.52 |
| 8 | 1295.74 | 472.5 | 4.3% | 7.1% | 12.76 |
| 16 | 1328.49 | 569.5 | 3.6% | 7.9% | 25.16 |
| 32 | 1400.86 | 655.5 | 3.6% | 6.4% | 50.27 |
| 64 | 1426.53 | 696.0 | 3.6% | 9.3% | 99.87 |

D2 is the empirical minimum, but the paired D1-to-D2 mean improvement is only 10.88 cp with a 95% position-bootstrap CI of roughly `[-60.66, 81.80]`, so H1 is not established. Deep recurrence is strongly worse in mean regret.

Two clues deserve focused investigation:

1. The recurrent-edges-at-5%-strength control remains near the D1/D2 quality level instead of suffering the same deep collapse.
2. The degree-preserving topology shuffle is much better than the original graph at some moderate depths.

The v1 readout is also weak and low-dimensional: the 256-dimensional activation cache has effective rank about 4.55, with weak candidate separation. The checkpoint was trained at D16, yet D2 performs best on the independent corpus. This raises a major mechanistic question: **does deeper recurrence destroy task information, or does it transform/rotate the representation so that the fixed D16 readout no longer tracks useful directions?**

---

# Discovery questions for today

The day should answer as many of these as possible with data, not intuition:

1. Does the D4+ collapse correspond to exploding/over-amplified recurrent dynamics, tanh saturation, candidate-state collapse, unstable winner switching, or some other measurable state transition?
2. Is useful chess information actually lost at depth, or can a depth-specific linear probe recover it?
3. Does lowering recurrent gain create a real “sweet spot” where additional depth helps rather than hurts?
4. Does clamping the sensory evidence every recurrent step cause part of the collapse? What changes under a single-pulse input?
5. Is the biological MaleCNS topology unusual compared with a *distribution* of valid degree-preserving shuffled controls, or was the earlier one-shuffle result just one random draw?
6. Does the current topology-shuffle implementation truly preserve the intended degree sequence after sparse duplicate coalescing?
7. Which kinds of chess positions benefit from additional recurrence and which are harmed? Is the effect related to phase, branching factor, teacher margin, tactics, baseline difficulty, or internal stability?
8. Can an adaptive internal stopping rule obtain better quality/compute tradeoffs by stopping before deep deterioration, without consulting Stockfish at inference time?
9. Does MaleCNS recurrence add useful linear information beyond the raw chess encoder or sensory projection alone?
10. Are the existing conclusions robust when uncertainty is clustered by source opening/game rather than incorrectly treating correlated offsets as fully independent positions?

Finding a clear negative answer is still a useful discovery. Do not optimize the reporting toward a positive MaleCNS result.

---

# Scientific boundaries

1. Start from current `main` and record the exact HEAD before work.
2. **Do not use GitHub Actions.** All tests and experiments run locally.
3. Do not edit `Research Goal.md`.
4. Preserve all v1 artifacts and the completed 140-position study unchanged.
5. Treat `data/chess_evaluation_corpus_v2.csv` as consumed evaluation data. Do **not** tune recurrent gain, clamp mode, readout, thresholds, population selection, or other choices on it.
6. New model/dynamics selection uses train/validation data or a newly created **development-only** corpus. Any positive headline result later requires a fresh held-out confirmation.
7. For causal depth comparisons, keep everything fixed except the explicitly named experimental factor(s).
8. Use all legal moves for scientific position-quality metrics.
9. Long runs must be resumable and flush results incrementally.
10. Record input hashes, graph/control seeds, engine versions/options, code commit, data split IDs, and experiment configuration.
11. Do not reroll a graph shuffle or random seed because its result is inconvenient.
12. Keep exploratory and confirmatory claims clearly separated.
13. Run focused tests before substantive commits and `pytest -q` before the day ends.

---

# Completion gate

Do not declare the plan complete merely because scripts exist. A complete day requires **executed results** for Blocks 1 through 8 below, a synthesis report, and updated documentation. Block 9 is conditional but should be attempted if the mechanism discovered earlier suggests a clear v2 improvement.

Expected output root:

```text
results/mechanistic_discovery_v1/
```

Use subdirectories and metadata so each experiment can be rerun independently.

---

# Block 0 — Preflight, repair known reporting debt, and freeze provenance

**Goal:** remove known analysis/reporting ambiguities before new discovery work.

Run:

```powershell
git checkout main
git pull --ff-only origin main
git status
git rev-parse HEAD
python -m pip install -e ".[chess,malecns,dashboard,dev]"
pytest -q
```

Mandatory tasks:

- create a new reproducibility manifest for this discovery day;
- verify SHA-256 of Stockfish, MaleCNS Feather inputs, populations, v1 checkpoint, training dataset, and the old 140-position corpus;
- fix the `serious_400_bounded` summary/report metadata inconsistency by deriving `max_plies`, candidate bounding/full-legal status, termination mix, and recurrent-pass interpretation from the raw game artifact rather than stale configuration fields;
- add a test that prevents this mismatch from recurring;
- do not reinterpret the short-game rating as normal Elo.

**Acceptance:** baseline tests pass, known report bug is fixed, manifest is written, and no old raw artifact is modified.

---

# Block 1 — Correct the statistics: cluster-aware uncertainty and selection-aware summaries

The old 140 positions come from only 20 source openings with several correlated offsets per opening. The current bootstrap resamples positions individually. Add cluster-aware analysis using `source_game_id` as the resampling unit.

For every depth and graph variant calculate:

- mean/median regret;
- D1-to-D paired improvement;
- ordinary position bootstrap CI for continuity with old reports;
- **cluster bootstrap CI by source_game_id** as the preferred uncertainty estimate;
- improved / unchanged / worsened fractions;
- teacher-best and top-3 agreement changes;
- rank and correlation metrics where available.

For original-vs-control difference-in-differences, also cluster bootstrap by source group.

Because D2 was selected after looking across depths, add an exploratory selection-aware summary. At minimum report both:

```text
predeclared depth-specific CIs
best-observed-depth effect (explicitly exploratory / selected on same data)
```

If practical, add a max-statistic or nested/bootstrap correction for selecting the best depth. Do not overstate nominal CIs after depth selection.

Write:

```text
results/mechanistic_discovery_v1/statistics/
```

with compact CSV/JSON and plots.

**Interesting-fact target:** determine whether any apparently strong control comparison survives clustered uncertainty.

---

# Block 2 — Audit the control graphs before trusting biological-specificity claims

The current “degree-preserving topology shuffle” randomizes destination indices and then converts back to CSR. Sparse duplicate coordinates can coalesce. Audit exactly what the control preserves *after* construction.

For original and current shuffled graph measure:

- neuron count;
- stored nonzero edge count;
- total edge-weight sum and absolute-weight sum;
- in-degree and out-degree sequence equality;
- weighted in/out-strength distributions;
- number of duplicate edge collisions induced by shuffle;
- self-loop count;
- connected-component / reachability summaries if computationally reasonable.

If the current control does not preserve the intended graph statistics exactly enough, implement a cleaner degree-preserving rewire, preferably directed double-edge swaps that preserve in-degree and out-degree without duplicate/self-loop artifacts unless explicitly allowed.

Add tests proving the invariants of the corrected control.

Do **not** delete the old control results. Name the corrected control separately, e.g.:

```text
degree_preserving_edge_swap_v2
```

**Interesting-fact target:** establish whether the surprising superiority of the old topology shuffle could partly be a control-construction artifact.

---

# Block 3 — Build recurrent-dynamics observability

Instrument the rate engine/experiment path so we can see *what changes internally as depth increases* without changing scoring semantics.

For each requested depth collect compact per-position/per-candidate summaries such as:

- hidden-state L2 norm;
- state delta norm and relative delta;
- recurrent-drive norm;
- sensory-drive norm;
- recurrent/sensory drive ratio;
- fraction of sampled/full units with `|state| >= 0.90`, `0.95`, `0.99`;
- readout-population mean/std/norm;
- candidate-state centroid norm;
- within-position candidate dispersion;
- mean pairwise cosine similarity among candidate states;
- effective rank of the candidate-by-readout representation;
- fly score dispersion and top1-top2 margin;
- identity of leading move at each depth;
- number/depth of winner switches;
- rank of the eventual D64 winner at earlier depths;
- whether the D2 winner later gets displaced;
- trajectory latency.

Do not dump complete 166k-neuron states for every position. Compute compact summaries online or retain only small sampled/readout matrices needed for diagnostics.

Validate that enabling observability changes neither candidate scores nor rankings beyond strict float tolerance.

Write a reusable API rather than one-off print statements.

**Interesting-fact target:** identify a measurable signature that appears around the D2→D4 collapse.

---

# Block 4 — Create a fresh development corpus large enough for discovery

Do not use the consumed 140-position evaluation corpus for tuning.

Create a **development-only** corpus with substantially more independent source groups. Target:

```text
64 source games/trajectories x 4 sampled positions = 256 development positions
```

If generation is very fast, stretch to 96 source groups / 384 positions. Prefer source diversity over many highly correlated offsets from only a few openings.

Requirements:

- deterministic generation seed;
- unique FENs;
- no FEN overlap with training/validation or old evaluation corpus;
- legal nonterminal positions;
- source-group ID retained;
- balanced early/middle/late sampling where practical;
- teacher score for every legal move;
- stronger teacher labeling than the old 1,000-node corpus if affordable; target 10,000 nodes, with a stability check against a higher-node sample;
- explicit `development_only` metadata.

Teacher stability sub-study: randomly/predeclared select at least 32 development positions and compare 1k-node vs 10k-node vs 50k-node teacher ordering/regret where affordable. Report best-move agreement and score/rank stability. This tells us whether teacher noise is materially affecting our conclusions.

Write under:

```text
data/chess_development_corpus_v1.csv
results/mechanistic_discovery_v1/teacher_stability/
```

Do not use this corpus for the final future confirmation.

---

# Block 5 — Main mechanistic factorial: recurrent gain × input mode × depth

This is the largest mandatory experiment of the day.

The 5%-recurrent control suggests current recurrence may simply be too strong. Test that hypothesis directly on the new development corpus.

## 5A. Broad gain screen

For the **original MaleCNS topology**, test relative recurrent-weight scales:

```text
0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25
```

where `1.00` reproduces the current baseline recurrence and all other dynamics remain fixed.

For every scale test both:

```text
clamped sensory input at every recurrent step
single-pulse sensory input only at depth 1
```

and depths:

```text
1, 2, 4, 8, 16, 32, 64
```

Use the multi-depth batch evaluator and all legal moves.

For each condition collect move quality plus the new internal-dynamics observability metrics.

This is deliberately much larger than the previous plan. Do not stop after a tiny smoke subset. Run at least 128 development positions for the entire grid; use all 256 if runtime permits within the day. Because D1 is mathematically shared across gain settings/modes where appropriate, reuse exact computations only when equivalence is proven.

## 5B. Focused confirmation inside development data

Split development source groups *before looking at results* into:

```text
dev_screen
dev_confirm
```

Use `dev_screen` to identify up to three interesting recurrent regimes, for example the best shallow-quality regime, the best deep-quality regime, and baseline 1.00. Then evaluate those choices on `dev_confirm` without retuning.

This is still development work, not the final held-out confirmation, but it prevents us from chasing pure noise.

**Interesting-fact targets:**

- Is there a recurrent-gain band where D8/D16/D32 actually improves over D1?
- Does single-pulse input avoid the D4+ collapse?
- Does the internal state show a clear transition from useful propagation to saturation/candidate collapse?
- Is “more thinking” beneficial only when recurrence is weak enough?

---

# Block 6 — Information preservation vs representation rotation: depth-specific probe matrix

This is a high-priority mechanistic experiment.

The current fixed readout was trained at D16, yet D2 is best. Determine whether deeper recurrence destroys chess information or merely moves it into directions the frozen readout does not use.

Using training/validation data only:

1. Extract or generate candidate activation caches at depths:

```text
1, 2, 4, 8, 16, 32, 64
```

2. Use the **same position-level train/validation split** for every depth.
3. Standardize features using training statistics only.
4. Train a separate linear ranking probe at each depth using a robust optimizer (L-BFGS or another deterministic convex solver is preferred if available).
5. Evaluate every trained probe on activations from every inference depth.

Produce a `7 x 7` cross-depth transfer matrix for at least:

- validation pair accuracy;
- validation top-1 teacher agreement;
- mean validation teacher regret if legal candidate labels are available.

Interpretation:

```text
Depth-specific probe remains strong at D16/D32 while frozen D16 probe fails elsewhere
    -> representation rotation / readout mismatch.

All probes become weak as depth grows
    -> task information is being destroyed/collapsed.

Probe quality improves with depth after gain tuning
    -> recurrence may create more linearly usable information under the right dynamics.
```

Also compute per-depth effective rank, within-position candidate distance, and score separability.

## 6B. Non-connectome baselines

Train comparable linear ranking probes on:

- raw `encode_board_move` features;
- sensory-projected features before recurrent propagation;
- MaleCNS activations at each depth.

Keep splits identical.

This answers a basic but important question: **is MaleCNS recurrence adding linearly useful chess signal beyond the encoder itself, or mostly degrading it?**

Write all matrices/plots under:

```text
results/mechanistic_discovery_v1/probe_transfer/
```

---

# Block 7 — Position-level discovery: where does recurrence help or hurt?

Use the development experiment to characterize the positions rather than only reporting global means.

For each position derive simple interpretable descriptors:

- phase / ply;
- side to move;
- legal move count;
- teacher best-vs-second margin;
- baseline D1 regret;
- whether teacher-best move is capture/check/promotion/castling;
- material balance if easy to compute;
- tactical volatility proxy from teacher score spread;
- internal state stability at D1/D2/D4;
- candidate representation dispersion/effective rank;
- number of leader switches across depth.

Analyze D1→D2, D1→D8, and D1→best-development-regime improvements against these descriptors.

Use grouped summaries, rank correlations, and simple regularized regression/classification if helpful. Bootstrap by source group. Mark this entire block **exploratory** and avoid p-value fishing.

Produce a ranked table of the most improved and most harmed positions with FEN, selected moves, teacher moves, regret change, and internal-dynamics metrics.

**Interesting-fact targets:** examples of concrete chess situations where recurrence helps, and a mechanistic signature predicting when extra depth becomes harmful.

---

# Block 8 — Replace one-off shuffled controls with control ensembles

One random shuffled graph is not a strong biological-specificity baseline. Build small ensembles.

On a predeclared subset of at least 64 development positions, evaluate:

```text
10 corrected degree-preserving topology shuffles
10 transmitter-sign shuffles
original MaleCNS
```

Use control seeds fixed in advance, e.g. `1000..1009`, and at minimum depths:

```text
2, 8, 16, 64
```

If Block 5 identifies a clearly better recurrent-gain regime, evaluate the ensemble both at baseline gain and that selected development regime if runtime permits.

Report the original graph as a percentile within the shuffle distribution for regret, teacher agreement, internal effective rank, candidate-state separation, and deep-collapse severity.

This is much more informative than asking whether the original beats one arbitrary shuffle.

**Interesting-fact target:** determine whether biological MaleCNS is an outlier, ordinary member, or poor member of the matched random-graph ensemble for these chess computations.

---

# Block 9 — Adaptive recurrent stopping from internal signals

Do not use Stockfish/teacher information at inference time for the stopping decision.

Using only `dev_screen`, test simple policies such as:

- stop when the same candidate leads for N consecutive requested depths;
- stop when score margin exceeds a threshold and remains stable;
- stop when state relative delta falls below a threshold;
- stop when candidate-state dispersion begins collapsing;
- stop when a learned lightweight rule over internal metrics predicts that another depth step is more likely to hurt than help.

Tune policy thresholds on `dev_screen`, then freeze and evaluate on `dev_confirm`.

Compare against fixed D1/D2/D4/D8/D16 baselines on:

- teacher regret;
- top-1/top-3 agreement;
- average recurrent passes;
- latency;
- fraction of positions assigned each stopping depth.

The interesting outcome is not merely speed. We want to know whether **adaptive depth beats every single fixed depth on quality/compute tradeoff**, which would be a genuine recurrent-thinking result even if “always think longer” fails.

Write results under:

```text
results/mechanistic_discovery_v1/adaptive_depth/
```

---

# Block 10 — Conditional v2 decoder experiment

This is secondary today. Only proceed after Blocks 1–9 have produced the mechanistic evidence.

If the probe-transfer experiment shows recoverable chess information at some depth/regime, train a better Stage-0 decoder using train/validation data only. Candidate improvements may include:

- feature standardization/whitening;
- stronger deterministic convex ranking optimization;
- more fitting positions;
- depth-specific training at the empirically informative development depth;
- a clearly versioned recurrent-gain regime if Block 5 supports one.

Do **not** evaluate v2 on the consumed 140-position corpus for selection, and do not claim a new headline result without a fresh confirmatory corpus.

Preserve v1 unchanged.

---

# Block 11 — Synthesis: turn the day into facts, not just files

Create:

```text
results/mechanistic_discovery_v1/discovery_report.md
results/mechanistic_discovery_v1/discovery_summary.json
```

The report should begin with a table of **new facts learned today**, each with:

```text
finding
evidence
sample size / source groups
uncertainty
whether exploratory or confirmatory
what competing explanation it rules out
what remains unresolved
```

At minimum synthesize:

- clustered reanalysis of the 140-position study;
- graph-control audit;
- mechanism of deep-depth deterioration;
- gain × clamp-mode response surface;
- probe-transfer result (information loss vs rotation);
- raw-feature/sensory/MaleCNS baseline comparison;
- position-type effects;
- shuffle-ensemble result;
- adaptive-depth result;
- teacher-label stability.

Update `docs/current_results.md`, `docs/experiment_protocol.md`, and `docs/chess_training.md` as needed. Keep `Research Goal.md` unchanged.

Finish with:

```powershell
pytest -q
git status
```

Commit small code, CSV/JSON summaries, plots, tests, and docs. Do not commit huge transient state dumps or engine/data binaries.

---

# If the mandatory work finishes early: extension ladder

Do not end the day early. Continue in this order until the workstation time budget is genuinely used:

## Extension A — Increase development sample size

Expand from 256 toward 512+ positions while increasing independent source groups, then rerun the most informative gain/input regimes and cluster-aware analysis.

## Extension B — More shuffle seeds

Increase from 10 to 25 or 50 corrected topology shuffles on a smaller fixed position subset to get a cleaner empirical null distribution for biological-topology specificity.

## Extension C — Leak × recurrent-gain interaction

For the top 2 gain regimes, screen leak values such as:

```text
0.00, 0.05, 0.15, 0.30, 0.50, 0.70
```

on `dev_screen`, then confirm the best few on `dev_confirm`. This tests whether the D4+ collapse is fundamentally a recurrence-strength problem or a memory/update-timescale problem.

## Extension D — Local stability / Jacobian probes

On a small predeclared position subset, estimate local recurrent amplification using Jacobian-vector products or finite-difference perturbation growth across depth. Compare original, attenuated, and shuffled graphs. Avoid massive dense Jacobians.

Question: does the onset of poor chess performance coincide with a local dynamical gain > 1 or rapidly amplifying perturbations?

## Extension E — Candidate perturbation sensitivity

Add tiny deterministic perturbations to the encoded candidate or sensory state and measure whether selected moves/scores become increasingly unstable with depth. Compare regimes.

## Extension F — LIF spot-check

On a small fixed subset only, repeat the most diagnostic baseline vs tuned-regime comparison under LIF dynamics to see whether the discovered mechanism is specific to the rate abstraction. Keep this explicitly secondary because more LIF steps also imply more simulated biological time.

---

# End-of-day decision tree

Use the evidence, not preference:

```text
If weaker/tuned recurrence + deeper depth improves move quality robustly:
    -> recurrent computation may be useful, but baseline dynamics were mis-scaled.
    -> freeze a tuned regime and plan a fresh held-out confirmation.

If depth-specific probes stay strong while the fixed readout collapses:
    -> information survives but representation rotates.
    -> improve readout/training rather than blaming recurrence itself.

If all probes lose information at depth and candidate states collapse/saturate:
    -> current recurrence destroys task information.
    -> focus on dynamics/normalization/input protocol before more chess training.

If original MaleCNS is not better than the shuffle ensemble:
    -> no biological-topology specificity claim.

If original is a robust outlier under corrected controls:
    -> this becomes a high-priority result for fresh confirmation.

If adaptive stopping beats fixed depths on dev-confirm:
    -> pursue adaptive test-time recurrent compute as a major research branch.

If raw encoder features beat MaleCNS features at every depth:
    -> connectome currently acts as a lossy transformation for chess.
    -> quantify the loss and redesign how chess information is injected/read out.
```

The priority for this day is **discovery**. Prefer learning something mechanistically sharp over producing another nominal Elo number or simply adding more training epochs.