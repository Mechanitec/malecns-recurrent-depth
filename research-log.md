# MaleCNS recurrent-depth research log

Status: the mechanistic-discovery goal was canceled on 2026-09-14. This log records the complete path taken up to cancellation. It separates executed evidence from prepared but unexecuted work.

## Repository and operating state

- Repository: `https://github.com/Mechanitec/malecns-recurrent-depth`
- Working branch: `main`
- Final repository commit before this log: `9a1498a` (`Fix factorial queue paths on Windows`)
- `main` and `origin/main` were synchronized before cancellation.
- Only `main` remains locally and on `origin`; temporary and unused branches were removed.
- The repository was made public during the project workflow.
- CSV tracking was corrected. The development corpus and its metadata are tracked, while large local datasets remain ignored.
- Existing unrelated untracked legacy results were preserved. They were not deleted because their ownership and retention scope were not explicit.
- Cancellation cleanup stopped all 8 processes belonging to `run_mechanistic_factorial.py`. No factorial worker remained after cleanup.

## Initial research and product path

The work began as an operational chess-engine and dashboard project and then narrowed into a reproducible recurrent-depth research program.

1. The repository was synchronized with GitHub and the public `main` branch became the authoritative working state.
2. Chess smoke tests, rating runs, dashboard runs, live-game state, and benchmark artifacts were inspected repeatedly.
3. The Fly agent was configured to play lower-level opponents first instead of beginning directly against Stockfish. The historical game and rating outputs were treated as diagnostic until the protocol was trustworthy.
4. A 1.5-second move-time setting was requested for Alfil/Fly play. The engine and benchmark path retained explicit time-control metadata rather than interpreting a short diagnostic game as chess strength.
5. The dashboard error shown in the attached screenshot was traced to training-history schema handling. `load_training_history` now accepts either `step` or `epoch`, synthesizes `step` from `epoch`, and normalizes optional columns. The training script writes a dashboard-compatible `training_history.csv`.
6. Pull, PR, merge, and branch-cleanup requests were handled through GitHub. The final branch policy was reduced to `main` only.

## Baseline diagnosis

The locked research contract is in `Research Goal.md`; it was preserved and not edited. The supporting protocol and current-results documents were updated to distinguish exploratory ratings from valid held-out position evidence.

The first important conclusion was that the existing Fly decoder was weak. The depth-16 rate checkpoint used a frozen MaleCNS graph, fixed sensory projector, fixed readout population, and a learned linear readout. Its diagnosis showed:

- 2,000 cached candidate rows from 65 positions;
- 256 readout features with effective rank about 4.55;
- mean within-position candidate distance about 0.0043;
- checkpoint validation pair accuracy about 58.2%;
- checkpoint validation top-1 agreement about 23.1%;
- recomputed all-cache top-1 agreement about 12.3%;
- no evidence that simple bounded-activation saturation or all-zero activity alone explained the weakness.

This established a decoder/interface limitation and prevented the project from treating the frozen readout as a strong chess player.

## Early chess benchmarks

An 8-position pilot initially suggested that deeper recurrence could help. On the original graph, teacher regret improved from 756.500 cp at D1 to 345.375 cp at D2 and reached 316.375 cp at D8/D16. The pilot was explicitly labeled exploratory because 8 positions made agreement statistics extremely coarse.

The larger independent study did not reproduce that strong effect. The historical short-game rating artifacts also contained forced-short-game adjudication and were therefore rejected as evidence for a genuine Elo claim.

## Independent 140-position study

The completed v1 position study used 140 legal, nonterminal positions from 20 source trajectories. It evaluated all legal candidate moves at depths 1, 2, 4, 8, 16, 32, and 64 for the original graph and three controls, producing 3,920 rows.

Original-graph results:

- D1 mean teacher regret: 581.39 cp.
- D2 mean teacher regret: 570.51 cp.
- D1-to-D2 paired improvement: 10.88 cp.
- Cluster-bootstrap interval for the paired improvement: approximately -92.48 to 106.33 cp.
- Regret worsened at D4 and remained higher through the deeper settings.

The position-level result did not support a reliable H1 claim that additional recurrence improves chess decisions. The control comparison also did not establish MaleCNS-specificity. The earlier one-shuffle result was considered insufficient because positions were clustered by source trajectory and the shuffle implementation needed an invariant audit.

## Corrected bounded-artifact reporting

The serious bounded rating artifact was reanalyzed from raw PGN and telemetry instead of trusting stale summary fields. The reporting fix derives legal-candidate coverage, observed maximum ply, candidate-evaluation count, termination mix, and recurrent-pass interpretation.

For the affected 400-game artifact, the corrected interpretation is:

- 400 games;
- observed maximum: 2 plies;
- termination: `MAX_PLIES_ADJUDICATION`;
- 1,600 bounded candidate evaluations;
- 8,000 legal candidates represented by the PGNs;
- 25,600 total recurrent passes.

A regression test uses a short PGN with a bounded candidate set and confirms that the report distinguishes full-legal candidates from bounded candidates.

## Mechanistic-discovery preflight

The GitHub plan was expanded into Blocks 0 through 11. Its purpose was to determine whether the deep-depth collapse came from state amplification, saturation, candidate-state collapse, winner switching, representation rotation, input clamping, or a generic rather than biological recurrent effect.

### Block 0: reproducibility and metadata

Executed. The preflight recorded repository, checkpoint, graph, projector, dataset, Stockfish, and artifact hashes. It also fixed the serious bounded-artifact metadata bug described above.

### Block 1: clustered uncertainty

Executed through `scripts/analyze_clustered_position_depth.py`. The analysis added source-group bootstrap intervals and a selection-aware depth summary using 2,000 bootstrap samples with seed 23.

Key values:

- Best observed depth: D2.
- D2 mean regret: 570.5071 cp.
- Nominal paired position-bootstrap interval: -59.40 to 80.25 cp.
- Cluster-bootstrap interval: -92.48 to 106.33 cp.
- Selection-aware interval: 0.00 to 104.91 cp, exploratory because D2 was selected after inspecting the same corpus.
- Selected-depth frequencies in the exploratory selection summary: D1 760, D2 1,149, D4 9, and D8 82.

Outputs are under `results/mechanistic_discovery_v1/statistics/`.

### Block 2: graph-control audit

Executed. The old destination-permutation shuffle was shown not to preserve the intended graph invariants after sparse duplicate coalescing:

- original graph: 164,620 neurons and 10,227,924 edges;
- old shuffle: 10,207,236 edges, 133 self-loops, non-exact degree sequences, and 20,770 duplicate-coordinate collisions;
- corrected shuffle: 10,227,924 edges, 42 self-loops, exact in/out-degree sequences, 0 duplicate-coordinate collisions, and 531 weak components.

The corrected directed double-edge swap is named `degree_preserving_edge_swap_v2`. Tests cover degree preservation, duplicate rejection, determinism, and weight handling. The old study outputs were preserved unchanged.

### Block 3: observability API

Executed in the rate engine. `run_batch_trajectory(..., observe=True)` now records per-candidate observables without changing plain-state results or scores:

- hidden-state norm and step delta;
- recurrent-drive norm and sensory-drive norm;
- recurrent/sensory drive ratio;
- saturation fraction;
- readout statistics and candidate geometry in the factorial runner;
- score spread, rank correlation, winner switches, D64 winner rank, D2 displacement, and latency.

A regression test confirms that observed and unobserved trajectories produce identical states.

An exact zero-gain optimization was added. When recurrent gain is exactly zero, the engine skips graph multiplication while retaining the same state update. Focused tests passed for the optimized path.

### Block 4: fresh development corpus and teacher stability

Executed. `data/chess_development_corpus_v1.csv` contains 256 legal, nonterminal positions from 64 independent source groups. The phase distribution is 64 early, 64 middle, and 128 late positions. The corpus contains 7,496 legal candidate rows and is tracked in Git with its metadata file.

The teacher-stability subset contains 32 predeclared positions and was evaluated with Stockfish 19 at 1,000, 10,000, and 50,000 nodes using one thread and a 128 MB hash. Best-move agreement with the 50,000-node result was:

- 1,000 nodes: 0.500;
- 10,000 nodes: 0.625;
- 50,000 nodes: 1.000 by definition.

Outputs are under `results/mechanistic_discovery_v1/teacher_stability/`.

### Block 5: gain and input-mode factorial

Started but canceled. The predeclared grid was 10 recurrent-gain scales, two input modes, seven depths, and all legal candidates across 256 development positions: 35,840 expected condition rows.

Completed before cancellation:

- gain 0.00, clamped sensory: 1,792 rows;
- gain 0.00, single pulse: 1,792 rows.

Partial resumable outputs at cancellation:

- gain 0.02, clamped sensory: 448 rows;
- gain 0.02, single pulse: 644 rows;
- gain 0.05, clamped sensory: 168 rows;
- gain 0.05, single pulse: 56 rows.

The gain 0.10 files had no rows. No merged factorial summary was claimed, and no gain regime was selected from the incomplete data.

The factorial queue was made resumable and bounded to four workers. A Windows path-quoting issue was found and fixed in the queue launcher, then the queue was stopped as part of cancellation cleanup.

## Prepared but not executed after cancellation

The following work was implemented or prepared but did not produce a completed research result:

- `scripts/run_probe_transfer.py` for a 7 x 7 cross-depth linear-probe matrix with train-only standardization and raw/sensory/MaleCNS baselines;
- `scripts/analyze_mechanistic_positions.py` for exploratory position descriptors, ranked improvements/harm, and grouped correlations;
- `scripts/run_control_ensemble.py` for corrected topology and transmitter-sign control members;
- `scripts/merge_control_ensemble.py` for original-graph percentiles across the control ensemble;
- `scripts/evaluate_adaptive_stopping.py` for dev-screen tuning and dev-confirm evaluation using internal signals only;
- `scripts/write_mechanistic_discovery_report.py` for final synthesis artifacts;
- `scripts/run_factorial_queue.ps1` for bounded resumable factorial scheduling.

Blocks 6 through 11 were therefore not completed. In particular, there is no valid completed probe-transfer matrix, control ensemble, adaptive-stopping result, discovery report, or final mechanistic conclusion from those blocks.

## Validation performed

- Focused engine, graph-control, reporting, and observability tests passed during implementation.
- The full suite passed with 55 tests before the final zero-gain optimization commit.
- The final post-optimization full-suite run was not completed before the goal was canceled.
- New analysis runners were syntax-checked with `py_compile`.
- `git fetch origin` confirmed that local `main` matched `origin/main` before this log was created.

## Final state and interpretation

The strongest completed evidence is that the original frozen v1 system has a weak decoder, D2 is only a small and uncertain improvement over D1 in the 140-position study, deep-depth performance deteriorates, and the original graph-control shuffle required correction before biological-specificity claims could be made. The fresh development corpus and teacher-stability preflight are complete, but the gain factorial and all downstream mechanistic blocks were canceled before completion.

The canceled work does not justify selecting a new gain, input mode, adaptive policy, decoder, or biological-specificity conclusion. Existing v1 artifacts remain preserved for a future restart.

## Plan 2 execution: neuron population study

Plan 2 is being executed on PR #9 in the isolated branch `plan2-neuron-population-study-v2`.

Completed preparation and preflight:

- Phase 0 built `results/population_study/malecns_region_atlas.csv` and `.json` from the filtered MaleCNS annotations. The atlas contains the requested MB/Kenyon, MBON, SMP/CRE/SIP, FB, hDelta, FC/PFN/PFR/PFL, central-complex, LAL, descending, motor/efferent and visual families.
- Phase 1 audited Population Baseline A and recorded graph distances, degree properties and activation diagnostics.
- Phase 2 created deterministic, hashed input/readout manifests under `data/populations_v2/manifests/`. Manifest indices were corrected to use the filtered graph index order rather than raw annotation-row indices.
- A new 256-position, 64-source-group confirmatory corpus was generated with a different seed and Stockfish 19 at 10,000 nodes. It contains 7,571 legal-move rows and separate source-group screen/confirm splits.
- The full test suite passed with 55 tests after the Plan 2 preparation changes.

Phase 3 is running as four contiguous 64-position extract-only shards. Each shard uses the full 164,620-neuron, 10,227,924-edge MaleCNS graph at depths 1, 2, 4, 8, 16, 32 and 64. The shard caches will be merged with position and candidate-row count validation before probes are fitted.

The first full single-process attempt exposed and fixed two reproducibility defects before valid results were accepted: local rather than global candidate-row offsets, and raw annotation indices used where filtered graph indices were required. No Phase 3 scientific result is claimed until the merged cache and probe outputs complete.

## Relevant commits

| Commit | Purpose |
|---|---|
| `7806171` | Complete the 140-position recurrent-depth study |
| `f233143` | Pin study provenance to the final commit |
| `33e498e` | Expand the mechanistic discovery plan |
| `3e2fb21` | Align current results with the plan |
| `0b99f80` | Strengthen the recurrent-depth protocol |
| `700f2a1` | Run mechanistic discovery preflight |
| `1bdbc13` | Keep only the intended development CSV tracked |
| `c2235ff` | Skip zero-gain recurrent propagation exactly |
| `014c329` | Add mechanistic discovery analysis runners |
| `9a1498a` | Fix the Windows factorial queue path handling |

