# MaleCNS Recurrent-Depth Chess: Codex Day Plan

**Execution target:** one full local workday on the Windows workstation using Codex / Luna High.

**Research scope:** `Research Goal.md` is LOCKED. Do not edit or reinterpret it. This plan advances Experiment A (frozen fly + recurrent depth) and prepares a stronger Stage-0 decoder for the next iteration without changing the overall research goal.

## Current state and evidence

The end-to-end real-MaleCNS chess pipeline works: real MaleCNS graph -> fixed chess projection -> recurrent rate dynamics -> trained linear readout -> legal-move ranking -> calibrated weak opponents / Stockfish analysis.

The first full-legal held-out sweep used only 8 positions. Its current original-graph results are:

| Depth | Teacher agreement | Teacher regret (cp) | Stockfish loss (cp) | Approx. position latency |
|---:|---:|---:|---:|---:|
| 1 | 0.000 | 756.5 | 685.125 | 1.12 s |
| 2 | 0.125 | 345.375 | 446.625 | 2.28 s |
| 4 | 0.125 | 354.875 | 453.0 | 4.67 s |
| 8 | 0.125 | 316.375 | 463.0 | 9.05 s |
| 16 | 0.125 | 316.375 | 463.0 | 16.67 s |
| 32 | 0.125 | 493.0 | 545.375 | 33.20 s |
| 64 | 0.125 | 519.875 | 520.5 | 69.98 s |

Interpretation: D1 -> D2/D8 shows a potentially large recurrent-compute benefit, but 8 positions are far too few. The degree-preserving shuffled control is sometimes as good as or better than the real graph, and 5%-recurrent-strength attenuation often matches the real graph. Therefore there is **no connectome-specific claim yet**.

The existing short-game rating numbers are diagnostic only. Runs capped after 2-4 plies are adjudicated as draws, so the repeated ~445 rating in compact sweeps and ~410 rating in the 402-game short protocol are not valid long-game chess-strength measurements.

The v1 readout is also weak: pairwise logistic loss remains close to `ln(2) ~= 0.6931`, validation top-1 agreement is usually around 8-23%, and pairwise candidate accuracy is only modestly above 0.5. Treat v1 as the frozen baseline checkpoint, not as the final decoder.

---

# Operating rules

1. Begin by synchronizing exactly:

   ```powershell
   git checkout main
   git pull --ff-only origin main
   git status
   git rev-parse HEAD
   ```

2. Run all work locally. **Do not use GitHub Actions.**
3. Do not edit `Research Goal.md`.
4. Do not overwrite or delete the v1 baseline results. New experiments get new directories/versioned names.
5. Do not tune on the held-out evaluation corpus. Hyperparameter/model selection uses train/validation only.
6. Keep graph topology, populations, projector, dynamics, checkpoint and data fixed inside each causal depth comparison. Only `D` may vary.
7. Use all legal moves for the position-quality experiment. Do not use `max_candidates=4` for headline position metrics.
8. Every long experiment must be resumable and must append/flush incremental results so an interruption does not discard completed work.
9. Record code commit, input hashes, checkpoint hash, engine hashes/options, seeds, graph variant, depths and dataset IDs in metadata.
10. Do not commit engine binaries, Feather files, large NPZ caches or huge raw traces. Commit small CSV/JSON summaries, plots and documentation.
11. A negative result is valid. Do not reroll controls, positions, seeds or reporting to make MaleCNS look better.
12. Before each substantive commit run focused tests; before ending the day run `pytest -q`.

---

# Today's primary question

Using a much larger set of unseen positions, does a fixed MaleCNS-derived system make better chess decisions when the **same recurrent block** is applied more times?

Primary causal quantity for each depth `D`:

```text
Delta_regret(D) = mean[regret(position, D=1) - regret(position, D)]
```

Positive `Delta_regret` means deeper recurrence improves move quality relative to D1.

Primary biological-specificity quantity for each control `C`:

```text
Delta_specific(D, C)
  = (regret_original_D1 - regret_original_D)
    - (regret_control_D1 - regret_control_D)
```

A MaleCNS-specific recurrent-depth claim requires a positive effect that is robust under paired uncertainty and is materially stronger than appropriate controls. If controls improve equally, report a generic recurrent-computation effect instead.

---

# Day deliverables

By end of day, aim to have all of the following:

1. An exact, tested high-throughput position evaluator that can score all legal candidates and collect **multiple recurrent depths from one trajectory** where mathematically equivalent.
2. A frozen independent evaluation corpus substantially larger than 8 positions (target 256; minimum useful target 128 if full-MaleCNS runtime is limiting).
3. A paired depth/control study over `D = 1,2,4,8,16,32,64` for original MaleCNS, degree-preserving topology shuffle, transmitter-sign shuffle, and recurrent edges attenuated to 5%.
4. Position-level raw results, aggregate CSV, paired bootstrap CIs and difference-in-differences control analysis.
5. Updated plots that emphasize move quality, not the invalid short-game rating.
6. A diagnosis of why v1 readout training stays near random pairwise loss.
7. If time allows, a v2 Stage-0 readout trained with a numerically stronger convex/normalized procedure, selected only on validation data.
8. Updated final/current-results documentation stating exactly what is and is not supported.
9. Full local test suite passing.

---

# Work Block 0 - Preflight and reproducibility snapshot

**Budget:** 15-30 minutes.

Run:

```powershell
git checkout main
git pull --ff-only origin main
python -m pip install -e ".[chess,malecns,dashboard,dev]"
pytest -q
```

Verify local paths for Stockfish, Minic, Gaia, the three MaleCNS Feather inputs, `data/chess_sensory_indices.npy`, `data/chess_readout_indices.npy`, and `results/training/v1_depth16_rate_large/readout_checkpoint.npz`.

Write a local run manifest before experimentation, including SHA-256 of all critical inputs. Do not continue silently if the checkpoint or MaleCNS inputs differ from those used for the baseline.

**Acceptance:** baseline tests pass and input hashes are recorded.

---

# Work Block 1 - Make large depth studies computationally feasible

**Budget:** 1-2 hours implementation + tests.

The current scalar sweep recomputes every candidate separately at every requested depth. That wastes most recurrent work. Implement an exact evaluation path optimized for the frozen rate engine.

## 1A. Multi-depth trajectory reuse

For one candidate input, run the recurrent state once up to max depth 64 and capture states/readout scores at `1,2,4,8,16,32,64`.

Do not independently rerun D1 + D2 + D4 + ... when the deterministic D64 trajectory already contains those intermediate states.

Add an engine/API helper rather than duplicating recurrent math in an experiment script if practical.

## 1B. Batch legal-candidate evaluation

Investigate and implement a batch path for the rate engine where all legal candidate states are propagated together using sparse-matrix x dense-matrix operations.

Expected state shape is roughly `[n_neurons, n_legal_candidates]`. This is acceptable for ~166k neurons and normal chess branching factors if float32 is used carefully.

Requirements:

- exact same graph and recurrence equations as scalar mode;
- deterministic ordering of legal moves;
- no change to chess feature encoding/projector semantics;
- no training or approximation introduced merely for speed.

## 1C. Equivalence tests

For several positions and graph variants, compare scalar versus optimized scores at every depth.

Acceptance tolerance should be strict, e.g. `np.allclose` with a justified float32 tolerance. Chosen move and complete ranking should match unless differences are below a documented numerical tie threshold.

Add tests for multi-depth snapshots, batch-vs-scalar candidate scores, original plus at least one control graph, all-legal candidate retention, and deterministic repeated runs.

## 1D. Benchmark throughput

Benchmark representative positions before/after optimization at D16 and D64. Save a small JSON/CSV summary.

**Success target:** >=2x total speedup for a complete 1..64 multi-depth position evaluation. If batching is slower on the actual sparse matrix, keep only the safe multi-depth optimization and record that honestly.

Commit code/tests before launching the long experiment.

---

# Work Block 2 - Freeze an independent evaluation corpus

**Budget:** 30-90 minutes plus teacher labeling compute.

Build a versioned evaluation corpus that is **never used to train or select the readout**.

Preferred target: `256` unique positions. Minimum for today's full control study: `128`. Stretch target after optimization: `512`.

Corpus requirements:

- no FEN overlap with train or validation positions used for v1/v2 readout fitting;
- no candidate-row leakage;
- deterministic source/seed;
- non-terminal legal positions;
- reasonable early/middle/late spread;
- exact FEN, side to move, source/game ID, ply, legal move count;
- Stockfish teacher score for every legal move from side-to-move perspective;
- exact Stockfish hash and deterministic nodes/depth limit;
- explicit `evaluation` / `test_only` status.

Prefer a suitable local PGN corpus with source-game grouping. If none exists, create a deterministic ordinary-position corpus using a documented engine/self-play or opening-expansion process. Do not use bizarre uniformly-random legal positions as the primary benchmark.

Programmatically verify all FENs/moves, one deterministic teacher-best move per position, no duplicate FENs, no overlap with fitting data, and correct score orientation. Write compact corpus metadata with hashes.

---

# Work Block 3 - Main paired position-depth/control study

**Budget:** largest block of the day.

Create or extend a dedicated resumable runner, preferably `scripts/run_position_depth_study.py`, separate from game tournaments. Resume at `(variant, position_id)` granularity and produce one raw row per position / variant / depth.

Hold fixed the v1 checkpoint, preprocessing, populations, projector, rate dynamics, clamp mode, evaluation corpus and control seeds.

Run all four graph variants:

```text
original
degree_preserving_topology_shuffle
transmitter_sign_shuffle
recurrent_edges_attenuated_0.05
```

Use one predeclared control seed. Do not reroll a shuffle because the result is inconvenient.

Run depths:

```text
1, 2, 4, 8, 16, 32, 64
```

Per-position/per-depth record at least selected move, teacher best, selected/best cp, regret cp, teacher-best agreement, selected teacher rank, fly rank of teacher best, top-3 agreement, candidate count, top1-top2 fly score margin, score dispersion, recurrent passes, latency, graph variant, depth and position ID. Where cheap, add Spearman/Kendall rank correlation between fly scores and teacher scores.

The root-move teacher scores are the primary move-quality metric. Independent post-move Stockfish evaluation is expensive; use a fixed predeclared subset such as 32 or 64 positions unless throughput allows more. It must never influence move selection.

Flush incremental outputs continuously, e.g.:

```text
results/position_depth_study_v2/raw_position_metrics.csv
results/position_depth_study_v2/metadata.json
results/position_depth_study_v2/progress.json
```

After the first 8 positions, estimate full runtime. If 256 x all variants will finish today, continue to 256. If not, guarantee at least 128 with all four variants and all depths. If 512 becomes feasible, use 512. Reduce position count before dropping depths or controls.

---

# Work Block 4 - Paired statistical analysis

Implement `scripts/analyze_position_depth_study.py` while the long study runs.

For each D > 1 on the original graph, calculate paired `regret_D1 - regret_D` and report mean, median, 95% paired bootstrap CI over positions, improved/unchanged/worsened fraction, teacher-best agreement difference, top-3 difference and rank difference.

For each control, calculate the paired difference-in-differences:

```text
(original D1 -> D improvement) - (control D1 -> D improvement)
```

Bootstrap positions and report 95% CIs.

Interpretation rules:

- original improves but controls improve similarly -> **generic recurrent-computation effect**;
- original materially exceeds controls with robust uncertainty -> evidence toward a **MaleCNS-structure-specific recurrent effect**;
- shuffled equals/beats original -> state that plainly;
- high depth worsens -> estimate an empirical optimum; do not assume monotonic scaling.

Generate plots for regret vs depth, paired D1-to-D improvement, original-vs-controls, teacher/top-3 agreement, latency, compute-quality Pareto, and per-position improvement distributions. Do not use compact short-game rating as the headline y-axis.

---

# Work Block 5 - Diagnose the weak v1 readout

**Budget:** 45-90 minutes; avoid competing with timing benchmarks.

Using the existing depth-16 activation cache, compute feature mean/std/dynamic range, zero/near-zero fractions, saturation, within-position candidate distances, readout-neuron variance, effective rank/singular spectrum on a tractable sample, pair balance, gradient norm, weight norm, score-margin distribution and train-vs-validation ranking metrics.

Write small diagnostics under `results/training/v1_depth16_rate_large/`.

Determine whether the main issue is tiny numerical scale, near-constant features, poor linear separability, optimizer behavior, insufficient data, or a label/split/orientation bug. Add tests before retraining if a bug is found.

---

# Work Block 6 - Stage-0 readout v2, if diagnostics justify it

Only the decoder/readout may improve; the MaleCNS graph and dynamics stay frozen.

Because fixed-feature pairwise logistic regression is convex, prefer a numerically robust optimizer rather than simply running Adam longer. Evaluate train-only feature standardization/scaling stored in the checkpoint, L-BFGS/scipy pairwise logistic + L2, a small predeclared L2 grid selected on validation, and best-validation checkpoint preservation.

Do **not** use the frozen evaluation corpus for hyperparameter selection.

v2 should clearly improve at least one validation measure without leakage: pairwise validation loss, pairwise accuracy, top-1 teacher agreement, plus deterministic checkpoint reload/inference. If it does not reliably beat v1, keep v1 and document the negative result.

If v2 succeeds and time remains, run a smaller confirmatory depth sweep on a predeclared subset such as 64 positions to see whether stronger decoding changes the depth optimum. Label it exploratory unless fully powered.

---

# Work Block 7 - Documentation and current-result correction

Before ending the day:

1. Update `README.md` so it no longer says Alfil is the default low-Elo engine or that chess still uses a random agent.
2. Update `docs/experiment_protocol.md` with the paired large-position recurrent-depth protocol and current interpretation boundary.
3. Update `docs/chess_training.md` with the observed weakness of v1 and Stage-0 v2 procedure.
4. Add/update `docs/current_results.md` with exact v1 8-position results, short-game rating caveat, control interpretation, training diagnosis, and latest large-study result if completed.
5. Update `results/final_experiment_report.md` only through its generator or with matching generator changes.
6. **Do not alter `Research Goal.md`.**

---

# End-of-day acceptance gate

The day is successful if the optimized evaluator is validated against scalar reference; at least 128 unseen positions have complete original + three-control depth results or there is a documented runtime blocker; all depths 1..64 are represented; raw rows and small summaries are preserved; paired bootstrap analysis is complete; reporting distinguishes generic recurrence from MaleCNS-specific recurrence; short-game pseudo-ratings are not presented as real Elo; v1 weakness is diagnosed; any v2 work uses train/validation only; docs match actual state; and `pytest -q` passes.

End with a concise Markdown summary containing commit SHA, positions completed, runtime, speedup, best original depth by regret, D1->best improvement + 95% CI, best control effect, MaleCNS-vs-control difference-in-differences + 95% CI, whether H1 received support, whether a MaleCNS-specific effect received support, v1 diagnosis, v2 status, and next recommended experiment.

---

# Do not spend today's compute on these

Unless the large position study and analysis are already complete, do not spend substantial time on another 400-game 2-ply/4-ply rating run, unrelated UI polish, arbitrary biological parameter training, topology-changing training, uncalibrated engine ladders, larger teacher-training jobs before diagnosing v1, or changing the locked research goal.

The highest-value use of today's workstation time is **better statistical evidence about recurrent depth and controls**, followed by a better Stage-0 decoder.
