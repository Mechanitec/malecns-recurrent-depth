# Plan 3 — Reward/Aversive Neuromodulated Chess Curriculum

**Status:** proposed experiment plan

**Relationship to `Research Goal.md`:** this does not change the locked research goal. It is an Experiment-B-style training study in which the MaleCNS topology remains fixed while a small, biologically motivated subset of existing synaptic strengths is allowed to adapt.

## Central question

Can a frozen-topology MaleCNS-derived recurrent network learn progressively harder chess behavior when good and bad decisions generate a biologically localized reward/aversive teaching signal whose magnitude is determined by Stockfish move quality?

The experiment is intentionally modest. It is a proof-of-concept for biologically localized reinforcement learning, not an attempt to train a strong chess engine.

The training curriculum should progress from:

1. basic chess-piece movement,
2. elementary endgames,
3. tactical combinations,
4. forced mates in 1, 2, and 3 moves.

The primary scientific comparison is:

> Does reward/aversive neuromodulated plasticity improve held-out chess behavior relative to the same MaleCNS network with the same decoder but frozen synapses?

A second question is whether learning produces useful behavior with only a small biological deviation from the original connectome.

---

## Terminology

Use **reward signal**, **aversive signal**, **neuromodulatory teaching signal**, or **reinforcement signal** in code and documentation.

Do not describe the simulated system as literally feeling pain. The aversive signal is a mathematical training variable inspired by biological reinforcement mechanisms; the current MaleCNS simulation is not evidence of subjective experience.

---

# 1. Experimental constraints

The following rules are fixed for this experiment unless a blocking implementation problem is documented.

- MaleCNS graph topology is fixed.
- No new edges may be created.
- Existing edge signs should remain fixed in the primary condition.
- Plasticity is initially restricted to a biologically motivated synapse class rather than the whole connectome.
- No backpropagation through the entire MaleCNS recurrent graph in the primary condition.
- Use deterministic seeds and save every training checkpoint.
- No GitHub Actions.
- Do not use the final held-out confirmation set for tuning.
- Keep the experiment small enough to execute on the current workstation in roughly one workday / overnight run rather than several days.

The first plastic synapse class should be **existing Kenyon-cell → MBON connections**, because this is a natural place to test reward-modulated associative plasticity while keeping the rest of MaleCNS unchanged.

If the MaleCNS annotations reveal that the exact KC→MBON mapping is incomplete, document the mapping and use the narrowest defensible MB/KC-like → MBON set. Do not silently broaden plasticity to the whole brain.

---

# 2. Overall architecture

Use the current chess encoder and recurrent MaleCNS engine.

Conceptually:

```text
board + candidate move
        ↓
chess feature encoder
        ↓
selected MaleCNS input population
        ↓
MaleCNS recurrent dynamics, depth D
        ↓
fixed decision/readout population
        ↓
move score / policy
        ↓
selected move
        ↓
Stockfish teacher evaluation
        ↓
reward or aversive scalar
        ↓
local KC→MBON plasticity on existing edges only
```

For the first implementation, use a single training depth rather than optimizing depth and plasticity simultaneously.

**Default training depth: D = 8.**

Reason: D8 is deep enough for information to reach central populations in the current graph, but is still computationally manageable. Depth should be frozen for the primary curriculum experiment.

After training is complete, evaluate the final checkpoint at D = 4, 8, and 16 as a small post-training generalization check. Do not tune on those results.

---

# 3. Stable policy/readout before plasticity

The primary experiment is meant to test internal synaptic plasticity, not continually retrain the decoder.

Therefore:

1. choose the input/readout interface before neuromodulated training;
2. fit a simple linear move-ranking decoder on the curriculum training split only;
3. freeze that decoder;
4. then train only the permitted MaleCNS synapses using the reward/aversive rule.

Preferred readout for this pilot:

- use the biologically motivated internal readout already defined by the population study if it exists and is frozen before this experiment;
- otherwise use `readout_mbon_smp_cre_sip_combined` at D8;
- record the exact manifest hash.

Do not choose a readout by looking at the final confirmation set.

Keep a **frozen-synapse control** using exactly the same fitted decoder.

---

# 4. Stockfish-derived reinforcement signal

For every training position, obtain a Stockfish score for each candidate move from the side-to-move perspective.

Let

```text
cp_best = Stockfish score of the best candidate
cp_move = Stockfish score of the selected candidate
regret_cp = max(0, cp_best - cp_move)
```

Map regret to a bounded signed teaching signal:

```text
signal = 1 - 2 * tanh(regret_cp / 400)
```

Interpretation:

- best move / 0 cp regret → approximately +1 reward;
- ~100 cp regret → positive but weaker reinforcement;
- ~200 cp regret → near-neutral;
- ~400 cp regret → clear aversive signal;
- very large blunder / lost mate → approaches -1.

Keep the scale constant at 400 cp for the primary run. Do not tune it on validation results.

For mate scores, allow the signal to saturate naturally near ±1. Preserve a separate flag indicating whether the selected move creates, preserves, loses, or allows a forced mate.

For stability, subtract a running reward baseline before plasticity:

```text
advantage = signal - running_mean_signal
```

Use an exponential running mean with a predeclared coefficient such as 0.95.

The raw `signal` and centered `advantage` must both be logged.

---

# 5. Local neuromodulated plasticity rule

Do not use full-network gradient descent in the primary condition.

For each selected move, accumulate a simple eligibility trace over the permitted KC→MBON edges during recurrent processing.

A suitable first rule is:

```text
eligibility_ij = mean_d( pre_i[d-1] * post_j[d] )
```

Then apply a reward-modulated Hebbian update:

```text
Δlog|w_ij| = learning_rate * advantage * normalized_eligibility_ij
```

with a weak restorative term toward the original biological weight:

```text
Δlog|w_ij| -= homeostasis * log(|w_ij| / |w0_ij|)
```

Reconstruct the weight using the original sign:

```text
w_ij = sign(w0_ij) * |w0_ij| * exp(log_ratio_ij)
```

Primary biological bounds:

```text
0.5 × |w0_ij| <= |w_ij| <= 2.0 × |w0_ij|
```

This guarantees:

- topology remains fixed;
- transmitter/sign interpretation remains fixed;
- synapses cannot grow without bound;
- biological deviation is directly measurable.

Suggested initial hyperparameters:

```text
learning_rate = 0.002
homeostasis = 0.0002
reward_baseline_decay = 0.95
max_weight_ratio = 2.0
min_weight_ratio = 0.5
```

These values are starting constants, not parameters to sweep in the first run.

Before the real curriculum, run a small 32-lesson smoke test. If weights instantly hit bounds or do not move numerically at all, correct the scaling problem. Do not perform a broad hyperparameter search.

---

# 6. Important issue: basic piece movement requires illegal candidates

The existing chess benchmark gives the agent only legal moves. Under that setup the network cannot actually learn how chess pieces move because illegal moves are never presented.

For Curriculum Stage 1, create a separate movement-learning task containing both legal and deliberately invalid candidate destinations.

Examples:

- rook: straight movement positive, diagonal movement negative;
- bishop: diagonal positive, orthogonal negative;
- knight: L-shaped positive, nearby non-knight destinations negative;
- queen: rook/bishop geometry positive, other geometry negative;
- king: one-square legal geometry positive, long jumps negative;
- pawn: color-sensitive forward/capture geometry, including blocked squares and illegal backwards motion.

Use minimal synthetic boards and include blockers where appropriate.

For this stage, legality supplies the target:

```text
legal candidate   → signal +1
illegal candidate → signal -1
```

Stockfish is not needed to decide movement legality. Stockfish-based grading begins with the endgame stage.

This stage tests whether the neuromodulated rule can learn a simple structured task before we spend compute on harder chess.

---

# 7. Curriculum

## Stage 1 — Piece movement and legality

Goal: verify that the system can learn simple chess rules.

Target dataset:

```text
training:   ~360 positions
validation: ~120 positions
```

Balance across the six piece types and both colors where relevant.

Each lesson should contain roughly 6–12 candidate destinations, mixing valid and invalid moves.

Primary metrics:

- legal-vs-illegal pair accuracy;
- fraction of selected candidates that are legal;
- per-piece accuracy;
- pawn-direction accuracy;
- blocker-sensitive sliding-piece accuracy.

Training passes: **2 maximum**.

If Stage 1 shows no improvement over the frozen-synapse control, stop and diagnose the plasticity rule before running the expensive stages.

---

## Stage 2 — Elementary endgames

Goal: teach move quality in positions with a small number of pieces and relatively clear objectives.

Target families:

- KQK;
- KRK;
- simple KPK positions;
- king opposition / pawn-race positions;
- simple conversion positions where one side is clearly winning.

Target dataset:

```text
training:   ~320 positions
validation: ~100 positions
```

For each training position use a small candidate set:

```text
1 Stockfish-best move
+ 4 hard negatives
```

Hard negatives should preferentially include the next-best moves and at least one clearly bad move when available.

Do not run every legal move during training unless the candidate count is already <= 6.

Use Stockfish with deterministic settings and a fixed node budget. Suggested teacher budget:

```text
20,000 nodes
Threads = 1
fixed hash size
```

The final evaluation can use a higher teacher budget if runtime permits, but training labels must be generated once and cached.

Primary metrics:

- top-1 best-move accuracy;
- top-3 accuracy;
- median and mean centipawn regret;
- >500 cp blunder rate;
- conversion success in a small number of played-out endgames against best defense.

Training passes: **2 maximum**.

---

## Stage 3 — Tactical combinations

Goal: teach short-range tactical discrimination.

Target motifs:

- hanging piece / simple capture;
- fork;
- pin;
- skewer;
- discovered attack;
- removal of defender;
- forcing check;
- simple exchange win.

Target dataset:

```text
training:   ~500 positions
validation: ~150 positions
```

Prefer positions where the teacher has a meaningful separation between the best move and alternatives. A practical screening criterion is a best-vs-median gap of at least ~100 cp, while keeping a range of difficulties.

Training candidate set:

```text
1 best move
+ 4 hard negatives
```

Primary metrics:

- top-1 / top-3 accuracy;
- median and trimmed-mean regret;
- >500 cp and >1000 cp blunder rates;
- accuracy by tactical motif;
- whether recurrent depth D8 remains better/worse than D4 and D16 after learning.

Training passes: **2 maximum**.

---

## Stage 4 — Mate curriculum, up to mate in 3

Goal: test whether the trained recurrent system can learn highly forcing multi-step structures.

Create a balanced curriculum of:

```text
mate in 1
mate in 2
mate in 3
```

Target dataset:

```text
training:   ~384 positions total
validation: ~120 positions total
```

Aim for approximately equal numbers of mate-in-1, mate-in-2, and mate-in-3 starting positions.

For mate-in-2 and mate-in-3, also add the intermediate positions along the teacher principal variation as separate curriculum lessons. This lets the network see the continuation after the opponent's best defense rather than receiving only isolated starting positions.

Verify mate labels with a stronger deterministic Stockfish pass than ordinary tactical labels. Record the mate distance separately from centipawn values.

Training candidate set:

```text
best forcing move
+ up to 4 hard alternatives
```

Final mate evaluation must be sequential:

1. give the fly the starting position;
2. let it choose a move;
3. let Stockfish play best defense;
4. ask the fly again;
5. continue until mate or the forced-mate property is lost.

Do not call a mate-in-3 problem solved merely because the first move matches Stockfish.

Primary metrics:

- mate-in-1 solve rate;
- mate-in-2 full-sequence solve rate;
- mate-in-3 full-sequence solve rate;
- fraction preserving a forced mate after each own move;
- number of positions where a forced mate is lost;
- mean Stockfish signal by curriculum stage.

Training passes: **2 maximum**.

---

# 8. Curriculum replay / catastrophic forgetting

After Stage 1, every later stage should reserve approximately **20% of training lessons for replay from earlier stages**.

Example:

```text
Stage 2 batches: 80% endings + 20% movement
Stage 3 batches: 80% tactics + 20% movement/endgame replay
Stage 4 batches: 80% mates + 20% earlier-stage replay
```

After each curriculum stage, rerun all earlier validation suites.

Record catastrophic forgetting explicitly.

If an earlier-stage metric drops by more than 10 percentage points, do not silently increase replay. Report the failure first. A single predeclared fallback may increase replay to 30% for one additional pass.

---

# 9. Controls

The first experiment does not need a giant control matrix. Keep it focused.

Required conditions:

### A. Frozen MaleCNS control

Same input population, same recurrent depth, same frozen decoder, no synaptic updates.

This is the primary baseline.

### B. Reward/aversive KC→MBON plasticity

The main experimental condition.

### C. Shuffled-teaching-signal control

Use the same training examples and the same number/magnitude distribution of reinforcement values, but deterministically permute the signals among lessons within each curriculum stage.

To control compute, this condition may use **50% of the training examples** for the initial study. If the primary condition looks strongly positive, run the full shuffled-signal control before making a strong claim.

Do not add a large family of alternative plasticity rules in this first experiment.

---

# 10. Candidate selection and compute control

Full legal-move evaluation is too expensive to use for every training update.

During training:

- use the teacher-best move plus four hard negatives for endings/tactics/mates;
- use 6–12 candidates for the movement task;
- batch candidate states whenever possible;
- cache Stockfish labels permanently;
- do not cache MaleCNS activations after synaptic plasticity begins, because the network weights are changing.

During validation/final evaluation:

- use all legal moves for normal chess positions;
- use the full generated candidate set for the movement task.

Use the existing batched recurrent trajectory implementation rather than evaluating candidates one by one.

---

# 11. Estimated experiment size and runtime budget

Approximate primary training set:

```text
Stage 1 movement      360 positions
Stage 2 endgames      320 positions
Stage 3 tactics       500 positions
Stage 4 mates         384 positions
----------------------------------
Total                1564 positions
```

With two passes maximum and approximately five candidates per ordinary lesson, the target is roughly **3,000–3,200 position-updates**, plus replay and validation.

Use D8 only during training.

Target compute budget on the current workstation:

```text
implementation + smoke tests: 1–2 hours of Codex work
Stockfish label generation:    ~15–45 minutes once cached
primary training run:          ~3–5 hours target
validation/evaluation:         ~1 hour target
shuffled-signal control:       ~1–2 additional hours at 50% scale
```

Target total wall-clock compute after implementation: approximately **5–8 hours**.

Hard cap for the first version: **10 hours** of local compute.

If projected runtime exceeds the cap, reduce the number of training positions proportionally while preserving all four curriculum stages. Do not remove the mate stage or change D8 merely to make the run finish.

The minimum acceptable scaled run is approximately:

```text
movement  180
endgames  160
tactics   250
mates     192
```

Do not go below this without documenting that the run has become a smoke test rather than the intended experiment.

---

# 12. Checkpointing

Save a checkpoint:

- before any neuromodulated training;
- after movement;
- after endgames;
- after tactics;
- after mate training;
- after any permitted replay fallback.

Each checkpoint must contain or reference:

- original graph/connectome hash;
- plastic-edge list hash;
- input/readout population manifest hashes;
- frozen decoder hash;
- recurrent parameters;
- training depth;
- current plastic weights;
- original weight values for the plastic edges;
- learning-rate/homeostasis parameters;
- curriculum stage;
- lesson count;
- RNG seeds;
- Stockfish executable hash and teacher settings.

The run must be resumable from the most recent completed checkpoint.

---

# 13. Biological-deviation measurements

For every saved checkpoint report:

```text
RMS log(|w| / |w0|)
median absolute log-ratio
95th percentile absolute log-ratio
fraction of plastic edges changed by >5%
fraction changed by >10%
fraction at lower bound
fraction at upper bound
```

Also report performance gain per unit biological deviation.

A useful result is not simply the highest chess score. The interesting question is how much behavior changes for how little modification of the original biological network.

If more than 10% of plastic edges are pinned at either weight bound, flag the run as plasticity-saturated and do not interpret further improvement as evidence for a well-calibrated biological learning rule.

---

# 14. Evaluation metrics

Do not rely on a single aggregate metric.

For ordinary chess positions report:

- top-1 best-move accuracy;
- top-3 accuracy;
- pairwise ranking accuracy;
- median centipawn regret;
- mean centipawn regret;
- 10%-trimmed mean regret;
- p90 and p95 regret;
- >500 cp blunder rate;
- >1000 cp blunder rate;
- mate-blunder count;
- selected-move teacher rank.

For the movement stage report legal/illegal classification and piece-specific accuracy.

For mate stages report complete sequence solve rate under best defense.

After the final checkpoint, evaluate at D4, D8, and D16 using the same frozen decoder. This asks whether learning has created a depth-specific representation or a more generally useful one.

---

# 15. Primary success criteria

This is an exploratory proof-of-concept, so avoid overfitting to arbitrary thresholds. Still define minimum evidence before calling the approach promising.

The main condition should satisfy at least three of the following on held-out curriculum validation data relative to the frozen-synapse baseline:

1. movement legality selection improves by at least 10 percentage points;
2. endgame median regret improves by at least 15%;
3. tactical median regret improves by at least 15% or top-1 improves by at least 5 percentage points;
4. mate-in-1/2/3 aggregate full-sequence solve rate improves by at least 10 percentage points;
5. >1000 cp blunder rate falls by at least 20% relative;
6. improvement is larger than the shuffled-teaching-signal control.

In addition, the run should not require gross biological distortion:

- fewer than 10% of plastic edges at weight bounds;
- report biological deviation even if performance is poor.

Failure to meet these criteria is a valid negative result.

---

# 16. Implementation tasks for Codex

Suggested implementation sequence:

## Milestone 1 — Plastic-edge audit

Create a reproducible audit of the KC→MBON edge set.

Outputs:

```text
results/neuromodulated_curriculum_v1/plastic_edge_audit.csv
results/neuromodulated_curriculum_v1/plastic_edge_metadata.json
```

Verify:

- all edges already exist in MaleCNS;
- no self-created edges;
- signs and original weights are preserved;
- source and destination neurons belong to the intended manifests.

## Milestone 2 — Plasticity engine

Add a small module such as:

```text
src/malecns_rd/neuromodulated_plasticity.py
```

It should:

- identify the permitted edge set;
- accumulate eligibility traces;
- apply signed reward/aversive updates;
- preserve topology and sign;
- apply weight-ratio bounds;
- save/load plastic-state checkpoints;
- expose biological-deviation metrics.

Add deterministic unit tests on a tiny synthetic graph before using real MaleCNS.

## Milestone 3 — Curriculum builder

Add a script such as:

```text
scripts/build_chess_curriculum_v1.py
```

It should generate/cache:

```text
data/curriculum_v1/movement.csv
data/curriculum_v1/endgames.csv
data/curriculum_v1/tactics.csv
data/curriculum_v1/mates.csv
```

Do not commit very large generated data if repository size becomes a concern; commit manifests/metadata and small reproducibility inputs instead.

## Milestone 4 — Teacher labels

Generate deterministic Stockfish labels once and save metadata/hash.

Do not repeatedly call Stockfish during training if labels can be cached.

## Milestone 5 — Freeze decoder

Fit the initial linear decoder using training data only, save it, and never update it during the primary plasticity run.

## Milestone 6 — 32-lesson smoke test

Run the real MaleCNS with 32 movement lessons.

Verify:

- reward signal has correct sign;
- weights move in the expected direction;
- no edge changes sign;
- no new edge appears;
- checkpoint restore is exact;
- runtime projection stays below the experiment cap.

## Milestone 7 — Full curriculum

Run Stages 1–4 sequentially with checkpoint/evaluation after each stage.

## Milestone 8 — Shuffled-signal control

Run the reduced 50% control with identical code and deterministic signal permutation.

## Milestone 9 — Final analysis

Produce a concise report with learning curves, forgetting curves, biological deviation, and the comparison to frozen and shuffled-signal controls.

---

# 17. Required output artifacts

Create under:

```text
results/neuromodulated_curriculum_v1/
```

At minimum:

```text
plastic_edge_audit.csv
plastic_edge_metadata.json
training_log.csv
stage_metrics.csv
movement_validation.csv
endgame_validation.csv
tactical_validation.csv
mate_validation.csv
biological_deviation.csv
reward_signal_distribution.csv
run_metadata.json
final_summary.md
```

Recommended plots:

```text
learning_curve.png
curriculum_retention.png
regret_distribution_by_stage.png
blunder_rate_by_stage.png
mate_solve_rate.png
biological_deviation_vs_performance.png
weight_ratio_distribution.png
```

Do not commit giant activation tensors or temporary caches.

---

# 18. Tests and verification

Before the full run:

- run focused tests for the new plasticity module;
- verify a positive reward strengthens the intended active pathway on a tiny test graph;
- verify a negative/aversive signal weakens it;
- verify inactive synapses do not receive large updates;
- verify sign preservation;
- verify bounds;
- verify deterministic replay from checkpoint;
- verify Stockfish score perspective is always side-to-move correct;
- verify mate labels using explicit mate-distance data rather than huge centipawn values alone.

After implementation run the complete local test suite:

```text
pytest -q
```

No GitHub Actions.

---

# 19. Interpretation rules

A positive result would support a narrow claim:

> A fixed-topology MaleCNS-derived recurrent network can acquire some chess-relevant behavior through localized reward/aversive-modulated synaptic plasticity under this curriculum.

It would **not** show that a biological fly understands chess, experiences punishment, or possesses human-like reasoning.

A negative result is also useful. It would constrain whether simple KC→MBON reward-modulated plasticity is sufficient for transferring symbolic chess information through this connectome and would motivate later experiments with different plastic synapse classes or learning rules.

Do not broaden plasticity beyond the predeclared synapses merely because the first run performs poorly. Finish and report the primary experiment first.

---

# 20. Stop conditions

Stop the expensive run and report the cause if any of the following occur:

- Stage 1 produces no measurable learning after two passes;
- more than 25% of plastic edges hit a bound during the smoke test or Stage 1;
- NaN/Inf appears in state, weights, or reward calculations;
- topology/sign invariants fail;
- estimated total compute exceeds 10 hours after batching/obvious optimization;
- teacher labels are inconsistent under deterministic rerun;
- checkpoint restore does not exactly reproduce the next-step update.

Do not hide a stopped run. Preserve its logs and make the failure mode part of the report.

---

# Expected scientific value

The key outcome is not merely whether chess performance improves. This experiment can answer several deeper questions:

- Can a biological-connectome-derived recurrent network learn from a scalar outcome signal without full-network backpropagation?
- Is a small, biologically localized plastic subsystem enough to alter whole-network behavior?
- Does learning transfer progressively from simple movement rules to endgames, tactics, and short forced-mate structures?
- Does the network catastrophically forget earlier lessons?
- Does recurrent depth become more useful after biologically localized learning?
- How much deviation from the original MaleCNS synaptic strengths is required for each increment of task capability?

That makes this a natural bridge between the current frozen-network recurrent-depth study and the locked research goal's later trained-fly experiments.
