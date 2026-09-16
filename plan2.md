# MaleCNS Neuron-Population Selection Study

**Purpose:** determine how strongly the choice of input and readout neurons controls the apparent computational capability of the MaleCNS recurrent system, with a specific emphasis on biologically motivated circuits that are the most natural fit for memory, state integration, goal comparison, and action selection.

**Research scope:** `Research Goal.md` remains LOCKED. This plan does not replace the overall program. It adds a controlled experimental axis: **where chess information enters the connectome, where task-relevant state is observed, and how those choices interact with recurrent depth.**

The current baseline uses a fixed seeded sample of 1,024 sensory neurons drawn uniformly from broad sensory superclasses and 256 readout neurons drawn uniformly from broad descending/motor/efferent superclasses. That baseline is reproducible, but it is only one interface into a ~165k-neuron recurrent network. The current negative recurrent-depth result therefore applies strictly to that frozen interface unless population robustness is demonstrated.

---

# Literature-informed biological prior

The population study should no longer treat all anatomical regions as equally likely candidates.

The strongest biological prior for the symbolic-chess experiment is a pathway of the form:

```text
symbolic chess features
        ↓
MB / Kenyon-cell-like sparse representation
        ↓
MBON + SMP / CRE / SIP convergence
        ↓
fan-shaped body / central-complex recurrent computation
        ↓
FC / PFN / PFR / PFL-type state comparison and action-selection signals
        ↓
LAL / descending neurons
```

The entire MaleCNS graph remains active. We are **not** replacing the connectome with this subnetwork. The experiment changes only where chess evidence is injected and where the resulting state is measured.

## Working biological interpretation

Use these roles as experimentally testable hypotheses, not as claims that flies play chess:

- **Mushroom body / Kenyon cells:** sparse distributed representation, pattern separation, associative memory and learned value.
- **MBONs:** learned-valence / action-bias outputs from the mushroom body.
- **SMP / CRE / SIP convergence circuits:** higher-order integration between mushroom-body outputs and central-brain circuits.
- **Fan-shaped body (FB):** highest-priority recurrent-computation region because of dense recurrent architecture, state integration, persistence, working-memory-like dynamics and context-dependent action selection.
- **hDelta-family / FB local neurons, especially hDeltaK where identifiable:** high-priority candidates for recurrent evidence integration and persistent internal state.
- **FC / PFN / PFR populations:** high-priority candidate populations for transformed state, vector-like integration and goal-related representation.
- **EPG / Delta7 / PB-EB network:** persistent attractor-like internal reference/state.
- **PFL populations, especially PFL3 where identifiable:** high-priority action-decision readout candidates because they combine internal state and goal-related signals before motor output.
- **LAL / descending neurons:** secondary downstream decision/output readout.
- **VNC motor neurons:** execution-stage control; low priority as a primary cognitive readout.
- **visual projection / optic-lobe sensory pathways:** biologically natural for pixel/visual input, but lower-priority for the current already-symbolic chess encoder.

This is a **prioritization**, not a guarantee. Every biological hypothesis must be compared with random and topology-matched controls.

---

# Central questions

1. Does task-relevant chess information become substantially more recoverable in FB/CX/MB-related populations than in the current motor/efferent readout?
2. Does the low effective rank of the current 256-neuron readout reflect a poor observation site rather than global information loss?
3. Does useful information move from MB-like populations toward FB/CX/PFL populations as recurrent depth increases?
4. Is the D4+ collapse true global information destruction, or does useful information remain recoverable elsewhere in the brain?
5. Does MB-oriented or dual MB+FB input placement create a recurrent-depth regime that is more useful than the current broad sensory input?
6. Is there a reproducible biologically informed interface for which moderate recurrent depth improves chess decisions?
7. Does the real MaleCNS anatomy outperform region-size-matched random and graph-matched controls?
8. Do graph metrics such as recurrent reach, cross-module participation, controllability proxies or observability proxies predict which neuron populations work best?
9. Does MaleCNS add linearly recoverable information beyond the raw chess encoder and projected sensory vector?
10. Can we identify a small set of neurons/regions that acts as a natural high-information observation window into the recurrent computation?

A positive result must not be described as finding “chess neurons.” Chess is an artificial probe. The claim is about accessibility of computation in a fixed biological architecture.

---

# Scientific rules

1. Start from current `main` and record exact HEAD and all relevant hashes.
2. **Do not use GitHub Actions.** Run locally.
3. Do not edit `Research Goal.md`.
4. Preserve the current seeded population selection as **Population Baseline A**.
5. Treat `data/chess_evaluation_corpus_v2.csv` as consumed evaluation data. Do not tune populations on it.
6. Population discovery and selection use training data plus a dedicated development corpus only.
7. Any strong final claim requires a fresh held-out confirmation corpus generated after the population-selection rule is frozen.
8. Never select a population because it happened to perform best on a test set.
9. Keep the internal MaleCNS graph fixed inside interface experiments unless a graph control is explicitly being tested.
10. Compare populations at equal or carefully energy-normalized sizes whenever possible.
11. Use multiple random seeds for stochastic population-selection baselines.
12. Record neuron IDs, annotation filters, selection rule, seed, graph metrics and hashes for every population.
13. Female-FlyWire or other published anatomical names must not be assumed to map automatically onto MaleCNS. Identify actual MaleCNS labels/cell types first and document the mapping.
14. If a named cell class such as hDeltaK, PFL3 or FC2 cannot be identified defensibly in MaleCNS, record it as unavailable instead of inventing a mapping.
15. Negative and null results are valid outcomes.

---

# Phase 0 — Build the MaleCNS chess-region atlas

Before running population sweeps, determine what the MaleCNS annotations actually support.

Inspect the available annotation fields and build a compact atlas of candidate chess-relevant regions/cell classes.

Minimum target families to search for:

```text
mushroom body / Kenyon cells
MBONs
DANs / modulatory MB populations
SMP
CRE
SIP
fan-shaped body
hDelta-family neurons
FC neurons
PFN neurons
PFR neurons
PFL neurons
EPG neurons
Delta7 neurons
protocerebral bridge / ellipsoid-body associated populations
LAL
central-complex general population
descending neurons
VNC motor/efferent populations
visual projection / optic-lobe sensory populations
```

For each family record:

- exact MaleCNS annotation filter used;
- number of neurons;
- body IDs;
- in/out degree and weighted strength distribution;
- recurrent/reciprocal connectivity statistics;
- SCC membership;
- graph distance to current sensory population;
- graph distance to current output population;
- overlap with other candidate families;
- confidence that the label is a valid anatomical mapping.

Create:

```text
results/population_study/malecns_region_atlas.csv
results/population_study/malecns_region_atlas.json
```

**Priority:** get the anatomical mapping right before optimizing anything.

---

# Phase 1 — Audit the current interface

For the existing 1,024 sensory and 256 readout neurons, record:

- body IDs / graph indices;
- superclass, class, cell type and region/neuropil fields;
- neurotransmitter class where available;
- in-degree, out-degree, weighted in/out strength;
- shortest-path statistics from input to readout;
- SCC participation;
- reciprocal-edge participation;
- recurrent-cycle proxy;
- PageRank / k-core / centrality where computationally feasible;
- distances to MB, FB/CX, PFL and LAL candidate populations;
- activation variance by depth;
- candidate discriminability by depth;
- fraction of readout neurons with nontrivial candidate-dependent activity.

Create:

```text
results/population_study/baseline_population_audit.csv
results/population_study/baseline_population_summary.json
```

This should answer whether the current interface is peripheral, redundant, low-variance or unusually distant from the recurrent central-brain circuits we care about.

---

# Phase 2 — Build reusable population manifests

Implement a versioned manifest abstraction for every input/readout population.

Minimum fields:

```text
population_id
role = input | readout
selection_method
anatomical_family
seed
requested_count
actual_count
body_ids
indices
annotation_filters
graph_metric_filters
source_data_hashes
selection_code_commit
```

Requirements:

- deterministic regeneration;
- no duplicate neurons;
- valid indices;
- annotation-filter tests;
- manifest hash stability;
- no overwrite of baseline population files.

---

# Phase 3 — Brain-region decoding atlas: highest priority

This is the first large experiment.

Keep the **current input population fixed** and read out different brain regions. Do not start by searching arbitrary task-selected neuron subsets. First ask where information is naturally observable.

## Priority readout families

Test these in this order where MaleCNS annotations permit:

1. **FB local / intrinsic neurons**, with hDelta-family subsets separated where possible.
2. **Combined FB population** including local, tangential and columnar neurons.
3. **FC / PFN / PFR populations** separately and as a combined set.
4. **PFL population**, with PFL3 separated where possible.
5. **MBON population**.
6. **SMP / CRE / SIP convergence population**, individually and combined.
7. **MBON + SMP/CRE/SIP combined readout**.
8. **FB + FC/PFN/PFR/PFL combined “central-computation panel.”**
9. **EPG / Delta7 / PB-EB attractor-state population.**
10. **LAL / descending population.**
11. **Baseline A current motor/efferent sample.**
12. **Whole-brain random matched-size controls.**
13. **Central-brain random matched-size controls.**
14. **10+ random seeds from the current motor/efferent candidate pool.**
15. **Topology-matched random controls** for the strongest biological region sets.

Do not force every group to 256 neurons if the biological family is smaller. In that case:

- evaluate the full family;
- include size-matched random controls;
- optionally test nested subsets if the family is large enough.

## Decoding protocol

For each population and each depth:

```text
D = 1, 2, 4, 8, 16, 32, 64
```

extract candidate activations on training/development positions and fit the **same standardized linear ranking probe** under identical optimization/regularization rules.

Report:

- top-1 teacher agreement;
- top-3 agreement;
- pairwise accuracy;
- teacher regret;
- selected teacher rank;
- Spearman/Kendall candidate-ranking correlation;
- effective rank;
- participation ratio / singular-value spectrum;
- within-position candidate-state distance;
- between-position state distance;
- score margin;
- fraction of near-zero / saturated units;
- train-vs-validation gap.

Primary output:

```text
results/population_study/brain_region_decoding_atlas.csv
```

Primary scientific question:

> Is useful chess information already present in FB/CX/MB-related populations but mostly invisible at the current motor/efferent readout?

---

# Phase 4 — Information-flow-through-depth study

For the strongest biological regions plus Baseline A, map how information changes with recurrent depth.

Train separate probes at each depth and produce:

```text
region × D_train × D_test
```

cross-depth transfer matrices.

Also compute, for each region and depth:

- effective rank;
- candidate-state separation;
- linear-probe performance;
- cosine similarity to earlier-depth representation;
- CKA or another tractable representation-similarity metric;
- winner-switch frequency;
- state norm and delta norm;
- saturation fraction.

Interpretation:

- deep representations fail for all probes/regions -> likely genuine task-information destruction;
- D-specific probes recover performance -> representation rotation;
- information appears sequentially in MB -> convergence areas -> FB/PFL with increasing depth -> strong evidence for recurrent propagation through biologically structured circuits;
- motor readout loses information while FB/PFL retain it -> observation-site bottleneck;
- all regions remain weak -> interface placement probably not the main problem.

Produce at minimum:

```text
results/population_study/depth_probe_transfer.csv
results/population_study/region_information_by_depth.csv
results/population_study/region_information_flow.png
```

---

# Phase 5 — High-priority biologically motivated input study

Only after readout mapping, test where symbolic chess information should enter.

Because the current encoder is already symbolic, **do not prioritize optic-lobe/retinal pathways** as the primary input route. Keep them as biological controls.

## Input families in priority order

1. **MB / Kenyon-cell-oriented input** where MaleCNS labels allow a defensible KC or MB-input target.
2. **Dual-stream MB + FB-context input**: split the fixed chess feature vector deterministically between an MB-oriented population and an FB/SMP-contextual population.
3. **SMP / CRE / SIP higher-order input**.
4. **FB tangential/contextual input** where annotations allow it.
5. **Current Baseline A sensory sample.**
6. **Broad multimodal biological sensory sample.**
7. **Ascending sensory population.**
8. **Visual projection / optic-lobe sensory population** as a modality-mismatch control for symbolic features.
9. **Whole-brain random outgoing-connected input control.**
10. **Topology-informed high-reach / controllability-proxy input set.**
11. **Topology-matched random control.**

## Input normalization

Make comparisons fair:

- same encoded feature vector;
- same total injected L2 energy where possible;
- same fanout or explicitly normalized total synaptic drive;
- same input population size when possible;
- deterministic feature-to-neuron mapping;
- no test-set-driven population choice.

Measure:

- fraction of brain reached after each depth;
- weighted activity spread;
- number/fraction of candidate readout neurons influenced;
- region-by-region activity onset depth;
- candidate-state separation;
- depth-specific probe performance;
- teacher regret/rank metrics on development positions.

Primary question:

> Does a biologically plausible MB/FB-oriented entry point allow recurrent processing to preserve and transform chess information more effectively than the current broad sensory injection?

---

# Phase 6 — Test the strongest a-priori architecture directly

Before doing a broad factorial sweep, explicitly test the biologically strongest predeclared configuration.

## Architecture H-BIO-1

```text
Input:
  MB/KC-oriented population
  + optional FB/SMP contextual branch

Core:
  full MaleCNS graph unchanged

Primary observation:
  FB local/intrinsic
  + MBON/SMP/CRE/SIP

Decision readout:
  FC/PFN/PFR/PFL-enriched population

Secondary readout:
  LAL/descending neurons
```

Compare against:

```text
Baseline A
size-matched central-brain random
size-matched whole-brain random
topology-matched random
same readout with no recurrence / D1
```

Evaluate the complete depth curve `1,2,4,8,16,32,64` on development data.

Do not call H-BIO-1 a success unless it beats the matched controls and improves under recurrence in a reproducible way.

---

# Phase 7 — Reduced input × readout matrix

After Phases 3–6, select approximately 4–6 input rules and 5–8 readout rules using development data only.

Always retain:

- Baseline A;
- H-BIO-1;
- random controls;
- topology-matched controls.

Run:

```text
input population
× readout population
× recurrent depth
```

with depths initially:

```text
1, 2, 4, 8, 16, 32
```

Add D64 where still informative.

Produce:

```text
results/population_study/interface_matrix.csv
results/population_study/interface_matrix_*.png
```

The goal is to identify whether particular biological input/output pairings unlock useful recurrent computation.

---

# Phase 8 — Population-size scaling

For the strongest biological and random-control rules, test size dependence.

Suggested readout sizes:

```text
32, 64, 128, 256, 512, 1024
```

Suggested input sizes:

```text
128, 256, 512, 1024, 2048
```

Use nested populations where possible.

Measure:

- quality;
- effective rank;
- candidate separation;
- compute cost;
- diminishing returns.

This determines whether the present low-dimensional readout is partly a sampling-size problem.

---

# Phase 9 — Robustness across seeds

For random/stochastic selection rules, run at least:

```text
20 input seeds
20 readout seeds
```

on a moderate development subset.

Estimate distributions of:

- teacher regret;
- top-1 / top-3 agreement;
- effective rank;
- D1-to-D2 and D1-to-D8 improvement;
- best empirical recurrent depth.

Report where Baseline A and H-BIO-1 lie in these distributions.

Important interpretations:

```text
Baseline A near median -> current baseline is representative
Baseline A in bottom tail -> previous null result may be interface-unlucky
H-BIO-1 near random median -> biological prior did not help
H-BIO-1 in top tail across metrics -> strong reason for fresh confirmation
very broad seed distribution -> neuron selection is itself a dominant experimental variable
```

---

# Phase 10 — Graph-theoretic explanation

Use the population sweep to identify structural properties associated with good interfaces.

Candidate predictors:

- in/out degree;
- weighted strength;
- shortest path from input to readout;
- reachable-neuron fraction by depth;
- SCC membership;
- reciprocal connectivity;
- local clustering;
- k-core level;
- PageRank/eigenvector centrality;
- betweenness approximation;
- cross-module participation coefficient;
- recurrent-cycle proxy;
- distance to MB;
- distance to FB/CX;
- distance to PFL/LAL/descending outputs;
- number of distinct anatomical modules reached by D.

Fit only simple development-set models and report effect sizes, not just p-values.

The ideal scientific result is a rule such as:

> Readout populations with high recurrent reach and MB-to-CX cross-module participation preserve more candidate information and benefit more from recurrent depth.

---

# Phase 11 — Baselines that bypass MaleCNS

Compare every strong biological interface against representations that do not require MaleCNS computation.

At minimum:

1. linear probe on raw chess encoder features;
2. linear probe on the projected input vector before recurrence;
3. identity/no-recurrence control;
4. random sparse recurrent graph matched for basic graph statistics where feasible;
5. original MaleCNS recurrence;
6. relevant shuffled-connectome controls.

Primary question:

```text
Does the recurrent MaleCNS create task information that is easier to decode than the original injected chess representation?
```

If a strong readout merely recovers unchanged input information, do not credit recurrence.

---

# Phase 12 — Freeze Population-v2 rule

Do **not** freeze the single best observed neuron list.

Freeze a reproducible selection rule, ideally:

```text
anatomical family
+ graph criterion
+ deterministic seed/tie-breaking
+ fixed size/energy normalization
```

Preferred candidate if supported by development results:

```text
input:
  MB/KC-oriented + FB/SMP contextual dual stream

readout:
  FB/MBON/SMP/CRE/SIP internal-state panel
  + FC/PFN/PFR/PFL decision panel
```

The manifests should be stored as:

```text
data/populations_v2/input_manifest.json
data/populations_v2/internal_readout_manifest.json
data/populations_v2/decision_readout_manifest.json
```

Document the decision before generating the new confirmation corpus.

---

# Phase 13 — Fresh confirmatory experiment

After the selection rule is frozen, generate a **new held-out source-diverse corpus** that has never been used for population selection, readout fitting or dynamics tuning.

Preferred design:

```text
>= 256 source-diverse positions
all legal moves
original MaleCNS + required graph controls
D = 1,2,4,8,16,32,64
cluster-aware bootstrap by source game/opening
```

Compare at least:

1. Baseline A;
2. frozen Population-v2 biological interface;
3. matched random interface;
4. topology-matched random interface;
5. raw/projected-feature baseline;
6. shuffled-connectome controls.

Primary confirmatory questions:

- Does Population-v2 improve held-out move quality over Baseline A?
- Does Population-v2 expose greater effective dimensionality?
- Does recurrent depth improve Population-v2 performance?
- Is the recurrent benefit stronger than matched random/shuffled controls?
- Does the effect survive source-cluster bootstrap?
- Does the anatomical information-flow pattern seen in development reproduce?

No interface tuning after inspecting this corpus.

---

# Recommended Codex execution order

Codex should execute in this order:

1. map actual MaleCNS annotations to the literature-prioritized regions/cell classes;
2. produce the MaleCNS chess-region atlas;
3. audit Baseline A relative to MB/FB/CX/PFL circuits;
4. implement versioned population manifests;
5. run the brain-region decoding atlas with current input fixed;
6. run depth-specific probes and cross-depth transfer matrices;
7. test MB/KC, FB-context and dual MB+FB input families;
8. execute H-BIO-1 directly;
9. run the reduced input × readout matrix;
10. run population-size scaling;
11. run 20-seed random-population robustness;
12. analyze graph predictors;
13. run raw/projection/no-recurrence/random-network baselines;
14. freeze Population-v2 rule;
15. generate a fresh confirmation corpus;
16. run final all-depth/control confirmation;
17. update `docs/current_results.md`, `docs/chess_training.md`, `docs/experiment_protocol.md`, and final summaries.

Do not stop after writing infrastructure. Execute experiments and commit compact results.

---

# Required compact artifacts

```text
results/population_study/malecns_region_atlas.csv
results/population_study/malecns_region_atlas.json
results/population_study/baseline_population_audit.csv
results/population_study/baseline_population_summary.json
results/population_study/brain_region_decoding_atlas.csv
results/population_study/region_information_by_depth.csv
results/population_study/depth_probe_transfer.csv
results/population_study/input_population_results.csv
results/population_study/h_bio_1_results.csv
results/population_study/interface_matrix.csv
results/population_study/population_size_scaling.csv
results/population_study/readout_seed_distribution.csv
results/population_study/input_seed_distribution.csv
results/population_study/graph_predictors.csv
results/population_study/baseline_comparison.csv
results/population_study/final_selection.json
results/population_study/final_confirmation_summary.json
results/population_study/*.png
```

Large activation/state caches may remain local and ignored, but preserve hashes, shapes and generation metadata.

---

# Interpretation guide

### Outcome A — FB/CX readouts strongly outperform motor output

Conclude that the previous experiment was substantially **observation-site limited**. The network may contain useful internal computation that downstream motor neurons compress or discard.

### Outcome B — MB-related representations are strong early, FB/PFL become stronger later

This would be particularly interesting. It would support an information-flow picture in which recurrent processing transforms sparse/learned representation into integrated state and action-selection signals.

### Outcome C — depth-specific probes recover strong information after the frozen readout fails

Conclude that deep recurrence causes **representation transformation/rotation**, not simple global information destruction.

### Outcome D — all internal regions lose decodable information at deep depth

This strengthens the hypothesis that current recurrent dynamics truly destroy task information.

### Outcome E — MB/FB-oriented input materially improves depth behavior

Conclude that **entry point / controllability** is a major determinant of useful recurrent computation.

### Outcome F — random populations match the biological prior

Then the biological labels do not provide detectable task-relevant advantage under the current artificial interface.

### Outcome G — H-BIO-1 beats matched random controls and benefits from recurrent depth

This is the strongest development-stage result this plan can produce. Freeze the rule and move immediately to fresh held-out confirmation.

### Outcome H — MaleCNS adds no value over raw/projected chess features

Then most measured chess capability is in the encoder/readout pipeline rather than the recurrent connectome. State that plainly.

### Outcome I — topology predicts good interfaces better than anatomy

This would itself be an interesting connectomics result: computationally useful observation/input locations may be determined more by network position than named biological role.

---

# Claims not allowed from this study alone

Do not claim:

- that selected neurons are biological “chess neurons”;
- that MaleCNS evolved for abstract reasoning;
- that MB or FB performs chess in a living fly;
- that a development-selected population proves biological specificity;
- that a best population found after many trials is valid without fresh confirmation;
- that higher probe accuracy alone proves recurrent thinking;
- that published female-FlyWire cell names automatically identify the same cells in MaleCNS without annotation support.

The goal is narrower and testable:

> **Determine whether biologically plausible memory, integration and action-selection circuits in MaleCNS expose more task-relevant recurrent computation than the current generic sensory-to-motor interface, and whether any such advantage survives matched controls and fresh held-out confirmation.**
