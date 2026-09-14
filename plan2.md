# MaleCNS Neuron-Population Selection Study

**Purpose:** determine how strongly the choice of sensory/input neurons and readout/output neurons affects the apparent computational capability of the MaleCNS recurrent system.

**Research scope:** `Research Goal.md` remains LOCKED. This plan does not replace the overall program. It adds a controlled experimental axis that is now important enough to study explicitly: **where chess information enters the connectome, where we observe the resulting neural state, and how those interface choices interact with recurrent depth.**

The current baseline uses a fixed seeded sample of 1,024 sensory neurons drawn uniformly from broad sensory superclasses and 256 readout neurons drawn uniformly from broad descending/motor/efferent superclasses. That is a reproducible biologically plausible interface, but it is only one interface into a ~165k-neuron recurrent network. The current negative recurrent-depth result therefore applies strictly to this frozen interface unless population robustness is demonstrated.

---

## Central questions

1. How sensitive is chess-relevant performance to the selected sensory/input neurons?
2. How sensitive is recoverable chess information to the selected readout neurons?
3. Does MaleCNS contain useful candidate-move information in regions that the current 256-neuron readout does not expose?
4. Does a biologically informed population outperform a uniformly sampled biological population?
5. Does a graph-topology-informed population outperform biological-label-only selection?
6. Is there a reproducible population configuration for which recurrent depth becomes more useful?
7. Does the apparent D4+ collapse represent true information loss, or only loss at the current observation site?
8. How much of any improvement comes from interface placement rather than the MaleCNS recurrent computation itself?

A positive result should not be interpreted as “these are the neurons the fly uses for chess.” Chess is an artificial task. The scientific claim is about accessibility of computation in a fixed biological architecture under different controlled interfaces.

---

## Why this matters

The current experiment has three conceptually distinct components:

```text
chess position + candidate move
          ↓
[input / sensory neuron population]
          ↓
      MaleCNS recurrence
          ↓
[readout / observed population]
          ↓
      trained linear decoder
```

A poor input population may inject information into weakly connected or poorly placed subcircuits. A poor readout population may observe only a low-dimensional projection of a richer internal state. Either failure can make the connectome look computationally weak even if useful information exists elsewhere.

The existing v1 readout diagnostics strengthen this concern: 256 readout features produced an effective rank of only about 4.55 and weak candidate separation. That may reflect real information collapse, but it may also reflect a narrow observation window.

---

# Scientific rules

1. Start from current `main` and record exact HEAD and all relevant hashes.
2. **Do not use GitHub Actions.** Run locally.
3. Do not edit `Research Goal.md`.
4. Preserve the current seeded population selection as **Population Baseline A**.
5. Do not tune population choices on `data/chess_evaluation_corpus_v2.csv`; that corpus is already consumed evaluation data.
6. Population discovery/selection uses training data plus a dedicated development corpus only.
7. Any strong final claim requires a fresh held-out confirmation corpus generated after the population-selection rule is frozen.
8. Never choose a population because it happened to perform best on a test set.
9. Keep the MaleCNS internal graph fixed inside population-interface experiments unless a graph control is explicitly being tested.
10. Compare methods with equal population sizes whenever possible.
11. Use multiple random seeds for stochastic population-selection baselines.
12. Record neuron IDs, annotations, selection rule, seed, graph metrics and hashes for every population.
13. Negative results and null differences are valid outcomes.

---

# Phase 0 — Audit the current population interface

Document exactly what the current selection script does and create a compact population report.

For the current 1,024 sensory and 256 readout neurons, record:

- body IDs / graph indices;
- superclass, class, cell type, region/neuropil fields available in MaleCNS annotations;
- neurotransmitter class where available;
- in-degree, out-degree, weighted in/out strength;
- shortest-path statistics from sensory to readout populations;
- fraction in largest strongly connected component;
- reciprocal-edge participation;
- recurrent-cycle participation proxy;
- betweenness / PageRank / eigenvector-style centrality where computationally feasible;
- distances to major central-brain, mushroom-body and central-complex populations where annotations permit;
- proportion of selected neurons that actually become nontrivially active under chess inputs at D1, D2, D8, D16 and D32;
- per-neuron activation variance and candidate discriminability.

Create:

```text
results/population_study/baseline_population_audit.csv
results/population_study/baseline_population_summary.json
```

This establishes whether the present interface is graphically peripheral, redundant or unusually low-variance.

---

# Phase 1 — Build a reusable population-definition system

Replace ad hoc population handling with a versioned manifest abstraction.

Implement a population manifest containing at least:

```text
population_id
role = input | readout
selection_method
seed
requested_count
actual_count
body_ids
indices
annotation filters
graph metric filters
source data hashes
selection code commit
```

Create code that can deterministically regenerate each population from the manifest.

Do not overwrite the current population files. Preserve them as the original baseline.

Add tests for:

- deterministic regeneration;
- no duplicate neurons;
- requested size;
- valid graph indices;
- annotation-filter correctness;
- exact manifest hash stability.

---

# Phase 2 — Readout-location mapping first

Prioritize readout mapping because changing where we observe the network is scientifically cleaner than immediately changing where chess evidence enters.

Keep the **current sensory population fixed** and compare multiple readout populations of matched size, preferably 256 neurons each.

Minimum readout families:

1. **Baseline A** — current seeded motor/efferent sample.
2. **Random biological output seeds** — at least 10 independent seeded samples from the same current readout candidate pool.
3. **Whole-brain random** — random neurons with nonzero incoming connectivity, matched size.
4. **Central-brain random** — where annotations permit.
5. **Descending-neuron only**.
6. **Motor/efferent only**.
7. **Central-complex / fan-shaped-body-associated** population where MaleCNS labels support a defensible mapping.
8. **Mushroom-body output / MB-associated** population where annotations support it.
9. **High-variance development-selected** — choose neurons with high candidate-dependent variance using training/development data only.
10. **Topology-informed** — e.g. high observability proxy / centrality / broad sensory reach, predeclared method.
11. **Random matched topology control** — matched degree/strength distribution to the topology-informed population where feasible.

If a named anatomical family cannot be identified reliably from MaleCNS annotation fields, do not invent labels. Record it as unavailable and proceed with defensible groups only.

## Readout experiment

For each population:

- keep current input population, encoder, projector, graph and dynamics fixed;
- extract candidate neural features at depths `1,2,4,8,16,32,64` on training/development positions;
- train the same standardized linear ranking probe using identical training procedure and regularization search;
- evaluate only on the development split during selection;
- report top-1 teacher agreement, top-3 agreement, pairwise accuracy, teacher regret, Spearman/Kendall ranking correlation, effective feature rank, within-position feature distance, score margin and calibration.

Use multiple random seeds for all stochastic population families.

Primary output:

```text
results/population_study/readout_population_results.csv
```

A central question is whether another readout family yields much higher effective rank and candidate separability than the current ~4.55-rank baseline.

---

# Phase 3 — Depth-specific information maps

For the best predeclared readout families plus Baseline A, determine whether useful information moves around the network with recurrent depth.

For each readout family, train separate probes at:

```text
D = 1, 2, 4, 8, 16, 32, 64
```

Then produce a cross-depth transfer matrix:

```text
probe trained at D_train × representation measured at D_test
```

Interpretation:

- performance collapses for all probes at deep D -> likely information destruction;
- D-specific probes recover strong performance while the frozen probe fails -> representation rotation / relocation;
- one anatomical readout family becomes informative only at later depth -> recurrent propagation into that circuit;
- the original readout loses information while another region gains it -> observation-site problem rather than global computation failure.

Create heatmaps for:

- top-1 accuracy;
- pairwise accuracy;
- teacher regret;
- effective rank;
- candidate-state separation.

---

# Phase 4 — Input-location study

After readout mapping, freeze a small set of readout families before exploring input placement.

Compare equal-size input populations, preferably 1,024 neurons unless a biological family is smaller.

Minimum input families:

1. **Baseline sensory sample**.
2. **10 random sensory seeds** from the same current candidate pool.
3. **Visual sensory / visual projection** where annotations permit.
4. **Ascending sensory**.
5. **VNC sensory**.
6. **Broad multimodal stratified sensory sample**.
7. **Whole-brain random input control** with outgoing connectivity.
8. **Topology-informed input set** using controllability/reachability proxy.
9. **Random topology-matched control** for the topology-informed set.

Keep encoder feature dimension, projector fanout/amplitude and total injected energy matched as closely as possible.

Do not let a larger anatomical population win simply because it receives more total input energy.

For each input population, measure at least:

- fraction of network reached after D recurrent steps;
- weighted activity spread;
- number/fraction of readout neurons influenced;
- candidate-state distance;
- depth-specific probe performance;
- teacher regret and rank metrics on development positions.

---

# Phase 5 — Input × readout interaction matrix

Do not exhaustively test every possible pair if the matrix is too large.

Select approximately the best 4–6 input rules and best 4–6 readout rules using development data only, while always retaining Baseline A and random controls.

Run the factorial matrix:

```text
input population × readout population × recurrent depth
```

with depths initially:

```text
1, 2, 4, 8, 16, 32
```

Add D64 only if it remains informative.

The purpose is to discover whether certain interfaces unlock recurrent computation that the current baseline misses.

Produce:

```text
results/population_study/interface_matrix.csv
results/population_study/interface_matrix_*.png
```

---

# Phase 6 — Population-size scaling

For a few representative population-selection rules, test size dependence rather than assuming 1,024 input / 256 readout is optimal.

Suggested readout sizes:

```text
32, 64, 128, 256, 512, 1024
```

Suggested input sizes:

```text
128, 256, 512, 1024, 2048
```

Use nested populations where possible so size comparisons add neurons rather than replacing the whole set.

Measure quality, effective rank, compute cost, and diminishing returns.

This tells us whether the current low-dimensional readout is caused partly by observing too few neurons.

---

# Phase 7 — Robustness over population seeds

A single random population is not enough.

For the current biological sampling rule, run at least:

```text
20 sensory seeds
20 readout seeds
```

on a moderate development subset.

Estimate the distribution of:

- teacher regret;
- top-1 agreement;
- effective rank;
- D1-to-D2 and D1-to-D8 improvement;
- best empirical recurrent depth.

Report where the current seed lies in that distribution.

Important interpretation:

```text
current seed near median -> baseline is reasonably representative
current seed in bottom tail -> prior negative result may be interface unlucky
current seed in top tail -> changing populations is unlikely to rescue much
very broad distribution -> neuron selection is itself a dominant experimental variable
```

---

# Phase 8 — Graph-theoretic predictors of good populations

Use the population sweep to ask whether performance can be predicted from network structure rather than chess labels.

Candidate predictors:

- mean in/out degree;
- weighted strength;
- sensory-to-readout shortest-path distribution;
- reachable-neuron fraction by depth;
- SCC membership;
- reciprocal connectivity;
- local clustering;
- k-core level;
- PageRank/eigenvector centrality;
- betweenness approximation;
- participation coefficient across anatomical modules;
- average recurrent-cycle count/proxy;
- graph distance to descending outputs;
- graph distance to central-complex / mushroom-body regions if annotated.

Fit simple development-only models to predict population quality.

The scientifically interesting outcome is not merely “population X is best,” but a rule such as:

> populations with broad recurrent reach and high cross-module participation preserve candidate information across depth better than peripheral populations.

Such a rule can later be frozen and tested on fresh held-out populations/data.

---

# Phase 9 — Baselines that bypass MaleCNS

To know whether the connectome adds value, compare against interface-matched baselines.

At minimum:

1. linear probe on raw chess encoder features;
2. linear probe on projected sensory vector before recurrence;
3. random sparse recurrent graph matched for size/degree/weight statistics where feasible;
4. identity/no-recurrence control;
5. current MaleCNS recurrence.

This asks:

```text
Does MaleCNS create more linearly recoverable chess information than the encoder/projection already contains?
```

If a high-performing readout is merely decoding the original injected features, do not credit recurrence.

---

# Phase 10 — Freeze a population-selection rule

Do **not** simply freeze the single best observed neuron set.

Prefer freezing a reproducible rule such as:

```text
anatomical filter
+ topology criterion
+ deterministic seed
+ fixed count
```

or, if task-informed selection is necessary:

```text
training/development-only feature criterion
+ regularization
+ deterministic tie-breaking
```

Document the exact selection decision before generating the final confirmation corpus.

The frozen rule should produce:

```text
data/populations_v2/input_manifest.json
data/populations_v2/readout_manifest.json
```

---

# Phase 11 — Fresh confirmatory experiment

After all population/interface choices are frozen, generate a **new held-out corpus** that has never been used for population selection, readout fitting or dynamics tuning.

Preferred design:

```text
>= 256 source-diverse positions
all legal moves
original MaleCNS + required graph controls
D = 1,2,4,8,16,32,64
cluster-aware bootstrap by source game/opening
```

Compare at least:

1. Baseline A interface;
2. frozen Population-v2 interface;
3. raw/projected-feature baseline;
4. appropriate shuffled-connectome controls.

Primary confirmatory questions:

- Does Population-v2 significantly improve held-out move quality over Baseline A?
- Does Population-v2 expose more effective dimensionality?
- Does recurrent depth help under Population-v2?
- Is any recurrent gain larger than matched random/shuffled controls?
- Does the result survive source-cluster bootstrap?

No further interface tuning is permitted after inspecting this corpus.

---

# Recommended implementation order

Codex should execute in this order:

1. audit current population selection and annotations;
2. implement versioned population manifests;
3. create readout population families;
4. run readout seed/anatomy/topology sweep;
5. run depth-specific probe and cross-depth transfer study;
6. create input population families;
7. run input sweep;
8. run reduced input × readout matrix;
9. run population-size scaling;
10. run 20-seed robustness study;
11. analyze graph predictors of population quality;
12. run raw-feature / projection / random-network baselines;
13. freeze v2 population-selection rule;
14. generate a new confirmation corpus;
15. run final all-depth/control confirmation;
16. update `docs/current_results.md`, `docs/chess_training.md`, `docs/experiment_protocol.md`, and final summaries.

Do not stop after infrastructure is implemented. Execute the experiments and commit compact results.

---

# Required compact artifacts

```text
results/population_study/baseline_population_audit.csv
results/population_study/baseline_population_summary.json
results/population_study/readout_population_results.csv
results/population_study/readout_seed_distribution.csv
results/population_study/depth_probe_transfer.csv
results/population_study/input_population_results.csv
results/population_study/interface_matrix.csv
results/population_study/population_size_scaling.csv
results/population_study/graph_predictors.csv
results/population_study/baseline_comparison.csv
results/population_study/final_selection.json
results/population_study/final_confirmation_summary.json
results/population_study/*.png
```

Large activation/state caches may remain local and ignored, but their hashes, shapes and generation metadata must be preserved.

---

# Interpretation guide

### Outcome A — readout placement matters enormously

If some biological or topology-informed readouts strongly outperform the current output sample while the graph is unchanged, conclude that the previous experiment was strongly **observation-site limited**.

### Outcome B — input placement matters enormously

If changing only sensory populations strongly alters depth behavior, conclude that **controllability / entry point** is a major determinant of useful recurrent computation.

### Outcome C — depth-specific probes recover information at deep depth

If the frozen readout fails at D16/D32 but a depth-specific probe succeeds, conclude that deep recurrence produces **representation transformation/rotation**, not simple global information loss.

### Outcome D — all populations fail similarly at deep depth

This strengthens the hypothesis that current recurrent dynamics truly destroy task-relevant information.

### Outcome E — random populations are as good as biologically informed ones

Then biological labels are not providing detectable task-relevant advantage under the current artificial interface.

### Outcome F — topology-informed selection generalizes

This would be especially interesting: it would suggest that useful interface locations can be predicted from connectome structure independently of chess labels.

### Outcome G — MaleCNS adds no value over projected chess features

Then the current chess capability is mostly in the encoder/readout pipeline, not the connectome. State that plainly.

### Outcome H — Population-v2 plus recurrent depth beats all matched baselines on fresh confirmation

This would justify a much stronger next step into the locked research program: test limited training of MaleCNS parameters while preserving the newly validated interface.

---

# Claims not allowed from this study alone

Do not claim:

- that selected neurons are biological “chess neurons”;
- that MaleCNS evolved for abstract reasoning;
- that a development-selected population proves biological specificity;
- that a best population found after many trials is valid without fresh confirmation;
- that higher probe accuracy alone proves recurrent thinking;
- that anatomical labels imply function unless supported by the released MaleCNS annotations/literature.

The goal is narrower and testable:

> **Quantify how much computational capability and recurrent-depth behavior depend on where information enters and where internal state is observed in the MaleCNS connectome.**
