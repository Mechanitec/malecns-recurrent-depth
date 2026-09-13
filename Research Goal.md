# MaleCNS Recurrent-Depth Chess Research Goal

**Status: LOCKED RESEARCH SCOPE**

This document defines the intended scientific scope, scale, and boundaries of the MaleCNS recurrent-depth chess project. It is the project-level research contract. Implementation plans, scripts, dashboards, benchmarks, and training procedures may evolve, but they should serve the questions defined here rather than silently changing them.

Any material change to the research questions, biological constraints, benchmark role, training scope, or interpretation rules below should require an explicit edit to this document.

---

## 1. Central research goal

The central goal is to determine whether a biologically derived, recurrent neural architecture based on the **MaleCNS fruit-fly connectome** can gain measurable computational capability from repeated internal computation, and how that capability changes when the network is allowed limited task-specific learning.

Chess is the primary controlled task because it provides:

- a very large space of structured decisions,
- an objective legal action set,
- strong teacher engines,
- scalable opponent difficulty,
- position-by-position evaluation,
- and a quantitative external performance measure through chess rating.

The project is **not** primarily an attempt to build the strongest possible chess engine. Chess is an experimental instrument for measuring capability in a recurrent biological architecture.

The primary object of study is the interaction among:

1. **biological network structure**,
2. **recurrent internal computation**, and
3. **task-specific learning**.

---

## 2. Core hypothesis

A fixed recurrent network may perform better when the same neural circuitry is allowed to update its internal state for additional recurrent steps before a decision is read out.

For a chess position and candidate move encoded as input `x`, the same network transition is repeatedly applied:

```text
h_1 = F(h_0, x)
h_2 = F(h_1, x)
h_3 = F(h_2, x)
...
h_D = F(h_(D-1), x)
```

where:

- `F` is the same MaleCNS-derived recurrent network at every step,
- `h` is the evolving neural state,
- and `D` is recurrent depth.

The first hypothesis is therefore:

> **H1 — Recurrent-depth hypothesis:** With network structure and learned parameters held fixed, increasing recurrent depth can improve chess decision quality and playing strength up to an architecture-dependent optimum.

A second hypothesis extends this to learning:

> **H2 — Learning/recurrent-computation interaction:** Limited chess-specific adaptation of a MaleCNS-derived network can increase chess strength, and the benefit of that learning may interact non-trivially with recurrent depth.

A third hypothesis concerns preservation of biology:

> **H3 — Biological-efficiency hypothesis:** A meaningful fraction of the attainable chess improvement may be achieved with only limited deviation from the original MaleCNS-derived parameters, rather than requiring the biological network to be effectively replaced by a generic chess network.

---

## 3. The four required headline experiments

The research program is organized around four experiments. These are all in scope and together define the intended scale of the project.

### Experiment A — Frozen fly: recurrent depth only

The MaleCNS-derived network remains fixed. The chess interface and trained readout remain fixed. Only recurrent depth changes.

Required depth sweep:

```text
D = 1, 2, 4, 8, 16, 32, 64
```

Additional depths may be added near an observed optimum.

This is the cleanest causal experiment. If performance changes, the change must come from allowing the same fixed recurrent system more internal computation.

Primary outputs:

- chess rating with uncertainty,
- candidate-move agreement with teacher engine,
- centipawn loss / evaluation loss,
- move latency,
- neural activity statistics,
- and recurrent-depth utilization.

### Experiment B — Trained fly: learning with controlled recurrent depth

Selected parameters inside the MaleCNS-derived neural system are allowed to adapt to chess while the biological topology remains fixed.

The purpose is to determine how much chess capability can be added through learning while preserving the architecture.

Training should progress from minimally invasive to increasingly flexible parameter classes rather than immediately allowing unrestricted edge-by-edge optimization.

### Experiment C — Trained fly plus recurrent thinking

Training amount and recurrent depth are varied together.

The target quantity is a performance surface:

```text
Chess strength = f(training amount, recurrent depth)
```

This experiment asks whether learning and repeated internal computation are:

- additive,
- synergistic,
- redundant,
- or antagonistic.

It must also determine whether the recurrent depth that is optimal for the native/frozen network changes after training.

### Experiment D — Biological deviation versus capability

The degree of departure from the original MaleCNS-derived parameters is explicitly measured.

The target quantity becomes:

```text
Performance = f(training amount, recurrent depth, biological deviation)
```

The aim is not merely to locate the maximum Elo. The scientifically preferred result is a **Pareto frontier** between chess capability and preservation of the original biological system.

A particularly important target is:

> the smallest biological modification that produces a large and reproducible increase in chess capability while retaining a meaningful benefit from recurrent depth.

---

## 4. What remains fixed in the frozen-fly experiment

For Experiment A, the following must be held constant across the recurrent-depth sweep:

- MaleCNS neuron set,
- connectome topology,
- edge provenance and sign convention,
- neural dynamics parameters,
- sensory population,
- sensory projection/encoder,
- readout population,
- trained readout weights,
- chess checkpoint,
- candidate-move scoring procedure,
- opening set,
- opponent configuration,
- engine versions,
- hardware-affecting engine options where relevant,
- random seeds where stochasticity is involved,
- and rating methodology.

Only recurrent depth should intentionally vary.

This separation is essential. A model separately trained at each depth may be useful as an engineering comparison, but it is **not** the primary causal recurrent-depth result.

---

## 5. What may learn

The project intentionally includes a progression from fixed biology to limited learned adaptation.

### Stage 0 — Decoder/readout learning only

The connectome itself remains unchanged. A small readout learns how to interpret the activity of the frozen network for chess candidate ranking.

This is considered an interface/decoder rather than a modification of the biological network.

### Stage 1 — Global or transmitter-specific gains

Examples include learnable gains for classes such as excitatory, inhibitory, or transmitter-defined connections.

This changes physiology at a coarse level while preserving individual connectivity.

### Stage 2 — Region-, cell-type-, or population-specific physiology

Examples include:

- gain,
- threshold,
- leak,
- excitability,
- recurrent scaling,
- or other neural-dynamics parameters.

### Stage 3 — Existing synaptic strengths may adapt

Individual or grouped connection strengths may be trained while preserving the original topology.

A preferred parameterization keeps the original weight as the reference point, for example:

```text
W_ij = W0_ij * exp(delta_ij)
```

with regularization that penalizes large `delta_ij`.

### Stage 4 — Highly flexible edge-level adaptation

This is allowed only as a later experiment and must be clearly labeled as a more strongly modified fly.

The purpose is to explore the upper end of the capability/deviation frontier, not to redefine the baseline system.

---

## 6. What should remain biologically fixed during training

Unless a later experiment explicitly relaxes a rule, the following constraints define the trained-fly research program:

- the MaleCNS neuron identities remain fixed,
- the connectome topology remains fixed,
- connections absent from MaleCNS are not created,
- training should not silently replace the recurrent architecture with a conventional chess network,
- the chess encoder and any auxiliary learned components must be reported separately from connectome learning,
- and every trained model must preserve enough metadata to quantify its distance from the original MaleCNS-derived system.

The default topology rule is:

```text
A_ij = 0  =>  W_ij = 0
```

A future topology-changing experiment would be outside the locked core scope unless this document is explicitly revised.

---

## 7. Biological deviation must be measured

Training strength alone is not an adequate description of how much the fly has changed.

Every trained-connectome experiment should report one or more explicit deviation measures relative to the original MaleCNS-derived parameters.

Possible measures include:

- normalized L2 parameter displacement,
- RMS log-ratio of synaptic weight magnitude,
- fraction of weights changed beyond defined thresholds,
- distribution of learned gains by transmitter/region/cell type,
- neuron-parameter displacement,
- and task-performance gain per unit of biological deviation.

For example, an edge-weight deviation measure may take the form:

```text
D_fly = sqrt(mean((log(|W_ij| / |W0_ij|))^2))
```

with appropriate handling of zero or signed values.

The exact metric may evolve, but biological deviation must remain a first-class experimental variable.

---

## 8. Definition of the optimization target

The project does **not** define success as maximum chess Elo alone.

The optimization target is multi-objective:

1. maximize chess capability,
2. preserve as much of the original MaleCNS-derived system as practical,
3. preserve a measurable benefit from recurrent internal computation,
4. minimize unnecessary computational cost,
5. and maintain reproducibility and interpretability of the experimental comparison.

The main scientific object is therefore a Pareto frontier rather than one scalar optimum.

Important candidate optima include:

- highest Elo regardless of biological deviation,
- highest Elo under a fixed deviation budget,
- largest Elo gain per unit of biological modification,
- highest Elo under a fixed recurrent-compute budget,
- and the smallest modification that produces a large recurrent-depth benefit.

---

## 9. Chess benchmark role

Chess remains the primary quantitative benchmark throughout the locked scope.

The project should measure at least:

- legal-move decision accuracy,
- teacher best-move agreement,
- candidate ranking quality,
- centipawn or WDL loss,
- actual game results,
- estimated chess rating with confidence intervals,
- performance by recurrent depth,
- and performance by training/deviation level.

Opponent strength should span from near-random chess through progressively stronger conventional engines.

Weak-engine Elo labels must not be invented. Low-strength opponents should be empirically calibrated or explicitly reported as engine-specific levels when cross-engine calibration is uncertain.

A separate strong Stockfish process may evaluate positions during games to show who is winning, but this evaluator must not influence the moves chosen by either the fly or its opponent.

---

## 10. Required recurrent-depth comparison

The central chart of the frozen-fly study is:

```text
Chess strength versus recurrent depth
```

The central chart of the learned-fly study is:

```text
Chess strength versus recurrent depth versus training/deviation level
```

At minimum, the final experimental dataset should support comparisons equivalent to:

```text
                   Recurrent depth
Training       1    2    4    8    16    32    64
--------------------------------------------------
Frozen         x    x    x    x     x     x     x
Light          x    x    x    x     x     x     x
Moderate       x    x    x    x     x     x     x
Heavy          x    x    x    x     x     x     x
```

The labels `Light`, `Moderate`, and `Heavy` must eventually be replaced or supplemented by quantitative parameter/deviation definitions.

---

## 11. Required controls

Any claim that the biological wiring is important requires controls.

The core controls should include, where computationally feasible:

- original MaleCNS topology,
- degree-preserving or otherwise structurally matched topology shuffle,
- transmitter/sign shuffle,
- recurrent-edge attenuation or removal,
- alternate/random sensory populations,
- alternate/random readout populations,
- and rate-state versus LIF-style dynamics where useful.

The crucial comparison is not simply:

> deeper recurrence improves performance.

It is:

> does the real MaleCNS-derived structure use recurrent depth differently or more effectively than appropriate control networks under matched conditions?

If shuffled or generic controls gain the same amount, that result must be reported plainly.

---

## 12. Interpretation boundaries

This project may support claims about computation in a MaleCNS-derived recurrent network under an artificial chess interface.

It must **not** automatically be interpreted as evidence that:

- a fruit fly understands chess,
- the simulation reproduces biological fly cognition,
- the connectome plus simplified dynamics constitutes a digital mind,
- recurrent depth is equivalent to human conscious thought,
- or observed chess performance directly reflects abilities of a living fly.

Terms such as **thinking**, **recurrent thinking**, or **internal reasoning** are useful shorthand for repeated hidden-state computation, but formal results should define the operation precisely.

The connectome is biologically measured structure; the simulated dynamics, encoding, training procedure, and chess interface are experimental computational constructs.

---

## 13. Scale of the research program

This is intended to be a substantial experimental research program rather than a one-off demo.

The planned scale includes:

- real MaleCNS data rather than only synthetic graphs,
- reproducible sensory/readout population selection,
- a teacher-labelled chess dataset large enough for meaningful train/validation/test separation,
- cached MaleCNS candidate activations,
- trained readout checkpoints,
- calibrated weak-engine opponents,
- hundreds to thousands of evaluation games for serious rating estimates,
- recurrent-depth sweeps,
- multiple levels of connectome adaptation,
- matched control networks,
- confidence intervals and statistical uncertainty,
- and experiment-level provenance sufficient to reproduce results later.

Exploratory runs may use smaller samples, but headline conclusions should not be based on tiny game counts or one stochastic run.

The expected end state is a dataset and analysis large enough to characterize a surface rather than a single score:

```text
performance = f(recurrent depth, training, biological deviation, architecture/control)
```

---

## 14. Reproducibility requirements

Every serious experiment must record enough information to reconstruct the tested system.

At minimum this includes:

- source code commit,
- MaleCNS dataset identity and hashes,
- neuron/population selections,
- edge preprocessing and normalization mode,
- dynamics type and parameters,
- sensory encoder/projector configuration,
- readout definition,
- checkpoint hash,
- training configuration,
- training-data split/provenance,
- recurrent depth,
- opponent engine/version/settings,
- evaluator engine/version/settings,
- opening/FEN source,
- random seed,
- game count,
- and hardware-relevant settings when they can materially affect results.

No GitHub Actions are required for this project. Large experiments are expected to run locally on the development workstation.

---

## 15. Success criteria

The project succeeds scientifically even if the result is negative, provided the experiment is controlled and reproducible.

Important possible outcomes include:

### Positive recurrent-depth result

The frozen MaleCNS-derived network gains statistically meaningful chess capability as recurrent depth increases, with an identifiable optimum.

### Positive biological-structure result

The real MaleCNS topology benefits from recurrence more than matched shuffled/control networks.

### Positive learning/recurrent synergy result

Limited connectome adaptation increases chess strength and also increases, preserves, or reorganizes the benefit of recurrent depth.

### Biological-efficiency result

A large performance gain occurs with only small measured deviation from the original MaleCNS-derived parameters.

### Negative but informative result

Recurrent depth provides little benefit, controls behave similarly, or useful chess performance requires such large modification that the biological initialization ceases to provide a meaningful advantage.

All of these are valid research outcomes.

---

## 16. Primary final results to report

The mature project should be capable of reporting, at minimum:

1. **Frozen MaleCNS Elo/MCR versus recurrent depth.**
2. **Teacher decision quality versus recurrent depth.**
3. **Real MaleCNS versus shuffled/control depth curves.**
4. **Trained MaleCNS performance versus amount/type of learning.**
5. **Training × recurrent-depth performance surface.**
6. **Chess capability versus biological deviation.**
7. **Optimal recurrent depth as a function of training/deviation.**
8. **Pareto frontier: capability versus biological preservation versus compute.**
9. **Live and post-game Stockfish evaluation traces for representative games.**
10. **A fully reproducible description of the best native, minimally adapted, and maximum-performance systems.**

A representative final scientific statement might take the form:

> The unmodified MaleCNS-derived recurrent network gained `X` rating points when recurrent depth increased from `D1` to `D2`. Under limited chess-specific adaptation with biological deviation `Z`, the same recurrent architecture gained `Y` additional rating points and achieved peak performance at depth `D*`. Matched topology/sign controls showed `C` recurrent-depth gain under the same protocol.

The actual values must come from experiment; no expected result is assumed in advance.

---

## 17. Explicit non-goals

The following are outside the locked core scope unless this document is deliberately revised:

- maximizing chess strength by replacing MaleCNS with a standard chess architecture,
- topology search that freely invents new non-biological connections,
- claiming digital consciousness or biological equivalence,
- optimizing primarily for commercial chess-engine strength,
- hiding negative controls or failed hypotheses,
- using opponent-engine assistance to select the fly's moves during evaluation,
- and changing multiple experimental variables while presenting the result as a recurrent-depth effect.

Other cognitive tasks may be explored later, but they should extend rather than replace the chess-centered core research program.

---

## 18. Scope lock

The project is considered to remain within scope when work contributes to one or more of the following:

```text
1. Frozen MaleCNS + recurrent depth
2. Controlled MaleCNS learning
3. Learning × recurrent-depth interaction
4. Biological-deviation / capability frontier
5. Matched architecture controls
6. Rigorous chess measurement and reproducibility
```

Infrastructure is justified only insofar as it enables these experiments.

The project should resist drifting into a generic chess-engine project or a generic neural-network optimization project.

**Locked central question:**

> **How much computational capability can a MaleCNS-derived recurrent biological architecture obtain from repeated internal computation, how does chess-specific learning alter that capability, and what combination of recurrent depth and limited biological adaptation produces the strongest performance while preserving meaningful continuity with the original connectome?**

That question defines the intended scope and scale of the research.