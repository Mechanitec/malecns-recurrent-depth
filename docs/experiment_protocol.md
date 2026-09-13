# Recurrent-depth experiment protocol

## Goal

Test whether additional applications of the same connectome dynamics block improve task performance under controlled conditions, without changing graph topology or model parameters.

The central claim under test is computational, not biological: a connectome may act as an iterative recurrent substrate whose useful computation scales with internal depth.

## Primary hypothesis

For tasks requiring progressively longer graph-mediated computation, accuracy should increase as recurrent depth increases while graph structure and neuron parameters remain fixed.

## Critical confound

A deeper spiking simulation also represents more membrane-integration time. Therefore, improved performance by itself does **not** establish latent reasoning.

Experiments must distinguish:

1. **Biological-time scaling** — run the LIF system for more integration steps while the environment evolves normally.
2. **Frozen-observation depth** — hold one observation fixed and permit additional internal recurrent passes before action/readout.
3. **Single-pulse propagation** — present evidence only on the first pass, then remove sensory drive.
4. **Rate-state recurrence** — use the non-spiking recurrent engine as a computational control.

A convincing recurrent-depth effect should survive controls that cannot be explained solely by extra exposure to changing sensory input.

## Current synthetic benchmark

The routing benchmark contains unique directed paths of lengths 2, 4, 8, 16 and 32 plus weak signed distractor connections. The correct target cannot receive causal information until enough applications of the shared dynamics block have occurred.

Sweep depths:

`1, 2, 4, 8, 16, 32, 40`

Record:

- overall accuracy;
- accuracy by required path length;
- depth used by adaptive stopping;
- total spikes for LIF;
- state norm/change for rate dynamics;
- readout margin.

## Controls for MaleCNS experiments

When the real MaleCNS graph is introduced, include at least these controls:

- original signed connectome;
- edge-shuffled graph preserving approximate degree statistics;
- transmitter-sign shuffled graph;
- recurrent edges removed or strongly attenuated;
- fixed-depth versus adaptive-depth compute;
- clamped observation versus single-pulse observation;
- matched-compute comparison between rate and LIF dynamics.

## Success criteria

A useful first result is not "the fly reasons." A useful result is one of the following:

- task accuracy improves monotonically over a meaningful depth range;
- longer causal tasks require systematically greater adaptive depth;
- the effect is stronger in the real connectome than in topology-destroying controls;
- useful depth scaling persists after matching sensory exposure and compute.

## Failure criteria

Treat any of these as evidence against the current formulation:

- performance improves only because sensory input is repeatedly injected;
- deeper recurrence only increases global activity without improving selectivity;
- adaptive stopping correlates with activity magnitude but not task difficulty;
- shuffled graphs perform as well as or better than the anatomical graph;
- results are highly unstable to small threshold/gain changes.

## Next real-data milestone

1. Load MaleCNS v1.0 signed connectivity.
2. Select a tractable sensory-to-descending-neuron subnetwork before full-CNS runs.
3. Define a task with an externally verifiable target readout.
4. Sweep depth under frozen-observation and single-pulse conditions.
5. Run topology and transmitter-sign controls.
6. Only then scale to the full graph.
