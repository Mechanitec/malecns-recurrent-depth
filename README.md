# MaleCNS-RD prototype

An executable research prototype for treating a connectome as a **shared recurrent-depth computational block**.

The central idea is:

`state[d+1] = F(state[d], sensory, W_connectome)`

where the **same graph and same dynamics are reused at every depth**. Sensory evidence may be held fixed during an internal computation period, and action/readout is taken only after the requested or adaptively selected depth.

## Implemented dynamics

Two engines now share the same `ConnectomeGraph`:

- **Rate-state recurrent engine** (`RecurrentDepthEngine`) using sparse signed propagation plus `tanh`/ReLU state updates.
- **Leaky-integrate-and-fire engine** (`LIFRecurrentDepthEngine`) using membrane decay, signed recurrent spike drive, threshold/reset behavior, optional refractory steps, spike-count readout, and adaptive depth.

Both support fixed-depth experiments. Both can keep sensory evidence clamped while recurrent computation continues.

## What is implemented

- Sparse signed directed connectome (`scipy.sparse`).
- Repeated shared recurrent update blocks.
- Fixed inference depth.
- Adaptive test-time depth controllers.
- Separation between sensory clamping and final readout.
- Controlled multi-hop benchmarks requiring 2/4/8/16/32-hop propagation.
- MaleCNS v1.0 Feather loader with neurotransmitter-based sign assignment.
- Downloader for the small official MaleCNS annotation + neurotransmitter files.
- Synthetic benchmark result files for both rate and LIF dynamics.
- Tests for propagation, determinism, adaptive stopping, inhibition and refractory behavior.
- An explicit experimental protocol in `docs/experiment_protocol.md`.

## Run

```bash
cd malecns-recurrent-depth
pip install -e '.[dev]'
python scripts/run_demo.py
python scripts/run_lif_demo.py
pytest -q
```

The benchmark scripts write:

- `results/depth_scaling.csv`
- `results/lif_depth_scaling.csv`

## Current LIF depth result

On the controlled noisy routing benchmark, the LIF engine produces the following overall accuracy as recurrent depth increases:

| Depth | Accuracy |
|---:|---:|
| 1 | 0.8% |
| 2 | 0.8% |
| 4 | 20.0% |
| 8 | 40.0% |
| 16 | 60.0% |
| 32 | 80.8% |
| 40 | 100.0% |

This demonstrates a causal depth effect in the prototype: longer paths cannot affect their target readouts until enough applications of the same recurrent block have occurred.

It does **not** establish that the fly connectome performs abstract reasoning.

## LIF example

```python
import numpy as np
from malecns_rd import ConnectomeGraph, LIFRecurrentDepthEngine

src = np.array([0, 1, 2])
dst = np.array([1, 2, 3])
weight = np.ones(3, dtype=np.float32)
graph = ConnectomeGraph.from_edges(4, src, dst, weight)

engine = LIFRecurrentDepthEngine(
    graph,
    dt_ms=1.0,
    tau_membrane_ms=10.0,
    threshold=0.5,
)

sensory = np.zeros(4, dtype=np.float32)
sensory[0] = 1.0
result = engine.run(sensory, max_depth=4, clamp_sensory=True)
print(result.spike_counts)
```

## Use the real MaleCNS graph

Janelia's official MaleCNS v1.0 flat files are expected:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

Install the optional Feather dependency:

```bash
pip install -e '.[malecns]'
python scripts/download_malecns_metadata.py
```

The large connectivity table is deliberately not downloaded automatically. Once placed under `data/`, load it with:

```python
from malecns_rd.malecns import load_malecns_feather
from malecns_rd import LIFRecurrentDepthEngine

graph, annotations = load_malecns_feather(
    'data/body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'data/body-neurotransmitters-male-cns-v1.0.feather',
    'data/connectome-weights-male-cns-v1.0-minconf-0.5.feather',
    traced_only=True,
    min_synapses=3,
)
engine = LIFRecurrentDepthEngine(graph)
```

## Adaptive recurrent depth

The rate engine uses state convergence/readout margin. The LIF engine uses accumulated spike-count margin plus winner stability.

The intent is **adaptive test-time compute**: easy cases may terminate after few applications of the same graph dynamics, while difficult cases can use more recurrent passes without adding parameters or connections.

## Scientific caution: depth is not automatically reasoning

For the LIF engine, one recurrent pass is also one membrane-integration microstep. Therefore, a deeper run also represents more internal simulated time. Improvement with depth alone is not enough to claim latent reasoning.

`docs/experiment_protocol.md` defines the controls required to separate:

- more biological simulation time;
- repeated sensory exposure;
- causal multi-hop propagation;
- useful extra internal recurrent computation.

The strongest future result would show that useful depth scaling survives matched sensory exposure and topology-destroying controls on the **actual MaleCNS graph**.

## Next milestone

The next target is a real-data experiment rather than another synthetic architecture feature:

1. load the signed MaleCNS graph;
2. select a tractable sensory-to-descending-neuron subnetwork;
3. define a verifiable readout task;
4. sweep recurrent depth with frozen-observation and single-pulse conditions;
5. compare against shuffled-topology and shuffled-transmitter controls;
6. scale to the full CNS only after the protocol is stable.
