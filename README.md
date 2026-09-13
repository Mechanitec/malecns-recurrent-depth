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
- Hybrid chess Elo benchmark: Alfil below Stockfish's Elo floor, Stockfish above it.
- Candidate-move chess encoding and deterministic sensory projection.
- Multi-opponent maximum-likelihood Elo estimator with approximate 95% interval.
- Tests for propagation, determinism, adaptive stopping, inhibition, refractory behavior, Elo estimation, opponent routing, and sensory projection.
- Experimental protocols in `docs/experiment_protocol.md` and `docs/chess_elo_protocol.md`.

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

## Chess Elo benchmark

The project uses a hybrid opponent ladder so very weak fly checkpoints can still receive a useful rating.

- **Below Stockfish's runtime-advertised Elo floor:** use Alfil.
- **At and above Stockfish's floor:** use Stockfish.

Current Stockfish builds commonly begin around 1320. Alfil publishes nominal `UCI_Elo` levels `0, 200, 400, ..., 3000`; therefore the default low ladder is `0, 200, 400, 600, 800, 1000, 1200`, followed by Stockfish `1320, 1400, 1500, ...`. The harness detects the installed Stockfish floor at runtime. Unsupported low values such as 1300 are rejected rather than silently rounded.

Every game record stores both `opponent_engine` and `opponent_elo`. Both engines use `UCI_LimitStrength=true` and `UCI_Elo=<target>`.

The Alfil and Stockfish numbers are nominal engine ratings, not guaranteed to lie on one perfectly aligned absolute scale. Serious experiments should cross-calibrate the two engines in their overlap region and freeze the exact binaries, settings, and time control.

The fly does not require one output neuron for every chess move. Instead, every legal move is evaluated as a candidate:

1. encode the board plus candidate move;
2. project the features into a frozen sensory population;
3. run the same connectome for a chosen recurrent depth;
4. score the candidate from a frozen readout population;
5. play the highest-scoring legal move.

This makes `depth = 1, 2, 4, 8, ...` directly comparable while keeping graph, chess interface, and readout fixed.

Install the optional chess dependency and run a harness smoke test with local Alfil and Stockfish binaries:

```bash
pip install -e '.[chess]'
python scripts/run_chess_benchmark.py \
  --alfil /path/to/alfil \
  --stockfish /path/to/stockfish \
  --elos 0,200,400,600,800,1000,1200,1320,1400,1500 \
  --games-per-elo 20
```

The current CLI uses a random legal-move agent only to verify tournament plumbing. The next checkpoint will replace that baseline with the real `FlyCandidateMoveAgent` plus trained/frozen chess readout parameters.

See `docs/chess_elo_protocol.md` for the full experimental design.

## Next milestone

The immediate target is the first actual MaleCNS chess checkpoint:

1. load the signed MaleCNS graph;
2. select and freeze sensory and readout populations;
3. train only the chess adapter/readout without changing the graph topology;
4. run the first hybrid Alfil + Stockfish Elo ladder;
5. sweep recurrent depth with the same frozen checkpoint;
6. compare against shuffled-topology and shuffled-transmitter controls;
7. optimize for reproducible Elo gain, not isolated wins.
