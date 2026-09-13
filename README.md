# MaleCNS-RD prototype

A small, executable prototype for treating a connectome as a **shared recurrent-depth reasoning block**.

The central update is

`h[d+1] = F(h[d], sensory, W_connectome)`

where the **same graph and same dynamics are reused at every depth**. Sensory input may be held fixed during an internal thinking period; action/motor readout is taken only after the requested depth.

## What is implemented

- Sparse signed directed connectome (`scipy.sparse`).
- Repeated shared recurrent update block.
- Fixed inference depth and adaptive convergence stopping.
- Separation between sensory clamping and final readout.
- A controlled multi-hop benchmark where deeper recurrence is required to reach increasingly distant targets.
- MaleCNS v1.0 Feather loader with transmitter-based sign assignment.
- Downloader for the small official MaleCNS annotation + neurotransmitter files.
- Tests.

## Run now

```bash
cd malecns-recurrent-depth
python scripts/run_demo.py
pytest -q
```

The demo writes `results/depth_scaling.csv`.

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

The edge table is ~1.1 GB and is deliberately not downloaded automatically. Once placed under `data/`, load it with:

```python
from malecns_rd.malecns import load_malecns_feather
from malecns_rd.engine import RecurrentDepthEngine

graph, annotations = load_malecns_feather(
    'data/body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'data/body-neurotransmitters-male-cns-v1.0.feather',
    'data/connectome-weights-male-cns-v1.0-minconf-0.5.feather',
    traced_only=True,
    min_synapses=3,
)
engine = RecurrentDepthEngine(graph)
```

## Scientific caution

This prototype tests a computational hypothesis. It does **not** claim that repeated graph updates reproduce real fly cognition. The connectome is measured anatomy, but the membrane/update rule, transmitter simplifications, gains, normalization, stopping rule and task objective are modeling choices.

A serious next stage should compare several dynamics (LIF, rate, conductance-based), preserve provenance for every filter, and evaluate held-out tasks while sweeping inference depth.

## Adaptive recurrent depth

`run_until_confident(...)` implements a first adaptive test-time-compute controller. It repeatedly applies the same connectome update and stops only when a chosen motor/readout population has a stable winner with sufficient margin. This is intentionally separate from the biological dynamics so that different stopping rules can be tested without changing the connectome.

## Current modeling choice

The prototype uses a sparse rate-state recurrence because it is easy to inspect and lets us test the recurrent-depth hypothesis independently of membrane-model details. The next biologically stronger version should add a LIF engine using MaleCNS neurotransmitter signs and compare fixed biological-time simulation against extra latent-depth passes under matched compute.
