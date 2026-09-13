from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .chess_agent import FlyCandidateMoveAgent
from .chess_features import HashedSensoryProjector
from .engine import RecurrentDepthEngine
from .lif import LIFRecurrentDepthEngine
from .malecns import load_malecns_feather
from .readout_training import ReadoutCheckpoint, load_readout_checkpoint


@dataclass(frozen=True)
class LoadedFlyAgent:
    agent: FlyCandidateMoveAgent
    checkpoint: ReadoutCheckpoint
    graph_neurons: int
    dynamics: str


def _metadata_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def load_fly_agent_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    connectome_weights_path: str | Path,
    sensory_indices_path: str | Path,
    min_synapses: int = 3,
    depth_override: int | None = None,
) -> LoadedFlyAgent:
    """Reconstruct a trained FlyCandidateMoveAgent from a readout checkpoint.

    The connectome itself is loaded fresh and remains frozen. Projector/dynamics
    settings are taken from the checkpoint metadata produced during activation
    extraction, preventing the benchmark from silently using a different chess
    adapter than training.
    """
    checkpoint = load_readout_checkpoint(checkpoint_path)
    metadata = checkpoint.metadata or {}

    graph, _ = load_malecns_feather(
        annotations_path,
        neurotransmitters_path,
        connectome_weights_path,
        traced_only=True,
        min_synapses=int(min_synapses),
    )

    sensory_indices = np.load(sensory_indices_path).astype(np.int64)
    if sensory_indices.ndim != 1 or sensory_indices.size == 0:
        raise ValueError("sensory indices must be a non-empty vector")
    if np.any(sensory_indices < 0) or np.any(sensory_indices >= graph.n_neurons):
        raise ValueError("sensory index outside loaded MaleCNS graph")

    fanout = int(metadata.get("projector_fanout", 4))
    amplitude = float(metadata.get("projector_amplitude", 1.0))
    seed = checkpoint.projector_seed
    if seed is None:
        seed = int(metadata.get("projector_seed", 0))
    clamp_sensory = _metadata_bool(metadata.get("clamp_sensory"), True)

    projector = HashedSensoryProjector(
        graph.n_neurons,
        sensory_indices,
        fanout=fanout,
        seed=int(seed),
        amplitude=amplitude,
    )

    dynamics_name = str(metadata.get("dynamics", "LIFRecurrentDepthEngine"))
    if dynamics_name in {"LIFRecurrentDepthEngine", "lif", "LIF"}:
        engine = LIFRecurrentDepthEngine(graph)
        normalized_dynamics = "lif"
    elif dynamics_name in {"RecurrentDepthEngine", "rate", "RATE"}:
        engine = RecurrentDepthEngine(graph)
        normalized_dynamics = "rate"
    else:
        raise ValueError(f"unsupported checkpoint dynamics: {dynamics_name!r}")

    depth = checkpoint.recurrent_depth if depth_override is None else int(depth_override)
    if depth < 1:
        raise ValueError("depth_override must be >= 1")

    agent = FlyCandidateMoveAgent(
        engine=engine,
        projector=projector,
        readout_indices=checkpoint.readout_indices,
        readout_weights=checkpoint.readout_weights,
        depth=depth,
        clamp_sensory=clamp_sensory,
        name="MaleCNS-RD",
    )
    return LoadedFlyAgent(
        agent=agent,
        checkpoint=checkpoint,
        graph_neurons=int(graph.n_neurons),
        dynamics=normalized_dynamics,
    )
