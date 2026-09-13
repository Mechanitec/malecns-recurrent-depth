from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .graph import ConnectomeGraph


@dataclass
class LIFRunResult:
    membrane: np.ndarray
    spikes: np.ndarray
    spike_counts: np.ndarray
    depth_used: int
    total_spikes: list[int]
    readout_margin: float | None = None


class LIFRecurrentDepthEngine:
    """Sparse discrete-time leaky-integrate-and-fire recurrent-depth engine.

    One recurrent-depth pass corresponds to one membrane integration microstep
    while the external observation may remain frozen. The same connectome and
    neuron parameters are reused at every pass.

    Connectome weights are interpreted as signed voltage increments after
    multiplication by ``recurrent_gain``. This is intentionally a simplified
    LIF model, not a conductance-based simulation.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        *,
        dt_ms: float = 1.0,
        tau_membrane_ms: float = 10.0,
        threshold: float = 0.5,
        reset_potential: float = 0.0,
        resting_potential: float = 0.0,
        recurrent_gain: float = 1.0,
        input_gain: float = 1.0,
        refractory_steps: int = 0,
    ) -> None:
        if dt_ms <= 0:
            raise ValueError("dt_ms must be > 0")
        if tau_membrane_ms <= 0:
            raise ValueError("tau_membrane_ms must be > 0")
        if threshold <= resting_potential:
            raise ValueError("threshold must exceed resting_potential")
        if refractory_steps < 0:
            raise ValueError("refractory_steps must be >= 0")
        self.graph = graph
        self.dt_ms = float(dt_ms)
        self.tau_membrane_ms = float(tau_membrane_ms)
        self.threshold = float(threshold)
        self.reset_potential = float(reset_potential)
        self.resting_potential = float(resting_potential)
        self.recurrent_gain = float(recurrent_gain)
        self.input_gain = float(input_gain)
        self.refractory_steps = int(refractory_steps)
        self.decay = float(math.exp(-self.dt_ms / self.tau_membrane_ms))

    def step(
        self,
        membrane: np.ndarray,
        previous_spikes: np.ndarray,
        sensory: np.ndarray,
        refractory: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.graph.n_neurons
        membrane = np.asarray(membrane, dtype=np.float32)
        previous_spikes = np.asarray(previous_spikes, dtype=np.float32)
        sensory = np.asarray(sensory, dtype=np.float32)
        if membrane.shape != (n,) or previous_spikes.shape != (n,) or sensory.shape != (n,):
            raise ValueError(f"all state vectors must have shape {(n,)}")
        if refractory is None:
            refractory = np.zeros(n, dtype=np.int32)
        else:
            refractory = np.asarray(refractory, dtype=np.int32)
            if refractory.shape != (n,):
                raise ValueError(f"refractory must have shape {(n,)}")

        recurrent = self.graph.weights @ previous_spikes
        active = refractory == 0
        next_membrane = np.full(n, self.reset_potential, dtype=np.float32)
        if np.any(active):
            decayed = self.resting_potential + self.decay * (membrane[active] - self.resting_potential)
            drive = self.recurrent_gain * recurrent[active] + self.input_gain * sensory[active]
            next_membrane[active] = decayed + drive

        spikes = (active & (next_membrane >= self.threshold)).astype(np.float32)
        next_membrane[spikes.astype(bool)] = self.reset_potential

        next_refractory = np.maximum(refractory - 1, 0).astype(np.int32, copy=False)
        if self.refractory_steps:
            next_refractory[spikes.astype(bool)] = self.refractory_steps
        return next_membrane, spikes, next_refractory

    def run(
        self,
        sensory: np.ndarray,
        *,
        max_depth: int = 16,
        initial_membrane: np.ndarray | None = None,
        initial_spikes: np.ndarray | None = None,
        clamp_sensory: bool = True,
    ) -> LIFRunResult:
        n = self.graph.n_neurons
        sensory = np.asarray(sensory, dtype=np.float32)
        if sensory.shape != (n,):
            raise ValueError(f"sensory must have shape {(n,)}")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        membrane = (
            np.full(n, self.resting_potential, dtype=np.float32)
            if initial_membrane is None
            else np.asarray(initial_membrane, dtype=np.float32).copy()
        )
        spikes = (
            np.zeros(n, dtype=np.float32)
            if initial_spikes is None
            else np.asarray(initial_spikes, dtype=np.float32).copy()
        )
        if membrane.shape != (n,) or spikes.shape != (n,):
            raise ValueError(f"initial states must have shape {(n,)}")

        refractory = np.zeros(n, dtype=np.int32)
        spike_counts = np.zeros(n, dtype=np.int32)
        total_spikes: list[int] = []
        zero_sensory = np.zeros_like(sensory)

        for depth in range(1, max_depth + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            membrane, spikes, refractory = self.step(membrane, spikes, step_input, refractory)
            spike_counts += spikes.astype(np.int32)
            total_spikes.append(int(spikes.sum()))

        return LIFRunResult(membrane, spikes, spike_counts, max_depth, total_spikes)

    def run_until_confident(
        self,
        sensory: np.ndarray,
        readout_indices: np.ndarray,
        *,
        max_depth: int = 64,
        min_depth: int = 2,
        count_margin: int = 1,
        stable_steps: int = 2,
        clamp_sensory: bool = True,
    ) -> LIFRunResult:
        """Spend additional recurrent depth until spike-count readout is stable."""
        n = self.graph.n_neurons
        sensory = np.asarray(sensory, dtype=np.float32)
        readout_indices = np.asarray(readout_indices, dtype=np.int64)
        if sensory.shape != (n,):
            raise ValueError(f"sensory must have shape {(n,)}")
        if readout_indices.ndim != 1 or len(readout_indices) < 2:
            raise ValueError("readout_indices must contain at least two neurons")
        if np.any(readout_indices < 0) or np.any(readout_indices >= n):
            raise ValueError("readout index outside graph")
        if not 1 <= min_depth <= max_depth:
            raise ValueError("require 1 <= min_depth <= max_depth")
        if count_margin < 1 or stable_steps < 1:
            raise ValueError("count_margin and stable_steps must be >= 1")

        membrane = np.full(n, self.resting_potential, dtype=np.float32)
        spikes = np.zeros(n, dtype=np.float32)
        refractory = np.zeros(n, dtype=np.int32)
        spike_counts = np.zeros(n, dtype=np.int32)
        total_spikes: list[int] = []
        zero_sensory = np.zeros_like(sensory)
        last_winner = -1
        stable = 0
        last_margin = 0.0

        for depth in range(1, max_depth + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            membrane, spikes, refractory = self.step(membrane, spikes, step_input, refractory)
            spike_counts += spikes.astype(np.int32)
            total_spikes.append(int(spikes.sum()))

            if depth < min_depth:
                continue
            counts = spike_counts[readout_indices]
            order = np.argsort(counts)
            winner = int(order[-1])
            top = int(counts[order[-1]])
            second = int(counts[order[-2]])
            last_margin = float(top - second)
            if top > 0 and top - second >= count_margin:
                if winner == last_winner:
                    stable += 1
                else:
                    last_winner = winner
                    stable = 1
                if stable >= stable_steps:
                    return LIFRunResult(
                        membrane, spikes, spike_counts, depth, total_spikes, last_margin
                    )
            else:
                last_winner = -1
                stable = 0

        return LIFRunResult(
            membrane, spikes, spike_counts, max_depth, total_spikes, last_margin
        )
