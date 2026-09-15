from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from .graph import ConnectomeGraph


@dataclass
class RunResult:
    state: np.ndarray
    depth_used: int
    deltas: list[float]
    state_norms: list[float]


@dataclass
class BatchTrajectoryResult:
    """Intermediate rate states and cumulative wall time by recurrent depth."""

    snapshots: dict[int, np.ndarray]
    latency_s: dict[int, float]
    observations: dict[int, dict[str, np.ndarray]] | None = None


class RecurrentDepthEngine:
    """Repeatedly applies one shared connectome dynamics block.

    This is intentionally a *latent recurrent-depth* abstraction, not a claim
    of biophysical fidelity. The same graph and update rule are reused at every
    depth step. Sensory input can be clamped during the internal "thinking"
    period while motor readout is withheld until the final depth.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        *,
        leak: float = 0.15,
        recurrent_gain: float = 1.35,
        input_gain: float = 1.0,
        activation: str = "tanh",
    ) -> None:
        if not 0.0 <= leak < 1.0:
            raise ValueError("leak must be in [0, 1)")
        self.graph = graph
        self.leak = float(leak)
        self.recurrent_gain = float(recurrent_gain)
        self.input_gain = float(input_gain)
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        self.activation = activation

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x).astype(np.float32, copy=False)
        return np.maximum(x, 0.0).astype(np.float32, copy=False)

    def step(self, state: np.ndarray, sensory: np.ndarray) -> np.ndarray:
        recurrent = np.zeros_like(state) if self.recurrent_gain == 0.0 else self.graph.weights @ state
        drive = self.recurrent_gain * recurrent + self.input_gain * sensory
        proposal = self._activate(drive)
        # Leaky residual state gives the recurrent block memory while keeping
        # the same F_theta at every depth.
        return (self.leak * state + (1.0 - self.leak) * proposal).astype(np.float32, copy=False)

    def run(
        self,
        sensory: np.ndarray,
        *,
        max_depth: int = 16,
        min_depth: int = 1,
        initial_state: np.ndarray | None = None,
        adaptive: bool = False,
        tolerance: float = 1e-4,
        patience: int = 3,
        clamp_sensory: bool = True,
    ) -> RunResult:
        n = self.graph.n_neurons
        sensory = np.asarray(sensory, dtype=np.float32)
        if sensory.shape != (n,):
            raise ValueError(f"sensory must have shape {(n,)}")
        if max_depth < 1 or not 1 <= min_depth <= max_depth:
            raise ValueError("require 1 <= min_depth <= max_depth")

        if initial_state is None:
            state = np.zeros(n, dtype=np.float32)
        else:
            state = np.asarray(initial_state, dtype=np.float32).copy()
            if state.shape != (n,):
                raise ValueError(f"initial_state must have shape {(n,)}")

        deltas: list[float] = []
        norms: list[float] = []
        stable = 0
        zero_sensory = np.zeros_like(sensory)

        for depth in range(1, max_depth + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            new_state = self.step(state, step_input)
            denom = max(float(np.linalg.norm(state)), 1e-6)
            delta = float(np.linalg.norm(new_state - state) / denom)
            deltas.append(delta)
            norms.append(float(np.linalg.norm(new_state)))
            state = new_state

            if adaptive and depth >= min_depth:
                if delta < tolerance:
                    stable += 1
                    if stable >= patience:
                        return RunResult(state, depth, deltas, norms)
                else:
                    stable = 0

        return RunResult(state, max_depth, deltas, norms)

    def run_batch(
        self,
        sensory: np.ndarray,
        *,
        max_depth: int = 16,
        clamp_sensory: bool = True,
    ) -> np.ndarray:
        """Run fixed-depth rate dynamics for several sensory inputs at once."""
        sensory = np.asarray(sensory, dtype=np.float32)
        n = self.graph.n_neurons
        if sensory.ndim != 2 or sensory.shape[0] != n or sensory.shape[1] == 0:
            raise ValueError(f"sensory must have shape ({n}, batch)")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        state = np.zeros_like(sensory)
        zero_sensory = np.zeros_like(sensory)
        for depth in range(1, max_depth + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            recurrent = np.zeros_like(state) if self.recurrent_gain == 0.0 else self.graph.weights @ state
            drive = self.recurrent_gain * recurrent + self.input_gain * step_input
            proposal = self._activate(drive)
            state = (self.leak * state + (1.0 - self.leak) * proposal).astype(
                np.float32, copy=False
            )
        return state

    def run_batch_trajectory(
        self,
        sensory: np.ndarray,
        *,
        depths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        clamp_sensory: bool = True,
        observe: bool = False,
    ) -> BatchTrajectoryResult:
        """Run one batched trajectory and capture exact intermediate states.

        The state at each requested depth is the same state that independent
        calls to :meth:`run_batch` would produce.  ``latency_s`` is cumulative
        wall time from the first recurrent pass through the requested depth.
        """
        sensory = np.asarray(sensory, dtype=np.float32)
        n = self.graph.n_neurons
        if sensory.ndim != 2 or sensory.shape[0] != n or sensory.shape[1] == 0:
            raise ValueError(f"sensory must have shape ({n}, batch)")
        requested = tuple(int(depth) for depth in depths)
        if not requested or any(depth < 1 for depth in requested):
            raise ValueError("depths must contain positive integers")
        if len(set(requested)) != len(requested):
            raise ValueError("depths must be unique")
        if tuple(sorted(requested)) != requested:
            raise ValueError("depths must be sorted")

        state = np.zeros_like(sensory)
        zero_sensory = np.zeros_like(sensory)
        snapshots: dict[int, np.ndarray] = {}
        latency_s: dict[int, float] = {}
        observations: dict[int, dict[str, np.ndarray]] | None = {} if observe else None
        started = time.perf_counter()
        wanted = set(requested)
        for depth in range(1, requested[-1] + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            recurrent = np.zeros_like(state) if self.recurrent_gain == 0.0 else self.graph.weights @ state
            drive = self.recurrent_gain * recurrent + self.input_gain * step_input
            proposal = self._activate(drive)
            previous_state = state
            state = (self.leak * state + (1.0 - self.leak) * proposal).astype(
                np.float32, copy=False
            )
            if depth in wanted:
                snapshots[depth] = state.copy()
                latency_s[depth] = time.perf_counter() - started
                if observations is not None:
                    state_norm = np.linalg.norm(state, axis=0).astype(np.float32)
                    denominator = np.maximum(np.linalg.norm(previous_state, axis=0), 1e-6)
                    delta = (np.linalg.norm(state - previous_state, axis=0) / denominator).astype(
                        np.float32
                    )
                    recurrent_norm = np.linalg.norm(recurrent, axis=0).astype(np.float32)
                    sensory_norm = np.linalg.norm(self.input_gain * step_input, axis=0).astype(
                        np.float32
                    )
                    observations[depth] = {
                        "state_norm": state_norm,
                        "delta": delta,
                        "recurrent_drive_norm": recurrent_norm,
                        "sensory_drive_norm": sensory_norm,
                        "recurrent_sensory_ratio": (
                            recurrent_norm / np.maximum(sensory_norm, 1e-6)
                        ).astype(np.float32),
                        "saturation_fraction": np.mean(np.abs(proposal) >= 0.95, axis=0).astype(
                            np.float32
                        ),
                    }
        return BatchTrajectoryResult(
            snapshots=snapshots, latency_s=latency_s, observations=observations
        )

    def run_trajectory(
        self,
        sensory: np.ndarray,
        *,
        depths: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        clamp_sensory: bool = True,
    ) -> dict[int, np.ndarray]:
        """Capture scalar intermediate states using the same recurrent block."""
        sensory = np.asarray(sensory, dtype=np.float32)
        n = self.graph.n_neurons
        if sensory.shape != (n,):
            raise ValueError(f"sensory must have shape {(n,)}")
        requested = tuple(int(depth) for depth in depths)
        if not requested or any(depth < 1 for depth in requested):
            raise ValueError("depths must contain positive integers")
        if len(set(requested)) != len(requested) or tuple(sorted(requested)) != requested:
            raise ValueError("depths must be sorted and unique")

        state = np.zeros(n, dtype=np.float32)
        zero_sensory = np.zeros_like(sensory)
        snapshots: dict[int, np.ndarray] = {}
        wanted = set(requested)
        for depth in range(1, requested[-1] + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            state = self.step(state, step_input)
            if depth in wanted:
                snapshots[depth] = state.copy()
        return snapshots

    def run_until_confident(
        self,
        sensory: np.ndarray,
        readout_indices: np.ndarray,
        *,
        max_depth: int = 64,
        min_depth: int = 2,
        margin: float = 0.05,
        min_activity: float = 1e-4,
        stable_steps: int = 2,
        clamp_sensory: bool = True,
    ) -> RunResult:
        """Use extra recurrent depth until a readout decision is stable/confident.

        This is a prototype of *adaptive test-time compute*: easy cases can exit
        early, while difficult cases consume more applications of the same
        connectome block.
        """
        n = self.graph.n_neurons
        sensory = np.asarray(sensory, dtype=np.float32)
        readout_indices = np.asarray(readout_indices, dtype=np.int64)
        if sensory.shape != (n,):
            raise ValueError(f"sensory must have shape {(n,)}")
        if readout_indices.ndim != 1 or len(readout_indices) < 2:
            raise ValueError("readout_indices must contain at least two neurons")

        state = np.zeros(n, dtype=np.float32)
        deltas: list[float] = []
        norms: list[float] = []
        last_winner = -1
        stable = 0
        zero_sensory = np.zeros_like(sensory)

        for depth in range(1, max_depth + 1):
            step_input = sensory if (clamp_sensory or depth == 1) else zero_sensory
            new_state = self.step(state, step_input)
            denom = max(float(np.linalg.norm(state)), 1e-6)
            deltas.append(float(np.linalg.norm(new_state - state) / denom))
            norms.append(float(np.linalg.norm(new_state)))
            state = new_state

            if depth < min_depth:
                continue
            logits = state[readout_indices]
            order = np.argsort(logits)
            winner = int(order[-1])
            top = float(logits[order[-1]])
            second = float(logits[order[-2]])
            decision_margin = top - second
            if top >= min_activity and decision_margin >= margin:
                if winner == last_winner:
                    stable += 1
                else:
                    stable = 1
                    last_winner = winner
                if stable >= stable_steps:
                    return RunResult(state, depth, deltas, norms)
            else:
                stable = 0
                last_winner = -1

        return RunResult(state, max_depth, deltas, norms)
