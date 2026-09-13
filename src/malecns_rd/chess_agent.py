from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import numpy as np

from .chess_features import HashedSensoryProjector, encode_board_move
from .engine import RecurrentDepthEngine
from .lif import LIFRecurrentDepthEngine


class ChessAgent(Protocol):
    name: str

    def choose_move(self, board): ...


@dataclass
class FlyCandidateMoveAgent:
    """Chess adapter that asks the same connectome to score every legal move.

    The chess adapter is deliberately small and explicit. The board and one
    candidate move are encoded, projected into a frozen sensory population, and
    propagated through the same connectome for ``depth`` recurrent passes. A
    fixed readout population produces one scalar candidate score. The legal move
    with the highest score is played.

    For scientific depth comparisons, projector seed, sensory/readout neurons,
    and readout weights MUST remain fixed while only depth/dynamics are varied.
    """

    engine: RecurrentDepthEngine | LIFRecurrentDepthEngine
    projector: HashedSensoryProjector
    readout_indices: np.ndarray
    readout_weights: np.ndarray
    depth: int = 16
    clamp_sensory: bool = True
    name: str = "MaleCNS-RD"
    last_decision: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.readout_indices = np.asarray(self.readout_indices, dtype=np.int64)
        self.readout_weights = np.asarray(self.readout_weights, dtype=np.float32)
        if self.readout_indices.ndim != 1 or self.readout_indices.size == 0:
            raise ValueError("readout_indices must be a non-empty vector")
        if self.readout_weights.shape != self.readout_indices.shape:
            raise ValueError("readout_weights must match readout_indices")
        n = self.engine.graph.n_neurons
        if np.any(self.readout_indices < 0) or np.any(self.readout_indices >= n):
            raise ValueError("readout index outside graph")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")

    def score_move(self, board, move) -> float:
        features = encode_board_move(board, move)
        sensory = self.projector.project(features)
        if isinstance(self.engine, RecurrentDepthEngine):
            state = self.engine.run(
                sensory, max_depth=self.depth, clamp_sensory=self.clamp_sensory
            ).state
            values = state[self.readout_indices]
        else:
            result = self.engine.run(
                sensory, max_depth=self.depth, clamp_sensory=self.clamp_sensory
            )
            # Spike counts preserve information accumulated across depth. A small
            # membrane term breaks ties without changing the dominant spike score.
            values = result.spike_counts[self.readout_indices].astype(np.float32)
            values += 1e-3 * result.membrane[self.readout_indices]
        return float(np.dot(values, self.readout_weights))

    def rank_moves(self, board) -> list[tuple[float, str, object]]:
        """Score and sort all legal candidates, strongest first."""
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("cannot choose a move in a terminal position")
        scored = [(self.score_move(board, move), move.uci(), move) for move in legal]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored

    def choose_move(self, board):
        scored = self.rank_moves(board)
        self.last_decision = {
            "depth": int(self.depth),
            "selected_move": scored[0][1],
            "selected_score": float(scored[0][0]),
            "candidates": [
                {"move": uci, "score": float(score)}
                for score, uci, _ in scored
            ],
        }
        return scored[0][2]


@dataclass
class RandomLegalAgent:
    seed: int = 0
    name: str = "RandomLegal"
    last_decision: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def choose_move(self, board):
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("cannot choose a move in a terminal position")
        move = legal[int(self._rng.integers(0, len(legal)))]
        self.last_decision = {
            "depth": None,
            "selected_move": move.uci(),
            "selected_score": None,
            "candidates": [],
        }
        return move
