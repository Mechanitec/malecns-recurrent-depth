from __future__ import annotations

import hashlib
import numpy as np

# 12 piece planes x 64 squares + side-to-move + 4 castling rights + 8 ep files
# + candidate from-square + to-square + 5 promotion classes + 3 move flags.
CHESS_FEATURE_DIM = 12 * 64 + 1 + 4 + 8 + 64 + 64 + 5 + 3


def _require_chess():
    try:
        import chess  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise RuntimeError(
            "Chess support requires the optional dependency: pip install -e '.[chess]'"
        ) from exc
    return chess


def encode_board_move(board, move) -> np.ndarray:
    """Encode a board plus one candidate legal move as a fixed float vector.

    This is deliberately chess-generic rather than MaleCNS-specific. A separate
    projector maps this vector onto whichever sensory neurons are chosen for an
    experiment, allowing the chess interface to remain frozen across connectome
    and recurrent-depth comparisons.
    """
    chess = _require_chess()
    if move not in board.legal_moves:
        raise ValueError("candidate move must be legal in the supplied board")

    x = np.zeros(CHESS_FEATURE_DIM, dtype=np.float32)
    offset = 0

    # Piece planes: white P,N,B,R,Q,K then black P,N,B,R,Q,K.
    for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            plane = color_index * 6 + (piece_type - 1)
            for square in board.pieces(piece_type, color):
                x[offset + plane * 64 + square] = 1.0
    offset += 12 * 64

    x[offset] = 1.0 if board.turn == chess.WHITE else -1.0
    offset += 1

    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for i, value in enumerate(castling):
        x[offset + i] = float(value)
    offset += 4

    if board.ep_square is not None:
        x[offset + chess.square_file(board.ep_square)] = 1.0
    offset += 8

    x[offset + move.from_square] = 1.0
    offset += 64
    x[offset + move.to_square] = 1.0
    offset += 64

    # 0 = no promotion, then knight/bishop/rook/queen.
    promo_index = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }.get(move.promotion)
    if promo_index is None:
        raise ValueError("unsupported promotion piece")
    x[offset + promo_index] = 1.0
    offset += 5

    x[offset] = float(board.is_capture(move))
    x[offset + 1] = float(board.is_castling(move))
    x[offset + 2] = float(board.is_en_passant(move))
    return x


class HashedSensoryProjector:
    """Deterministic sparse feature -> sensory-neuron projection.

    Each active chess feature is projected to ``fanout`` sensory neurons with a
    deterministic +/- sign. The mapping is seed-controlled and must be frozen
    when comparing recurrent-depth variants.
    """

    def __init__(
        self,
        n_neurons: int,
        sensory_indices: np.ndarray,
        *,
        fanout: int = 4,
        seed: int = 0,
        amplitude: float = 1.0,
    ) -> None:
        self.n_neurons = int(n_neurons)
        self.sensory_indices = np.asarray(sensory_indices, dtype=np.int64)
        if self.sensory_indices.ndim != 1 or self.sensory_indices.size == 0:
            raise ValueError("sensory_indices must be a non-empty vector")
        if np.any(self.sensory_indices < 0) or np.any(self.sensory_indices >= self.n_neurons):
            raise ValueError("sensory index outside graph")
        if fanout < 1:
            raise ValueError("fanout must be >= 1")
        self.fanout = int(fanout)
        self.seed = int(seed)
        self.amplitude = float(amplitude)

    def _slot(self, feature_index: int, branch: int) -> tuple[int, float]:
        payload = f"{self.seed}:{feature_index}:{branch}".encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        value = int.from_bytes(digest[:8], "little")
        sign_bit = digest[8] & 1
        slot = value % self.sensory_indices.size
        sign = -1.0 if sign_bit else 1.0
        return int(self.sensory_indices[slot]), sign

    def project(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (CHESS_FEATURE_DIM,):
            raise ValueError(f"features must have shape {(CHESS_FEATURE_DIM,)}")
        sensory = np.zeros(self.n_neurons, dtype=np.float32)
        for feature_index in np.flatnonzero(features):
            value = float(features[feature_index]) * self.amplitude / self.fanout
            for branch in range(self.fanout):
                neuron, sign = self._slot(int(feature_index), branch)
                sensory[neuron] += value * sign
        return sensory
