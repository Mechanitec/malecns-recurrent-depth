import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.chess_features import CHESS_FEATURE_DIM, HashedSensoryProjector, encode_board_move, encode_board_move_unchecked
from malecns_rd.chess_agent import FlyCandidateMoveAgent
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.graph import ConnectomeGraph


def test_hashed_projection_is_deterministic():
    indices = np.arange(10, 110, dtype=np.int64)
    projector = HashedSensoryProjector(200, indices, fanout=4, seed=42)
    x = np.zeros(CHESS_FEATURE_DIM, dtype=np.float32)
    x[[0, 17, 900]] = [1.0, 1.0, -1.0]
    a = projector.project(x)
    b = projector.project(x)
    np.testing.assert_array_equal(a, b)
    assert np.count_nonzero(a) > 0
    assert np.all(a[:10] == 0)
    assert np.all(a[110:] == 0)


def test_projector_rejects_wrong_feature_shape():
    projector = HashedSensoryProjector(20, np.arange(10), seed=1)
    try:
        projector.project(np.zeros(10, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_unchecked_encoder_accepts_invalid_movement_candidate():
    import chess

    board = chess.Board()
    move = chess.Move.from_uci("e2e5")
    try:
        encode_board_move(board, move)
    except ValueError:
        pass
    else:
        raise AssertionError("expected legal encoder to reject invalid move")
    assert encode_board_move_unchecked(board, move).shape == (CHESS_FEATURE_DIM,)


def test_multi_depth_move_ranking_matches_independent_rankings():
    import chess

    graph = ConnectomeGraph.from_edges(
        64,
        np.arange(63, dtype=np.int64),
        np.arange(1, 64, dtype=np.int64),
        np.full(63, 0.01, dtype=np.float32),
        normalize_incoming=False,
    )
    projector = HashedSensoryProjector(64, np.arange(8, dtype=np.int64), fanout=2, seed=4)
    agent = FlyCandidateMoveAgent(
        RecurrentDepthEngine(graph),
        projector,
        np.array([20, 21, 22], dtype=np.int64),
        np.array([0.2, -0.1, 0.05], dtype=np.float32),
        depth=4,
    )
    board = chess.Board()
    multi = agent.rank_moves_by_depth(board, depths=(1, 2, 4))
    for depth, ranked in multi.items():
        independent = FlyCandidateMoveAgent(
            RecurrentDepthEngine(graph),
            projector,
            agent.readout_indices,
            agent.readout_weights,
            depth=depth,
        ).rank_moves(board)
        assert [item[1] for item in ranked] == [item[1] for item in independent]
        np.testing.assert_allclose(
            [item[0] for item in ranked],
            [item[0] for item in independent],
            rtol=0,
            atol=1e-6,
        )
