from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malecns_rd.chess_features import HashedSensoryProjector, encode_board_move
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.lif import LIFRecurrentDepthEngine
from malecns_rd.readout_training import ActivationCache, save_activation_cache


def _require_chess():
    try:
        import chess  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Activation extraction requires: pip install -e '.[chess]'") from exc
    return chess


def extract_candidate_activations(
    teacher_csv: str | Path,
    *,
    engine: RecurrentDepthEngine | LIFRecurrentDepthEngine,
    projector: HashedSensoryProjector,
    readout_indices: np.ndarray,
    recurrent_depth: int,
    clamp_sensory: bool = True,
    max_rows: int | None = None,
) -> ActivationCache:
    """Run frozen connectome dynamics once per teacher-labelled candidate move."""
    if recurrent_depth < 1:
        raise ValueError("recurrent_depth must be >= 1")
    chess = _require_chess()
    frame = pd.read_csv(teacher_csv)
    required = {"position_id", "fen", "move_uci", "teacher_target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher dataset missing columns: {sorted(missing)}")
    if max_rows is not None:
        frame = frame.iloc[: int(max_rows)].copy()
    indices = np.asarray(readout_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("readout_indices must be a non-empty vector")
    if np.any(indices < 0) or np.any(indices >= engine.graph.n_neurons):
        raise ValueError("readout index outside graph")

    rows: list[np.ndarray] = []
    kept: list[int] = []
    for row_idx, row in frame.iterrows():
        board = chess.Board(str(row["fen"]))
        move = chess.Move.from_uci(str(row["move_uci"]))
        if move not in board.legal_moves:
            continue
        sensory = projector.project(encode_board_move(board, move))
        if isinstance(engine, RecurrentDepthEngine):
            result = engine.run(sensory, max_depth=recurrent_depth, clamp_sensory=clamp_sensory)
            values = result.state[indices]
        else:
            result = engine.run(sensory, max_depth=recurrent_depth, clamp_sensory=clamp_sensory)
            values = result.spike_counts[indices].astype(np.float32)
            values = values + 1e-3 * result.membrane[indices]
        rows.append(np.asarray(values, dtype=np.float32))
        kept.append(int(row_idx))
    if not rows:
        raise ValueError("no legal candidate rows were extracted")
    out_frame = frame.loc[kept].reset_index(drop=True)
    return ActivationCache(
        features=np.vstack(rows).astype(np.float32),
        frame=out_frame,
        readout_indices=indices.copy(),
        metadata={
            "recurrent_depth": int(recurrent_depth),
            "projector_seed": int(projector.seed),
            "projector_fanout": int(projector.fanout),
            "projector_amplitude": float(projector.amplitude),
            "clamp_sensory": bool(clamp_sensory),
            "dynamics": type(engine).__name__,
        },
    )


def extract_and_save_candidate_activations(
    output_path: str | Path,
    teacher_csv: str | Path,
    **kwargs,
) -> ActivationCache:
    cache = extract_candidate_activations(teacher_csv, **kwargs)
    save_activation_cache(
        output_path,
        cache.features,
        cache.frame,
        readout_indices=cache.readout_indices,
        metadata=cache.metadata,
    )
    return cache
