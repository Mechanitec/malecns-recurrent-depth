from pathlib import Path
import numpy as np
import pandas as pd

from malecns_rd.readout_training import (
    build_best_vs_rest_pairs,
    load_activation_cache,
    load_readout_checkpoint,
    pair_accuracy,
    save_activation_cache,
    save_readout_checkpoint,
    stable_position_split,
    top1_accuracy,
    train_pairwise_readout,
)


def synthetic_cache(seed=3):
    rng = np.random.default_rng(seed)
    true_w = np.array([1.2, -0.8, 0.5, 0.3], dtype=np.float32)
    rows = []
    feats = []
    for p in range(30):
        for m in range(4):
            x = rng.normal(size=4).astype(np.float32)
            target = float(x @ true_w)
            rows.append({"position_id": f"p{p}", "move_uci": f"m{m}", "teacher_target": target})
            feats.append(x)
    return np.vstack(feats), pd.DataFrame(rows), true_w


def test_split_stable():
    ids = [f"p{i}" for i in range(20)]
    a = stable_position_split(ids, 0.25, seed=9)
    b = stable_position_split(ids, 0.25, seed=9)
    np.testing.assert_array_equal(a, b)


def test_pairs_and_metrics():
    x, frame, true_w = synthetic_cache()
    pairs, _ = build_best_vs_rest_pairs(x, frame)
    assert len(pairs) == 30 * 3
    assert pair_accuracy(pairs, true_w) == 1.0
    assert top1_accuracy(x, frame, true_w) == 1.0


def test_pairwise_training_learns_ranking():
    x, frame, _ = synthetic_cache()
    result = train_pairwise_readout(
        x,
        frame,
        validation_fraction=0.25,
        epochs=80,
        learning_rate=0.04,
        l2=1e-4,
        batch_size=32,
        seed=11,
    )
    assert result.best_epoch >= 1
    assert result.best_weights.shape == result.weights.shape
    assert result.train_pair_accuracy > 0.95
    assert result.validation_pair_accuracy > 0.90
    assert result.validation_top1_accuracy > 0.80
    assert result.history.iloc[-1]["loss"] < result.history.iloc[0]["loss"]


def test_cache_roundtrip(tmp_path: Path):
    x, frame, _ = synthetic_cache()
    path = tmp_path / "cache.npz"
    indices = np.array([10, 11, 12, 13], dtype=np.int64)
    save_activation_cache(path, x, frame, readout_indices=indices, metadata={"recurrent_depth": 16})
    cache = load_activation_cache(path)
    np.testing.assert_allclose(cache.features, x)
    assert cache.frame["move_uci"].tolist() == frame["move_uci"].tolist()
    np.testing.assert_array_equal(cache.readout_indices, indices)
    assert cache.metadata["recurrent_depth"] == 16


def test_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "checkpoint.npz"
    save_readout_checkpoint(
        path,
        readout_weights=np.array([1.0, -2.0], dtype=np.float32),
        readout_indices=np.array([10, 20], dtype=np.int64),
        recurrent_depth=32,
        projector_seed=7,
        metadata={"validation_top1": 0.61},
    )
    ckpt = load_readout_checkpoint(path)
    np.testing.assert_allclose(ckpt.readout_weights, [1.0, -2.0])
    np.testing.assert_array_equal(ckpt.readout_indices, [10, 20])
    assert ckpt.recurrent_depth == 32
    assert ckpt.projector_seed == 7
    assert ckpt.metadata["validation_top1"] == 0.61
