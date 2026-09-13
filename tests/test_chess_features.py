import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.chess_features import CHESS_FEATURE_DIM, HashedSensoryProjector


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
