from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.lif import LIFRecurrentDepthEngine
from malecns_rd.synthetic import make_multihop_routing_benchmark

DEPTHS = (1, 2, 4, 8, 16, 32, 40)


def evaluate(depth: int):
    bench = make_multihop_routing_benchmark()
    engine = LIFRecurrentDepthEngine(bench.graph, threshold=0.5)
    rows = []
    all_correct = []

    for source, target, path_length in zip(
        bench.sources, bench.targets, bench.path_lengths
    ):
        sensory = np.zeros(bench.graph.n_neurons, dtype=np.float32)
        sensory[source] = 1.0
        result = engine.run(sensory, max_depth=depth, clamp_sensory=True)
        prediction = int(
            bench.targets[np.argmax(result.spike_counts[bench.targets])]
        )
        correct = prediction == int(target)
        all_correct.append(correct)
        rows.append((int(path_length), correct))

    by_path = {}
    for length in sorted(set(bench.path_lengths.tolist())):
        vals = [ok for path, ok in rows if path == length]
        by_path[int(length)] = float(np.mean(vals))
    return float(np.mean(all_correct)), by_path


def main() -> None:
    out = ROOT / "results" / "lif_depth_scaling.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["depth", "accuracy", "path2", "path4", "path8", "path16", "path32"]
        )
        for depth in DEPTHS:
            accuracy, by_path = evaluate(depth)
            writer.writerow(
                [depth, f"{accuracy:.6f}"]
                + [f"{by_path[n]:.6f}" for n in (2, 4, 8, 16, 32)]
            )
            print(depth, accuracy, by_path)


if __name__ == "__main__":
    main()
