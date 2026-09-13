from __future__ import annotations

import csv
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.synthetic import make_multihop_routing_benchmark


def evaluate(depth: int, bench, engine) -> tuple[float, dict[int, float]]:
    target_set = bench.targets
    correct = 0
    by_length: dict[int, list[bool]] = {}

    for src, target, length in zip(bench.sources, bench.targets, bench.path_lengths):
        sensory = np.zeros(bench.graph.n_neurons, dtype=np.float32)
        sensory[src] = 1.0
        result = engine.run(sensory, max_depth=depth, clamp_sensory=True)
        pred = int(target_set[np.argmax(result.state[target_set])])
        ok = pred == int(target)
        correct += int(ok)
        by_length.setdefault(int(length), []).append(ok)

    per_len = {k: float(np.mean(v)) for k, v in sorted(by_length.items())}
    return correct / len(bench.sources), per_len


def main() -> None:
    bench = make_multihop_routing_benchmark(noise_edges_per_neuron=0.05)
    engine = RecurrentDepthEngine(
        bench.graph,
        leak=0.15,
        recurrent_gain=1.15,
        input_gain=1.5,
        activation="relu",
    )

    depths = [1, 2, 4, 8, 16, 32, 40]
    rows = []
    print(f"graph: {bench.graph.n_neurons:,} neurons, {bench.graph.n_edges:,} edges")
    print("depth | accuracy | by required path length")
    print("------|----------|------------------------")
    for depth in depths:
        acc, per_len = evaluate(depth, bench, engine)
        print(f"{depth:>5} | {acc:>8.1%} | " + ", ".join(f"L{k}:{v:.0%}" for k, v in per_len.items()))
        row = {"depth": depth, "accuracy": acc}
        row.update({f"length_{k}": v for k, v in per_len.items()})
        rows.append(row)

    out = ROOT / "results" / "depth_scaling.csv"
    out.parent.mkdir(exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
