"""Run teacher-labelled candidates through a frozen MaleCNS model and cache readout activations."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from malecns_rd.activation_extraction import extract_and_save_candidate_activations
from malecns_rd.chess_features import HashedSensoryProjector
from malecns_rd.engine import RecurrentDepthEngine
from malecns_rd.lif import LIFRecurrentDepthEngine
from malecns_rd.malecns import load_malecns_feather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True, help=".npy graph-index vector")
    parser.add_argument("--readout-indices", type=Path, required=True, help=".npy graph-index vector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamics", choices=["rate", "lif"], default="lif")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--projector-seed", type=int, default=0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--min-synapses", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--single-pulse", action="store_true", help="Do not clamp sensory evidence after depth 1")
    args = parser.parse_args()

    graph, _ = load_malecns_feather(
        args.annotations,
        args.neurotransmitters,
        args.weights,
        traced_only=True,
        min_synapses=args.min_synapses,
    )
    sensory_indices = np.load(args.sensory_indices).astype(np.int64)
    readout_indices = np.load(args.readout_indices).astype(np.int64)
    projector = HashedSensoryProjector(
        graph.n_neurons,
        sensory_indices,
        fanout=args.fanout,
        seed=args.projector_seed,
        amplitude=args.amplitude,
    )
    engine = LIFRecurrentDepthEngine(graph) if args.dynamics == "lif" else RecurrentDepthEngine(graph)
    cache = extract_and_save_candidate_activations(
        args.output,
        args.teacher_csv,
        engine=engine,
        projector=projector,
        readout_indices=readout_indices,
        recurrent_depth=args.depth,
        clamp_sensory=not args.single_pulse,
        max_rows=args.max_rows,
    )
    print(f"cached {len(cache.frame)} candidates x {cache.features.shape[1]} readout neurons")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
