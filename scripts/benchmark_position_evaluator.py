"""Compare repeated scalar depth evaluation with one batched trajectory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_agent import FlyCandidateMoveAgent
from malecns_rd.engine import RecurrentDepthEngine

from run_position_depth_study import DEPTHS, load_corpus


def scalar_agent(base, depth: int) -> FlyCandidateMoveAgent:
    if not isinstance(base.engine, RecurrentDepthEngine):
        raise RuntimeError("benchmark requires a rate checkpoint")
    engine = RecurrentDepthEngine(
        base.engine.graph,
        leak=base.engine.leak,
        recurrent_gain=base.engine.recurrent_gain,
        input_gain=base.engine.input_gain,
        activation=base.engine.activation,
    )
    return FlyCandidateMoveAgent(
        engine=engine,
        projector=base.projector,
        readout_indices=base.readout_indices,
        readout_weights=base.readout_weights,
        depth=depth,
        clamp_sensory=base.clamp_sensory,
        max_candidates=None,
        name=base.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--connectome-weights", type=Path, required=True)
    parser.add_argument("--sensory-indices", type=Path, required=True)
    parser.add_argument("--evaluation-corpus", type=Path, default=Path("data/chess_evaluation_corpus_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/position_depth_study_v2/evaluator_benchmark.json"))
    args = parser.parse_args()
    item = load_corpus(args.evaluation_corpus)[0]
    loaded = load_fly_agent_from_checkpoint(
        args.checkpoint,
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectome_weights_path=args.connectome_weights,
        sensory_indices_path=args.sensory_indices,
        min_synapses=3,
    )
    base = loaded.agent
    scalar_times = {}
    started = time.perf_counter()
    for depth in DEPTHS:
        agent = scalar_agent(base, depth)
        depth_started = time.perf_counter()
        agent.rank_moves(item["board"])
        scalar_times[depth] = time.perf_counter() - depth_started
    scalar_total = time.perf_counter() - started
    optimized_agent = scalar_agent(base, 64)
    optimized_started = time.perf_counter()
    optimized_agent.rank_moves_by_depth(item["board"], DEPTHS)
    optimized_total = time.perf_counter() - optimized_started
    payload = {
        "position_id": item["position_id"],
        "candidate_count": len(item["candidates"]),
        "depths": list(DEPTHS),
        "scalar_times_s": scalar_times,
        "scalar_total_s": scalar_total,
        "optimized_trajectory_total_s": optimized_total,
        "speedup_scalar_over_optimized": scalar_total / optimized_total if optimized_total else None,
        "note": "Scalar baseline recomputes each requested depth independently. Optimized path runs one batched trajectory to depth 64 and snapshots requested depths.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
