"""MaleCNS chess benchmark with fast low-Elo opponents and live analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

from malecns_rd.checkpoint_agent import load_fly_agent_from_checkpoint
from malecns_rd.chess_agent import RandomLegalAgent
from malecns_rd.chess_benchmark import run_elo_tournament, save_tournament
from malecns_rd.fast_opponents import (
    run_fast_elo_tournament,
    run_fast_elo_tournament_with_analysis,
)
from malecns_rd.position_analysis import AnalysisConfig, run_elo_tournament_with_analysis


def parse_elos(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one Elo")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish executable")
    parser.add_argument(
        "--minic",
        help="Path to Minic executable; used for the fast Elo 0 random-mover anchor",
    )
    parser.add_argument(
        "--gaia",
        help="Path to Gaia 4 executable; used for fast 580-1300 low-rating anchors",
    )
    parser.add_argument(
        "--alfil",
        help="Legacy Alfil executable; used only with --legacy-alfil",
    )
    parser.add_argument(
        "--legacy-alfil",
        action="store_true",
        help="Use the old Alfil-below-Stockfish ladder instead of the fast Minic/Gaia ladder",
    )
    parser.add_argument(
        "--elos",
        type=parse_elos,
        default=parse_elos("0,580,700,820,940,1060,1180,1300,1320,1400,1500"),
        help=(
            "Opponent nominal Elo ladder. Fast mode supports 0, Gaia anchors "
            "580/700/820/940/1060/1180/1300, and Stockfish ratings at/above its floor."
        ),
    )
    parser.add_argument(
        "--stockfish-floor",
        type=int,
        default=None,
        help="Override Stockfish minimum Elo; by default it is detected from UCI_Elo",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Measured low-Elo calibration CSV used for sub-Stockfish routing",
    )
    parser.add_argument("--games-per-elo", type=int, default=20)
    parser.add_argument("--move-time", type=float, default=0.05)
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("results/chess_smoke"))
    parser.add_argument(
        "--live-state",
        type=Path,
        default=None,
        help="Live JSON path; defaults to <output>/live_state.json",
    )

    fly = parser.add_argument_group("trained fly checkpoint")
    fly.add_argument("--checkpoint", type=Path, default=None, help="Trained readout_checkpoint.npz")
    fly.add_argument("--annotations", type=Path, default=None)
    fly.add_argument("--neurotransmitters", type=Path, default=None)
    fly.add_argument("--connectome-weights", type=Path, default=None)
    fly.add_argument("--sensory-indices", type=Path, default=None, help=".npy graph-index vector used during training")
    fly.add_argument("--min-synapses", type=int, default=3)
    fly.add_argument(
        "--fly-depth",
        type=int,
        default=None,
        help="Override checkpoint recurrent depth for inference-depth experiments",
    )
    fly.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Bounded diagnostic only: score at most this many tactical legal moves; unset scores all legal moves",
    )

    analysis = parser.add_argument_group("independent Stockfish analysis")
    analysis.add_argument(
        "--analysis-stockfish",
        default=None,
        help="Full-strength Stockfish binary used only for position evaluation; defaults to --stockfish",
    )
    analysis.add_argument(
        "--analysis-depth",
        type=int,
        default=18,
        help="Fixed Stockfish analysis depth after every move (default: 18)",
    )
    analysis.add_argument("--analysis-threads", type=int, default=1)
    analysis.add_argument("--analysis-hash-mb", type=int, default=128)
    analysis.add_argument(
        "--no-position-eval",
        action="store_true",
        help="Disable the independent Stockfish position observer",
    )
    parser.add_argument("--seed", type=int, default=7, help="RandomLegal fallback seed")
    args = parser.parse_args()

    live_state = args.live_state or (args.output / "live_state.json")

    if args.checkpoint is None:
        agent = RandomLegalAgent(seed=args.seed)
        print("Agent: RandomLegal smoke-test fallback")
    else:
        required = {
            "--annotations": args.annotations,
            "--neurotransmitters": args.neurotransmitters,
            "--connectome-weights": args.connectome_weights,
            "--sensory-indices": args.sensory_indices,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--checkpoint requires " + ", ".join(missing))
        loaded = load_fly_agent_from_checkpoint(
            args.checkpoint,
            annotations_path=args.annotations,
            neurotransmitters_path=args.neurotransmitters,
            connectome_weights_path=args.connectome_weights,
            sensory_indices_path=args.sensory_indices,
            min_synapses=args.min_synapses,
            depth_override=args.fly_depth,
            max_candidates=args.max_candidates,
        )
        agent = loaded.agent
        print(
            f"Agent: MaleCNS-RD checkpoint={args.checkpoint} "
            f"dynamics={loaded.dynamics} depth={agent.depth} neurons={loaded.graph_neurons}"
        )

    if args.legacy_alfil:
        if args.no_position_eval:
            result = run_elo_tournament(
                agent,
                stockfish_executable=args.stockfish,
                alfil_executable=args.alfil,
                stockfish_floor=args.stockfish_floor,
                opponent_elos=args.elos,
                games_per_elo=args.games_per_elo,
                move_time_s=args.move_time,
                max_plies=args.max_plies,
                live_state_path=live_state,
                run_id=args.output.name,
            )
        else:
            analysis_config = AnalysisConfig(
                executable=args.analysis_stockfish or args.stockfish,
                depth=args.analysis_depth,
                threads=args.analysis_threads,
                hash_mb=args.analysis_hash_mb,
            )
            result = run_elo_tournament_with_analysis(
                agent,
                stockfish_executable=args.stockfish,
                alfil_executable=args.alfil,
                stockfish_floor=args.stockfish_floor,
                opponent_elos=args.elos,
                games_per_elo=args.games_per_elo,
                move_time_s=args.move_time,
                max_plies=args.max_plies,
                live_state_path=live_state,
                evaluation_state_path=args.output / "position_eval.json",
                evaluation_history_path=args.output / "evaluation_history.csv",
                analysis_config=analysis_config,
                run_id=args.output.name,
            )
    else:
        if args.no_position_eval:
            result = run_fast_elo_tournament(
                agent,
                stockfish_executable=args.stockfish,
                minic_executable=args.minic,
                gaia_executable=args.gaia,
                stockfish_floor=args.stockfish_floor,
                opponent_elos=args.elos,
                games_per_elo=args.games_per_elo,
                move_time_s=args.move_time,
                max_plies=args.max_plies,
                live_state_path=live_state,
                run_id=args.output.name,
                calibration_path=args.calibration,
            )
        else:
            analysis_config = AnalysisConfig(
                executable=args.analysis_stockfish or args.stockfish,
                depth=args.analysis_depth,
                threads=args.analysis_threads,
                hash_mb=args.analysis_hash_mb,
            )
            result = run_fast_elo_tournament_with_analysis(
                agent,
                stockfish_executable=args.stockfish,
                minic_executable=args.minic,
                gaia_executable=args.gaia,
                stockfish_floor=args.stockfish_floor,
                opponent_elos=args.elos,
                games_per_elo=args.games_per_elo,
                move_time_s=args.move_time,
                max_plies=args.max_plies,
                live_state_path=live_state,
                evaluation_state_path=args.output / "position_eval.json",
                evaluation_history_path=args.output / "evaluation_history.csv",
                analysis_config=analysis_config,
                run_id=args.output.name,
                calibration_path=args.calibration,
            )

    save_tournament(result, args.output)
    print(f"Stockfish floor: {result.stockfish_floor}")
    print(f"Live state: {live_state}")
    if not args.no_position_eval:
        print(f"Position eval: {args.output / 'position_eval.json'}")
        print(f"Eval history: {args.output / 'evaluation_history.csv'}")
    print(result.elo)


if __name__ == "__main__":
    main()
