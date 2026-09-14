"""Measure teacher-label stability on a predeclared development subset."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from malecns_rd.chess_teacher import StockfishTeacher


def load_positions(path: Path, count: int) -> list[dict[str, str]]:
    import chess

    grouped: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["position_id"], row)
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in grouped.values():
        by_source.setdefault(row["source_game_id"], []).append(row)
    selected: list[dict[str, str]] = []
    for source_id in sorted(by_source)[::2]:
        candidates = sorted(by_source[source_id], key=lambda row: int(row["ply"]))
        preferred = [row for row in candidates if row["phase"] == "middle"]
        selected.append((preferred or candidates)[0])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"could not select {count} source-independent positions")
    for row in selected:
        board = chess.Board(row["fen"])
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"selected terminal position {row['position_id']}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--nodes", default="1000,10000,50000")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    args = parser.parse_args()
    node_budgets = tuple(int(value) for value in args.nodes.split(","))
    if args.count < 1 or any(value < 1 for value in node_budgets):
        parser.error("count and node budgets must be positive")
    selected = load_positions(args.corpus, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    labels: dict[int, dict[str, dict[str, float | str]]] = {}
    metadata: dict[str, object] = {
        "selection_policy": "every other source group, middle phase when available",
        "selected_position_ids": [row["position_id"] for row in selected],
        "node_budgets": list(node_budgets),
        "threads": args.threads,
        "hash_mb": args.hash_mb,
    }
    for nodes in node_budgets:
        with StockfishTeacher(
            str(args.stockfish), nodes=nodes, threads=args.threads, hash_mb=args.hash_mb
        ) as teacher:
            metadata.setdefault("teacher", {})[str(nodes)] = teacher.metadata()
            labels[nodes] = {}
            for position in selected:
                import chess

                candidates = teacher.evaluate_position(
                    chess.Board(position["fen"]), position_id=position["position_id"]
                )
                if not candidates:
                    raise RuntimeError(f"no labels for {position['position_id']} at {nodes} nodes")
                best = max(candidates, key=lambda candidate: (candidate.teacher_cp, candidate.move_uci))
                labels[nodes][position["position_id"]] = {
                    "best_move": best.move_uci,
                    "best_cp": best.teacher_cp,
                    "candidate_count": len(candidates),
                }
    reference_nodes = max(node_budgets)
    for position in selected:
        position_id = position["position_id"]
        reference = labels[reference_nodes][position_id]
        for nodes in node_budgets:
            label = labels[nodes][position_id]
            rows.append({
                "position_id": position_id,
                "source_game_id": position["source_game_id"],
                "phase": position["phase"],
                "node_budget": nodes,
                "best_move": label["best_move"],
                "best_cp": label["best_cp"],
                "candidate_count": label["candidate_count"],
                "agreement_with_max_nodes": int(label["best_move"] == reference["best_move"]),
                "cp_delta_from_max_nodes": float(label["best_cp"] - reference["best_cp"]),
            })
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata.update({
        "positions": len(selected),
        "rows": len(rows),
        "max_node_budget": reference_nodes,
        "best_move_agreement_by_nodes": {
            str(nodes): sum(
                int(labels[nodes][position["position_id"]]["best_move"]
                    == labels[reference_nodes][position["position_id"]]["best_move"])
                for position in selected
            ) / len(selected)
            for nodes in node_budgets
        },
    })
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
