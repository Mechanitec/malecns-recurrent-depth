"""Assemble the mechanistic-discovery outputs into a concise evidence report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _number(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/mechanistic_discovery_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanistic_discovery_v1"))
    args = parser.parse_args()
    root = args.root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary_json = _json(root / "gain_factorial" / "summary_by_condition.json")
    factorial_summary = root / "gain_factorial" / "summary_by_condition.csv"
    factorial = pd.read_csv(factorial_summary) if factorial_summary.exists() else pd.DataFrame()
    stats = _json(root / "statistics" / "selection_aware_summary.json")
    graph = _json(root / "graph_control_audit" / "audit.json")
    stability = _json(root / "teacher_stability" / "stability.csv.metadata.json")
    probe = _json(root / "probe_transfer" / "metadata.json")
    position = _json(root / "position_discovery" / "metadata.json")
    adaptive = _json(root / "adaptive_stopping" / "metadata.json")
    control = _json(root / "control_ensemble" / "metadata.json")

    facts = [
        ("Old-study clustered uncertainty", f"D1 to D2 paired improvement CI: {stats.get('paired_improvement_cluster_bootstrap_ci', 'n/a')} cp", "Block 1"),
        ("Graph control audit", f"Corrected shuffle preserves degree sequences: {graph.get('corrected', {}).get('degree_sequences_exact', 'n/a')}", "Block 2"),
        ("Teacher stability", f"Best-move agreement at 1k/10k/50k nodes: {stability.get('agreement_with_max_nodes', 'n/a')}", "Block 4"),
        ("Gain/depth response", "See the complete gain x input-mode x depth table below.", "Block 5"),
        ("Probe transfer", f"Status: {probe.get('status', 'not run')}; standardized train-only features and cross-depth transfer are recorded.", "Block 6"),
        ("Position discovery", f"Status: {position.get('status', 'not run')}; exploratory descriptors and ranked positions are recorded.", "Block 7"),
        ("Control ensemble", f"Status: {control.get('status', 'not run')}; corrected topology and sign-shuffle members are recorded.", "Block 8"),
        ("Adaptive stopping", f"Status: {adaptive.get('status', 'not run')}; selection uses internal signals only.", "Block 9"),
    ]
    lines = [
        "# Mechanistic discovery report",
        "",
        "This report summarizes the executed mechanistic-discovery plan. The original research goal and v1 artifacts remain unchanged.",
        "",
        "## New facts learned today",
        "",
        "| Finding | Evidence | Block |",
        "|---|---|---|",
    ]
    lines.extend(f"| {finding} | {evidence} | {block} |" for finding, evidence, block in facts)
    lines.extend(["", "## Gain and depth response", ""])
    if factorial.empty:
        lines.append("The factorial summary is not available yet.")
    else:
        lines.extend([
            "| Gain | Input mode | Depth | Mean regret (cp) | Teacher agreement | Mean latency (s) |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for _, row in factorial.sort_values(["gain_scale", "input_mode", "depth"]).iterrows():
            lines.append(
                f"| {row['gain_scale']} | {row['input_mode']} | {int(row['depth'])} | "
                f"{_number(row['regret_cp'])} | {_number(row['teacher_best_agreement'])} | {_number(row.get('latency_s'))} |"
            )
    lines.extend([
        "",
        "## Methods and interpretation",
        "",
        "- Block 0 records hashes, legal-candidate coverage, termination metadata, and the corrected bounded-artifact interpretation.",
        "- Block 1 uses ordinary and source-cluster bootstrap intervals. Clustered intervals are the primary uncertainty estimate.",
        "- Block 2 audits exact directed degree preservation and edge-weight preservation. The corrected v2 shuffle is the control used downstream.",
        "- Block 3 records hidden-state, drive, saturation, readout, candidate-geometry, ranking, switching, and latency observables without changing scores.",
        "- Blocks 4 and 5 use the frozen, non-overlapping development corpus and split source groups before inspecting outcomes.",
        "- Blocks 6 through 9 are interpreted as representation, position, control-ensemble, and stopping evidence. They do not justify changing the research goal.",
        "",
        "## Artifact index",
        "",
        "- `statistics/` — clustered uncertainty and selection-aware summaries.",
        "- `graph_control_audit/` — original versus corrected control audit.",
        "- `gain_factorial/` — raw resumable workers and merged gain/depth summaries.",
        "- `probe_transfer/` — depth-specific probes and baseline comparisons.",
        "- `position_discovery/` — exploratory descriptors, correlations, and ranked positions.",
        "- `control_ensemble/` — corrected topology/sign controls and original percentiles.",
        "- `adaptive_stopping/` — screen tuning and held-out confirmation.",
        "- `teacher_stability/` — fixed-node teacher agreement.",
    ])
    (output / "discovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "complete" if factorial_summary.exists() and control.get("status") == "complete" else "in_progress",
        "blocks": {"1": bool(stats), "2": bool(graph), "4": bool(stability), "5": bool(summary_json or not factorial.empty), "6": bool(probe), "7": bool(position), "8": bool(control), "9": bool(adaptive)},
        "selected_regimes": summary_json.get("selected_regimes", []),
        "report": "results/mechanistic_discovery_v1/discovery_report.md",
    }
    (output / "discovery_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
