"""Build compact comparison tables and plots from completed population studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FROZEN_RULE = {
    "input_population_id": "input_mb_kenyon_cells",
    "internal_readout_population_id": "readout_mbon_smp_cre_sip_combined",
    "decision_readout_population_id": "readout_central_computation_panel",
    "confirmation_readout_population_id": "readout_central_computation_panel",
    "confirmation_depth": 8,
    "seed": 7001,
}


def _freeze_population_v2(root: Path, manifest_dir: Path) -> None:
    manifests = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in manifest_dir.glob("*.json")
    }
    required = (
        FROZEN_RULE["input_population_id"],
        FROZEN_RULE["internal_readout_population_id"],
        FROZEN_RULE["decision_readout_population_id"],
    )
    missing = [name for name in required if name not in manifests]
    if missing:
        raise FileNotFoundError(f"frozen Population-v2 manifests are missing: {missing}")
    freeze_metadata = {
        "freeze_status": "frozen_before_fresh_confirmation",
        "selection_rule": "predeclared anatomical families with deterministic manifest selection and fixed seed; no fresh-corpus tuning",
        "source_manifest_directory": str(manifest_dir),
        "rule": FROZEN_RULE,
    }
    output_names = {
        "input_manifest.json": FROZEN_RULE["input_population_id"],
        "internal_readout_manifest.json": FROZEN_RULE["internal_readout_population_id"],
        "decision_readout_manifest.json": FROZEN_RULE["decision_readout_population_id"],
    }
    manifest_dir.parent.mkdir(parents=True, exist_ok=True)
    for filename, source_name in output_names.items():
        payload = dict(manifests[source_name])
        payload["frozen_population_v2"] = freeze_metadata
        (manifest_dir.parent / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _plot_region_flow(root: Path) -> None:
    path = root / "depth_probe_transfer.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    subset = frame[frame["train_depth"] == 1].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, group in subset.groupby("population_id"):
        group = group.sort_values("test_depth")
        ax.plot(group["test_depth"], group["validation_mean_teacher_regret_cp"], marker="o", label=name.removeprefix("readout_"))
    ax.set(xlabel="Test recurrent depth", ylabel="Validation teacher regret (cp)", title="Population information flow: D1-trained probes")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "region_information_flow.png", dpi=160)
    plt.close(fig)


def _plot_interface(root: Path, matrix: pd.DataFrame) -> None:
    if matrix.empty:
        return
    depth = matrix[matrix["depth"].isin([1, 8])].copy()
    if depth.empty:
        return
    pivot = depth.pivot_table(index="input_population_id", columns=["readout_population_id", "depth"], values="validation_mean_teacher_regret_cp", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set(xticks=range(len(pivot.columns)), xticklabels=[f"{readout.removeprefix('readout_')} D{depth}" for readout, depth in pivot.columns], yticks=range(len(pivot.index)), yticklabels=[name.removeprefix("input_") for name in pivot.index], title="Reduced input × readout matrix: validation regret")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    fig.colorbar(image, ax=ax, label="Teacher regret (cp)")
    fig.tight_layout()
    fig.savefig(root / "interface_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/population_study"))
    parser.add_argument("--manifests", type=Path, default=Path("data/populations_v2/manifests"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input_population_results.csv"
    matrix = pd.DataFrame()
    if input_path.exists():
        matrix = pd.read_csv(input_path)
    hbio_path = root / "h_bio_1_results.csv"
    hbio = pd.DataFrame()
    if hbio_path.exists():
        hbio = pd.read_csv(hbio_path)
        hbio = hbio.rename(columns={"architecture": "input_population_id", "readout_population_id": "readout_population_id"})
        hbio["input_population_id"] = "h_bio_1"
        hbio.to_csv(root / "h_bio_1_results.csv", index=False)
    interface = matrix.copy()
    if not hbio.empty:
        interface = pd.concat([interface, hbio], ignore_index=True, sort=False)
    if not interface.empty:
        interface.to_csv(root / "interface_matrix.csv", index=False)
    _plot_region_flow(root)
    _plot_interface(root, interface)
    if not matrix.empty:
        baseline = matrix[matrix["input_population_id"] == "input_baseline_a"].copy()
        bypass_path = root / "bypass_baseline_results.csv"
        if bypass_path.exists():
            bypass = pd.read_csv(bypass_path)
            bypass["comparison_family"] = "bypass"
            baseline["comparison_family"] = "MaleCNS"
            baseline = pd.concat([baseline, bypass], ignore_index=True, sort=False)
        if not baseline.empty:
            baseline.to_csv(root / "baseline_comparison.csv", index=False)
        _freeze_population_v2(root, args.manifests)
        selection = {
            "status": "development_rule_frozen", "selection_rule": FROZEN_RULE,
            "selected_input_population_id": FROZEN_RULE["input_population_id"],
            "selected_readout_population_id": FROZEN_RULE["confirmation_readout_population_id"],
            "selected_depth": FROZEN_RULE["confirmation_depth"],
            "source": "predeclared biological rule; development artifacts are descriptive only",
            "fresh_confirmation_required": True,
        }
        (root / "final_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "interface_matrix": (root / "interface_matrix.csv").exists(), "region_plot": (root / "region_information_flow.png").exists(), "final_selection": (root / "final_selection.json").exists()}, indent=2))


if __name__ == "__main__":
    main()
