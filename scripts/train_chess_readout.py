"""Train the frozen-connectome chess readout from a cached activation matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from malecns_rd.readout_training import (
    load_activation_cache,
    save_readout_checkpoint,
    train_pairwise_readout,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache = load_activation_cache(args.cache)
    if cache.readout_indices is None:
        raise RuntimeError("activation cache does not contain readout_indices")
    result = train_pairwise_readout(
        cache.features,
        cache.frame,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        batch_size=args.batch_size,
        min_teacher_margin=args.min_teacher_margin,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth = int(cache.metadata.get("recurrent_depth", 1))
    projector_seed = cache.metadata.get("projector_seed")
    checkpoint_path = args.output_dir / "readout_checkpoint.npz"
    metrics = {
        "train_pair_accuracy": result.train_pair_accuracy,
        "validation_pair_accuracy": result.validation_pair_accuracy,
        "train_top1_accuracy": result.train_top1_accuracy,
        "validation_top1_accuracy": result.validation_top1_accuracy,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "recurrent_depth": depth,
    }
    save_readout_checkpoint(
        checkpoint_path,
        readout_weights=result.weights,
        readout_indices=cache.readout_indices,
        recurrent_depth=depth,
        projector_seed=int(projector_seed) if projector_seed is not None else None,
        metadata={**cache.metadata, **metrics},
    )

    h = result.history.copy()
    dashboard = pd.DataFrame({
        "step": h["epoch"].astype(int),
        "epoch": h["epoch"],
        "loss": h["loss"],
        "validation_loss": h["validation_loss"],
        "teacher_agreement": h["validation_top1_accuracy"],
        "candidate_accuracy": h["validation_pair_accuracy"],
        "elo": pd.NA,
        "elo_ci_low": pd.NA,
        "elo_ci_high": pd.NA,
        "recurrent_depth": depth,
        "learning_rate": args.learning_rate,
        "checkpoint": "",
    })
    dashboard.loc[dashboard.index[-1], "checkpoint"] = checkpoint_path.name
    dashboard.to_csv(args.output_dir / "training_history.csv", index=False)
    result.history.to_csv(args.output_dir / "training_diagnostics.csv", index=False)
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
