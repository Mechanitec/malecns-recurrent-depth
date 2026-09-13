from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class TrainingMetric:
    step: int
    epoch: float | None = None
    loss: float | None = None
    validation_loss: float | None = None
    teacher_agreement: float | None = None
    candidate_accuracy: float | None = None
    elo: float | None = None
    elo_ci_low: float | None = None
    elo_ci_high: float | None = None
    recurrent_depth: int | None = None
    learning_rate: float | None = None
    checkpoint: str | None = None


def append_training_metric(path: str | Path, metric: TrainingMetric) -> None:
    """Append one dashboard-compatible training row to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(metric)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
