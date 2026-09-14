"""Merge isolated per-variant position-depth study outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    args = parser.parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not variants:
        parser.error("--variants must not be empty")
    all_rows = []
    fields = None
    for variant in variants:
        paths = sorted((args.input_root / variant).rglob("raw_position_metrics.csv"))
        if not paths:
            raise FileNotFoundError(args.input_root / variant)
        for path in paths:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if fields is None:
                    fields = reader.fieldnames
                elif reader.fieldnames != fields:
                    raise ValueError(f"CSV schema mismatch: {path}")
                rows = list(reader)
            if rows and {row["variant"] for row in rows} != {variant}:
                raise ValueError(f"unexpected variant rows in {path}")
            all_rows.extend(rows)
    assert fields is not None
    keys = [(row["variant"], row["position_id"], int(row["depth"])) for row in all_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate variant-position-depth rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"merged {len(all_rows)} rows from {len(variants)} variants into {args.output}")


if __name__ == "__main__":
    main()
