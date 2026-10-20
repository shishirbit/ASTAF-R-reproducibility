"""Fail-fast audit for raw-observation leakage across temporal splits."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from revision_data import dataset_summary, prepare_dataset
from run_revision_experiments import DEFAULT_DATA_DIR


def used_observations(bundle, split: str) -> set[int]:
    used: set[int] = set()
    span = bundle.seq_length + bundle.horizon
    for start in bundle.split_indices[split]:
        used.update(range(int(start), int(start) + span))
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "round2_split_audit.json",
    )
    args = parser.parse_args()
    report = {}
    for dataset in ("METR-LA", "PeMS-BAY"):
        bundle = prepare_dataset(args.data_dir, dataset)
        used = {split: used_observations(bundle, split) for split in ("train", "val", "test")}
        intersections = {
            "train_validation": len(used["train"] & used["val"]),
            "train_test": len(used["train"] & used["test"]),
            "validation_test": len(used["val"] & used["test"]),
        }
        if any(intersections.values()):
            raise AssertionError(f"Raw-observation leakage in {dataset}: {intersections}")
        report[dataset] = {
            "intersections": intersections,
            "dataset_summary": dataset_summary(bundle),
        }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
