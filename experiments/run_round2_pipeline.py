"""Leakage-free, validation-locked experiment pipeline for review round two.

Architecture selection is completed and written to disk before the test-set
runs begin. The locked test stage never compares candidate variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_improvement_screen import CANDIDATES, run_candidate
from run_revision_experiments import DEFAULT_DATA_DIR, aggregate_results, run_one


SELECTION_CANDIDATES = [
    "plus_fast64",
    "plus_resattn64",
    "plus_mandatory64",
    "plus_decomp_off64",
    "plus_decomp_fixed64",
    "plus_topk40_64",
    "plus_graph1_64",
    "plus_no_residual64",
    "plus_scales13_64",
]

CONTROL_CANDIDATES = [
    "plus_uniform_graph64",
    "plus_shuffled_graph64",
    "plus_correlation_graph64",
    "plus_identity_graph64",
    "plus_no_graph64",
    "plus_weights_equal64",
    "plus_weights_trend64",
    "plus_scales1_64",
    "plus_scales3_64",
    "plus_scales16_64",
    "plus_scales36_64",
    "plus_graph3_64",
    "plus_no_context64",
    *[f"cumulative_{name}64" for name in (
        "temporal", "history", "context", "decomposition", "graph",
        "attention", "residual",
    )],
]

BASELINES = [
    "persistence",
    "stid",
    "dcrnn",
    "stgcn",
    "transformer",
    "graphwavenet",
    "agcrn",
    "astaf",
]


def load_or_screen(args, candidate: str, dataset: str, seed: int) -> dict:
    run_id = f"{dataset}_{candidate}_seed{seed}".replace("-", "_").lower()
    path = args.output_dir / "validation" / candidate / "runs" / f"{run_id}.json"
    if path.exists():
        print(f"REUSE validation {run_id}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    screen_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=args.output_dir / "validation",
        batch_size=args.batch_size,
        max_epochs=args.selection_epochs,
        patience=args.selection_patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return run_candidate(screen_args, dataset, candidate, seed)


def select_architecture(results: list[dict]) -> dict:
    by_candidate: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if result["candidate"] not in SELECTION_CANDIDATES:
            continue
        by_candidate.setdefault(result["candidate"], {}).setdefault(
            result["dataset"], []
        ).append(float(result["validation"]["MAE"]))
    datasets = sorted({dataset for values in by_candidate.values() for dataset in values})
    means = {
        candidate: {
            dataset: float(np.mean(by_dataset[dataset]))
            for dataset in datasets
        }
        for candidate, by_dataset in by_candidate.items()
        if all(dataset in by_dataset for dataset in datasets)
    }
    best_by_dataset = {
        dataset: min(values[dataset] for values in means.values()) for dataset in datasets
    }
    scores = {
        candidate: float(
            np.mean([values[dataset] / best_by_dataset[dataset] for dataset in datasets])
        )
        for candidate, values in means.items()
    }
    winner = min(scores, key=lambda candidate: (scores[candidate], candidate))
    return {
        "selection_split": "validation only; test outcomes unavailable at selection time",
        "criterion": "mean across-dataset relative validation MAE; lower is better",
        "datasets": datasets,
        "candidate_validation_mae": means,
        "relative_scores": scores,
        "selected_candidate": winner,
        "selected_specification": CANDIDATES[winner],
    }


def load_existing(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "round2_leakage_free",
    )
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PeMS-BAY"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--selection-epochs", type=int, default=40)
    parser.add_argument("--selection-patience", type=int, default=8)
    parser.add_argument("--test-epochs", type=int, default=60)
    parser.add_argument("--test-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--selection-only", action="store_true", help="Stop before locked test evaluation."
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_results = []
    for dataset in args.datasets:
        for candidate in SELECTION_CANDIDATES:
            for seed in args.seeds:
                validation_results.append(load_or_screen(args, candidate, dataset, seed))
        # Diagnostic graph, scale, and cumulative controls are deliberately
        # validation-only. The principal design choices above receive the full
        # three-seed analysis; these broader controls use the first declared
        # seed to keep the computational budget auditable and finite.
        for candidate in CONTROL_CANDIDATES:
            validation_results.append(
                load_or_screen(args, candidate, dataset, args.seeds[0])
            )
    selection = select_architecture(validation_results)
    selection_path = args.output_dir / "architecture_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(validation_results, indent=2), encoding="utf-8"
    )
    print(f"LOCKED ARCHITECTURE: {selection['selected_candidate']}", flush=True)
    if args.selection_only:
        return

    selected = CANDIDATES[selection["selected_candidate"]]
    test_results = []
    main_dir = args.output_dir / "locked_test"
    for dataset in args.datasets:
        for model in ["astafr_plus", *BASELINES]:
            seeds = [args.seeds[0]] if model == "persistence" else args.seeds
            for seed in seeds:
                run_id = f"{dataset}_{model}_seed{seed}".replace("-", "_").lower()
                existing_path = main_dir / "runs" / f"{run_id}.json"
                existing = load_existing(existing_path)
                if existing is not None:
                    print(f"REUSE locked test {run_id}", flush=True)
                    test_results.append(existing)
                    continue
                overrides = selected["overrides"] if model == "astafr_plus" else None
                hidden_dim = selected["hidden_dim"] if model == "astafr_plus" else 64
                test_results.append(
                    run_one(
                        dataset_name=dataset,
                        model_name=model,
                        seed=seed,
                        data_dir=args.data_dir,
                        output_dir=main_dir,
                        max_epochs=args.test_epochs,
                        patience=args.test_patience,
                        batch_size=args.batch_size,
                        hidden_dim=hidden_dim,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        overrides=overrides,
                    )
                )
    (main_dir / "main_summary.json").write_text(
        json.dumps(aggregate_results(test_results), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
