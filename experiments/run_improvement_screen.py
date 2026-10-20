"""Validation-only screening for improved ASTAF-R candidates.

This script deliberately never constructs a test-set evaluation. Candidate
selection is based only on the chronological validation split. A separate
locked confirmation command is used after the architecture is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from revision_data import dataset_summary, make_loaders, prepare_dataset
from revision_models import build_model, count_parameters
from run_revision_experiments import (
    DEFAULT_DATA_DIR,
    environment_metadata,
    evaluate_model,
    graph_artifacts,
    save_run,
    set_seed,
    train_model,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "improvement_screen"


CANDIDATES = {
    "combo32": {
        "model": "astafr",
        "hidden_dim": 32,
        "overrides": {"gamma": 0.9999, "attention_mode": "mandatory"},
    },
    "plus_fast32": {
        "model": "astafr_plus",
        "hidden_dim": 32,
        "overrides": {
            "attention_mode": "off",
            "graph_top_k": 20,
            "graph_layers": 2,
            "use_context": True,
            "residual_output": True,
        },
    },
    "plus_resattn32": {
        "model": "astafr_plus",
        "hidden_dim": 32,
        "overrides": {
            "attention_mode": "residual",
            "graph_top_k": 20,
            "graph_layers": 2,
            "use_context": True,
            "residual_output": True,
        },
    },
    "plus_mandatory32": {
        "model": "astafr_plus",
        "hidden_dim": 32,
        "overrides": {
            "attention_mode": "mandatory",
            "graph_top_k": 20,
            "graph_layers": 2,
            "use_context": True,
            "residual_output": True,
        },
    },
    "plus_fast64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {
            "attention_mode": "off",
            "graph_top_k": 20,
            "graph_layers": 2,
            "use_context": True,
            "residual_output": True,
        },
    },
    "plus_resattn64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {
            "attention_mode": "residual",
            "graph_top_k": 20,
            "graph_layers": 2,
            "use_context": True,
            "residual_output": True,
        },
    },
    "plus_mandatory64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "mandatory", "graph_top_k": 20},
    },
    "plus_decomp_off64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "decomposition_mode": "off"},
    },
    "plus_decomp_fixed64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "decomposition_mode": "fixed"},
    },
    "plus_weights_equal64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {
            "attention_mode": "residual",
            "decomposition_mode": "fixed",
            "decomposition_weights": (0.2333, 0.2333, 0.2333),
        },
    },
    "plus_weights_trend64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {
            "attention_mode": "residual",
            "decomposition_mode": "fixed",
            "decomposition_weights": (0.50, 0.15, 0.05),
        },
    },
    "plus_topk10_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_top_k": 10},
    },
    "plus_topk40_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_top_k": 40},
    },
    "plus_graph1_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_layers": 1},
    },
    "plus_graph3_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_layers": 3},
    },
    "plus_no_context64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "use_context": False},
    },
    "plus_no_residual64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "residual_output": False},
    },
    "plus_scales1_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "temporal_scales": (1,)},
    },
    "plus_scales3_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "temporal_scales": (3,)},
    },
    "plus_scales13_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "temporal_scales": (1, 3)},
    },
    "plus_scales16_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "temporal_scales": (1, 6)},
    },
    "plus_scales36_64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "temporal_scales": (3, 6)},
    },
    "plus_uniform_graph64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_mode": "uniform_learned_topk"},
    },
    "plus_shuffled_graph64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_mode": "shuffled_topk"},
    },
    "plus_correlation_graph64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_mode": "correlation_topk"},
    },
    "plus_identity_graph64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_mode": "identity"},
    },
    "plus_no_graph64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "graph_mode": "none"},
    },
    "cumulative_temporal64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "off", "decomposition_mode": "off", "graph_mode": "none", "use_history": False, "use_node_embedding": False, "use_context": False, "residual_output": False},
    },
    "cumulative_history64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "off", "decomposition_mode": "off", "graph_mode": "none", "use_node_embedding": False, "use_context": False, "residual_output": False},
    },
    "cumulative_context64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "off", "decomposition_mode": "off", "graph_mode": "none", "residual_output": False},
    },
    "cumulative_decomposition64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "off", "graph_mode": "none", "residual_output": False},
    },
    "cumulative_graph64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "off", "residual_output": False},
    },
    "cumulative_attention64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "residual_output": False},
    },
    "cumulative_residual64": {
        "model": "astafr_plus",
        "hidden_dim": 64,
        "overrides": {"attention_mode": "residual", "residual_output": True},
    },
}


def run_candidate(args, dataset_name: str, candidate_name: str, seed: int) -> dict:
    specification = CANDIDATES[candidate_name]
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = prepare_dataset(args.data_dir, dataset_name)
    loaders = make_loaders(bundle, batch_size=args.batch_size, seed=seed)
    overrides = dict(specification["overrides"])
    if overrides.get("graph_mode") == "correlation_topk":
        train_end = bundle.split_time_bounds["train"][1]
        train_values = bundle.raw[:train_end].astype(np.float64, copy=True)
        train_valid = bundle.valid[:train_end]
        sensor_means = np.divide(
            (train_values * train_valid).sum(axis=0),
            train_valid.sum(axis=0),
            out=np.full(train_values.shape[1], bundle.mean, dtype=np.float64),
            where=train_valid.sum(axis=0) > 0,
        )
        train_values[~train_valid] = np.broadcast_to(
            sensor_means, train_values.shape
        )[~train_valid]
        correlation = np.nan_to_num(np.abs(np.corrcoef(train_values, rowvar=False)))
        np.fill_diagonal(correlation, -np.inf)
        top_k = 20
        indices = np.argpartition(correlation, -top_k, axis=1)[:, -top_k:]
        reference = np.zeros_like(correlation, dtype=np.float32)
        np.put_along_axis(reference, indices, 1.0 / top_k, axis=1)
        overrides["reference_adjacency"] = reference
    model = build_model(
        specification["model"],
        num_nodes=bundle.raw.shape[1],
        adjacency=bundle.adjacency,
        seq_length=bundle.seq_length,
        horizon=bundle.horizon,
        hidden_dim=specification["hidden_dim"],
        **overrides,
    ).to(device)

    run_id = f"{dataset_name}_{candidate_name}_seed{seed}".replace("-", "_").lower()
    candidate_dir = args.output_dir / candidate_name
    checkpoint_path = candidate_dir / "checkpoints" / f"{run_id}.pt"
    print(
        f"\nSCREEN {run_id}: parameters={count_parameters(model):,} device={device}",
        flush=True,
    )
    training = train_model(
        model,
        loaders,
        device,
        checkpoint_path,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    validation, arrays = evaluate_model(
        model,
        loaders["val"],
        device,
        mean=bundle.mean,
        std=bundle.std,
    )
    artifacts = graph_artifacts(model)
    result = {
        "run_id": run_id,
        "stage": "validation_only_screen",
        "test_set_evaluated": False,
        "dataset": dataset_name,
        "candidate": candidate_name,
        "model": specification["model"],
        "seed": seed,
        "protocol": {
            "selection_split": "chronological validation split only",
            "normalization": "training-period valid values only",
            "loss": "masked MAE",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "hidden_dim": specification["hidden_dim"],
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "overrides": {
                key: value
                for key, value in overrides.items()
                if key != "reference_adjacency"
            },
        },
        "dataset_summary": dataset_summary(bundle),
        "environment": environment_metadata(device),
        "parameters": count_parameters(model),
        "training": training,
        "validation": validation,
        "learned_values": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in artifacts.items()
            if key not in {"learned_adjacency", "effective_adjacency", "reference_adjacency"}
        },
    }
    save_run(candidate_dir / "runs", run_id, result, arrays, artifacts)
    print(
        f"SCREEN DONE {run_id}: validation MAE={validation['MAE']:.4f} "
        f"RMSE={validation['RMSE']:.4f}",
        flush=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["METR-LA"])
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a completed validation JSON instead of retraining it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = sorted(set(args.candidates) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset_name in args.datasets:
        for candidate_name in args.candidates:
            for seed in args.seeds:
                run_id = f"{dataset_name}_{candidate_name}_seed{seed}".replace(
                    "-", "_"
                ).lower()
                existing = (
                    args.output_dir
                    / candidate_name
                    / "runs"
                    / f"{run_id}.json"
                )
                if args.skip_existing and existing.exists():
                    print(f"SKIP completed {run_id}", flush=True)
                    results.append(json.loads(existing.read_text(encoding="utf-8")))
                else:
                    results.append(run_candidate(args, dataset_name, candidate_name, seed))
    summary_path = args.output_dir / "screen_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
