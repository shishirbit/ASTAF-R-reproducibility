"""Run reproducible full-split experiments requested during peer review."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import scipy
import torch
import torch.nn as nn

from revision_data import dataset_summary, make_loaders, prepare_dataset
from revision_models import ASTAFR, build_model, count_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parent
REPOSITORY_DATA_DIR = REVISION_DIR / "data"
LEGACY_DATA_DIR = REVISION_DIR.parent / "experiment_package" / "Paper-5" / "data"
DEFAULT_DATA_DIR = (
    REPOSITORY_DATA_DIR if REPOSITORY_DATA_DIR.exists() else LEGACY_DATA_DIR
)
DEFAULT_OUTPUT_DIR = REVISION_DIR / "results"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    valid = mask.bool()
    if not torch.any(valid):
        return (prediction - target).abs().mean() * 0
    return (prediction[valid] - target[valid]).abs().mean()


def model_forward(model: nn.Module, x: torch.Tensor, starts: torch.Tensor):
    if getattr(model, "requires_starts", False):
        return model(x, starts=starts)
    return model(x)


def environment_metadata(device: torch.device) -> dict:
    metadata = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "cuda_runtime": torch.version.cuda,
                "gpu_total_memory_gib": round(
                    torch.cuda.get_device_properties(device).total_memory / (1024**3), 3
                ),
            }
        )
    return metadata


def optimizer_for(model: nn.Module, learning_rate: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return None
    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_model(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    checkpoint_path: Path,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    max_batches: Optional[int] = None,
) -> dict:
    optimizer = optimizer_for(model, learning_rate, weight_decay)
    if optimizer is None:
        return {
            "epochs_completed": 0,
            "best_epoch": 0,
            "best_validation_mae_normalized": None,
            "history": [],
            "training_seconds": 0.0,
        }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    start_time = time.perf_counter()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_index, (x, y, mask, starts) in enumerate(loaders["train"]):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                prediction = model_forward(model, x, starts)
                loss = masked_mae(prediction, y, mask)
                if hasattr(model, "regularization_loss"):
                    loss = loss + model.regularization_loss()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach())
            train_count += 1

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch_index, (x, y, mask, starts) in enumerate(loaders["val"]):
                if max_batches is not None and batch_index >= max_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                starts = starts.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    prediction = model_forward(model, x, starts)
                    loss = masked_mae(prediction, y, mask)
                validation_total += float(loss)
                validation_count += 1
        train_loss = train_total / max(train_count, 1)
        validation_loss = validation_total / max(validation_count, 1)
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_mae_normalized": train_loss,
                "validation_mae_normalized": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} train={train_loss:.5f} "
            f"val={validation_loss:.5f} best={best_loss:.5f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break

    elapsed = time.perf_counter() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mae_normalized": best_loss,
        "history": history,
        "training_seconds": elapsed,
    }


def _masked_metric_sums(prediction, target, mask):
    valid = mask.astype(bool)
    error = prediction - target
    absolute = np.abs(error)
    squared = error**2
    percentage = absolute / np.maximum(np.abs(target), 1e-3)
    return (
        float(absolute[valid].sum()),
        float(squared[valid].sum()),
        float(percentage[valid].sum()),
        int(valid.sum()),
    )


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    mean: float,
    std: float,
    max_batches: Optional[int] = None,
) -> tuple[dict, dict]:
    model.eval()
    overall_sums = np.zeros(4, dtype=np.float64)
    horizon_sums = None
    sample_errors = []
    starts_all = []
    batch_times_ms = []
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_index, (x, y, mask, starts) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            starts_device = starts.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                prediction = model_forward(model, x, starts_device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - batch_start) * 1000
            if batch_index >= 2:
                batch_times_ms.append(elapsed_ms)

            prediction = prediction.float().cpu().numpy() * std + mean
            target = y.numpy() * std + mean
            valid = mask.numpy().astype(bool)
            overall_sums += np.asarray(_masked_metric_sums(prediction, target, valid))
            if horizon_sums is None:
                horizon_sums = np.zeros((prediction.shape[1], 4), dtype=np.float64)
            for horizon_index in range(prediction.shape[1]):
                horizon_sums[horizon_index] += np.asarray(
                    _masked_metric_sums(
                        prediction[:, horizon_index],
                        target[:, horizon_index],
                        valid[:, horizon_index],
                    )
                )
            absolute = np.abs(prediction - target)
            numerator = (absolute * valid).sum(axis=(1, 2, 3))
            denominator = valid.sum(axis=(1, 2, 3)).clip(min=1)
            sample_errors.extend((numerator / denominator).tolist())
            starts_all.extend(starts.numpy().tolist())

    def finalize(sums):
        count = max(float(sums[3]), 1.0)
        return {
            "MAE": float(sums[0] / count),
            "RMSE": float(math.sqrt(sums[1] / count)),
            "MAPE_percent": float(100 * sums[2] / count),
            "valid_targets": int(sums[3]),
        }

    metrics = finalize(overall_sums)
    metrics["per_horizon"] = {
        str(index + 1): finalize(horizon_sums[index])
        for index in range(horizon_sums.shape[0])
    }
    metrics["reported_horizons"] = {
        "15_min": metrics["per_horizon"]["3"],
        "30_min": metrics["per_horizon"]["6"],
        "60_min": metrics["per_horizon"]["12"],
    }
    metrics["mean_inference_ms_per_batch"] = (
        float(np.mean(batch_times_ms)) if batch_times_ms else None
    )
    metrics["inference_ms_per_sample"] = (
        float(np.sum(batch_times_ms) / max(len(sample_errors), 1))
        if batch_times_ms
        else None
    )
    metrics["peak_gpu_memory_mib"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    arrays = {
        "sample_mae": np.asarray(sample_errors, dtype=np.float32),
        "window_start": np.asarray(starts_all, dtype=np.int64),
    }
    return metrics, arrays


def estimate_flops(
    model: nn.Module,
    example: torch.Tensor,
    device: torch.device,
    example_starts: Optional[torch.Tensor] = None,
):
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        model.eval()
        with torch.no_grad(), torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
        ) as profile:
            if example_starts is None:
                example_starts = torch.zeros(
                    example.shape[0], dtype=torch.long, device=device
                )
            model_forward(model, example, example_starts)
        flops = sum(event.flops for event in profile.key_averages() if event.flops)
        return int(flops) if flops else None
    except Exception as exc:
        print(f"FLOP profiling unavailable: {exc}", flush=True)
        return None


def graph_artifacts(model: nn.Module) -> dict:
    artifacts = {}
    if hasattr(model, "learned_adjacency"):
        with torch.no_grad():
            learned = model.learned_adjacency().detach().float().cpu().numpy()
            effective = model.effective_adjacency().detach().float().cpu().numpy()
        artifacts["learned_adjacency"] = learned
        artifacts["effective_adjacency"] = effective
        if hasattr(model, "fixed_adjacency"):
            artifacts["reference_adjacency"] = (
                model.fixed_adjacency.detach().float().cpu().numpy()
            )
    if hasattr(model, "gamma_value"):
        artifacts["gamma"] = float(model.gamma_value().detach().cpu())
    if hasattr(model, "decomposition_weight_values"):
        artifacts["decomposition_weights"] = (
            model.decomposition_weight_values().detach().cpu().numpy()
        )
    if getattr(model, "attention_gate", None) is not None:
        artifacts["attention_gate"] = float(
            torch.sigmoid(model.attention_gate).detach().cpu()
        )
    return artifacts


def save_run(
    output_dir: Path,
    run_id: str,
    result: dict,
    arrays: dict,
    artifacts: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    npz_payload = dict(arrays)
    for key, value in artifacts.items():
        if isinstance(value, np.ndarray):
            npz_payload[key] = value
    np.savez_compressed(output_dir / f"{run_id}.npz", **npz_payload)


def run_one(
    dataset_name: str,
    model_name: str,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    max_batches: Optional[int] = None,
    overrides: Optional[dict] = None,
) -> dict:
    overrides = overrides or {}
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = prepare_dataset(data_dir, dataset_name)
    loaders = make_loaders(bundle, batch_size=batch_size, seed=seed)
    model = build_model(
        model_name,
        num_nodes=bundle.raw.shape[1],
        adjacency=bundle.adjacency,
        seq_length=bundle.seq_length,
        horizon=bundle.horizon,
        hidden_dim=hidden_dim,
        **overrides,
    ).to(device)
    run_id = f"{dataset_name}_{model_name}_seed{seed}".replace("-", "_").lower()
    checkpoint_path = output_dir / "checkpoints" / f"{run_id}.pt"
    print(
        f"\nRUN {run_id}: parameters={count_parameters(model):,} device={device}",
        flush=True,
    )
    training = train_model(
        model,
        loaders,
        device,
        checkpoint_path,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_batches=max_batches,
    )
    metrics, arrays = evaluate_model(
        model,
        loaders["test"],
        device,
        mean=bundle.mean,
        std=bundle.std,
        max_batches=max_batches,
    )
    example_batch = next(iter(loaders["test"]))
    example = example_batch[0][:1].to(device)
    example_starts = example_batch[3][:1].to(device)
    flops = estimate_flops(model, example, device, example_starts)
    artifacts = graph_artifacts(model)
    result = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "protocol": {
            "normalization": "global z-score estimated on valid training-period values only",
            "missing_values": "zeros treated as unavailable; mean-imputed in inputs and masked in targets",
            "loss": "masked MAE",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "gradient_clip_norm": 5.0,
            "random_seed": seed,
            "mixed_precision": device.type == "cuda",
            "overrides": overrides,
        },
        "dataset_summary": dataset_summary(bundle),
        "environment": environment_metadata(device),
        "parameters": count_parameters(model),
        "profiled_flops_batch1": flops,
        "training": training,
        "test": metrics,
        "learned_values": {
            key: (
                value.tolist() if isinstance(value, np.ndarray) and value.ndim == 1 else value
            )
            for key, value in artifacts.items()
            if key
            not in {
                "learned_adjacency",
                "effective_adjacency",
                "reference_adjacency",
            }
        },
    }
    save_run(output_dir / "runs", run_id, result, arrays, artifacts)
    print(
        f"DONE {run_id}: MAE={metrics['MAE']:.4f} "
        f"RMSE={metrics['RMSE']:.4f} MAPE={metrics['MAPE_percent']:.2f}%",
        flush=True,
    )
    return result


def aggregate_results(results: Iterable[dict]) -> dict:
    groups = defaultdict(list)
    for result in results:
        groups[(result["dataset"], result["model"])].append(result)
    summary = {}
    for (dataset, model), runs in sorted(groups.items()):
        key = f"{dataset}/{model}"
        metric_names = ["MAE", "RMSE", "MAPE_percent"]
        summary[key] = {
            "seeds": [run["seed"] for run in runs],
            "n": len(runs),
            "parameters": runs[0]["parameters"],
        }
        for metric in metric_names:
            values = np.asarray([run["test"][metric] for run in runs], dtype=float)
            summary[key][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "values": values.tolist(),
            }
        for horizon_name in ("15_min", "30_min", "60_min"):
            values = np.asarray(
                [run["test"]["reported_horizons"][horizon_name]["MAE"] for run in runs]
            )
            summary[key][f"{horizon_name}_MAE"] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        times = np.asarray([run["training"]["training_seconds"] for run in runs])
        summary[key]["training_seconds"] = {
            "mean": float(times.mean()),
            "std": float(times.std(ddof=1)) if times.size > 1 else 0.0,
        }
    return summary


SENSITIVITY_VARIANTS = {
    "gamma_0": {"gamma": 0.0},
    "gamma_005": {"gamma": 0.05},
    "gamma_015": {"gamma": 0.15},
    "gamma_030": {"gamma": 0.30},
    "gamma_050": {"gamma": 0.50},
    "gamma_100": {"gamma": 0.9999},
    "weights_equal": {"decomposition_weights": (0.2333, 0.2333, 0.2333)},
    "weights_default": {"decomposition_weights": (0.35, 0.25, 0.10)},
    "weights_trend_heavy": {"decomposition_weights": (0.50, 0.15, 0.05)},
    "weights_learned": {"decomposition_mode": "learned"},
    "decomposition_off": {"decomposition_mode": "off"},
    "attention_off": {"attention_mode": "off"},
    "attention_gated": {"attention_mode": "gated"},
    "attention_mandatory": {"attention_mode": "mandatory"},
    "scales_1": {"temporal_scales": (1,)},
    "scales_3": {"temporal_scales": (3,)},
    "scales_136": {"temporal_scales": (1, 3, 6)},
    "graph_identity": {"graph_mode": "identity"},
    "graph_fixed": {"graph_mode": "fixed"},
    "graph_fixed_self_loop": {"graph_mode": "fixed_self_loop"},
    "graph_learned": {"graph_mode": "learned"},
    "graph_regularized": {"graph_mode": "regularized"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "sensitivity"), default="main")
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PeMS-BAY"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "persistence",
        "stid",
        "astaf",
        "dcrnn",
            "stgcn",
            "transformer",
            "graphwavenet",
            "agcrn",
            "astafr_attention",
            "astafr",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--variants", nargs="+", default=list(SENSITIVITY_VARIANTS))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing completed JSON run instead of retraining it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if args.mode == "main":
        for dataset_name in args.datasets:
            for model_name in args.models:
                model_seeds = [args.seeds[0]] if model_name == "persistence" else args.seeds
                for seed in model_seeds:
                    run_id = f"{dataset_name}_{model_name}_seed{seed}".replace("-", "_").lower()
                    existing = args.output_dir / "runs" / f"{run_id}.json"
                    if args.skip_existing and existing.exists():
                        print(f"SKIP completed {run_id}", flush=True)
                        results.append(json.loads(existing.read_text(encoding="utf-8")))
                        continue
                    results.append(
                        run_one(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            seed=seed,
                            data_dir=args.data_dir,
                            output_dir=args.output_dir,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            batch_size=args.batch_size,
                            hidden_dim=args.hidden_dim,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            max_batches=args.max_batches,
                        )
                    )
        summary = aggregate_results(results)
        (args.output_dir / "main_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    else:
        dataset_name = args.datasets[0]
        seed = args.seeds[0]
        for variant in args.variants:
            if variant not in SENSITIVITY_VARIANTS:
                raise ValueError(f"Unknown sensitivity variant: {variant}")
            result = run_one(
                dataset_name=dataset_name,
                model_name="astafr",
                seed=seed,
                data_dir=args.data_dir,
                output_dir=args.output_dir / "sensitivity" / variant,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_batches=args.max_batches,
                overrides=SENSITIVITY_VARIANTS[variant],
            )
            result["sensitivity_variant"] = variant
            results.append(result)
        (args.output_dir / "sensitivity_summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
