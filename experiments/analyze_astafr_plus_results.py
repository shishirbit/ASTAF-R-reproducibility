"""Analyze the locked ASTAF-R+ confirmation against common-code baselines.

Expanded-budget common-code runs replace matching earlier baseline runs, and
the frozen confirmation runs in ``results/final_plus`` provide ASTAF-R. The
earlier directory remains available only for deterministic persistence and any
non-overlapping context rows. Validation-only architecture-screening artifacts
are deliberately excluded.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from revision_data import prepare_dataset
from revision_models import row_normalize_adjacency
from run_revision_experiments import DEFAULT_DATA_DIR


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNS = ROOT / "results" / "main" / "runs"
TUNED_BASELINE_RUNS = ROOT / "results" / "final_baselines_64" / "runs"
PROPOSED_RUNS = ROOT / "results" / "final_plus" / "runs"
OUTPUT = ROOT / "results" / "final_analysis"
FIGURES = ROOT / "figures" / "final_plus"
TABLES = ROOT / "tables" / "final_plus"

DISPLAY = {
    "persistence": "Persistence",
    "stid": "STID-style",
    "dcrnn": "DCRNN-style",
    "stgcn": "STGCN-style",
    "transformer": "Transformer-style",
    "graphwavenet": "Graph WaveNet-style",
    "agcrn": "AGCRN-style",
    "astaf": "ASTAF backbone",
    "astafr_attention": "Earlier ASTAF-R",
    "astafr": "Earlier ASTAF-R (attention off)",
    "astafr_plus": "ASTAF-R (proposed)",
}

PALETTE = {
    "astafr_plus": "#b2182b",
    "agcrn": "#2166ac",
    "graphwavenet": "#4393c3",
    "stid": "#92c5de",
    "dcrnn": "#d1e5f0",
}

PRIMARY_MODELS = {
    "persistence", "stid", "dcrnn", "stgcn", "transformer",
    "graphwavenet", "agcrn", "astaf", "astafr_plus",
}


def load_runs() -> list[dict]:
    indexed: dict[tuple[str, str, int], dict] = {}
    # Later directories replace an earlier run with the same dataset/model/seed.
    # This makes expanded-budget baselines authoritative without duplicating rows.
    for directory in (BASELINE_RUNS, TUNED_BASELINE_RUNS, PROPOSED_RUNS):
        for path in sorted(directory.glob("*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            if "test" not in run:
                continue
            npz_path = path.with_suffix(".npz")
            run["_path"] = str(path)
            run["_arrays"] = dict(np.load(npz_path)) if npz_path.exists() else {}
            indexed[(run["dataset"], run["model"], int(run["seed"]))] = run
    return list(indexed.values())


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def aggregate(runs: list[dict]) -> pd.DataFrame:
    groups = defaultdict(list)
    for run in runs:
        if run["model"] not in PRIMARY_MODELS:
            continue
        groups[(run["dataset"], run["model"])].append(run)
    rows = []
    for (dataset, model), group in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "model": model,
            "display_name": DISPLAY.get(model, model),
            "n_seeds": len(group),
            "parameters": group[0].get("parameters"),
            "FLOPs_batch1": group[0].get("profiled_flops_batch1"),
        }
        for metric in ("MAE", "RMSE", "MAPE_percent"):
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(
                [entry["test"][metric] for entry in group]
            )
        for label in ("15_min", "30_min", "60_min"):
            row[f"{label}_MAE_mean"], row[f"{label}_MAE_sd"] = mean_sd(
                [entry["test"]["reported_horizons"][label]["MAE"] for entry in group]
            )
        row["training_seconds_mean"], row["training_seconds_sd"] = mean_sd(
            [entry["training"]["training_seconds"] for entry in group]
        )
        inference = [
            entry["test"].get("inference_ms_per_sample") for entry in group
            if entry["test"].get("inference_ms_per_sample") is not None
        ]
        row["inference_ms_per_sample_mean"], row["inference_ms_per_sample_sd"] = (
            mean_sd(inference) if inference else (np.nan, np.nan)
        )
        memory = [
            entry["test"].get("peak_gpu_memory_mib") for entry in group
            if entry["test"].get("peak_gpu_memory_mib") is not None
        ]
        row["peak_gpu_memory_mib_mean"], row["peak_gpu_memory_mib_sd"] = (
            mean_sd(memory) if memory else (np.nan, np.nan)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def horizon_table(runs: list[dict]) -> pd.DataFrame:
    grouped = defaultdict(list)
    for run in runs:
        if run["model"] not in PRIMARY_MODELS:
            continue
        for horizon, metrics in run["test"]["per_horizon"].items():
            grouped[(run["dataset"], run["model"], int(horizon))].append(metrics)
    rows = []
    for (dataset, model, horizon), values in sorted(grouped.items()):
        row = {
            "dataset": dataset,
            "model": model,
            "display_name": DISPLAY.get(model, model),
            "horizon": horizon,
            "minutes": 5 * horizon,
            "n_seeds": len(values),
        }
        for metric in ("MAE", "RMSE", "MAPE_percent"):
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(
                [value[metric] for value in values]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_daily_tests(runs: list[dict]) -> pd.DataFrame:
    index = {(r["dataset"], r["model"], r["seed"]): r for r in runs}
    rows = []
    for dataset in sorted({r["dataset"] for r in runs}):
        proposed = index.get((dataset, "astafr_plus", 11))
        if proposed is None:
            continue
        p_starts = proposed["_arrays"]["window_start"]
        p_error = proposed["_arrays"]["sample_mae"]
        dataset_rows = []
        t_ps, w_ps = [], []
        for model in sorted({r["model"] for r in runs if r["dataset"] == dataset}):
            if model == "astafr_plus" or model not in PRIMARY_MODELS:
                continue
            comparator = index.get((dataset, model, 11))
            if comparator is None:
                continue
            c_starts = comparator["_arrays"]["window_start"]
            c_error = comparator["_arrays"]["sample_mae"]
            common, ip, ic = np.intersect1d(p_starts, c_starts, return_indices=True)
            # The archive stores input-window starts. Statistical blocks are
            # assigned by forecast origin, which is 12 five-minute steps later.
            day = (common + 12) // 288
            days = np.unique(day)
            p_daily = np.asarray([p_error[ip][day == d].mean() for d in days])
            c_daily = np.asarray([c_error[ic][day == d].mean() for d in days])
            difference = p_daily - c_daily
            t = stats.ttest_rel(p_daily, c_daily)
            try:
                w = stats.wilcoxon(difference, zero_method="pratt")
                w_p = float(w.pvalue)
            except ValueError:
                w_p = 1.0
            t_p = float(t.pvalue)
            t_ps.append(t_p)
            w_ps.append(w_p)
            dataset_rows.append({
                "dataset": dataset,
                "comparison": f"ASTAF-R (proposed) vs {DISPLAY.get(model, model)}",
                "paired_windows": len(common),
                "paired_days": len(days),
                "mean_daily_MAE_difference": float(difference.mean()),
                "paired_t_statistic": float(t.statistic),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            })
        if dataset_rows:
            for row, t_adj, w_adj in zip(dataset_rows, holm(t_ps), holm(w_ps)):
                row["paired_t_p_holm"] = float(t_adj)
                row["wilcoxon_p_holm"] = float(w_adj)
                rows.append(row)
    return pd.DataFrame(rows)


def graph_tables(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    def strongest_positive(row: np.ndarray, limit: int = 20) -> set[int]:
        positive = np.flatnonzero(row > 0)
        if len(positive) <= limit:
            return set(positive.tolist())
        local = np.argpartition(row[positive], -limit)[-limit:]
        return set(positive[local].tolist())

    references = {
        dataset: row_normalize_adjacency(
            prepare_dataset(DEFAULT_DATA_DIR, dataset).adjacency,
            add_self_loops=False,
        ).numpy()
        for dataset in ("METR-LA", "PeMS-BAY")
    }
    graphs = {}
    rows = []
    for run in runs:
        if run["model"] != "astafr_plus":
            continue
        adjacency = run["_arrays"].get("learned_adjacency")
        if adjacency is None:
            continue
        graphs[(run["dataset"], run["seed"])] = adjacency.astype(float)
        mask = ~np.eye(adjacency.shape[0], dtype=bool)
        reference = references[run["dataset"]]
        entropy = -(adjacency * np.log(np.clip(adjacency, 1e-12, None))).sum(1).mean()
        overlaps = []
        for node in range(adjacency.shape[0]):
            learned_neighbors = strongest_positive(adjacency[node])
            reference_neighbors = strongest_positive(reference[node])
            union = learned_neighbors | reference_neighbors
            overlaps.append(len(learned_neighbors & reference_neighbors) / len(union) if union else 1.0)
        rows.append({
            "dataset": run["dataset"],
            "seed": run["seed"],
            "row_entropy": float(entropy),
            "mean_off_diagonal_weight": float(adjacency[mask].mean()),
            "max_off_diagonal_weight": float(adjacency[mask].max()),
            "nonzero_fraction": float((adjacency > 0).mean()),
            "top10_probability_mass": float(np.sort(adjacency, axis=1)[:, -10:].sum(1).mean()),
            "reference_adjacency_pearson": float(
                stats.pearsonr(adjacency[mask], reference[mask]).statistic
            ),
            "reference_top20_jaccard": float(np.mean(overlaps)),
        })
    stability = []
    for dataset in sorted({key[0] for key in graphs}):
        seeds = sorted(seed for ds, seed in graphs if ds == dataset)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                a, b = graphs[(dataset, seed_a)], graphs[(dataset, seed_b)]
                mask = ~np.eye(a.shape[0], dtype=bool)
                overlaps = []
                for node in range(a.shape[0]):
                    aa = strongest_positive(a[node])
                    bb = strongest_positive(b[node])
                    union = aa | bb
                    overlaps.append(len(aa & bb) / len(union) if union else 1.0)
                stability.append({
                    "dataset": dataset,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "adjacency_pearson": float(stats.pearsonr(a[mask], b[mask]).statistic),
                    "mean_top20_jaccard": float(np.mean(overlaps)),
                })
    return pd.DataFrame(rows), pd.DataFrame(stability)


def save_figure(figure, name):
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=320, bbox_inches="tight")
    plt.close(figure)


def plot_main(table: pd.DataFrame):
    for dataset in table["dataset"].unique():
        subset = table[table["dataset"] == dataset].sort_values("MAE_mean")
        figure, axis = plt.subplots(figsize=(10, 5.2))
        colors = [PALETTE.get(model, "#bdbdbd") for model in subset["model"]]
        axis.bar(subset["display_name"], subset["MAE_mean"],
                 yerr=subset["MAE_sd"], capsize=4, color=colors, edgecolor="white")
        axis.set_ylabel("Test MAE (mph)")
        axis.set_title(f"{dataset}: common-code full-split test comparison")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=.22)
        save_figure(figure, f"{dataset.lower().replace('-', '_')}_mae_comparison")


def plot_normalized_metrics(table: pd.DataFrame):
    """Compare differently scaled aggregate metrics without mixing their units."""
    metrics = ["MAE_mean", "RMSE_mean", "MAPE_percent_mean"]
    labels = ["MAE", "RMSE", "MAPE"]
    for dataset in table["dataset"].unique():
        subset = table[table["dataset"] == dataset].sort_values("MAE_mean").copy()
        values = subset[metrics].to_numpy(dtype=float)
        normalized = values / np.nanmin(values, axis=0, keepdims=True)
        x = np.arange(len(metrics))
        width = 0.82 / len(subset)
        figure, axis = plt.subplots(figsize=(10.2, 5.6))
        for index, (_, row) in enumerate(subset.iterrows()):
            offset = (index - (len(subset) - 1) / 2) * width
            axis.bar(
                x + offset,
                normalized[index],
                width=width,
                label=row["display_name"],
                color=PALETTE.get(
                    row["model"], plt.cm.tab20(index / max(1, len(subset) - 1))
                ),
                edgecolor="white",
            )
        axis.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Metric / best metric value (lower is better)")
        axis.set_title(f"{dataset}: normalized aggregate error profile")
        axis.grid(axis="y", alpha=.22)
        axis.legend(frameon=False, ncol=3, fontsize=8)
        save_figure(figure, f"{dataset.lower().replace('-', '_')}_normalized_metrics")


def plot_uniform_ranks(table: pd.DataFrame):
    """Show whether any model is consistently first across metrics and horizons."""
    columns = [
        "MAE_mean", "RMSE_mean", "MAPE_percent_mean",
        "15_min_MAE_mean", "30_min_MAE_mean", "60_min_MAE_mean",
    ]
    labels = ["MAE", "RMSE", "MAPE", "15-min MAE", "30-min MAE", "60-min MAE"]
    for dataset in table["dataset"].unique():
        subset = table[table["dataset"] == dataset].sort_values("MAE_mean").reset_index(drop=True)
        ranks = subset[columns].rank(method="min", ascending=True).to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(9.2, 6.0))
        image = axis.imshow(ranks, aspect="auto", cmap="YlGnBu", vmin=1, vmax=len(subset))
        for row in range(ranks.shape[0]):
            for column in range(ranks.shape[1]):
                axis.text(column, row, f"{int(ranks[row, column])}", ha="center", va="center",
                          color="white" if ranks[row, column] > len(subset) / 2 else "black",
                          fontsize=8)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(subset)), subset["display_name"])
        axis.set_title(f"{dataset}: rank consistency across metrics and horizons")
        figure.colorbar(image, ax=axis, fraction=.035, pad=.03, label="Rank (1 is best)")
        save_figure(figure, f"{dataset.lower().replace('-', '_')}_uniform_ranks")


def plot_horizons(horizons: pd.DataFrame):
    selected = {"astafr_plus", "agcrn", "graphwavenet", "stid", "dcrnn"}
    for dataset in horizons["dataset"].unique():
        figure, axis = plt.subplots(figsize=(7.8, 5.2))
        subset = horizons[(horizons["dataset"] == dataset) & horizons["model"].isin(selected)]
        for model, group in subset.groupby("model"):
            group = group.sort_values("minutes")
            axis.plot(group["minutes"], group["MAE_mean"], marker="o", linewidth=2.2,
                      label=DISPLAY.get(model, model), color=PALETTE.get(model))
            if group["MAE_sd"].max() > 0:
                axis.fill_between(group["minutes"], group["MAE_mean"] - group["MAE_sd"],
                                  group["MAE_mean"] + group["MAE_sd"], alpha=.12,
                                  color=PALETTE.get(model))
        axis.set_xlabel("Forecast horizon (minutes)")
        axis.set_ylabel("Test MAE (mph)")
        axis.set_title(f"{dataset}: error across all 12 horizons")
        axis.grid(alpha=.22)
        axis.legend(frameon=False)
        save_figure(figure, f"{dataset.lower().replace('-', '_')}_horizon_mae")


def plot_efficiency(table: pd.DataFrame):
    for dataset in table["dataset"].unique():
        subset = table[(table["dataset"] == dataset) & (table["parameters"] > 0)].dropna(
            subset=["parameters"]
        )
        figure, axis = plt.subplots(figsize=(8, 5.4))
        for _, row in subset.iterrows():
            axis.scatter(row["parameters"], row["MAE_mean"], s=80,
                         color=PALETTE.get(row["model"], "#969696"))
            axis.annotate(row["display_name"], (row["parameters"], row["MAE_mean"]),
                          xytext=(5, 4), textcoords="offset points", fontsize=8)
        axis.set_xscale("log")
        axis.set_xlabel("Trainable parameters (log scale)")
        axis.set_ylabel("Test MAE (mph; lower is better)")
        axis.set_title(f"{dataset}: accuracy--parameter trade-off")
        axis.grid(alpha=.22)
        save_figure(figure, f"{dataset.lower().replace('-', '_')}_efficiency")


def plot_training(runs: list[dict]):
    for dataset in sorted({r["dataset"] for r in runs if r["model"] == "astafr_plus"}):
        figure, axis = plt.subplots(figsize=(7.8, 5.0))
        for run in runs:
            if run["dataset"] != dataset or run["model"] != "astafr_plus":
                continue
            history = run["training"]["history"]
            epochs = [entry["epoch"] for entry in history]
            axis.plot(epochs, [entry["validation_mae_normalized"] for entry in history],
                      label=f"validation, seed {run['seed']}", linewidth=2)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Normalized validation MAE")
        axis.set_title(f"{dataset}: ASTAF-R convergence across seeds")
        axis.grid(alpha=.22)
        axis.legend(frameon=False)
        save_figure(figure, f"{dataset.lower().replace('-', '_')}_training_curves")


def plot_adjacency(runs: list[dict]):
    for run in runs:
        if run["model"] != "astafr_plus" or run["seed"] != 11:
            continue
        adjacency = run["_arrays"].get("learned_adjacency")
        if adjacency is None:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 6.2))
        image = axis.imshow(adjacency, aspect="auto", cmap="magma")
        axis.set_xlabel("Source sensor")
        axis.set_ylabel("Target sensor")
        axis.set_title(f"{run['dataset']}: learned sparse adaptive adjacency (seed 11)")
        figure.colorbar(image, ax=axis, fraction=.046, pad=.04, label="Row-normalized weight")
        save_figure(figure, f"{run['dataset'].lower().replace('-', '_')}_learned_adjacency")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    proposed = [r for r in runs if r["model"] == "astafr_plus"]
    expected = {(dataset, seed) for dataset in ("METR-LA", "PeMS-BAY") for seed in (11, 22, 33)}
    found = {(r["dataset"], r["seed"]) for r in proposed}
    if found != expected:
        raise RuntimeError(f"Locked confirmation incomplete: expected {expected}, found {found}")
    tuned_expected = {
        (dataset, model, seed)
        for dataset in ("METR-LA", "PeMS-BAY")
        for model in ("stid", "graphwavenet", "agcrn", "dcrnn", "transformer", "stgcn", "astaf")
        for seed in (11, 22, 33)
    }
    tuned_found = set()
    for path in TUNED_BASELINE_RUNS.glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        tuned_found.add((run["dataset"], run["model"], int(run["seed"])))
    if tuned_found != tuned_expected:
        raise RuntimeError(
            f"Expanded-budget baseline confirmation incomplete: expected {tuned_expected}, "
            f"found {tuned_found}"
        )

    main_table = aggregate(runs)
    horizons = horizon_table(runs)
    significance = paired_daily_tests(runs)
    graph, stability = graph_tables(runs)
    for name, table in (
        ("main_results", main_table),
        ("horizon_results", horizons),
        ("significance_results", significance),
        ("graph_results", graph),
        ("graph_stability", stability),
    ):
        table.to_csv(TABLES / f"{name}.csv", index=False)
    plot_main(main_table)
    plot_normalized_metrics(main_table)
    plot_uniform_ranks(main_table)
    plot_horizons(horizons)
    plot_efficiency(main_table)
    plot_training(runs)
    plot_adjacency(runs)
    summary = {
        "included_baseline_directory": str(BASELINE_RUNS),
        "included_tuned_baseline_directory": str(TUNED_BASELINE_RUNS),
        "included_proposed_directory": str(PROPOSED_RUNS),
        "validation_screens_included": False,
        "run_count": len(runs),
        "proposed_run_count": len(proposed),
        "tables": [str(path) for path in sorted(TABLES.glob("*.csv"))],
        "figures": [str(path) for path in sorted(FIGURES.glob("*.pdf"))],
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
