"""Verify and analyse the leakage-free round-two experiment archive.

This module keeps validation-only architecture selection separate from the
locked test archive.  It reports seed variation, day-block uncertainty, graph
stability at full precision, and regenerates the comparison figures used by
the manuscript.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze_astafr_plus_results as legacy


ROOT = Path(__file__).resolve().parents[1]
ROUND2 = ROOT / "results" / "round2_leakage_free"
RUNS = ROUND2 / "locked_test" / "runs"
OUTPUT = ROOT / "results" / "round2_analysis"
TABLES = ROOT / "tables" / "round2"
FIGURES = ROOT / "figures" / "round2"


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in run:
            continue
        archive = path.with_suffix(".npz")
        run["_path"] = str(path)
        run["_arrays"] = dict(np.load(archive)) if archive.exists() else {}
        runs.append(run)
    return runs


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    means = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def seed_tests(runs: list[dict]) -> pd.DataFrame:
    indexed = {(r["dataset"], r["model"], int(r["seed"])): r for r in runs}
    rows = []
    for dataset in ("METR-LA", "PeMS-BAY"):
        for model in sorted({r["model"] for r in runs if r["dataset"] == dataset}):
            if model == "astafr_plus":
                continue
            paired = []
            for seed in (11, 22, 33):
                proposed = indexed.get((dataset, "astafr_plus", seed))
                comparator = indexed.get((dataset, model, seed))
                if proposed is not None and comparator is not None:
                    paired.append(float(proposed["test"]["MAE"] - comparator["test"]["MAE"]))
            if len(paired) != 3:
                continue
            paired_array = np.asarray(paired)
            test = stats.ttest_1samp(paired_array, 0.0)
            rows.append({
                "dataset": dataset,
                "comparison": f"ASTAF-R vs {legacy.DISPLAY.get(model, model)}",
                "n_seeds": 3,
                "mean_seed_MAE_difference": float(paired_array.mean()),
                "sd_seed_MAE_difference": float(paired_array.std(ddof=1)),
                "paired_t_statistic": float(test.statistic),
                "paired_t_p": float(test.pvalue),
                "exact_seed_sign_flip_p": exact_sign_flip_p(paired_array),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["paired_t_p_holm"] = np.nan
        frame["exact_seed_sign_flip_p_holm"] = np.nan
        for dataset, indices in frame.groupby("dataset").groups.items():
            frame.loc[indices, "paired_t_p_holm"] = legacy.holm(
                frame.loc[indices, "paired_t_p"].to_numpy()
            )
            frame.loc[indices, "exact_seed_sign_flip_p_holm"] = legacy.holm(
                frame.loc[indices, "exact_seed_sign_flip_p"].to_numpy()
            )
    return frame


def block_tests(runs: list[dict], bootstrap_repetitions: int = 10000) -> pd.DataFrame:
    indexed = {(r["dataset"], r["model"], int(r["seed"])): r for r in runs}
    rng = np.random.default_rng(20261023)
    rows = []
    for dataset in ("METR-LA", "PeMS-BAY"):
        models = sorted({r["model"] for r in runs if r["dataset"] == dataset})
        for model in models:
            if model == "astafr_plus":
                continue
            blocks_by_seed = []
            paired_windows = 0
            for seed in (11, 22, 33):
                proposed = indexed.get((dataset, "astafr_plus", seed))
                comparator = indexed.get((dataset, model, seed))
                if proposed is None or comparator is None:
                    continue
                pa, ca = proposed["_arrays"], comparator["_arrays"]
                common, ip, ic = np.intersect1d(
                    pa["window_start"], ca["window_start"], return_indices=True
                )
                day = (common + 12) // 288
                differences = pa["sample_mae"][ip] - ca["sample_mae"][ic]
                daily = np.asarray([differences[day == d].mean() for d in np.unique(day)])
                blocks_by_seed.append(daily)
                paired_windows += len(common)
            if len(blocks_by_seed) != 3:
                continue
            seed_means = np.asarray([blocks.mean() for blocks in blocks_by_seed])
            boot = np.empty(bootstrap_repetitions)
            for index in range(bootstrap_repetitions):
                sampled = [rng.choice(blocks, len(blocks), replace=True).mean() for blocks in blocks_by_seed]
                boot[index] = np.mean(sampled)
            rows.append({
                "dataset": dataset,
                "comparison": f"ASTAF-R vs {legacy.DISPLAY.get(model, model)}",
                "seeds": 3,
                "calendar_day_blocks_per_seed": ";".join(str(len(x)) for x in blocks_by_seed),
                "paired_windows_across_seeds": paired_windows,
                "mean_daily_block_MAE_difference": float(seed_means.mean()),
                "seed_to_seed_sd_of_daily_mean_difference": float(seed_means.std(ddof=1)),
                "clustered_day_bootstrap_CI_low": float(np.quantile(boot, 0.025)),
                "clustered_day_bootstrap_CI_high": float(np.quantile(boot, 0.975)),
                "bootstrap_repetitions": bootstrap_repetitions,
            })
    return pd.DataFrame(rows)


def validation_table() -> pd.DataFrame:
    results = json.loads((ROUND2 / "validation_summary.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["dataset"], result["candidate"])].append(
            float(result["validation"]["MAE"])
        )
    rows = []
    for (dataset, candidate), values in sorted(grouped.items()):
        array = np.asarray(values)
        rows.append({
            "dataset": dataset,
            "candidate": candidate,
            "n_seeds": len(values),
            "validation_MAE_mean": float(array.mean()),
            "validation_MAE_sd": float(array.std(ddof=1)) if len(array) > 1 else np.nan,
            "validation_MAE_values": ";".join(f"{value:.12g}" for value in values),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing metr-la.h5 and pems-bay.h5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if len(runs) != 50:
        raise RuntimeError(f"Expected 50 locked-test records, found {len(runs)}")
    expected = {
        (dataset, model, seed)
        for dataset in ("METR-LA", "PeMS-BAY")
        for model in legacy.PRIMARY_MODELS
        for seed in ((11,) if model == "persistence" else (11, 22, 33))
    }
    found = {(r["dataset"], r["model"], int(r["seed"])) for r in runs}
    if found != expected:
        raise RuntimeError(f"Locked archive mismatch: missing={expected-found}, extra={found-expected}")
    for run in runs:
        summary = run["dataset_summary"]
        if summary.get("raw_observation_overlap_across_splits") is not False:
            raise RuntimeError(f"Leakage audit failed for {run['run_id']}")

    legacy.FIGURES = FIGURES
    legacy.DEFAULT_DATA_DIR = args.data_dir
    main_table = legacy.aggregate(runs)
    horizons = legacy.horizon_table(runs)
    daily_seed11 = legacy.paired_daily_tests(runs)
    seed_level = seed_tests(runs)
    day_blocks = block_tests(runs)
    graph, stability = legacy.graph_tables(runs)
    selection = validation_table()
    tables = {
        "main_results": main_table,
        "horizon_results": horizons,
        "daily_seed11_tests": daily_seed11,
        "seed_level_tests": seed_level,
        "day_block_bootstrap": day_blocks,
        "graph_results": graph,
        "graph_stability_full_precision": stability,
        "validation_candidates": selection,
    }
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.12g")

    legacy.plot_main(main_table)
    legacy.plot_normalized_metrics(main_table)
    legacy.plot_uniform_ranks(main_table)
    legacy.plot_horizons(horizons)
    legacy.plot_efficiency(main_table)
    legacy.plot_training(runs)
    legacy.plot_adjacency(runs)

    manifest = {
        "validation_run_count": 94,
        "locked_test_run_count": len(runs),
        "leakage_check": "passed for every locked-test record",
        "architecture_selection": json.loads(
            (ROUND2 / "architecture_selection.json").read_text(encoding="utf-8")
        ),
        "tables": [
            (TABLES / f"{name}.csv").relative_to(ROOT).as_posix()
            for name in tables
        ],
        "figures": [
            path.relative_to(ROOT).as_posix()
            for path in sorted(FIGURES.glob("*.pdf"))
        ],
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
