"""Generate validation-sensitivity assets used by the expanded manuscript.

The input is the immutable validation-candidate summary produced by the
round-two pipeline. Test metrics are intentionally not read by this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "tables" / "round2" / "validation_candidates.csv"
HORIZONS = ROOT / "tables" / "round2" / "horizon_results.csv"
MANUSCRIPT = ROOT / ("round2_submission" if (ROOT / "round2_submission").exists() else "manuscript")
OUT = MANUSCRIPT / "figures"
OUT_TABLES = MANUSCRIPT / "tables"

LABELS = {
    "plus_decomp_off64": "decomposition off",
    "plus_decomp_fixed64": "fixed decomposition",
    "plus_resattn64": "residual attention, k=20",
    "plus_mandatory64": "mandatory attention",
    "plus_graph1_64": "one graph block",
    "plus_graph3_64": "three graph blocks",
    "plus_topk40_64": "residual attention, k=40",
    "plus_no_graph64": "graph disabled",
    "plus_identity_graph64": "identity graph",
    "plus_uniform_graph64": "uniform sparse graph",
    "plus_shuffled_graph64": "shuffled reference graph",
    "plus_scales1_64": "scale {1}",
    "plus_scales3_64": "scale {3}",
    "plus_scales13_64": "scales {1,3}",
    "plus_scales16_64": "scales {1,6}",
    "plus_scales36_64": "scales {3,6}",
}

ORDER = list(LABELS)


def main() -> None:
    df = pd.read_csv(TABLE)
    df = df[df["candidate"].isin(ORDER)].copy()
    df["candidate"] = pd.Categorical(df["candidate"], categories=ORDER, ordered=True)
    df = df.sort_values(["dataset", "candidate"])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8), sharey=True)
    for ax, dataset in zip(axes, ["METR-LA", "PeMS-BAY"]):
        part = df[df["dataset"] == dataset].copy()
        best = part["validation_MAE_mean"].min()
        part["relative"] = 100.0 * (part["validation_MAE_mean"] / best - 1.0)
        y = range(len(part))
        colors = ["#2155a5" if n == 3 else "#8aa9d6" for n in part["n_seeds"]]
        ax.barh(list(y), part["relative"], color=colors, edgecolor="white")
        ax.set_title(dataset)
        ax.set_xlabel("Validation MAE above best candidate (%)")
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
        if ax is axes[0]:
            ax.set_yticks(list(y), [LABELS[str(c)] for c in part["candidate"]], fontsize=8.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
        for yi, (_, row) in enumerate(part.iterrows()):
            ax.text(row["relative"] + 0.08, yi, f'n={int(row["n_seeds"])}', va="center", fontsize=7.5)

    fig.suptitle("Validation-only architectural sensitivity")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "validation_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "validation_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    horizons = pd.read_csv(HORIZONS)
    horizons = horizons[horizons["model"].isin(["astafr_plus", "stid"])].copy()
    table_path = OUT_TABLES / "full_horizons.tex"
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Horizon & ASTAF-R MAE & STID-style MAE & Difference\\",
        r"\midrule",
    ]
    for dataset in ["METR-LA", "PeMS-BAY"]:
        part = horizons[horizons["dataset"] == dataset]
        ast = part[part["model"] == "astafr_plus"].set_index("horizon")
        stid = part[part["model"] == "stid"].set_index("horizon")
        for horizon in range(1, 13):
            a = ast.loc[horizon]
            s = stid.loc[horizon]
            dataset_cell = dataset if horizon == 1 else ""
            diff = float(a["MAE_mean"] - s["MAE_mean"])
            lines.append(
                f"{dataset_cell} & {horizon} ({int(a['minutes'])} min) & "
                f"{a['MAE_mean']:.3f}$\\pm${a['MAE_sd']:.3f} & "
                f"{s['MAE_mean']:.3f}$\\pm${s['MAE_sd']:.3f} & {diff:.3f}\\\\"
            )
        if dataset == "METR-LA":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    horizon_all = pd.read_csv(HORIZONS)
    model_order = [
        ("astafr_plus", "ASTAF-R"),
        ("stid", "STID"),
        ("dcrnn", "DCRNN"),
        ("graphwavenet", "GWN"),
        ("agcrn", "AGCRN"),
        ("stgcn", "STGCN"),
        ("transformer", "Trans."),
        ("astaf", "ASTAF"),
        ("persistence", "Persist."),
    ]
    for dataset, suffix in [("METR-LA", "metr"), ("PeMS-BAY", "pems")]:
        part = horizon_all[horizon_all["dataset"] == dataset]
        by_model = {
            model: part[part["model"] == model].set_index("horizon")
            for model, _ in model_order
        }
        all_lines = [
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            "Horizon & " + " & ".join(label for _, label in model_order) + r"\\",
            r"\midrule",
        ]
        for horizon in range(1, 13):
            minutes = int(by_model["astafr_plus"].loc[horizon, "minutes"])
            values = [
                f"{by_model[model].loc[horizon, 'MAE_mean']:.3f}"
                for model, _ in model_order
            ]
            all_lines.append(f"{horizon} ({minutes} min) & " + " & ".join(values) + r"\\")
        all_lines.extend([r"\bottomrule", r"\end{tabular}"])
        (OUT_TABLES / f"all_models_horizon_{suffix}.tex").write_text(
            "\n".join(all_lines) + "\n", encoding="utf-8"
        )

    all_candidates = pd.read_csv(TABLE)
    for dataset, suffix in [("METR-LA", "metr"), ("PeMS-BAY", "pems")]:
        part = all_candidates[all_candidates["dataset"] == dataset].sort_values(
            "validation_MAE_mean"
        )
        candidate_lines = [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Candidate identifier & Seeds & Validation MAE (mph)\\",
            r"\midrule",
        ]
        for _, row in part.iterrows():
            code = str(row["candidate"]).replace("_", r"\_")
            if int(row["n_seeds"]) > 1:
                metric = f"{row['validation_MAE_mean']:.3f}$\\pm${row['validation_MAE_sd']:.3f}"
            else:
                metric = f"{row['validation_MAE_mean']:.3f}"
            candidate_lines.append(
                rf"\texttt{{{code}}} & {int(row['n_seeds'])} & {metric}\\"
            )
        candidate_lines.extend([r"\bottomrule", r"\end{tabular}"])
        (OUT_TABLES / f"validation_all_{suffix}.tex").write_text(
            "\n".join(candidate_lines) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
