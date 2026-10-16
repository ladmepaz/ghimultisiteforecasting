"""
Plot the manuscript's results figures: for each country and metric (MAE, RMSE), one panel
per forecast horizon with the percent change of the joint LSTM's and the GNN's error
relative to the individual LSTM at each site (negative = more accurate than the individual
LSTM), plus the average across sites. Asterisks mark differences from the individual LSTM
that are significant in a Diebold-Mariano test after Holm correction (for "Mean", the
country-level test).

Reads results/{country}/comparison/performance_summary.csv (evaluate.py) and
results/{country}/significance/dm_tests.csv (significance_tests.py), and writes
results/figures/{MAE,RMSE}_{country}.png.

Usage:
  python plot_results.py                 # all four figures
  python plot_results.py --country india --metric MAE
"""

import argparse
import os
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODEL_ORDER = ["individual", "joint", "gnn"]
MODEL_COLORS = {"individual": "tab:blue", "joint": "tab:orange", "gnn": "tab:green"}
HORIZON_ORDER = ["1d", "3d", "7d"]


def _significant_sites(significance_path, metric, model):
    """{(city NFC, horizon)} where `model` differs significantly from the individual LSTM
    after Holm correction, for the loss matching `metric`. The country-level test is
    returned under the name "Mean"."""
    tests = pd.read_csv(significance_path)
    loss = "abs" if metric == "MAE" else "squared"
    result = "result_holm" if "result_holm" in tests.columns else "result"
    rows = tests[(tests.loss == loss) & (tests.model_a == model) & (tests.model_b == "individual")
                 & (tests[result] != "no significant difference")]
    return {(unicodedata.normalize("NFC", "Mean" if c == "ALL SITES" else c), h)
            for c, h in zip(rows.city, rows.horizon)}


def plot_relative(df, metric, output_path, significance_path=None):
    cities = sorted(df["City"].unique())
    base = df[df["Model_Type"] == "individual"].set_index("City").loc[cities]
    compared = [("joint", "Joint LSTM"), ("gnn", "GNN")]
    significant = ({m: _significant_sites(significance_path, metric, m) for m, _ in compared}
                   if significance_path else None)

    labels = cities + ["Mean"]
    x = np.arange(len(labels), dtype=float)
    x[-1] += 0.6  # set the average apart from the sites
    width = 0.38

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(len(HORIZON_ORDER), 1, figsize=(10, 8), sharex=True)
    for ax, horizon in zip(axes, HORIZON_ORDER):
        col = f"{metric}_{horizon}"
        for offset, (model, _) in zip((-width / 2, width / 2), compared):
            values = df[df["Model_Type"] == model].set_index("City").loc[cities, col]
            change = 100 * (values - base[col]) / base[col]
            heights = list(change) + [change.mean()]
            # Every bar is drawn in full color, since the sign is informative even where a
            # single site's difference is not significant; significant ones get an asterisk
            ax.bar(x + offset, heights, width, color=MODEL_COLORS[model])
            if significant is not None:
                for i, (city, height) in enumerate(zip(labels, heights)):
                    if (unicodedata.normalize("NFC", city), horizon) in significant[model]:
                        ax.annotate("*", (x[i] + offset, height), xytext=(0, 1 if height >= 0 else -1),
                                    textcoords="offset points", ha="center",
                                    va="bottom" if height >= 0 else "top", fontsize=13, fontweight="bold")
        ax.margins(y=0.12)  # room for the asterisks
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline((x[-2] + x[-1]) / 2, color="grey", linewidth=0.6, linestyle="--")
        ax.set_ylabel(f"{horizon[0]}-day ahead\n{metric} change (%)")
        ax.grid(True, axis="y", alpha=0.4)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    # The caption explains the figure, so no title; the legend sits above the top panel
    # so the plots can use the full width
    handles = [Patch(color=MODEL_COLORS[m], label=f"{name} vs individual LSTM") for m, name in compared]
    if significant is not None:
        handles.append(Line2D([], [], linestyle="None", marker="$*$", color="black", markersize=9,
                              label="significant (Diebold-Mariano, Holm)"))
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(handles), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot the per-site change in MAE/RMSE relative to the individual LSTM.")
    parser.add_argument("--country", choices=["india", "colombia"], help="Default: both")
    parser.add_argument("--metric", choices=["MAE", "RMSE"], help="Default: both")
    args = parser.parse_args()

    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    for country in [args.country] if args.country else ["india", "colombia"]:
        df = pd.read_csv(os.path.join("results", country, "comparison", "performance_summary.csv"))
        significance = os.path.join("results", country, "significance", "dm_tests.csv")
        for metric in [args.metric] if args.metric else ["MAE", "RMSE"]:
            plot_relative(df, metric, os.path.join("results", "figures", f"{metric}_{country}.png"),
                          significance if os.path.exists(significance) else None)


if __name__ == "__main__":
    main()
