"""
Plot the per-site cloud-type distribution heatmaps of the manuscript's Study Area
section ("Cloud type in Colombia/India"). Cells show the percentage of daytime hourly
observations (Solar Zenith Angle < 90 degrees) at each Cloud Type value, per site, the
same daytime restriction used by compute_cloud_climate_stats.py.

Usage:
  python plot_cloud_type_heatmap.py                 # India
  python plot_cloud_type_heatmap.py --is-colombia   # Colombia
  (default output: results/figures/cloud_type_{india,colombia}.png)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import CONFIG, load_data


def compute_distribution(locations, cities):
    rows = {}
    for city in cities:
        df = load_data(locations, city)
        daytime = df[df["Solar Zenith Angle"] < 90]
        pct = 100 * daytime["Cloud Type"].value_counts(normalize=True)
        rows[city] = pct
    dist = pd.DataFrame(rows).T.fillna(0.0)
    dist = dist[sorted(dist.columns, key=int)]
    return dist.round(1)


def main():
    parser = argparse.ArgumentParser(description="Plot per-city daytime cloud-type distribution heatmap.")
    parser.add_argument("--is-colombia", action="store_true", help="Use Colombia locations; default is India.")
    parser.add_argument("--output", default=None,
                        help="Path to save the PNG (default: results/figures/cloud_type_{country}.png)")
    parser.add_argument("--csv_output", default=None, help="Optional path to also save the underlying percentages as CSV")
    args = parser.parse_args()

    if args.is_colombia:
        locations = CONFIG["data_locations_colombia"]
    else:
        locations = CONFIG["data_locations"]
    cities = sorted(locations.keys())
    if args.output is None:
        country = "colombia" if args.is_colombia else "india"
        os.makedirs(os.path.join("results", "figures"), exist_ok=True)
        args.output = os.path.join("results", "figures", f"cloud_type_{country}.png")

    dist = compute_distribution(locations, cities)

    if args.csv_output:
        dist.to_csv(args.csv_output)
        print(f"Saved: {args.csv_output}")

    fig, ax = plt.subplots(figsize=(11, 8))
    values = dist.to_numpy()
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(dist.columns)))
    ax.set_xticklabels(dist.columns)
    ax.set_yticks(range(len(dist.index)))
    ax.set_yticklabels(dist.index)
    ax.set_xlabel("Cloud Type")
    ax.set_ylabel("City")

    vmax = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "white" if v > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
