# make_results_tables.py
"""
Build the manuscript's results tables (Tables tab:india_results and tab:colombia_results)
from the comparison CSVs written by evaluate.py, and print the
summary numbers the Results text cites (averages, best-model counts, exceptions).

Usage
-----
    python make_results_tables.py

Reads results/{india,colombia}/comparison/performance_summary.csv and writes the
LaTeX tables to results/tables/results_india.tex and results/tables/results_colombia.tex.
"""

import argparse
import os

import pandas as pd
from scipy import stats

HORIZONS = [("1d", "1-Day Ahead"), ("3d", "3-Day Ahead"), ("7d", "7-Day Ahead")]
METRICS = [("MAE", "MAE↓", "{:.1f}", min), ("RMSE", "RMSE↓", "{:.1f}", min),
           ("R²", "R$^2$↑", "{:.3f}", max), ("Correlation", "Corr↑", "{:.3f}", max)]
MODELS = [("individual", "Individual"), ("joint", "Joint"), ("gnn", "GNN")]
LATEX_NAMES = {"Acandi": "Acand\\'{i}"}
ACCENTS = {"á": "\\'{a}", "é": "\\'{e}", "í": "\\'{i}", "ó": "\\'{o}", "ú": "\\'{u}", "ñ": "\\~{n}"}


def latex_name(city):
    if city in LATEX_NAMES:
        return LATEX_NAMES[city]
    return "".join(ACCENTS.get(c, c) for c in city)


def build_table(df, country, label):
    lines = [
        "\\begin{sidewaystable}[htbp]",
        "\\centering",
        f"\\caption{{Forecast performance for {country}. MAE and RMSE in W/m$^2$. Arrows indicate "
        "optimization direction: ↓ = lower is better, ↑ = higher is better. The best value at each site "
        "is in bold.}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrrrrrrrrrr}",
        "\\hline",
        "\\multirow{2}{*}{Location} & \\multirow{2}{*}{Model} &",
        " & ".join(f"\\multicolumn{{4}}{{c}}{{{title}}}" for _, title in HORIZONS) + " \\\\",
        "\\cline{3-6}\\cline{7-10}\\cline{11-14}",
        " & & " + " & ".join(m[1] for _ in HORIZONS for m in METRICS) + " \\\\",
        "\\hline",
    ]
    for city in sorted(df.City.unique(), key=lambda c: latex_name(c).replace("\\", "")):
        site = df[df.City == city].set_index("Model_Type")
        best = {}
        for h, _ in HORIZONS:
            for name, _, fmt, pick in METRICS:
                col = f"{name}_{h}"
                values = {m: float(fmt.format(site.loc[m, col])) for m, _ in MODELS}
                best[col] = pick(values.values())
        lines.append(f"\\multirow{{3}}{{*}}{{{latex_name(city)}}}")
        for m, title in MODELS:
            cells = []
            for h, _ in HORIZONS:
                for name, _, fmt, _ in METRICS:
                    col = f"{name}_{h}"
                    text = fmt.format(site.loc[m, col])
                    cells.append(f"\\textbf{{{text}}}" if float(text) == best[col] else text)
            lines.append(f" & {title:<11s} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{sidewaystable}", ""]
    return "\n".join(lines)


def sign_test(successes, n):
    """Two-sided sign (binomial, p = 0.5) test of how often a model beats the baseline."""
    if hasattr(stats, "binomtest"):
        return stats.binomtest(successes, n, 0.5).pvalue
    return stats.binom_test(successes, n, 0.5)


def summarize(df, country):
    print(f"\n=== {country} ===")
    print("Average across sites (W/m2):")
    for m, title in MODELS:
        d = df[df.Model_Type == m]
        print(f"  {title:<11s} MAE " + "  ".join(f"{h} {d[f'MAE_{h}'].mean():6.1f}" for h, _ in HORIZONS)
              + " | RMSE " + "  ".join(f"{h} {d[f'RMSE_{h}'].mean():6.1f}" for h, _ in HORIZONS))
    for name in ("MAE", "RMSE"):
        counts = []
        for h, _ in HORIZONS:
            best = df.pivot(index="City", columns="Model_Type", values=f"{name}_{h}").idxmin(axis=1)
            counts.append(f"{h}: " + ", ".join(f"{t} {int((best == m).sum())}" for m, t in MODELS))
        print(f"Lowest {name}, number of sites: " + " | ".join(counts))
    print("Sites where the joint LSTM / GNN is more accurate than the individual LSTM "
          "(two-sided sign test across sites):")
    for name in ("MAE", "RMSE"):
        for m, title in MODELS[1:]:
            parts = []
            for h, _ in HORIZONS:
                p = df.pivot(index="City", columns="Model_Type", values=f"{name}_{h}")
                better = int((p[m] < p["individual"]).sum())
                parts.append(f"{h} {better}/{len(p)} (p = {sign_test(better, len(p)):.3g})")
            print(f"  {name:<4s} {title:<6s} " + " | ".join(parts))
    print("Site-horizons where the GNN does not have the lowest MAE / RMSE:")
    for name in ("MAE", "RMSE"):
        for h, _ in HORIZONS:
            p = df.pivot(index="City", columns="Model_Type", values=f"{name}_{h}")
            for city in p.index[p.idxmin(axis=1) != "gnn"]:
                r = p.loc[city]
                print(f"  {name:<4s} {h} {city:<14s} individual {r['individual']:.1f}, joint {r['joint']:.1f}, gnn {r['gnn']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Build the manuscript's results tables")
    parser.parse_args()

    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    for country, country_dir, label, out in [("India", "india", "tab:india_results", "results_india.tex"),
                                             ("Colombia", "colombia", "tab:colombia_results", "results_colombia.tex")]:
        df = pd.read_csv(os.path.join("results", country_dir, "comparison", "performance_summary.csv"))
        missing = df[df[[f"MAE_{h}" for h, _ in HORIZONS]].isna().any(axis=1)]
        if len(missing):
            raise ValueError(f"{country}: missing metrics for {missing[['City', 'Model_Type']].values.tolist()}")
        path = os.path.join("results", "tables", out)
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_table(df, country, label))
        print(f"Wrote {path}")
        summarize(df, country)


if __name__ == "__main__":
    main()
