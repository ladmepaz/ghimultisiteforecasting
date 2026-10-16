# significance_tests.py
"""
Diebold-Mariano tests of whether the differences in forecast accuracy between the
Individual LSTM, Joint LSTM and GNN models are statistically significant.

Inputs are the hourly test predictions each model's evaluation writes, all scored on
the same test hours (see evaluate.py):
    results/{india,colombia}/individual/{city}/test_predictions.csv
    results/{india,colombia}/joint/{city}/test_predictions.csv
    results/{india,colombia}/gnn/{city}/test_predictions.csv

Method
------
For each site, horizon (1, 3, 7 days) and pair of models A vs B, the loss differential
is d_t = L(e_A,t) - L(e_B,t), with L the absolute error (MAE) or squared error (MSE,
the RMSE comparison). Hourly errors are strongly autocorrelated within a day, so d_t
is averaged per calendar day and the test is run on the daily series. An h-day-ahead
forecast has overlapping errors over h-1 days, so the long-run variance uses
autocovariances up to lag h-1 (Diebold & Mariano, 1995), with the Harvey, Leybourne
and Newbold (1997) small-sample correction and a t(n-1) reference distribution.

A negative mean differential means model A is more accurate. A country-level test
averages d_t across sites for each day before testing. p-values are also reported after
a Holm correction for multiple testing (p_holm): across the 18 country-level tests, and
across sites for each pair, horizon and loss.

Usage
-----
    python significance_tests.py                 # India
    python significance_tests.py --is-colombia   # Colombia

Output: results/{india,colombia}/significance/dm_tests.csv (one row per site, horizon, loss
and pair, plus the country-level rows with city = "ALL SITES").
"""

import argparse
import os
import unicodedata
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from utils import CONFIG, country_name, results_path

HORIZONS = {"1d": 1, "3d": 3, "7d": 7}
PAIRS = [("gnn", "individual"), ("gnn", "joint"), ("joint", "individual")]
LOSSES = {"abs": lambda e: np.abs(e), "squared": lambda e: e ** 2}
ALPHA = 0.05


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def load_predictions(root, city):
    """Read root/city/test_predictions.csv, matching the city name across Unicode forms."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found - run evaluate.py first")
    for name in os.listdir(root):
        if nfc(name) == nfc(city):
            path = os.path.join(root, name, "test_predictions.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["datetime"])
                return df.set_index(pd.DatetimeIndex(df["datetime"]).floor("h")).drop(columns="datetime")
    raise FileNotFoundError(f"No test_predictions.csv for {city} in {root}")


def dm_test(d, h):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.
    d: loss differential series; h: forecast horizon in the units of d (days)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    d_bar = d.mean()
    dc = d - d_bar
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    long_run_var = gamma[0] + 2 * sum(gamma[1:])
    if long_run_var <= 0:
        return d_bar, np.nan, np.nan
    dm = d_bar / np.sqrt(long_run_var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return d_bar, dm, p


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (NaN entries are left as NaN)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def verdict(mean_diff, p, a, b):
    if np.isnan(p) or p >= ALPHA:
        return "no significant difference"
    return f"{a} better" if mean_diff < 0 else f"{b} better"


def main():
    parser = argparse.ArgumentParser(description="Diebold-Mariano tests between the forecasting models")
    parser.add_argument("--is-colombia", action="store_true", help="Test the Colombia models (default: India)")
    args = parser.parse_args()

    country_dir = country_name(args.is_colombia)
    roots = {m: os.path.join("results", country_dir, m) for m in ("individual", "joint", "gnn")}
    cities = list(CONFIG["data_locations_colombia" if args.is_colombia else "data_locations"].keys())
    country = "Colombia" if args.is_colombia else "India"

    rows = []
    daily = {}  # (horizon, loss, pair) -> list of per-site daily differential series
    for city in cities:
        preds = {m: load_predictions(root, city) for m, root in roots.items()}
        common = preds["gnn"].index
        for m in ("individual", "joint"):
            common = common.intersection(preds[m].index)
        if any(len(p.index.difference(common)) for p in preds.values()):
            counts = {m: len(p) for m, p in preds.items()}
            print(f"WARNING: {city}: test hours differ between models {counts}; using the {len(common)} shared hours")
        preds = {m: p.loc[common] for m, p in preds.items()}

        actual = preds["gnn"][[f"actual_{h}" for h in HORIZONS]]
        for m in ("individual", "joint"):
            diff = np.abs(preds[m][[f"actual_{h}" for h in HORIZONS]].values - actual.values).max()
            if diff > 1e-3:
                print(f"WARNING: {city}: {m} actual values differ from the GNN's by up to {diff:.3f} W/m2")

        for (h, h_days), (loss_name, loss), (a, b) in product(HORIZONS.items(), LOSSES.items(), PAIRS):
            y = preds["gnn"][f"actual_{h}"]
            d = loss(preds[a][f"predicted_{h}"] - y) - loss(preds[b][f"predicted_{h}"] - y)
            d_daily = d.groupby(d.index.date).mean()
            daily.setdefault((h, loss_name, a, b), []).append(d_daily.rename(city))
            mean_diff, dm, p = dm_test(d_daily.values, h_days)
            rows.append({"country": country, "city": city, "horizon": h, "loss": loss_name,
                         "model_a": a, "model_b": b, "mean_loss_diff": mean_diff, "dm_stat": dm,
                         "p_value": p, "n_days": len(d_daily), "n_hours": len(d),
                         "result": verdict(mean_diff, p, a, b)})

    # Country level: average the daily differentials across sites
    for (h, loss_name, a, b), series in daily.items():
        d_daily = pd.concat(series, axis=1).mean(axis=1)
        mean_diff, dm, p = dm_test(d_daily.values, HORIZONS[h])
        rows.append({"country": country, "city": "ALL SITES", "horizon": h, "loss": loss_name,
                     "model_a": a, "model_b": b, "mean_loss_diff": mean_diff, "dm_stat": dm,
                     "p_value": p, "n_days": len(d_daily), "n_hours": np.nan,
                     "result": verdict(mean_diff, p, a, b)})

    out = pd.DataFrame(rows)

    # Holm correction for multiple testing. Country level: one family of 18 tests
    # (3 pairs x 3 horizons x 2 losses). Site level: one family per pair, horizon and
    # loss, across the sites (the "at how many sites" count).
    out["p_holm"] = np.nan
    pooled_rows = out.city == "ALL SITES"
    out.loc[pooled_rows, "p_holm"] = holm(out.loc[pooled_rows, "p_value"])
    for _, idx in out[~pooled_rows].groupby(["horizon", "loss", "model_a", "model_b"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_value"])
    out["result_holm"] = [verdict(d, p, a, b) for d, p, a, b in
                          zip(out.mean_loss_diff, out.p_holm, out.model_a, out.model_b)]

    out_dir = results_path(country_dir, "significance")
    out.to_csv(os.path.join(out_dir, "dm_tests.csv"), index=False)

    sites = out[out.city != "ALL SITES"]
    for label, col in [("unadjusted p-values", "result"), ("Holm-adjusted p-values", "result_holm")]:
        p_col = "p_value" if col == "result" else "p_holm"
        print(f"\nDiebold-Mariano tests, {country} (alpha = {ALPHA}, {label}); counts of sites")
        for loss_name in LOSSES:
            print(f"\nLoss: {'absolute error (MAE)' if loss_name == 'abs' else 'squared error (RMSE)'}")
            for a, b in PAIRS:
                for h in HORIZONS:
                    s = sites[(sites.loss == loss_name) & (sites.model_a == a) & (sites.model_b == b) & (sites.horizon == h)]
                    c = s[col].value_counts()
                    pooled = out[(out.city == "ALL SITES") & (out.loss == loss_name) & (out.model_a == a)
                                 & (out.model_b == b) & (out.horizon == h)].iloc[0]
                    print(f"  {a:>5s} vs {b:<10s} {h}: {a} better {c.get(f'{a} better', 0):2d}, "
                          f"{b} better {c.get(f'{b} better', 0):2d}, n.s. {c.get('no significant difference', 0):2d} | "
                          f"all sites: {pooled[col]} (p = {pooled[p_col]:.3g})")
    print(f"\nSaved {os.path.join(out_dir, 'dm_tests.csv')}")


if __name__ == "__main__":
    main()
