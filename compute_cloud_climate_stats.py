"""
Compute the cloud-type and climate statistics per site used in the manuscript's Study
Area section. Both groups are restricted to daytime hours (Solar Zenith Angle < 90
degrees) unless noted:

  - Cloud-type shares: percentage of daytime hourly observations classified as
    Cloud Type 0 (clear) and Cloud Type 7 (thin/semi-transparent cirrus).
  - Seasonal humidity: average Relative Humidity during the monsoon months
    (June-September) versus the dry months (November-February), plus the overall
    average Temperature (all hours, not just daytime).

Usage:
  python compute_cloud_climate_stats.py [--is-colombia] [--city CITY] [--output PATH]
  (default output: results/tables/cloud_climate_{india,colombia}.csv)
"""

import argparse
import os
import pandas as pd

from utils import CONFIG, load_data

MONSOON_MONTHS = [6, 7, 8, 9]
DRY_MONTHS = [11, 12, 1, 2]


def compute_stats(df):
    """Compute cloud-type and seasonal climate statistics for one city's combined data."""
    daytime = df[df["Solar Zenith Angle"] < 90]
    n_day = len(daytime)

    pct_clear = 100 * (daytime["Cloud Type"] == 0).sum() / n_day
    pct_cirrus = 100 * (daytime["Cloud Type"] == 7).sum() / n_day

    avg_temperature = df["Temperature"].mean()
    rh_monsoon = df.loc[df["Month"].isin(MONSOON_MONTHS), "Relative Humidity"].mean()
    rh_dry = df.loc[df["Month"].isin(DRY_MONTHS), "Relative Humidity"].mean()

    return {
        "n_daytime_obs": n_day,
        "pct_clear": pct_clear,
        "pct_cirrus": pct_cirrus,
        "avg_temperature": avg_temperature,
        "avg_rh_monsoon_jun_sep": rh_monsoon,
        "avg_rh_dry_nov_feb": rh_dry,
    }


def run_for_location(locations, city, verbose=True):
    df = load_data(locations, city)
    stats = compute_stats(df)
    stats["location"] = city
    if verbose:
        print(f"\n--- {city} ---")
        print(f"Clear-sky (daytime):   {stats['pct_clear']:.1f}%")
        print(f"Cirrus (daytime):      {stats['pct_cirrus']:.1f}%")
        print(f"Avg temperature:       {stats['avg_temperature']:.1f} C")
        print(f"Avg RH, Jun-Sep:       {stats['avg_rh_monsoon_jun_sep']:.0f}%")
        print(f"Avg RH, Nov-Feb:       {stats['avg_rh_dry_nov_feb']:.0f}%")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute cloud-type shares and seasonal climate statistics per location."
    )
    parser.add_argument(
        "--is-colombia",
        action="store_true",
        help="Use Colombia locations; default is India.",
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Compute for a single city. If omitted, run for all locations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save the per-city summary CSV to this path (default: results/tables/cloud_climate_{country}.csv).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output.",
    )
    args = parser.parse_args()

    if args.is_colombia:
        locations = CONFIG["data_locations_colombia"]
        region = "Colombia"
    else:
        locations = CONFIG["data_locations"]
        region = "India"

    cities = [args.city] if args.city else list(locations.keys())
    verbose = not args.quiet

    if verbose:
        print(f"Region: {region}")
        print(f"Locations: {cities}")

    rows = []
    for city in cities:
        try:
            rows.append(run_for_location(locations, city, verbose=verbose))
        except Exception as e:
            print(f"Error for {city}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

    if not rows:
        print("No data processed.")
        return

    summary_df = pd.DataFrame(rows).set_index("location")

    if args.output is None and not args.city:
        os.makedirs(os.path.join("results", "tables"), exist_ok=True)
        args.output = os.path.join("results", "tables", f"cloud_climate_{region.lower()}.csv")
    if args.output:
        summary_df.to_csv(args.output)
        if verbose:
            print(f"\nSaved: {args.output}")

    if verbose:
        print("\n--- Summary ---")
        print(summary_df.to_string())
        print(f"\nRegion average clear-sky share: {summary_df['pct_clear'].mean():.1f}%")

    return summary_df


if __name__ == "__main__":
    main()
