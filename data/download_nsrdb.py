"""
Download the NSRDB data used in the study from NREL's public archive on AWS S3.

For every site in data/sites.csv and every year (2017-2019), this script reads the
site's grid point (its NSRDB Location ID) from the yearly HDF5 files and writes
data/data_{site}_{year}.csv, the format utils.load_data() reads. No API key or AWS
account is needed: only the bytes for each site's columns are fetched, using HTTP
range requests, from files that are about 1 TB each.

Sources
-------
- Colombia: NSRDB Physical Solar Model (PSM) v3.2.2, from GOES satellites, 30-minute data
  https://nrel-pds-nsrdb.s3.amazonaws.com/GOES/aggregated/v3.2.2/nsrdb_{year}.h5
- India: NSRDB Meteosat Indian Ocean Data Coverage (IODC), 15-minute data, with the
  variables split across files
  https://nrel-pds-nsrdb.s3.amazonaws.com/meteosat/meteosat_{group}_{year}.h5

The files are kept at hourly resolution by selecting the :30 UTC time steps, and the
timestamps are written in UTC (Time Zone = 0 in the metadata row).

Usage
-----
    python data/download_nsrdb.py                          # all sites, 2017-2019
    python data/download_nsrdb.py --country colombia
    python data/download_nsrdb.py --sites Cali Bangalore --years 2018
    python data/download_nsrdb.py --workers 16             # more parallel downloads

Site-years already present in data/ are skipped, so an interrupted run can simply be
restarted. Downloading everything takes several hours, depending on the connection.
"""

import argparse
import csv
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SITES_FILE = os.path.join(DATA_DIR, "sites.csv")
YEARS = (2017, 2018, 2019)
BUCKET = "https://nrel-pds-nsrdb.s3.amazonaws.com"

# Output column -> NSRDB dataset name
VARIABLES = [
    ("GHI", "ghi"),
    ("Temperature", "air_temperature"),
    ("Relative Humidity", "relative_humidity"),
    ("Pressure", "surface_pressure"),
    ("Precipitable Water", "total_precipitable_water"),
    ("Wind Direction", "wind_direction"),
    ("Wind Speed", "wind_speed"),
    ("Cloud Type", "cloud_type"),
    ("Solar Zenith Angle", "solar_zenith_angle"),
]
INTEGER_COLUMNS = {"Cloud Type"}
TIME_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute"]

# The Meteosat (India) archive splits the variables into groups of files
MSG_GROUPS = {
    "ghi": "irradiance",
    "air_temperature": "pv",
    "wind_direction": "pv",
    "wind_speed": "pv",
    "relative_humidity": "csp",
    "surface_pressure": "csp",
    "total_precipitable_water": "ancillary_b",
    "solar_zenith_angle": "ancillary_b",
    "cloud_type": "clouds",
}


def source_url(country, dataset, year):
    """URL of the HDF5 file holding `dataset` for `country` and `year`."""
    if country == "colombia":
        return f"{BUCKET}/GOES/aggregated/v3.2.2/nsrdb_{year}.h5"
    if country == "india":
        return f"{BUCKET}/meteosat/meteosat_{MSG_GROUPS[dataset]}_{year}.h5"
    raise ValueError(f"Unknown country: {country}")


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, with a small block cache."""

    def __init__(self, url, block=1 << 20, retries=5):
        self.url, self.block, self.retries = url, block, retries
        self.pos, self.cache = 0, {}
        self.session = requests.Session()
        self.size = int(self.session.head(url, timeout=60).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            start = i * self.block
            end = min(start + self.block, self.size) - 1
            for attempt in range(self.retries):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == self.retries - 1:
                        raise
                    time.sleep(2 ** attempt)
            if len(self.cache) > 256:
                self.cache.clear()
            self.cache[i] = r.content
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        out, got = memoryview(b), 0
        while got < n:
            i, off = divmod(self.pos + got, self.block)
            data = self._block(i)[off:off + n - got]
            out[got:got + len(data)] = data
            got += len(data)
        self.pos += n
        return n


def read_column(job):
    """Return (column, time_index, physical values) of one dataset at one grid point."""
    column, url, dataset, gid = job
    with h5py.File(HTTPRangeFile(url), "r") as h:
        ds = h[dataset]
        values = ds[:, gid].astype(np.float64) / float(ds.attrs.get("psm_scale_factor", 1.0))
        time_index = pd.to_datetime(h["time_index"][:].astype(str)).tz_localize(None)
    return column, time_index, values


def read_meta(url, gid):
    with h5py.File(HTTPRangeFile(url), "r") as h:
        row = h["meta"][gid]
    return {k: (row[k].decode() if isinstance(row[k], bytes) else row[k]) for k in row.dtype.names}


def output_path(site, year):
    return os.path.join(DATA_DIR, f"data_{site}_{year}.csv")


def is_complete(path):
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 3 >= 8760


def download_site_year(site, year, pool):
    country, gid = site["country"], int(site["nsrdb_gid"])
    meta = read_meta(source_url(country, "ghi", year), gid)
    if (abs(float(meta["latitude"]) - float(site["latitude"])) > 0.011
            or abs(float(meta["longitude"]) - float(site["longitude"])) > 0.011):
        raise ValueError(f"{site['site']}: grid point {gid} is at ({meta['latitude']:.2f}, {meta['longitude']:.2f}), "
                         f"not at the coordinates in sites.csv ({site['latitude']}, {site['longitude']})")

    jobs = [(column, source_url(country, dataset, year), dataset, gid) for column, dataset in VARIABLES]
    data, time_index = {}, None
    for column, ti, values in pool.map(read_column, jobs):
        data[column] = values
        time_index = ti

    df = pd.DataFrame(data, index=time_index)
    df = df[df.index.minute == 30]  # hourly, at :30 UTC
    for name in TIME_COLUMNS:
        df[name] = getattr(df.index, name.lower())
    for column in INTEGER_COLUMNS:
        df[column] = df[column].round().astype(int)
    value_columns = [c for c, _ in VARIABLES if c not in INTEGER_COLUMNS]
    df[value_columns] = df[value_columns].round(4)
    df = df[TIME_COLUMNS + [c for c, _ in VARIABLES]]

    path = output_path(site["site"], year)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Location ID", "City", "State", "Country", "Latitude", "Longitude",
                         "Time Zone", "Elevation", "Local Time Zone"])
        writer.writerow(["NSRDB", gid, site["site"], meta.get("state", "-"), meta.get("country", "-"),
                         round(float(meta["latitude"]), 2), round(float(meta["longitude"]), 2),
                         0, meta["elevation"], meta.get("timezone", "-")])
        df.to_csv(f, index=False)
    os.replace(tmp, path)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Download the study's NSRDB data from NREL's public S3 archive")
    parser.add_argument("--country", choices=["india", "colombia"], help="Default: both")
    parser.add_argument("--sites", nargs="+", help="Subset of sites (names as in data/sites.csv)")
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (default: 8)")
    args = parser.parse_args()

    with open(SITES_FILE, encoding="utf-8") as f:
        sites = list(csv.DictReader(f))
    if args.country:
        sites = [s for s in sites if s["country"] == args.country]
    if args.sites:
        unknown = set(args.sites) - {s["site"] for s in sites}
        if unknown:
            raise SystemExit(f"Not in data/sites.csv: {sorted(unknown)}")
        sites = [s for s in sites if s["site"] in args.sites]

    todo = [(s, y) for s in sites for y in args.years if not is_complete(output_path(s["site"], y))]
    print(f"{len(todo)} site-year(s) to download ({len(sites) * len(args.years) - len(todo)} already present)")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (site, year) in enumerate(todo, start=1):
            start = time.time()
            n = download_site_year(site, year, pool)
            print(f"[{k}/{len(todo)}] {site['country']:<8s} {site['site']:<14s} {year}: "
                  f"{n} hours ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
