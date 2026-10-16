# Data

The data are not stored in this repository. `download_nsrdb.py` downloads them from the
NREL National Solar Radiation Database (NSRDB) on NREL's public AWS S3 archive, which needs
no API key or AWS account.

```bash
python data/download_nsrdb.py                                 # all sites, 2017-2019
python data/download_nsrdb.py --country india                 # one country
python data/download_nsrdb.py --sites Cali Bangalore --years 2018
python data/download_nsrdb.py --workers 16                    # more parallel downloads
```

Each site-year is written to `data/data_{site}_{year}.csv`. Files that are already present
are skipped, so an interrupted download can be restarted with the same command. The yearly
NSRDB files are about 1 TB each, but only the bytes of each site's columns are read (with
HTTP range requests). A full download takes a few hours, depending on the connection.

## Sites

`sites.csv` lists the 33 sites with their NSRDB grid point (`nsrdb_gid`, the Location ID),
coordinates and elevation. The downloader checks that each grid point is at the listed
coordinates. The order of the rows fixes the order of the GNN's nodes.

## Sources

| Country | NSRDB product | Native resolution | S3 files |
|---|---|---|---|
| Colombia | Physical Solar Model (PSM) v3.2.2, GOES | 30 min | `GOES/aggregated/v3.2.2/nsrdb_{year}.h5` |
| India | Meteosat Indian Ocean Data Coverage (IODC) | 15 min | `meteosat/meteosat_{irradiance,pv,csp,ancillary_b,clouds}_{year}.h5` |

Bucket: `https://nrel-pds-nsrdb.s3.amazonaws.com`. Data are kept at hourly resolution by
selecting the :30 UTC time steps.

## Format

Each CSV follows the NSRDB download format: a metadata header and row (Location ID,
latitude, longitude, elevation, and `Time Zone = 0`, since timestamps are in UTC), then
one row per hour with these columns:

| Column | Unit |
|---|---|
| Year, Month, Day, Hour, Minute | UTC |
| GHI | W/m² |
| Temperature | °C |
| Relative Humidity | % |
| Pressure | mbar |
| Precipitable Water | cm |
| Wind Direction | degrees |
| Wind Speed | m/s |
| Cloud Type | NSRDB cloud-type code (0 = clear, 7 = cirrus, …) |
| Solar Zenith Angle | degrees |

NSRDB data are provided by the National Renewable Energy Laboratory (Sengupta et al.,
2018, *The National Solar Radiation Data Base (NSRDB)*, Renewable and Sustainable Energy
Reviews 89, 51–60).
