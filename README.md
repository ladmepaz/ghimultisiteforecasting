# Multi-site GHI forecasting

Code to reproduce the experiments of

> J. F. Ruiz-Munoz, L. S. Hoyos-Gómez, V. A. Tikkiwal. *Beyond Single-Site Models:
> Multi-Location Data Improves GHI Forecasting Across Contrasting Climates.*
> Manuscript under review.

The study forecasts hourly Global Horizontal Irradiance (GHI) 1, 3 and 7 days ahead at
17 sites in India and 16 sites in Colombia, and compares three models:

| Model | Script | Description |
|---|---|---|
| Individual LSTM | `train_individual.py` | One LSTM per site, trained on that site's data only |
| Joint LSTM | `train_joint.py` | One LSTM trained on all sites of a country, with a one-hot site ID |
| GNN | `train_gnn.py` | Graph convolutional network over all sites of a country, with edges weighted by geographic (Haversine) distance |

Both LSTMs share the same architecture (three LSTM layers with a residual connection, an
attention readout and a dense head). All models use the previous 24 hours of GHI, lagged
meteorological variables and cyclical time features, and are split 70/15/15 by day into
training, validation and test sets. The LSTMs are scored on the GNN's test hours, so all
three models are compared on exactly the same observations, and the differences between
them are tested with Diebold-Mariano tests corrected for multiple testing (Holm).

Main findings:

- In both countries, models that learn from multiple sites outperform the site-specific
  LSTMs at most sites, with an average MAE up to 6.9% lower.
- In clearer-skied India, pooling the data of all sites (joint LSTM) captures most of the
  gain.
- In cloudier Colombia, the GNN is the most accurate model at most sites and significantly
  outperforms the joint LSTM at the 3- and 7-day horizons.

## Setup

Python 3.8 with TensorFlow 2.6 and Spektral 1.3:

```bash
conda create -n ghimultisite python=3.8.13
conda activate ghimultisite
pip install -r requirements.txt
```

A GPU is recommended for training.

## Data

The data come from the NREL National Solar Radiation Database (NSRDB), 2017–2019, and are
downloaded from NREL's public archive on AWS S3 (no API key needed):

```bash
python data/download_nsrdb.py
```

This writes one CSV per site and year to `data/` (99 files). See [data/README.md](data/README.md)
for the sources, variables and options (for example, downloading a single country).

## Reproducing the experiments

Run every step for both countries with one command (stops at the first failure):

```bash
bash run_all.sh                  # Linux / macOS
powershell -ExecutionPolicy Bypass -File run_all.ps1   # Windows
```

or step by step, adding `--is-colombia` to run a step for Colombia instead of India:

```bash
# 1. Train the three models
python train_individual.py [--is-colombia]
python train_joint.py [--is-colombia]
python train_gnn.py [--is-colombia]          # resumes from its checkpoint, --fresh to restart

# 2. Evaluate all models on the same test hours
python evaluate.py [--is-colombia]

# 3. Diebold-Mariano tests between the models
python significance_tests.py [--is-colombia]

# 4. Tables and figures for both countries
python make_results_tables.py
python plot_results.py

# 5. Study Area statistics and figures
python compute_cloud_climate_stats.py [--is-colombia]
python plot_cloud_type_heatmap.py [--is-colombia]
```

Trained models are saved to `models/{india,colombia}/` and all outputs to `results/`:

| Output | Manuscript |
|---|---|
| `results/tables/results_india.tex`, `results_colombia.tex` | Results tables for India and Colombia |
| `results/figures/MAE_india.png`, `RMSE_india.png` | MAE and RMSE relative to the individual LSTM, India |
| `results/figures/MAE_colombia.png`, `RMSE_colombia.png` | MAE and RMSE relative to the individual LSTM, Colombia |
| `results/{country}/significance/dm_tests.csv` | Significance tests (asterisks in the figures) |
| `results/figures/cloud_type_{india,colombia}.png` | Cloud-type distribution figures (Study Area) |
| `results/tables/cloud_climate_{india,colombia}.csv` | Cloud-type and climate statistics (Study Area) |
| `results/{country}/comparison/performance_summary.csv` | All metrics per site, model and horizon |
| `results/{country}/{individual,joint,gnn}/{site}/test_predictions.csv` | Hourly test predictions of each model |

## Notes

- Timestamps are converted to UTC when the data are loaded, so the sites of a country
  share the same hours.
- The downloader writes the files in UTC. Some of the files used for the manuscript were
  downloaded in local time (UTC+5 in India, UTC-5 in Colombia), so a fresh download has the same values but
  starts and ends each year a few hours apart from them, which can change the results
  slightly.
- The order of the sites in `data/sites.csv` fixes the order of the GNN's nodes, so keep it
  unchanged to reproduce the results.
- Random seeds are fixed, but results can still differ slightly between runs and hardware
  because of non-deterministic GPU operations.

## License

MIT, see [LICENSE](LICENSE).
