"""
Shared utilities and configuration for GHI forecasting: site lists, data loading,
and the paths where models and results are stored.
"""

import os
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SITES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sites.csv")
YEARS = (2017, 2018, 2019)


def generate_config(sites_file=SITES_FILE):
    """Generate the configuration dictionary, with the sites of each country read from
    data/sites.csv in file order. The order matters: it fixes the order of the GNN's
    graph nodes and output heads."""
    sites = pd.read_csv(sites_file, encoding="utf-8")
    data_locations = {s: {y: None for y in YEARS} for s in sites[sites.country == "india"].site}
    data_locations_colombia = {s: {y: None for y in YEARS} for s in sites[sites.country == "colombia"].site}

    config = {
        "random_seed": 42,
        "data_locations": data_locations,
        "data_locations_colombia": data_locations_colombia,
    }
    return config

CONFIG = generate_config()


def country_name(is_colombia):
    """'colombia' or 'india', as used in the models/ and results/ paths."""
    return "colombia" if is_colombia else "india"


def results_path(country, *parts):
    """Path under results/{country}/, creating the parent directory."""
    path = os.path.join("results", country, *parts)
    os.makedirs(path if not os.path.splitext(path)[1] else os.path.dirname(path), exist_ok=True)
    return path


def model_path(country, *parts):
    """Path under models/{country}/, creating the parent directory."""
    path = os.path.join("models", country, *parts)
    os.makedirs(path if not os.path.splitext(path)[1] else os.path.dirname(path), exist_ok=True)
    return path


def normalize_name_for_path(name):
    """
    Normalize a location/city name for use in file paths so that names with
    accented characters (e.g. Coveñas, Chinchiná) match across different
    Unicode representations (NFC vs NFD). Use this when building model
    filenames and when checking for existing model files.
    """
    if not isinstance(name, str):
        return name
    return unicodedata.normalize("NFC", name)


def _normalize_city_equal(a, b):
    """True if two city names are equal when normalized (NFC or NFD)."""
    if a is None or b is None:
        return False
    a_nfc = unicodedata.normalize("NFC", str(a))
    b_nfc = unicodedata.normalize("NFC", str(b))
    return a_nfc == b_nfc


def _strip_accents(s):
    """Return string with combining characters removed (e.g. 'Coveñas' -> 'Covenas') for fallback matching."""
    if not isinstance(s, str):
        s = str(s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def resolve_individual_model_path(city, country):
    """
    Return the path to the individual model of `city`, models/{country}/individual/{city}.h5,
    matching accented names (e.g. Coveñas) across Unicode forms: tries the NFC and NFD
    file names, then an accent-insensitive search of the directory.
    """
    models_dir = os.path.join("models", country, "individual")
    if not isinstance(city, str):
        city = str(city)
    city_nfc = unicodedata.normalize("NFC", city)
    city_nfd = unicodedata.normalize("NFD", city)
    # Try direct paths with both normalizations (OS/filesystem may store either)
    for name in (city_nfc, city_nfd):
        path = os.path.join(models_dir, f"{name}.h5")
        if os.path.exists(path):
            return path
    if not os.path.isdir(models_dir):
        return os.path.join(models_dir, f"{city_nfc}.h5")
    city_ascii = _strip_accents(city)
    for f in os.listdir(models_dir):
        if not f.endswith(".h5"):
            continue
        candidate = f[:-len(".h5")]
        if _normalize_city_equal(candidate, city) or _strip_accents(candidate) == city_ascii:
            return os.path.join(models_dir, f)
    return os.path.join(models_dir, f"{city_nfc}.h5")


def resolve_data_file_path(city, year, data_dir="data"):
    """
    Return the path to data_{city}_{year}.csv, using the same Unicode
    normalization and directory search as resolve_individual_model_path
    so that accented names (e.g. Coveñas) find existing files.
    """
    if not isinstance(city, str):
        city = str(city)
    city_nfc = unicodedata.normalize("NFC", city)
    city_nfd = unicodedata.normalize("NFD", city)
    for name in (city_nfc, city_nfd):
        path = os.path.join(data_dir, f"data_{name}_{year}.csv")
        if os.path.exists(path):
            return path
    if not os.path.isdir(data_dir):
        return os.path.join(data_dir, f"data_{city_nfc}_{year}.csv")
    prefix = f"data_"
    suffix_s = f"_{year}.csv"
    city_ascii = _strip_accents(city)
    for f in os.listdir(data_dir):
        if not f.startswith(prefix) or not f.endswith(suffix_s):
            continue
        candidate = f[len(prefix) : -len(suffix_s)]
        if _normalize_city_equal(candidate, city) or _strip_accents(candidate) == city_ascii:
            return os.path.join(data_dir, f)
    return os.path.join(data_dir, f"data_{city_nfc}_{year}.csv")


def get_city_coordinates(locations, data_dir="data"):
    """
    Return {city: (latitude, longitude)} for every city in `locations` (a
    dict keyed by city name, as in CONFIG["data_locations"] or
    CONFIG["data_locations_colombia"]), read from the NSRDB metadata row
    (the second line) of any available yearly CSV for that city.
    """
    coordinates = {}
    for city in locations:
        for year in (2017, 2018, 2019):
            path = resolve_data_file_path(city, year, data_dir=data_dir)
            if os.path.exists(path):
                meta = pd.read_csv(path, nrows=1)
                coordinates[city] = (float(meta.loc[0, "Latitude"]), float(meta.loc[0, "Longitude"]))
                break
    return coordinates


def load_data(locations, city):
    """
    Load and preprocess the GHI data from multiple years for a specific city,
    from the NSRDB CSVs in data/. Timestamps are converted to UTC.
    """
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Loading data for {city}")
        print(f"{'='*60}")
        dfs = []
        
        for year in [2017, 2018, 2019]:
            try:
                local_file = resolve_data_file_path(city, year)
                print(f"\nProcessing {year} data...")
                
                # Data are read only from data/ (tracked in git). The Google Drive
                # copies are outdated, so a missing file is an error rather than
                # a silent download of an old version.
                if not (os.path.exists(local_file) and os.path.getsize(local_file) > 0):
                    raise FileNotFoundError(f"Data file not found or empty: {local_file} (run python data/download_nsrdb.py first)")
                print(f"✓ Found local file: {local_file} ({os.path.getsize(local_file)/1024:.2f} KB)")
                
                # Try reading the file
                print("Reading data file...")
                try:
                    # First check the file content
                    with open(local_file, 'r', encoding='utf-8') as f:
                        first_lines = [next(f) for _ in range(5)]
                    print("First few lines of the file:")
                    for i, line in enumerate(first_lines):
                        print(f"Line {i+1}: {line.strip()}")
                    
                    # Now try reading with pandas
                    df = pd.read_csv(local_file, skiprows=2)
                    
                    if len(df) == 0:
                        raise ValueError("File is empty after reading")
                    
                    # Validate data using comprehensive validation function
                    validation_result = validate_data(df, city_name=f"{city}_{year}", strict_mode=True)
                    if not validation_result['valid']:
                        raise ValueError(f"Data validation failed for {city} ({year}). See issues above.")
                    
                    # Offset of this file's timestamps from UTC, from the NSRDB
                    # metadata row. Files downloaded at different times use
                    # different conventions (UTC, or local time such as UTC+5
                    # for India and UTC-5 for Colombia).
                    df["utc_offset_hours"] = float(pd.read_csv(local_file, nrows=1).loc[0, "Time Zone"])

                    print(f"✓ Successfully loaded {len(df)} rows (timestamps at UTC{df['utc_offset_hours'].iloc[0]:+g})")
                    print(f"Columns: {df.columns.tolist()}")
                    dfs.append(df)
                    
                except Exception as e:
                    print(f"× Error reading file: {str(e)}")
                    raise

            except Exception as e:
                # A missing or unreadable year is fatal: skipping it would silently
                # train on fewer years than the other sites
                print(f"× Error processing {year}: {str(e)}")
                raise
        
        if not dfs:
            raise ValueError(f"No data was loaded successfully for {city}")
        
        # Combine all years
        print("\nCombining data from all years...")
        df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined dataset: {len(df)} total rows")
        
        # Process datetime
        print("\nProcessing datetime...")
        # Convert every file to UTC so all sites share one clock, and rewrite the
        # date components (used for the hour/month encodings) from the UTC time
        df["datetime"] = (pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
                          - pd.to_timedelta(df["utc_offset_hours"], unit="h"))
        df = df.drop(columns="utc_offset_hours")
        df["Year"] = df["datetime"].dt.year
        df["Month"] = df["datetime"].dt.month
        df["Day"] = df["datetime"].dt.day
        df["Hour"] = df["datetime"].dt.hour
        df["Minute"] = df["datetime"].dt.minute
        df = df.sort_values("datetime").reset_index(drop=True)
        # Add city name to the DataFrame
        df["City"] = city
        print("✓ Datetime processing complete")
        
        # Validate final dataset
        print("\nValidating final dataset:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        
        # Final comprehensive validation
        final_validation = validate_data(df, city_name=city, strict_mode=True)
        if not final_validation['valid']:
            raise ValueError(f"Final data validation failed for {city}. See issues above.")
        
        return df
        
    except Exception as e:
        print(f"\n× Error loading data for {city}:")
        print(str(e))
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise

def plot_results(y_true, y_pred, title="LSTM Model: Predicted vs. True GHI"):
    """Plot comparison of predicted vs true values."""
    plt.figure(figsize=(10, 5))
    # Plot only first 100 points for clarity
    x_axis = range(100)
    plt.plot(x_axis, y_true[:100], label="True Next-Day GHI", marker="o", linestyle="-")
    plt.plot(x_axis, y_pred[:100], label="Predicted Next-Day GHI", marker="s", linestyle="--")
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("GHI (W/m²)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_loss_history(history):
    """Plot training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    return plt.gcf()

def prepare_lstm_input(df, feature_columns, target_column):
    """Prepare data for LSTM input format."""
    X = df[feature_columns].values
    y = df[target_column].values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y

def validate_data(df, city_name="", strict_mode=True):
    """
    Validate data completeness and quality before processing.
    
    Args:
        df: DataFrame to validate
        city_name: Name of the city (for error messages)
        strict_mode: If True, raise errors for missing columns. If False, return warnings.
    
    Returns:
        dict: Validation results with 'valid' (bool) and 'issues' (list of strings)
    
    Raises:
        ValueError: If strict_mode=True and critical issues are found
    """
    issues = []
    warnings = []
    
    city_prefix = f"[{city_name}] " if city_name else ""
    
    # Required basic columns
    required_basic = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'GHI', 'Temperature']
    missing_basic = [col for col in required_basic if col not in df.columns]
    
    if missing_basic:
        error_msg = f"{city_prefix}Missing required basic columns: {missing_basic}"
        if strict_mode:
            raise ValueError(error_msg)
        issues.append(error_msg)
    
    # Required meteorological columns
    required_met = [
        "Relative Humidity", 
        "Pressure",
        "Precipitable Water", 
        "Wind Direction", 
        "Wind Speed"
    ]
    missing_met = [col for col in required_met if col not in df.columns]
    
    if missing_met:
        error_msg = f"{city_prefix}Missing required meteorological columns: {missing_met}"
        if strict_mode:
            raise ValueError(error_msg)
        issues.append(error_msg)
        warnings.append(f"{city_prefix}Model will work with reduced features, but performance may be affected")
    
    # Check for empty dataframe
    if len(df) == 0:
        error_msg = f"{city_prefix}DataFrame is empty"
        if strict_mode:
            raise ValueError(error_msg)
        issues.append(error_msg)
    
    # Check for datetime column (should be created after loading)
    if 'datetime' not in df.columns:
        # Check if we can create it
        if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour', 'Minute']):
            warnings.append(f"{city_prefix}datetime column not found, but can be created from date components")
        else:
            error_msg = f"{city_prefix}datetime column not found and cannot be created"
            if strict_mode:
                raise ValueError(error_msg)
            issues.append(error_msg)
    
    # Check data quality for key columns
    if 'GHI' in df.columns:
        # Check for reasonable GHI values (should be 0-1500 W/m² typically)
        invalid_ghi = df[(df['GHI'] < 0) | (df['GHI'] > 2000)]
        if len(invalid_ghi) > 0:
            warnings.append(f"{city_prefix}Found {len(invalid_ghi)} GHI values outside reasonable range [0, 2000] W/m²")
        
        # Check for excessive missing values
        ghi_missing_pct = df['GHI'].isna().sum() / len(df) * 100
        if ghi_missing_pct > 10:
            warnings.append(f"{city_prefix}GHI has {ghi_missing_pct:.1f}% missing values")
    
    if 'Temperature' in df.columns:
        # Check for reasonable temperature values (should be -50 to 60°C typically)
        invalid_temp = df[(df['Temperature'] < -50) | (df['Temperature'] > 60)]
        if len(invalid_temp) > 0:
            warnings.append(f"{city_prefix}Found {len(invalid_temp)} Temperature values outside reasonable range [-50, 60]°C")
    
    # Check for sufficient data points (need at least 24 hours for lag features)
    if len(df) < 24:
        error_msg = f"{city_prefix}Insufficient data: {len(df)} rows (need at least 24 for lag features)"
        if strict_mode:
            raise ValueError(error_msg)
        issues.append(error_msg)
    
    # Check date range consistency
    if all(col in df.columns for col in ['Year', 'Month', 'Day']):
        years = df['Year'].unique()
        if len(years) == 0:
            error_msg = f"{city_prefix}No valid years found in data"
            if strict_mode:
                raise ValueError(error_msg)
            issues.append(error_msg)
        elif len(years) < 3:
            warnings.append(f"{city_prefix}Only {len(years)} year(s) of data available: {years}")
    
    # Compile results
    is_valid = len(issues) == 0
    
    result = {
        'valid': is_valid,
        'issues': issues,
        'warnings': warnings,
        'missing_basic': missing_basic if 'missing_basic' in locals() else [],
        'missing_met': missing_met if 'missing_met' in locals() else []
    }
    
    # Print summary
    if issues:
        print(f"\n{city_prefix}VALIDATION ISSUES FOUND:")
        for issue in issues:
            print(f"  ✗ {issue}")
    
    if warnings:
        print(f"\n{city_prefix}VALIDATION WARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    if is_valid and not warnings:
        print(f"{city_prefix}✓ Data validation passed")
    
    return result

def evaluate_model(y_true, y_pred):
    """Calculate various metrics for model evaluation."""
    # Create mask for non-zero true GHI values
    non_zero_mask = y_true > 0
    
    # Filter out zero values
    y_true_nonzero = y_true[non_zero_mask]
    y_pred_nonzero = y_pred[non_zero_mask]
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate metrics for all values
    metrics.update({
        "Mean Absolute Error (All)": float(mean_absolute_error(y_true, y_pred)),
        "Mean Squared Error (All)": float(mean_squared_error(y_true, y_pred)),
        "Root Mean Squared Error (All)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R² Score (All)": float(r2_score(y_true, y_pred)),
        "Mean Absolute Percentage Error (All)": float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100),
        "Mean Squared Percentage Error (All)": float(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))) * 100)
    })
    
    # Calculate percentage of non-zero values
    metrics["Non-zero Values Percentage"] = float(100 * len(y_true_nonzero) / len(y_true))
    
    # Only calculate non-zero metrics if there are non-zero values
    if len(y_true_nonzero) > 0:
        metrics.update({
            "Mean Absolute Error (Non-zero)": float(mean_absolute_error(y_true_nonzero, y_pred_nonzero)),
            "Mean Squared Error (Non-zero)": float(mean_squared_error(y_true_nonzero, y_pred_nonzero)),
            "Root Mean Squared Error (Non-zero)": float(np.sqrt(mean_squared_error(y_true_nonzero, y_pred_nonzero))),
            "R² Score (Non-zero)": float(r2_score(y_true_nonzero, y_pred_nonzero)),
            "Mean Absolute Percentage Error (Non-zero)": float(np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / (y_true_nonzero + epsilon))) * 100),
            "Mean Squared Percentage Error (Non-zero)": float(np.mean(np.square((y_true_nonzero - y_pred_nonzero) / (y_true_nonzero + epsilon))) * 100)
        })
    else:
        # If no non-zero values, set these metrics to NaN
        metrics.update({
            "Mean Absolute Error (Non-zero)": float('nan'),
            "Mean Squared Error (Non-zero)": float('nan'),
            "Root Mean Squared Error (Non-zero)": float('nan'),
            "R² Score (Non-zero)": float('nan'),
            "Mean Absolute Percentage Error (Non-zero)": float('nan'),
            "Mean Squared Percentage Error (Non-zero)": float('nan')
        })
    
    return metrics 