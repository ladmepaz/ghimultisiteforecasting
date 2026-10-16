import os
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from utils import (CONFIG, load_data, resolve_individual_model_path, normalize_name_for_path,
                   country_name, results_path, model_path as model_file_path)

# Custom InputLayer that accepts legacy 'batch_shape' config (for loading models saved with older Keras)
class _InputLayerCompat(tf.keras.layers.InputLayer):
    """InputLayer subclass that accepts batch_shape in from_config for backward compatibility."""
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        batch_shape = config.pop("batch_shape", None)
        if batch_shape is not None:
            # New API uses input_shape (shape without batch) and optional batch_size
            config["input_shape"] = tuple(batch_shape[1:]) if len(batch_shape) > 1 else (batch_shape[0],)
            if batch_shape[0] is not None:
                config["batch_size"] = batch_shape[0]
        return super(tf.keras.layers.InputLayer, cls).from_config(config)

# BatchNormalization compat: strip 'synchronized' and other unknown kwargs (saved with older Keras)
class _BatchNormCompat(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("synchronized", None)
        config.pop("virtual_batch_size", None)  # sometimes present in older configs
        return super(tf.keras.layers.BatchNormalization, cls).from_config(config)


def _build_joint_model_architecture(input_shape, n_horizons=3):
    """Build joint LSTM model architecture (same as train_joint.create_joint_model).
    Used for load_weights fallback when full model load fails; avoids importing train_joint.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    lstm2 = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm2 = tf.keras.layers.Dropout(0.3)(lstm2)
    x = tf.keras.layers.Add()([x, lstm2])
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(64)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    x = tf.keras.layers.Multiply()([x, attention_weights])
    x = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
    dense1 = tf.keras.layers.Dense(32, activation='relu')(x)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.3)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    x = tf.keras.layers.Add()([dense1, dense2])
    outputs = tf.keras.layers.Dense(n_horizons, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Three time horizons: 1d (24h), 3d (72h), 7d (168h) at same hour
HORIZONS = ["1d", "3d", "7d"]
TARGET_COLUMNS = ["target_GHI_1d", "target_GHI_3d", "target_GHI_7d"]

def create_features(df):
    """Create features for prediction."""
    # Create time features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Create GHI lag features
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    
    # Create meteorological lag features - only use columns that exist
    met_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    # Filter to only include variables that exist in the dataframe
    available_met_features = [feature for feature in met_features if feature in df.columns]
    missing_met_features = [feature for feature in met_features if feature not in df.columns]
    
    if missing_met_features:
        print(f"WARNING: Missing meteorological features: {missing_met_features}")
        print(f"Using available features: {available_met_features}")
    
    for feature in available_met_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    
    return df.dropna().reset_index(drop=True)

def create_sequences_joint(df, sequence_length, input_config):
    """Create sequences for joint models based on input configuration."""
    # Initialize feature columns based on configuration
    feature_columns = []
    
    if input_config in ['ghi_only', 'ghi_met']:
        # Add GHI lag features (24 features)
        feature_columns.extend([f"GHI_lag_{lag}" for lag in range(1, 25)])
        # Add current GHI
        feature_columns.append("GHI")
    
    if input_config in ['met_only', 'ghi_met']:
        # Add meteorological features
        met_features = [
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ]
        feature_columns.extend(met_features)
    
    # Always add time features
    time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    feature_columns.extend(time_features)
    
    # Get all data at once
    data = df[feature_columns].values
    target = df["GHI"].values
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
    # Pre-allocate arrays
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros(n_sequences)
    
    # Create sequences using vectorized operations
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = target[i + sequence_length]
    
    # Filter out sequences where target is zero (night time)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    return X, y

def create_sequences_individual(df, sequence_length):
    """Create sequences for individual models (single target at next step)."""
    # Get all required features
    feature_columns = (
        [f"GHI_lag_{lag}" for lag in range(1, 25)] +  # GHI lag features
        ["GHI"] +  # Current GHI
        [  # Meteorological features
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ] +
        [  # Time features
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ]
    )
    
    # Get all data at once
    data = df[feature_columns].values
    target = df["GHI"].values
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
    # Pre-allocate arrays
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros(n_sequences)
    
    # Create sequences using vectorized operations
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = target[i + sequence_length]
    
    # Filter out sequences where target is zero (night time)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    return X, y

def create_sequences_individual_multi_horizon(df, sequence_length, target_columns=None, return_dates=False):
    """Create sequences for individual models with multiple horizons (1d, 3d, 7d).
    
    Returns:
        If return_dates False: (X, y) where y has shape (n_samples, n_horizons)
        If return_dates True: (X, y, dates) where dates are target timestamps
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS
    n_horizons = len(target_columns)
    feature_columns = (
        [f"GHI_lag_{lag}" for lag in range(1, 25)] +
        ["GHI"] +
        [
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ] +
        ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    )
    feature_columns = [c for c in feature_columns if c in df.columns]
    for col in target_columns:
        if col not in df.columns:
            raise ValueError(f"Target column {col} not found.")
    data = df[feature_columns].values
    targets = df[target_columns].values  # (n, n_horizons)
    has_datetime = "datetime" in df.columns
    if return_dates and has_datetime:
        datetimes = pd.to_datetime(df["datetime"].values)
    n_sequences = len(df) - sequence_length
    if n_sequences <= 0:
        if return_dates and has_datetime:
            return np.zeros((0, sequence_length, len(feature_columns))), np.zeros((0, n_horizons)), pd.DatetimeIndex([])
        return np.zeros((0, sequence_length, len(feature_columns))), np.zeros((0, n_horizons))
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros((n_sequences, n_horizons))
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = targets[i + sequence_length]
    mask = np.any(y > 0, axis=1)
    X = X[mask]
    y = y[mask]
    if return_dates and has_datetime:
        # Target for sequence i is at index sequence_length + i (before mask)
        seq_target_indices = np.where(mask)[0] + sequence_length
        dates = pd.DatetimeIndex(datetimes[seq_target_indices])
        return X, y, dates
    return X, y

def split_data_by_days(df, train_ratio=0.7, val_ratio=0.15):
    """Split data by days to maintain temporal consistency."""
    # Convert datetime to date for grouping
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Get unique dates
    unique_dates = sorted(df['date'].unique())
    
    # Calculate split indices
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]
    
    # Split data based on dates
    train_df = df[df['date'].isin(train_dates)]
    val_df = df[df['date'].isin(val_dates)]
    test_df = df[df['date'].isin(test_dates)]
    
    return train_df, val_df, test_df

def calculate_daily_metrics(dates, actual, predicted):
    """Calculate daily metrics for evaluation."""
    # Convert dates to pandas datetime
    dates_pd = pd.to_datetime(dates)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates_pd,
        'actual': actual,
        'predicted': predicted
    })
    
    # Group by date and calculate daily metrics
    daily_stats = []
    for date, group in df.groupby(df['date'].dt.date):
        # Filter out zero values
        non_zero_mask = group['actual'] > 0
        if non_zero_mask.sum() > 0:
            actual_nonzero = group.loc[non_zero_mask, 'actual']
            predicted_nonzero = group.loc[non_zero_mask, 'predicted']
            
            # Calculate correlation
            correlation = actual_nonzero.corr(predicted_nonzero)
            if pd.isna(correlation):
                correlation = 0
            
            # Calculate other metrics
            mae = mean_absolute_error(actual_nonzero, predicted_nonzero)
            rmse = np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero))
            r2 = r2_score(actual_nonzero, predicted_nonzero)
            
            # Clamp R² to [0, 1] and convert to Python float to avoid structseq/numpy scalar issues
            r2 = float(max(0, min(1, r2)))
            daily_stats.append({
                'date': date,
                'correlation': float(correlation),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': r2,
                'num_points': int(len(actual_nonzero)),
                'actual_max': float(actual_nonzero.max()),
                'actual_mean': float(actual_nonzero.mean()),
                'actual_std': float(actual_nonzero.std()),
                'predicted_max': float(predicted_nonzero.max()),
                'predicted_mean': float(predicted_nonzero.mean()),
                'predicted_std': float(predicted_nonzero.std())
            })
    
    return pd.DataFrame(daily_stats).sort_values('date')

def plot_daily_correlations(daily_metrics, city, model_type, config, save_dir):
    """Create plots showing daily correlation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    if not isinstance(daily_metrics, pd.DataFrame) or len(daily_metrics) == 0:
        return
    # Check which columns are available
    available_columns = daily_metrics.columns.tolist()
    
    # Create figure with subplots based on available data
    if 'correlation' in available_columns and 'r2' in available_columns:
        # Full plot with correlation and R²
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot correlation and R²
        ax1.plot(daily_metrics['date'], daily_metrics['correlation'], 'b-', label='Correlation')
        ax1.plot(daily_metrics['date'], daily_metrics['r2'], 'r-', label='R²')
        ax1.set_title(f'Daily Correlation and R² - {city} ({model_type}, {config})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE and RMSE
        if 'mae' in available_columns and 'rmse' in available_columns:
            ax2.plot(daily_metrics['date'], daily_metrics['mae'], 'g-', label='MAE')
            ax2.plot(daily_metrics['date'], daily_metrics['rmse'], 'm-', label='RMSE')
            ax2.set_title(f'Daily MAE and RMSE - {city} ({model_type}, {config})')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Error (W/m²)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'MAE/RMSE data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Daily MAE and RMSE - {city} ({model_type}, {config})')
        
        # Plot GHI statistics
        if all(col in available_columns for col in ['actual_max', 'predicted_max', 'actual_mean', 'predicted_mean']):
            ax3.plot(daily_metrics['date'], daily_metrics['actual_max'], 'b-', label='Actual Max')
            ax3.plot(daily_metrics['date'], daily_metrics['predicted_max'], 'r--', label='Predicted Max')
            ax3.plot(daily_metrics['date'], daily_metrics['actual_mean'], 'g-', label='Actual Mean')
            ax3.plot(daily_metrics['date'], daily_metrics['predicted_mean'], 'm--', label='Predicted Mean')
        else:
            # For GNN, we only have mean values
            ax3.plot(daily_metrics['date'], daily_metrics['actual_mean'], 'g-', label='Actual Mean')
            ax3.plot(daily_metrics['date'], daily_metrics['predicted_mean'], 'm--', label='Predicted Mean')
        
        ax3.set_title(f'Daily GHI Statistics - {city} ({model_type}, {config})')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('GHI (W/m²)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'daily_correlations_{city}_{model_type}_{config}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    elif 'correlation' in available_columns:
        # Simplified plot for GNN (only correlation available)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot correlation only
        ax1.plot(daily_metrics['date'], daily_metrics['correlation'], 'b-', label='Correlation')
        ax1.set_title(f'Daily Correlation - {city} ({model_type}, {config})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot GHI statistics (mean values only)
        ax2.plot(daily_metrics['date'], daily_metrics['actual_mean'], 'g-', label='Actual Mean')
        ax2.plot(daily_metrics['date'], daily_metrics['predicted_mean'], 'm--', label='Predicted Mean')
        ax2.set_title(f'Daily GHI Statistics - {city} ({model_type}, {config})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('GHI (W/m²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'daily_correlations_{city}_{model_type}_{config}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save daily metrics to CSV
    daily_metrics.to_csv(os.path.join(save_dir, f'daily_correlations_{city}_{model_type}_{config}.csv'), 
                        index=False)

def plot_correlation_analysis(daily_metrics, dates, actual, predicted, city, model_type, config, save_dir):
    """Create scatter plots comparing actual vs predicted values."""
    os.makedirs(save_dir, exist_ok=True)
    if not isinstance(daily_metrics, pd.DataFrame) or len(daily_metrics) == 0 or 'correlation' not in daily_metrics.columns:
        return
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Convert dates to pandas Series for proper date handling
    dates = pd.Series(pd.to_datetime(dates))
    # Use Python date for comparisons to avoid structseq/numpy scalar issues
    def to_py_date(x):
        try:
            return pd.Timestamp(x).date() if x is not None else None
        except Exception:
            return x
    # Plot 1: Day with highest correlation
    best_day_idx = daily_metrics['correlation'].idxmax()
    best_day = daily_metrics.loc[best_day_idx]
    best_date = to_py_date(best_day['date'])
    # Get the actual data for this day
    mask = dates.dt.date == best_date
    actual_data = np.array(actual)[mask]
    predicted_data = np.array(predicted)[mask]
    
    if len(actual_data) > 0:
        ax1.scatter(actual_data, predicted_data, alpha=0.5)
        max_val = float(max(actual_data.max(), predicted_data.max()))
        ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax1.set_title(f'Best Day (Correlation = {float(best_day["correlation"]):.3f})')
        ax1.set_xlabel('Actual GHI')
        ax1.set_ylabel('Predicted GHI')
        ax1.legend()
    
    # Plot 2: Day with lowest correlation
    worst_day_idx = daily_metrics['correlation'].idxmin()
    worst_day = daily_metrics.loc[worst_day_idx]
    worst_date = to_py_date(worst_day['date'])
    # Get the actual data for this day
    mask = dates.dt.date == worst_date
    actual_data = np.array(actual)[mask]
    predicted_data = np.array(predicted)[mask]
    
    if len(actual_data) > 0:
        ax2.scatter(actual_data, predicted_data, alpha=0.5)
        max_val = float(max(actual_data.max(), predicted_data.max()))
        ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax2.set_title(f'Worst Day (Correlation = {float(worst_day["correlation"]):.3f})')
        ax2.set_xlabel('Actual GHI')
        ax2.set_ylabel('Predicted GHI')
        ax2.legend()
    
    # Calculate R² if available, otherwise use correlation (use float() to avoid numpy scalar issues)
    if 'r2' in daily_metrics.columns:
        r2_value = float(daily_metrics["r2"].mean())
        title_text = f'Correlation Analysis for {city} ({model_type}, {config})\nR² = {r2_value:.3f}'
    else:
        corr_value = float(daily_metrics["correlation"].mean())
        title_text = f'Correlation Analysis for {city} ({model_type}, {config})\nAvg Correlation = {corr_value:.3f}'
    
    plt.suptitle(title_text)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_analysis_{city}_{model_type}_{config}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_metrics, country):
    """Create a comprehensive summary table with per-horizon (1d, 3d, 7d) metrics only."""
    summary_data = []
    for (city, model_type, config), metrics in all_metrics.items():
        row = {
            'City': city,
            'Model_Type': model_type,
            'Config': config
        }
        for h in HORIZONS:
            row[f'MAE_{h}'] = metrics.get(f'mae_{h}', np.nan)
            row[f'RMSE_{h}'] = metrics.get(f'rmse_{h}', np.nan)
            row[f'R²_{h}'] = metrics.get(f'r2_{h}', np.nan)
            row[f'Correlation_{h}'] = metrics.get(f'correlation_{h}', np.nan)
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)
    summary_file = results_path(country, "comparison", "performance_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")
    print(summary_df)
    return summary_df

def create_sequences_joint_all_features(df, sequence_length, locations, target_columns=None):
    """Create sequences for joint models with all features including location encoding.
    
    Args:
        df: DataFrame with features and target columns
        sequence_length: Length of input sequences
        locations: List of location names
        target_columns: List of target column names (e.g. TARGET_COLUMNS for 1d, 3d, 7d).
                        If None, uses TARGET_COLUMNS.
    
    Returns:
        tuple: (X, y, dates) where X is input sequences, y is targets (n_samples, n_horizons),
               and dates are corresponding datetimes
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS
    n_horizons = len(target_columns)
    
    # Initialize feature columns
    base_feature_columns = (
        [f"GHI_lag_{lag}" for lag in range(1, 25)] +  # GHI lag features
        ["GHI"] +  # Current GHI
        [  # Meteorological features
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ] +
        [  # Time features
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ]
    )
    
    # Filter to only include columns that exist in the dataframe
    feature_columns = [col for col in base_feature_columns if col in df.columns]
    missing_columns = [col for col in base_feature_columns if col not in df.columns]
    
    if missing_columns:
        print(f"WARNING: Missing feature columns: {missing_columns}")
        print(f"Using available columns: {len(feature_columns)} out of {len(base_feature_columns)}")
    
    for col in target_columns:
        if col not in df.columns:
            raise ValueError(f"Target column {col} not found in dataframe.")
    
    # Initialize lists to store sequences and dates
    all_X_sequences = []
    all_y_sequences = []
    all_dates = []
    
    # Process each location separately
    for location in locations:
        df_loc = df[df['location'] == location].copy()
        
        print(f"  Processing {location}: {len(df_loc)} samples")
        
        if len(df_loc) == 0:
            print(f"  No data for {location}, skipping")
            continue
            
        # Create one-hot encoded location vector for the current location
        location_vector = np.zeros(len(locations))
        location_vector[locations.index(location)] = 1
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Get all required data at once (only use available columns)
        data = df_loc[feature_columns].values
        targets = df_loc[target_columns].values  # Shape: (n_rows, n_horizons)
        dates = pd.to_datetime(df_loc["datetime"].values)  # Extract dates
    
        # Calculate number of sequences
        n_sequences = len(df_loc) - sequence_length
    
        # Pre-allocate arrays
        X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
        y = np.zeros((n_sequences, n_horizons))
        seq_dates = []
    
        # Create sequences using vectorized operations
        for i in range(n_sequences):
            X[i] = data[i:i + sequence_length]
            y[i] = targets[i + sequence_length]  # All horizons at target time step
            seq_dates.append(dates[i + sequence_length])
    
        # Filter out sequences where all targets are zero (night time)
        mask = np.any(y > 0, axis=1)
        X = X[mask]
        y = y[mask]
        seq_dates = [seq_dates[i] for i in range(len(seq_dates)) if mask[i]]
    
        # Add location features
        location_features = np.tile(location_vector, (len(X), sequence_length, 1))
        X = np.concatenate([X, location_features], axis=2)
    
        print(f"  Created {len(X)} sequences for {location} ({n_horizons} horizons)")
        all_X_sequences.append(X)
        all_y_sequences.append(y)
        all_dates.extend(seq_dates)
    
    if not all_X_sequences:
        raise ValueError("No valid sequences were created for any location")
    
    # Combine sequences from all locations
    X = np.concatenate(all_X_sequences, axis=0)
    y = np.concatenate(all_y_sequences, axis=0)
    dates_array = pd.to_datetime(all_dates)
    
    print(f"Total sequences created: {len(X)} (y shape: {y.shape})")
    return X, y, dates_array

def evaluate_joint_models(country, is_colombia=False, gnn_hours=None):
    """Evaluate joint models with different configurations."""
    print("\nEvaluating joint models...")
    
    # Choose data locations based on flag
    if is_colombia:
        data_locations = CONFIG["data_locations_colombia"]
        print(f"Using Colombia data locations: {list(data_locations.keys())}")
    else:
        data_locations = CONFIG["data_locations"]
        print(f"Using India data locations: {list(data_locations.keys())}")
    
    # Load data
    all_dfs = []
    for city in data_locations.keys():
        df = load_data(data_locations, city)
        df = create_features(df)
        df['location'] = city
        all_dfs.append(df)
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    locations = sorted(df_all['location'].unique().tolist())
    
    model_path = model_file_path(country, "joint", "model.h5")
    if not os.path.exists(model_path):
        print(f"Joint model not found: {model_path}")
        return {}
    
    # Try to load the full model first; on 'function_type' or similar we'll fall back to build + load_weights later
    model = None
    print(f"Loading joint model from {model_path}...")
    custom_objs = {
        'custom_ghi_loss': custom_ghi_loss,
        'InputLayer': _InputLayerCompat,
        'BatchNormalization': _BatchNormCompat,
        'BatchNorm': _BatchNormCompat,
    }
    _policy_cls = (
        getattr(tf.keras, 'DTypePolicy', None)
        or getattr(tf.keras.mixed_precision, 'Policy', None)
    )
    if _policy_cls is None:
        try:
            import keras as _keras
            _policy_cls = getattr(_keras, 'DTypePolicy', None) or getattr(
                getattr(_keras, 'mixed_precision', None), 'Policy', None
            )
        except ImportError:
            pass
    if _policy_cls is not None:
        custom_objs['DTypePolicy'] = _policy_cls
        custom_objs['Policy'] = _policy_cls
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
        print(f"✓ Joint model loaded successfully")
        model.summary()
    except Exception as e:
        print(f"✗ Error loading joint model: {e}")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
            print(f"✓ Joint model loaded (compile=False)")
            model.summary()
        except Exception as e2:
            if "dtype policy" in str(e2).lower() or "DTypePolicy" in str(e2):
                try:
                    set_global_policy = getattr(tf.keras.mixed_precision, "set_global_policy", None)
                    if set_global_policy is not None:
                        set_global_policy("float32")
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
                    print(f"✓ Joint model loaded (float32 policy)")
                    model.summary()
                except Exception as e3:
                    model = None
            else:
                model = None
            if model is None:
                print(f"  Will try build + load_weights after preparing data (fallback for function_type/Lambda).")
    
    # Load target scaler
    scaler_path = model_file_path(country, "joint", "target_scaler.pkl")
    # Scaler will be loaded or fitted after we have df_all with targets (below)
    
    # Evaluate the joint model
    all_metrics = {}
    
    print(f"\nEvaluating joint model...")
    
    # Create sequences using the same approach as train_joint.py
    print("Creating joint features...")
    
    # Add time-based features
    df_all["hour_sin"] = np.sin(2 * np.pi * df_all["Hour"] / 24)
    df_all["hour_cos"] = np.cos(2 * np.pi * df_all["Hour"] / 24)
    df_all["month_sin"] = np.sin(2 * np.pi * df_all["Month"] / 12)
    df_all["month_cos"] = np.cos(2 * np.pi * df_all["Month"] / 12)
    
    # Targets and lags are shifted within each location: df_all stacks the sites one
    # after another, so shifting the whole frame would give each site's last week the
    # targets of the next site's first days (and its first rows the previous site's lags)
    by_location = df_all.groupby("location", sort=False)

    # Create targets for three horizons: 1d, 3d, 7d at same hour
    df_all["target_GHI_1d"] = by_location["GHI"].shift(-24)
    df_all["target_GHI_3d"] = by_location["GHI"].shift(-72)
    df_all["target_GHI_7d"] = by_location["GHI"].shift(-168)

    # Create lag features for previous 24 hours of GHI
    for lag in range(1, 25):
        df_all[f"GHI_lag_{lag}"] = by_location["GHI"].shift(lag)
    
    # Create lag features for meteorological variables - only use columns that exist
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    # Filter to only include variables that exist in the dataframe
    available_met_features = [feature for feature in meteorological_features if feature in df_all.columns]
    missing_met_features = [feature for feature in meteorological_features if feature not in df_all.columns]
    
    if missing_met_features:
        print(f"WARNING: Missing meteorological features: {missing_met_features}")
        print(f"Using available features: {available_met_features}")
    
    for feature in available_met_features:
        df_all[f"{feature}_lag_24"] = by_location[feature].shift(24)
    
    # Create location-specific features (one-hot encoding)
    from sklearn.preprocessing import OneHotEncoder
    try:
        # Try new scikit-learn version first
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        # Fall back to old scikit-learn version
        encoder = OneHotEncoder(sparse=False)
    
    # Ensure encoder sees all unique locations
    unique_locations = df_all['location'].unique()
    print(f"Unique locations in data: {unique_locations}")
    
    # Fit encoder on all unique locations
    encoder.fit(df_all[['location']])
    location_encoded = encoder.transform(df_all[['location']])
    
    # IMPORTANT: Use encoder categories order (sorted alphabetically by OneHotEncoder)
    encoder_categories = encoder.categories_[0].tolist()
    location_df = pd.DataFrame(location_encoded, 
                             columns=[f"location_{loc}" for loc in encoder_categories])
    
    print(f"Location encoding created:")
    print(f"  Encoder categories (sorted): {encoder_categories}")
    print(f"  Locations list (original order): {locations}")
    print(f"  Location columns: {location_df.columns.tolist()}")
    
    # Check encoding for each location
    print(f"  Location encoding verification:")
    for loc in unique_locations:
        loc_mask = df_all['location'] == loc
        if loc_mask.sum() > 0:
            sample_encoding = location_df.loc[loc_mask].iloc[0].values
            category_idx = encoder_categories.index(loc)
            print(f"    {loc}: category index {category_idx}, encoding: {sample_encoding}")
    
    # Combine with original data
    df_all = pd.concat([df_all, location_df], axis=1)
    
    # Clean data - fill NaN in lag features, then drop rows where target is NaN
    print("\nCleaning data...")
    lag_columns = [col for col in df_all.columns if '_lag_' in col]
    print(f"Filling NaN in {len(lag_columns)} lag feature columns...")
    for col in lag_columns:
        df_all[col] = df_all.groupby("location", sort=False)[col].transform(lambda s: s.ffill().bfill()).fillna(0.0)
    
    # Now drop rows where any target horizon is NaN
    df_all = df_all.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    
    # Split data by days (same as individual models)
    train_df, val_df, test_df = split_data_by_days(df_all)
    
    # Load or fit target scaler (after we have df_all with targets, for fallback)
    target_scaler = None
    if os.path.exists(scaler_path):
        print(f"Loading target scaler from {scaler_path}...")
        try:
            # NumPy 2.x pickles reference numpy._core.*; older NumPy has numpy.core
            import sys
            if 'numpy._core' not in sys.modules and hasattr(np, 'core'):
                _np_core = np.core
                sys.modules['numpy._core'] = _np_core
                for _sub in ('umath', 'multiarray', 'numeric', '_internal'):
                    _attr = getattr(_np_core, _sub, None)
                    if _attr is not None:
                        sys.modules[f'numpy._core.{_sub}'] = _attr
            with open(scaler_path, 'rb') as f:
                target_scaler = pickle.load(f)
            print(f"✓ Target scaler loaded (range: [{target_scaler.data_min_.min():.2f}, {target_scaler.data_max_.max():.2f}])")
        except Exception as e:
            print(f"✗ Could not load scaler ({e}), will fit from data.")
    if target_scaler is None:
        print("Fitting target scaler on full data (same as training)...")
        target_scaler = MinMaxScaler()
        target_scaler.fit(df_all[TARGET_COLUMNS])
        print(f"✓ Target scaler fitted (range: [{target_scaler.data_min_.min():.2f}, {target_scaler.data_max_.max():.2f}])")
    
    # Scale only target columns (inputs GHI and lags stay raw, same as training)
    test_df_scaled = test_df.copy()
    test_df_scaled[TARGET_COLUMNS] = target_scaler.transform(test_df[TARGET_COLUMNS])
    
    # Debug: Check location distribution in test data
    print(f"\nTest data location distribution:")
    for loc in encoder_categories:  # Use encoder categories order
        count = (test_df_scaled['location'] == loc).sum()
        print(f"  {loc}: {count} samples")
    
    # Create sequences for test data - use encoder_categories order to match training (three horizons)
    X_test, y_test, test_dates = create_sequences_joint_all_features(
        test_df_scaled, 24, encoder_categories, target_columns=TARGET_COLUMNS
    )
    
    if len(X_test) == 0:
        print("No test sequences created for joint model")
        return {}
    
    print(f"Created {len(X_test)} test sequences for joint model")
    print(f"Date range: {test_dates.min()} to {test_dates.max()}")
    
    # Fallback: if model failed to load (e.g. 'function_type' from Lambda layer), build from code and load weights
    if model is None:
        print("Building joint model from code and loading weights (fallback)...")
        try:
            input_shape = (X_test.shape[1], X_test.shape[2])
            model = _build_joint_model_architecture(input_shape, n_horizons=len(HORIZONS))
            model.load_weights(model_path)
            print(f"✓ Joint model loaded via load_weights (input_shape={input_shape})")
        except Exception as e_fw:
            print(f"✗ Build + load_weights failed: {e_fw}")
            return {}
    
    # Make predictions (model outputs shape (n, 3) for 1d, 3d, 7d)
    print(f"Making predictions on {len(X_test)} test sequences...")
    y_pred = model.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] != len(HORIZONS):
        raise ValueError(f"Model output has {y_pred.shape[1]} units; expected {len(HORIZONS)} for horizons {HORIZONS}.")
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    # Inverse transform (y_test and y_pred are (n, 3))
    y_test_original = target_scaler.inverse_transform(y_test)
    y_pred_original = target_scaler.inverse_transform(y_pred_clipped)
    
    print(f"Original test range: [{np.min(y_test_original):.2f}, {np.max(y_test_original):.2f}]")
    print(f"Original pred range: [{np.min(y_pred_original):.2f}, {np.max(y_pred_original):.2f}]")
    
    # Calculate metrics for each city and each horizon
    expected_cities = list(data_locations.keys())
    print(f"\nEvaluating joint model for each city and horizon (1d, 3d, 7d)...")
    print(f"Expected cities: {expected_cities}")
    print(f"Encoder categories (used for location encoding): {encoder_categories}")
    print(f"Total predictions: {len(y_pred_original)}")
    print(f"Location features shape: {X_test[:, 0, -len(encoder_categories):].shape}")
    
    for city in expected_cities:
        print(f"\nEvaluating {city}...")
        
        if city not in encoder_categories:
            print(f"  WARNING: {city} not found in encoder categories")
            placeholder = {f'mae_{h}': np.nan for h in HORIZONS}
            placeholder.update({f'rmse_{h}': np.nan for h in HORIZONS})
            placeholder.update({f'r2_{h}': np.nan for h in HORIZONS})
            placeholder.update({f'correlation_{h}': np.nan for h in HORIZONS})
            placeholder['daily_metrics'] = pd.DataFrame()
            placeholder['mae'] = np.nan
            placeholder['rmse'] = np.nan
            placeholder['r2'] = np.nan
            placeholder['correlation'] = np.nan
            for h in HORIZONS:
                placeholder[f'actual_{h}'] = np.array([])
                placeholder[f'predicted_{h}'] = np.array([])
            placeholder['dates'] = pd.DatetimeIndex([])
            all_metrics[(city, 'joint', 'standard')] = placeholder
            continue
        
        city_idx = encoder_categories.index(city)
        city_mask = X_test[:, 0, -len(encoder_categories):][:, city_idx] == 1
        print(f"  City mask for {city}: {city_mask.sum()} samples out of {len(city_mask)}")
        # Score on the same hours as the GNN
        city_mask[city_mask] = gnn_hours_mask(city, test_dates[city_mask], gnn_hours, "joint")
        
        if city_mask.sum() > 0:
            city_dates = test_dates[city_mask]
            save_test_predictions(results_path(country, "joint"), city, city_dates,
                                  y_test_original[city_mask], y_pred_original[city_mask])
            city_metrics = {
                'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'correlation': np.nan,  # overall not computed for joint
                'daily_metrics': pd.DataFrame(),
                'dates': city_dates
            }
            for h_idx, h in enumerate(HORIZONS):
                city_actual_h = y_test_original[city_mask, h_idx]
                city_predicted_h = y_pred_original[city_mask, h_idx]
                mae = float(mean_absolute_error(city_actual_h, city_predicted_h))
                rmse = float(np.sqrt(mean_squared_error(city_actual_h, city_predicted_h)))
                r2 = float(r2_score(city_actual_h, city_predicted_h))  # not clipped, same as the GNN
                corr = np.corrcoef(city_actual_h, city_predicted_h)[0, 1] if len(city_actual_h) > 1 else np.nan
                if np.isnan(corr):
                    corr = 0.0
                else:
                    corr = float(corr)
                city_metrics[f'mae_{h}'] = mae
                city_metrics[f'rmse_{h}'] = rmse
                city_metrics[f'r2_{h}'] = r2
                city_metrics[f'correlation_{h}'] = corr
                city_metrics[f'actual_{h}'] = np.asarray(city_actual_h)
                city_metrics[f'predicted_{h}'] = np.asarray(city_predicted_h)
                print(f"  {h}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, Corr={corr:.4f}")
            all_metrics[(city, 'joint', 'standard')] = city_metrics
        else:
            placeholder = {f'mae_{h}': np.nan for h in HORIZONS}
            placeholder.update({f'rmse_{h}': np.nan for h in HORIZONS})
            placeholder.update({f'r2_{h}': np.nan for h in HORIZONS})
            placeholder.update({f'correlation_{h}': np.nan for h in HORIZONS})
            placeholder['mae'] = np.nan
            placeholder['rmse'] = np.nan
            placeholder['r2'] = np.nan
            placeholder['correlation'] = np.nan
            placeholder['daily_metrics'] = pd.DataFrame()
            for h in HORIZONS:
                placeholder[f'actual_{h}'] = np.array([])
                placeholder[f'predicted_{h}'] = np.array([])
            placeholder['dates'] = pd.DatetimeIndex([])
            all_metrics[(city, 'joint', 'standard')] = placeholder
            print(f"  No test data found for {city}")
    
    return all_metrics

def load_gnn_test_hours(country):
    """Return {city (NFC): DatetimeIndex} of the hours each site's GNN was tested on,
    read from results/{country}/gnn/{city}/test_predictions.csv.

    The GNN splits the dates shared by the graph as a whole, so its test period can
    start a day later than a single site's (e.g. Colombia after the UTC conversion,
    where Puerto Nare's data reach one extra calendar date). Scoring the LSTMs on
    these same hours keeps the comparison like for like without retraining."""
    root = os.path.join("results", country, "gnn")
    hours = {}
    if not os.path.isdir(root):
        return hours
    for name in os.listdir(root):
        path = os.path.join(root, name, "test_predictions.csv")
        if os.path.exists(path):
            hours[unicodedata.normalize("NFC", name)] = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path)["datetime"]))
    return hours


def gnn_hours_mask(city, dates, gnn_hours, model_type):
    """Boolean mask selecting the LSTM test samples whose hour is in the GNN's test set."""
    if not gnn_hours:
        return np.ones(len(dates), dtype=bool)
    city_hours = gnn_hours.get(unicodedata.normalize("NFC", city))
    if city_hours is None:
        print(f"  WARNING: no GNN test hours for {city}; scoring the {model_type} model on all its test hours")
        return np.ones(len(dates), dtype=bool)
    lstm_hours = pd.DatetimeIndex(dates).floor("h")
    mask = lstm_hours.isin(city_hours)
    missing = city_hours.difference(lstm_hours)
    print(f"  Matching GNN test hours: keeping {mask.sum()} of {len(mask)} {model_type} samples"
          + (f"; WARNING: {len(missing)} GNN hours have no {model_type} sample" if len(missing) else ""))
    return mask


def save_test_predictions(results_dir, city, dates, actual, predicted):
    """Write hourly test predictions as results_dir/city/test_predictions.csv, in the
    same format as the GNN's, so the models can be compared hour by hour
    (significance_tests.py)."""
    city_dir = os.path.join(results_dir, normalize_name_for_path(city))
    os.makedirs(city_dir, exist_ok=True)
    out = pd.DataFrame({"datetime": pd.DatetimeIndex(dates).floor("h")})
    for h_idx, h in enumerate(HORIZONS):
        out[f"actual_{h}"] = actual[:, h_idx]
        out[f"predicted_{h}"] = predicted[:, h_idx]
    out.to_csv(os.path.join(city_dir, "test_predictions.csv"), index=False)


def _individual_placeholder_metrics():
    """Return a placeholder metrics dict with NaN for all horizon fields (so location still appears in summary)."""
    placeholder = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'correlation': np.nan, 'dates': pd.DatetimeIndex([])}
    for h in HORIZONS:
        placeholder[f'mae_{h}'] = np.nan
        placeholder[f'rmse_{h}'] = np.nan
        placeholder[f'r2_{h}'] = np.nan
        placeholder[f'correlation_{h}'] = np.nan
        placeholder[f'actual_{h}'] = np.array([])
        placeholder[f'predicted_{h}'] = np.array([])
    placeholder['daily_metrics'] = pd.DataFrame()
    return placeholder


def evaluate_individual_models(country, is_colombia=False, gnn_hours=None):
    """Evaluate individual models."""
    print("\nEvaluating individual models...")
    
    # Choose data locations based on flag
    if is_colombia:
        data_locations = CONFIG["data_locations_colombia"]
        print(f"Using Colombia data locations: {list(data_locations.keys())}")
    else:
        data_locations = CONFIG["data_locations"]
        print(f"Using India data locations: {list(data_locations.keys())}")
    
    all_metrics = {}
    
    for city in data_locations.keys():
        print(f"\nEvaluating {city}...")
        # Resolve model path with Unicode normalization so names like Coveñas match saved files
        model_path = resolve_individual_model_path(city, country)
        print(f"  Checking for model file: {model_path}")
        
        # Load data
        try:
            df = load_data(data_locations, city)
            df = create_features(df)
            # Add three-horizon targets (same as train_individual)
            df["target_GHI_1d"] = df["GHI"].shift(-24)
            df["target_GHI_3d"] = df["GHI"].shift(-72)
            df["target_GHI_7d"] = df["GHI"].shift(-168)
            df = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
        except Exception as e:
            print(f"  Error loading/preparing data for {city}: {e}")
            all_metrics[(city, 'individual', 'standard')] = _individual_placeholder_metrics()
            continue
        
        # Load model (path may have been resolved from a differently-normalized filename)
        if not os.path.exists(model_path):
            print(f"  Model not found for {city}; adding placeholder so location appears in summary.")
            all_metrics[(city, 'individual', 'standard')] = _individual_placeholder_metrics()
            continue
        
        print(f"  Loading model from {model_path}...")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'custom_ghi_loss': custom_ghi_loss})
            print(f"  ✓ Model loaded successfully")
            print(f"  Model summary:")
            model.summary()
        except Exception as e:
            # The individual model uses the joint network, whose Lambda layer cannot always
            # be deserialized (e.g. a model saved under another Python version); in that
            # case the network is rebuilt below and only the weights are loaded
            print(f"  ✗ Error loading full model ({e}); will rebuild the network and load the weights.")
            model = None
        
        # Split data by days (same as training)
        train_df, val_df, test_df = split_data_by_days(df)
        
        # Scaler fitted on all target columns (same as training)
        print(f"  Creating scaler for {city} (fitted on all target horizons)...")
        target_scaler = MinMaxScaler()
        target_scaler.fit(df[TARGET_COLUMNS])
        print(f"  Scaler range: [{target_scaler.data_min_.min():.2f}, {target_scaler.data_max_.max():.2f}]")
        
        # Scale only target columns for test (inputs stay raw like in training)
        test_df_scaled = test_df.copy()
        test_df_scaled[TARGET_COLUMNS] = target_scaler.transform(test_df[TARGET_COLUMNS])
        
        # Create sequences with three horizons (and get target dates)
        X_test, y_test, dates = create_sequences_individual_multi_horizon(
            test_df_scaled, 24, TARGET_COLUMNS, return_dates=True
        )
        
        if len(X_test) == 0:
            print(f"  No test sequences for {city}; adding placeholder so location appears in summary.")
            all_metrics[(city, 'individual', 'standard')] = _individual_placeholder_metrics()
            continue

        if model is None:
            try:
                model = _build_joint_model_architecture((X_test.shape[1], X_test.shape[2]), n_horizons=len(HORIZONS))
                model.load_weights(model_path)
                print(f"  ✓ Individual model loaded via load_weights (input_shape={X_test.shape[1:]})")
            except Exception as e_fw:
                print(f"  ✗ Rebuild + load_weights failed ({e_fw}); adding placeholder so location appears in summary.")
                all_metrics[(city, 'individual', 'standard')] = _individual_placeholder_metrics()
                continue
        
        # Make predictions (output shape (n, 3))
        print(f"  Making predictions on {len(X_test)} test sequences...")
        y_pred = model.predict(X_test)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_pred.shape[1] != len(HORIZONS):
            print(f"  WARNING: Model output has {y_pred.shape[1]} units; expected {len(HORIZONS)}. Adding placeholder for {city}.")
            all_metrics[(city, 'individual', 'standard')] = _individual_placeholder_metrics()
            continue
        y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
        print(f"  Predictions shape: {y_pred.shape}")
        
        # Inverse transform (y_test and y_pred are (n, 3))
        y_test_original = target_scaler.inverse_transform(y_test)
        y_pred_original = target_scaler.inverse_transform(y_pred_clipped)

        # Score on the same hours as the GNN
        keep = gnn_hours_mask(city, dates, gnn_hours, "individual")
        y_test_original, y_pred_original = y_test_original[keep], y_pred_original[keep]
        dates = pd.DatetimeIndex(dates)[keep]
        save_test_predictions(results_path(country, "individual"), city, dates, y_test_original, y_pred_original)

        # Per-horizon metrics only (no overall)
        city_metrics = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'correlation': np.nan, 'dates': dates}
        for h_idx, h in enumerate(HORIZONS):
            actual_h = y_test_original[:, h_idx]
            pred_h = y_pred_original[:, h_idx]
            mae = mean_absolute_error(actual_h, pred_h)
            rmse = np.sqrt(mean_squared_error(actual_h, pred_h))
            r2 = r2_score(actual_h, pred_h)  # not clipped, same as the GNN
            corr = np.corrcoef(actual_h, pred_h)[0, 1] if len(actual_h) > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0
            city_metrics[f'mae_{h}'] = mae
            city_metrics[f'rmse_{h}'] = rmse
            city_metrics[f'r2_{h}'] = r2
            city_metrics[f'correlation_{h}'] = corr
            city_metrics[f'actual_{h}'] = actual_h
            city_metrics[f'predicted_{h}'] = pred_h
            print(f"  {h}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, Corr={corr:.4f}")
        daily_metrics = calculate_daily_metrics(dates, y_test_original[:, 0], y_pred_original[:, 0])
        city_metrics['daily_metrics'] = daily_metrics
        all_metrics[(city, 'individual', 'standard')] = city_metrics
        
        # Create plots (using 1d horizon for daily correlation view)
        plot_daily_correlations(daily_metrics, city, 'individual', 'standard', results_path(country, "individual"))
        plot_correlation_analysis(daily_metrics, dates, city_metrics['actual_1d'], city_metrics['predicted_1d'],
                               city, 'individual', 'standard', results_path(country, "individual"))
    
    return all_metrics

def custom_ghi_loss(y_true, y_pred):
    """Custom loss function for GHI prediction."""
    # Standard Huber loss
    huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
    
    # Penalty for predictions outside [0, 1] range
    range_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 1.0) + tf.maximum(0.0, -y_pred))
    
    return huber_loss + range_penalty

def evaluate_gnn_models(country, is_colombia=False):
    """Evaluate GNN models."""
    print("\nEvaluating GNN models...")
    
    # Choose data locations based on flag
    if is_colombia:
        data_locations = CONFIG["data_locations_colombia"]
        print(f"Using Colombia data locations: {list(data_locations.keys())}")
    else:
        data_locations = CONFIG["data_locations"]
        print(f"Using India data locations: {list(data_locations.keys())}")
    
    # Check if GNN results exist
    gnn_summary_path = os.path.join("results", country, "gnn", "summary.csv")
    if not os.path.exists(gnn_summary_path):
        print(f"GNN results not found: {gnn_summary_path}")
        print("Please run train_gnn.py first.")
        return {}
    
    # Load GNN summary
    gnn_summary = pd.read_csv(gnn_summary_path)
    
    # Get all cities from summary file (should include all cities that were trained)
    cities_in_summary = gnn_summary['City'].unique().tolist()
    print(f"Cities found in GNN summary: {cities_in_summary}")
    
    all_metrics = {}
    
    # Evaluate ALL expected cities from data_locations
    expected_cities = list(data_locations.keys())
    print(f"Expected cities to evaluate: {expected_cities}")
    
    for city in expected_cities:
        print(f"\nEvaluating {city}...")
        
        # Check if city exists in summary
        city_summary = gnn_summary[gnn_summary['City'] == city]
        
        if len(city_summary) == 0:
            print(f"  WARNING: {city} not found in GNN summary file")
            placeholder = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'correlation': np.nan,
                           'daily_metrics': pd.DataFrame(), 'dates': pd.DatetimeIndex([])}
            for h in HORIZONS:
                placeholder[f'mae_{h}'] = placeholder[f'rmse_{h}'] = placeholder[f'r2_{h}'] = placeholder[f'correlation_{h}'] = np.nan
            all_metrics[(city, 'gnn', 'standard')] = placeholder
            continue
        
        # Get per-horizon metrics from GNN summary CSV (overall not computed)
        def get_col(name, default=np.nan):
            return city_summary[name].iloc[0] if name in city_summary.columns else default
        
        # Try to load daily metrics if available
        city_dir = os.path.join("results", country, "gnn", city)
        daily_metrics_path = f"{city_dir}/daily_metrics.csv"
        
        if os.path.exists(daily_metrics_path):
            try:
                daily_metrics = pd.read_csv(daily_metrics_path)
                if len(daily_metrics) > 0:
                    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
                    
                    # For GNN with 24-hour forecasting, each row represents one hour prediction
                    # The daily metrics already contain the actual and predicted values
                    if 'actual_mean' in daily_metrics.columns and 'predicted_mean' in daily_metrics.columns:
                        actual = daily_metrics['actual_mean'].values
                        predicted = daily_metrics['predicted_mean'].values
                        dates = daily_metrics['date'].values
                        print(f"  ✓ Loaded daily metrics: {len(daily_metrics)} days")
                    else:
                        # Fallback: use summary metrics only
                        print(f"  WARNING: Daily metrics file exists but missing required columns")
                        actual = np.array([])
                        predicted = np.array([])
                        dates = pd.DatetimeIndex([])
                else:
                    print(f"  WARNING: Daily metrics file is empty for {city}")
                    daily_metrics = pd.DataFrame(columns=['date', 'correlation', 'mae', 'rmse', 'r2', 
                                                           'num_points', 'actual_max', 'actual_mean', 
                                                           'actual_std', 'predicted_max', 'predicted_mean', 
                                                           'predicted_std'])
                    actual = np.array([])
                    predicted = np.array([])
                    dates = pd.DatetimeIndex([])
            except Exception as e:
                print(f"  WARNING: Error reading daily metrics file for {city}: {e}")
                daily_metrics = pd.DataFrame(columns=['date', 'correlation', 'mae', 'rmse', 'r2', 
                                                       'num_points', 'actual_max', 'actual_mean', 
                                                       'actual_std', 'predicted_max', 'predicted_mean', 
                                                       'predicted_std'])
                actual = np.array([])
                predicted = np.array([])
                dates = pd.DatetimeIndex([])
        else:
            print(f"  WARNING: Daily metrics file not found for {city} at {daily_metrics_path}")
            print(f"  This usually means GNN training/evaluation didn't complete for this city.")
            print(f"  Using summary metrics only (from summary.csv)")
            # Create empty DataFrame with expected columns to avoid KeyError
            daily_metrics = pd.DataFrame(columns=['date', 'correlation', 'mae', 'rmse', 'r2', 
                                                   'num_points', 'actual_max', 'actual_mean', 
                                                   'actual_std', 'predicted_max', 'predicted_mean', 
                                                   'predicted_std'])
            actual = np.array([])
            predicted = np.array([])
            dates = pd.DatetimeIndex([])
        
        gnn_metrics = {
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'correlation': np.nan,
            'daily_metrics': daily_metrics,
            'dates': pd.to_datetime(dates) if len(dates) > 0 else pd.DatetimeIndex([])
        }
        for h in HORIZONS:
            gnn_metrics[f'mae_{h}'] = get_col(f'MAE_{h}')
            gnn_metrics[f'rmse_{h}'] = get_col(f'RMSE_{h}')
            gnn_metrics[f'r2_{h}'] = get_col(f'R²_{h}')
            gnn_metrics[f'correlation_{h}'] = get_col(f'Corr_{h}')
        all_metrics[(city, 'gnn', 'standard')] = gnn_metrics
        
        # Create plots if we have daily metrics
        if len(daily_metrics) > 0 and 'date' in daily_metrics.columns:
            try:
                plot_daily_correlations(daily_metrics, city, 'gnn', 'standard', results_path(country, "gnn"))
                if len(actual) > 0 and len(predicted) > 0:
                    plot_correlation_analysis(daily_metrics, dates, actual, predicted, 
                                           city, 'gnn', 'standard', results_path(country, "gnn"))
            except Exception as e:
                print(f"  WARNING: Could not create plots for {city}: {e}")
    
    return all_metrics

def create_model_comparison_plots(summary_df, save_dir):
    """Create comparison plots for different models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Model comparison plot
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Get unique cities and model types
    cities = summary_df['City'].unique()
    model_types = summary_df['Model_Type'].unique()
    
    # Set up colors for different model types
    colors = {'individual': 'blue', 'joint': 'red', 'gnn': 'green'}
    # Use mean of per-horizon metrics for comparison (no overall columns in summary)
    mae_cols = [f'MAE_{h}' for h in HORIZONS if f'MAE_{h}' in summary_df.columns]
    rmse_cols = [f'RMSE_{h}' for h in HORIZONS if f'RMSE_{h}' in summary_df.columns]
    r2_cols = [f'R²_{h}' for h in HORIZONS if f'R²_{h}' in summary_df.columns]
    corr_cols = [f'Correlation_{h}' for h in HORIZONS if f'Correlation_{h}' in summary_df.columns]
    
    # Plot 1: MAE comparison (mean across horizons)
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        vals = model_data[mae_cols].mean(axis=1) if mae_cols else [np.nan] * len(model_data)
        ax1.bar([f"{city}_{model_type}" for city in model_data['City']], 
                vals, label=model_type, color=colors[model_type], alpha=0.7)
    ax1.set_title('MAE Comparison (mean 1d/3d/7d)')
    ax1.set_ylabel('MAE (W/m²)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: RMSE comparison (mean across horizons)
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        vals = model_data[rmse_cols].mean(axis=1) if rmse_cols else [np.nan] * len(model_data)
        ax2.bar([f"{city}_{model_type}" for city in model_data['City']], 
                vals, label=model_type, color=colors[model_type], alpha=0.7)
    ax2.set_title('RMSE Comparison (mean 1d/3d/7d)')
    ax2.set_ylabel('RMSE (W/m²)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: R² comparison (mean across horizons)
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        vals = model_data[r2_cols].mean(axis=1) if r2_cols else [np.nan] * len(model_data)
        ax3.bar([f"{city}_{model_type}" for city in model_data['City']], 
                vals, label=model_type, color=colors[model_type], alpha=0.7)
    ax3.set_title('R² Comparison (mean 1d/3d/7d)')
    ax3.set_ylabel('R²')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Correlation comparison (mean across horizons)
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        vals = model_data[corr_cols].mean(axis=1) if corr_cols else [np.nan] * len(model_data)
        ax4.bar([f"{city}_{model_type}" for city in model_data['City']], 
                vals, label=model_type, color=colors[model_type], alpha=0.7)
    ax4.set_title('Correlation Comparison (mean 1d/3d/7d)')
    ax4.set_ylabel('Correlation')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Location comparison plot
    plt.figure(figsize=(15, 10))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot by location (mean of per-horizon metrics)
    mae_cols = [f'MAE_{h}' for h in HORIZONS if f'MAE_{h}' in summary_df.columns]
    rmse_cols = [f'RMSE_{h}' for h in HORIZONS if f'RMSE_{h}' in summary_df.columns]
    r2_cols = [f'R²_{h}' for h in HORIZONS if f'R²_{h}' in summary_df.columns]
    corr_cols = [f'Correlation_{h}' for h in HORIZONS if f'Correlation_{h}' in summary_df.columns]
    for city in cities:
        city_data = summary_df[summary_df['City'] == city]
        mae_vals = city_data[mae_cols].mean(axis=1) if mae_cols else [np.nan] * len(city_data)
        rmse_vals = city_data[rmse_cols].mean(axis=1) if rmse_cols else [np.nan] * len(city_data)
        r2_vals = city_data[r2_cols].mean(axis=1) if r2_cols else [np.nan] * len(city_data)
        corr_vals = city_data[corr_cols].mean(axis=1) if corr_cols else [np.nan] * len(city_data)
        # MAE by location
        ax1.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                mae_vals, label=city, alpha=0.7)
        ax1.set_title('MAE by Location (mean 1d/3d/7d)')
        ax1.set_ylabel('MAE (W/m²)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        # RMSE by location
        ax2.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                rmse_vals, label=city, alpha=0.7)
        ax2.set_title('RMSE by Location (mean 1d/3d/7d)')
        ax2.set_ylabel('RMSE (W/m²)')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        # R² by location
        ax3.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                r2_vals, label=city, alpha=0.7)
        ax3.set_title('R² by Location (mean 1d/3d/7d)')
        ax3.set_ylabel('R²')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        # Correlation by location
        ax4.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                corr_vals, label=city, alpha=0.7)
        ax4.set_title('Correlation by Location (mean 1d/3d/7d)')
        ax4.set_ylabel('Correlation')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'location_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-horizon comparison (MAE and R² for 1d, 3d, 7d) with grouped bars per city
    if all(f'MAE_{h}' in summary_df.columns for h in HORIZONS):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        n_models = len(model_types)
        bar_width = 0.8 / n_models
        for col_idx, h in enumerate(HORIZONS):
            mae_col = f'MAE_{h}'
            r2_col = f'R²_{h}'
            # One row per city; get cities from first model type
            cities_ordered = summary_df[summary_df['Model_Type'] == model_types[0]]['City'].tolist()
            x = np.arange(len(cities_ordered))
            for i, model_type in enumerate(model_types):
                model_data = summary_df[summary_df['Model_Type'] == model_type].set_index('City')
                mae_vals = [model_data.loc[c, mae_col] if c in model_data.index else np.nan for c in cities_ordered]
                r2_vals = [model_data.loc[c, r2_col] if c in model_data.index else np.nan for c in cities_ordered]
                offset = (i - n_models / 2 + 0.5) * bar_width
                axes[0, col_idx].bar(x + offset, mae_vals, bar_width, label=model_type, color=colors[model_type], alpha=0.7)
                axes[1, col_idx].bar(x + offset, r2_vals, bar_width, label=model_type, color=colors[model_type], alpha=0.7)
            axes[0, col_idx].set_title(f'MAE - {h} horizon')
            axes[0, col_idx].set_ylabel('MAE (W/m²)')
            axes[0, col_idx].set_xticks(x)
            axes[0, col_idx].set_xticklabels(cities_ordered, rotation=45, ha='right')
            axes[0, col_idx].legend()
            axes[1, col_idx].set_title(f'R² - {h} horizon')
            axes[1, col_idx].set_ylabel('R²')
            axes[1, col_idx].set_xticks(x)
            axes[1, col_idx].set_xticklabels(cities_ordered, rotation=45, ha='right')
            axes[1, col_idx].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_by_horizon.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to evaluate all model configurations."""
    print("Starting comprehensive model evaluation...")
    
    is_colombia = args.is_colombia
    country = country_name(is_colombia)

    if is_colombia:
        print("Evaluating Colombia models")
    else:
        print("Evaluating India models")

    # Evaluate all model types
    all_metrics = {}

    # Evaluate individual models
    print("\n" + "="*60)
    print("EVALUATING INDIVIDUAL MODELS")
    print("="*60)
    # Hours each site's GNN was tested on; the LSTMs are scored on the same hours
    gnn_hours = load_gnn_test_hours(country)
    if gnn_hours:
        print(f"Scoring the LSTM models on the GNN's test hours ({len(gnn_hours)} sites)")
    else:
        print("WARNING: no GNN test_predictions.csv found; LSTMs are scored on all their own test hours")

    individual_metrics = evaluate_individual_models(country, is_colombia, gnn_hours)
    print(f"Individual metrics returned: {len(individual_metrics)} entries")
    all_metrics.update(individual_metrics)

    # Evaluate joint models
    joint_metrics = evaluate_joint_models(country, is_colombia, gnn_hours)
    all_metrics.update(joint_metrics)

    # Evaluate GNN models
    gnn_metrics = evaluate_gnn_models(country, is_colombia)
    all_metrics.update(gnn_metrics)

    # Create comprehensive summary
    if all_metrics:
        summary_df = create_summary_table(all_metrics, country)
        create_model_comparison_plots(summary_df, results_path(country, "comparison"))
        print("\nComprehensive evaluation completed!")
    else:
        print("\nNo metrics were calculated. Please ensure all models are trained and available.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--is-colombia', action='store_true',
                      help='Evaluate Colombia models instead of India models')
    args = parser.parse_args()

    main()