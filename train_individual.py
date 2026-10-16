"""
GHI Forecasting using LSTM with Individual Training
This script trains separate LSTM models for each location.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import datetime
import pickle

# Configure TensorFlow to use GPU if available
import tensorflow as tf

# Then import ML libraries
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import shared utilities and configurations
from utils import (
    CONFIG, 
    plot_results, 
    plot_loss_history,
    prepare_lstm_input,
    evaluate_model,
    load_data,
    normalize_name_for_path,
    country_name,
    results_path,
    model_path,
)



def load_all_data(locations):
    """
    Load and combine data from all locations.
    
    Args:
        locations: Dictionary of location data from CONFIG
    
    Returns:
        pd.DataFrame: Combined dataset with location identifier
    """
    all_dfs = []
    
    for city in locations.keys():
        print(f"\n{'='*60}")
        print(f"Loading data for {city}")
        print(f"{'='*60}")
        try:
            # Use existing load_data function
            df = load_data(locations, city)
            
            # Add location identifier
            df['location'] = city
            
            # Print detailed information about the loaded data
            print(f"\nData loaded for {city}:")
            print(f"Number of rows: {len(df)}")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            print(f"\nMissing values per column:")
            print(missing_values[missing_values > 0])
            
            # Check for zero values in GHI
            zero_ghi = (df['GHI'] == 0).sum()
            print(f"\nZero GHI values: {zero_ghi} ({zero_ghi/len(df)*100:.2f}%)")
            
            all_dfs.append(df)
            print(f"✓ Successfully loaded {len(df)} rows for {city}")
            
        except Exception as e:
            print(f"× Error loading data for {city}:")
            print(str(e))
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            raise

    if not all_dfs:
        raise ValueError("No data was loaded successfully for any city")
    
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {len(combined_df)} total rows")
    print(f"Locations: {combined_df['location'].unique().tolist()}")
    
    return combined_df

def create_features(df):
    """
    Create features for individual model training.
    
    Args:
        df: DataFrame with data
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("\nCreating features...")
    print(f"Initial shape: {df.shape}")
    print(f"Initial date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Initial memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Create time-based features
    print("\nCreating time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    print("✓ Time-based features created")
    
    # Create targets: GHI for 1, 3, and 7 days ahead at the same hour
    print("\nCreating target variables for multiple horizons...")
    df["target_GHI_1d"] = df["GHI"].shift(-24)   # 1 day ahead (24 hours)
    df["target_GHI_3d"] = df["GHI"].shift(-72)   # 3 days ahead (72 hours)
    df["target_GHI_7d"] = df["GHI"].shift(-168)  # 7 days ahead (168 hours)
    print("✓ Target variables created for 1, 3, and 7 day horizons")
    
    # Create lag features for previous 24 hours of GHI
    print("\nCreating GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    print("✓ GHI lag features created")
    
    # Create lag features for meteorological variables - only use columns that exist
    print("\nCreating meteorological lag features...")
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    # Filter to only include variables that exist in the dataframe
    available_met_features = [feature for feature in meteorological_features if feature in df.columns]
    missing_met_features = [feature for feature in meteorological_features if feature not in df.columns]
    
    if missing_met_features:
        print(f"WARNING: Missing meteorological features: {missing_met_features}")
        print(f"Using available features: {available_met_features}")
    
    for feature in available_met_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    print("✓ Meteorological lag features created")
    
    # Clean data - fill NaN in lag features, then drop rows where target is NaN
    print("\nCleaning data...")
    print(f"Shape before cleaning: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Fill NaN in lag features (but not target columns) using forward fill, backward fill, then 0
    lag_columns = [col for col in df.columns if '_lag_' in col]
    print(f"Filling NaN in {len(lag_columns)} lag feature columns...")
    for col in lag_columns:
        # Forward fill, then backward fill, then fill remaining with 0
        df[col] = df[col].ffill().bfill().fillna(0.0)
    
    # Now drop rows where any target is NaN (these are the last 168 rows and any gaps)
    # We need all three targets for training
    target_columns = ['target_GHI_1d', 'target_GHI_3d', 'target_GHI_7d']
    df_clean = df.dropna(subset=target_columns).reset_index(drop=True)
    
    # Validate we have enough data
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after feature creation. This usually means:\n"
                        "1. Data has too many gaps (missing 168+ consecutive hours)\n"
                        "2. Data is too short (need at least 192 hours for lag features + 7-day target)\n"
                        "3. Missing values in critical columns (GHI, target_GHI_1d, target_GHI_3d, target_GHI_7d)")
    
    if len(df_clean) < 100:
        print(f"WARNING: Only {len(df_clean)} rows remaining after cleaning. This may not be enough for training.")
    
    print(f"\nShape after cleaning: {df_clean.shape}")
    print(f"Date range after cleaning: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
    missing_after = df_clean.isnull().sum()
    if missing_after.sum() > 0:
        print(f"Missing values after cleaning:\n{missing_after[missing_after > 0]}")
    else:
        print("No missing values after cleaning")
    print(f"Memory usage after cleaning: {df_clean.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nFeatures created: {df_clean.columns.tolist()}")
    return df_clean

def split_and_scale_data(df, sequence_length=24, target_columns=["target_GHI_1d", "target_GHI_3d", "target_GHI_7d"]):
    """
    Split and scale the data for individual model training with multiple horizons.
    
    Args:
        df: DataFrame with data
        sequence_length: Length of input sequences
        target_columns: List of target column names (for multiple horizons)
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler)
    """
    print("\nSplitting and scaling data:")
    print(f"Total samples: {len(df)}")
    
    # Validate we have data
    if len(df) == 0:
        raise ValueError("Cannot split and scale empty DataFrame. Check data loading and feature creation.")
    
    # Sort by datetime to ensure chronological order
    df = df.sort_values("datetime")
    
    # Calculate split indices
    n_samples = len(df)
    train_end = int(n_samples * 0.7)  # 70% for training
    val_end = int(n_samples * 0.85)   # 15% for validation
    
    # Split the data
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print("\nSplit sizes:")
    print(f"Train: {len(df_train)} samples")
    print(f"Validation: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Validate we have data in each split
    if len(df_train) == 0:
        raise ValueError("Training set is empty. Need at least some data for training.")
    
    # IMPORTANT FIX: Use a more robust scaling approach
    # Fit scaler on all data to ensure consistent scaling across splits
    # Use the same scaler for all horizons (they're all GHI values)
    target_scaler = MinMaxScaler()
    
    # Fit on all target columns from all data to get the full range
    all_targets = df[target_columns].values
    target_scaler.fit(all_targets)
    
    # Now transform each split
    df_train[target_columns] = target_scaler.transform(df_train[target_columns])
    df_val[target_columns] = target_scaler.transform(df_val[target_columns])
    df_test[target_columns] = target_scaler.transform(df_test[target_columns])
    
    # Print scaling statistics
    print(f"\nScaling statistics:")
    print(f"Original GHI range: [{target_scaler.data_min_.min():.2f}, {target_scaler.data_max_.max():.2f}] W/m²")
    print(f"Scaled GHI range: [0.0, 1.0]")
    for col in target_columns:
        print(f"  {col} - Train: [{df_train[col].min():.4f}, {df_train[col].max():.4f}], "
              f"Val: [{df_val[col].min():.4f}, {df_val[col].max():.4f}], "
              f"Test: [{df_test[col].min():.4f}, {df_test[col].max():.4f}]")
    
    # Create sequences
    X_train, y_train = create_sequences(df_train, sequence_length, target_columns)
    X_val, y_val = create_sequences(df_val, sequence_length, target_columns)
    X_test, y_test = create_sequences(df_test, sequence_length, target_columns)
    
    print("\nFinal sequence shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def create_sequences(df, sequence_length, target_columns):
    """
    Create sequences for individual model training with multiple horizons.
    
    Args:
        df: DataFrame with data
        sequence_length: Length of input sequences
        target_columns: List of target column names (for multiple horizons)
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values (n_samples, n_horizons)
    """
    print(f"\nCreating sequences for {len(df)} samples")
    
    # Get all possible feature columns (but only use those that exist)
    # Note: We use "GHI" as the current GHI value, not the target columns
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
        print(f"Using {len(feature_columns)} out of {len(base_feature_columns)} columns")
    
    # Validate we have enough features
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in dataframe. Check data loading and feature creation.")
    
    # Validate target columns exist
    for col in target_columns:
        if col not in df.columns:
            raise ValueError(f"Target column {col} not found in dataframe.")
    
    # Get all data at once
    data = df[feature_columns].values
    targets = df[target_columns].values  # Shape: (n_samples, n_horizons)
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
    if n_sequences <= 0:
        print(f"WARNING: Not enough data to create sequences. Need at least {sequence_length + 1} samples, got {len(df)}")
        return np.array([]).reshape(0, sequence_length, len(feature_columns)), np.array([]).reshape(0, len(target_columns))
    
    # Pre-allocate arrays
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros((n_sequences, len(target_columns)))  # Multiple targets
    
    # Create sequences using vectorized operations
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = targets[i + sequence_length]  # Get all horizons at once
    
    # Filter out sequences where all targets are zero (night time)
    # Keep sequences where at least one target is non-zero
    mask = np.any(y > 0, axis=1)
    X = X[mask]
    y = y[mask]
    
    print(f"Created {len(X)} valid sequences with {len(target_columns)} horizons")
    return X, y

def custom_ghi_loss(y_true, y_pred):
    """
    Custom loss function that penalizes unrealistic GHI predictions.

    No longer used for training (the individual model now uses the joint model's
    Huber loss); kept so that models trained with it can still be loaded.

    Args:
        y_true: True values (scaled to [0,1])
        y_pred: Predicted values (scaled to [0,1])
    
    Returns:
        Loss value
    """
    # Standard Huber loss
    huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
    
    # Penalty for predictions outside [0, 1] range (since we use sigmoid)
    # This should be minimal since sigmoid constrains output to [0, 1]
    range_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 1.0) + tf.maximum(0.0, -y_pred))
    
    return huber_loss + range_penalty

def create_model(input_shape, n_horizons=3):
    """Create and compile the LSTM for one location.

    Uses exactly the joint model's network, loss, and optimizer
    (train_joint.create_joint_model), so the individual and joint LSTMs differ
    only in their training data: one site versus all sites, plus the joint model's
    one-hot site identifier. Any difference in accuracy then reflects pooling the data,
    not the architecture.
    """
    from train_joint import create_joint_model
    return create_joint_model(input_shape, n_horizons=n_horizons)

def train_model(model, X_train, y_train, X_val, y_val, location, country, batch_size=32, epochs=50):
    """Train the individual model with early stopping and model checkpointing."""
    print(f"\nTraining model for {location}")
    print(f"Training data shape: {X_train.shape}")
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Add TensorBoard callback for monitoring
    log_dir = f"logs/fit/{location}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Add custom callback to monitor predictions
    class PredictionMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Check every 5 epochs
                # Get predictions for a few samples (shape: n_samples, n_horizons)
                sample_preds = self.model.predict(X_val[:5], verbose=0)
                print(f"\nEpoch {epoch} - Sample predictions:")
                print(f"Predictions shape: {sample_preds.shape}")
                print(f"Predictions (1d, 3d, 7d):\n{sample_preds}")
                print(f"Actual values (1d, 3d, 7d):\n{y_val[:5]}")
                print(f"Prediction stats - Min: {np.min(sample_preds):.4f}, Max: {np.max(sample_preds):.4f}, Mean: {np.mean(sample_preds):.4f}")
                
                # Check if predictions are within reasonable bounds
                if np.any(sample_preds < 0) or np.any(sample_preds > 1):
                    print(f"Warning: Predictions outside [0,1] range detected!")
    
    # Create model checkpoint callback (normalize location name for paths with ñ etc.)
    checkpoint_path = model_path(country, "individual", "checkpoints", f"{normalize_name_for_path(location)}.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            reduce_lr,
            tensorboard_callback,
            model_checkpoint,
            PredictionMonitor()
        ],
        verbose=1
    )
    
    return history, model

def evaluate_model(model, X_test, y_test, target_scaler, location, country, horizons=["1d", "3d", "7d"]):
    """
    Evaluate the model on test data for multiple horizons.
    
    Args:
        model: Trained model
        X_test: Test input sequences
        y_test: Test target values (n_samples, n_horizons)
        target_scaler: Scaler used for target values
        location: Location name
        country: "india" or "colombia" (output directory)
        horizons: List of horizon names
    
    Returns:
        dict: Dictionary with evaluation metrics for each horizon
    """
    print(f"\nEvaluating model for {location}")
    
    # Make predictions (shape: n_samples, n_horizons)
    y_pred = model.predict(X_test)
    
    # Print prediction statistics before inverse transform
    print(f"Raw predictions shape: {y_pred.shape}")
    print(f"Raw predictions - Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}, Mean: {np.mean(y_pred):.4f}")
    
    # Clip predictions to [0, 1] range (since we use sigmoid)
    y_pred_clipped = np.clip(y_pred, 0, 1)
    
    # Inverse transform predictions and actual values
    y_test_original = target_scaler.inverse_transform(y_test)  # Shape: (n_samples, n_horizons)
    y_pred_original = target_scaler.inverse_transform(y_pred_clipped)  # Shape: (n_samples, n_horizons)
    
    # Clip predictions to realistic range
    max_realistic_ghi = 1200  # Maximum realistic GHI value
    y_pred_original = np.clip(y_pred_original, 0, max_realistic_ghi)
    
    # Calculate metrics for each horizon
    results = {
        'location': location,
        'horizons': horizons
    }
    
    print(f"\nMetrics for {location}:")
    print(f"{'Horizon':<10} {'MAE (W/m²)':<15} {'RMSE (W/m²)':<15} {'R²':<10} {'Correlation':<12}")
    print("-" * 70)
    
    for i, horizon in enumerate(horizons):
        y_true_h = y_test_original[:, i]
        y_pred_h = y_pred_original[:, i]
        
        mae = mean_absolute_error(y_true_h, y_pred_h)
        rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        r2 = r2_score(y_true_h, y_pred_h)
        correlation = np.corrcoef(y_true_h, y_pred_h)[0, 1]
        
        results[f'mae_{horizon}'] = mae
        results[f'rmse_{horizon}'] = rmse
        results[f'r2_{horizon}'] = r2
        results[f'correlation_{horizon}'] = correlation
        results[f'actual_{horizon}'] = y_true_h
        results[f'predicted_{horizon}'] = y_pred_h
        
        print(f"{horizon:<10} {mae:<15.2f} {rmse:<15.2f} {r2:<10.4f} {correlation:<12.4f}")
    
    # Create and save plots
    plot_results(results, location, country)
    
    return results

def plot_results(results, location, country):
    """
    Create and save plots for model evaluation with multiple horizons.
    
    Args:
        results: Dictionary with evaluation results
        location: Location name
        country: "india" or "colombia" (output directory)
    """
    # Create directory for plots
    plot_dir = results_path(country, "training", "individual", location)
    
    horizons = results['horizons']
    
    # Create scatter plots for each horizon
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 6))
    if len(horizons) == 1:
        axes = [axes]
    
    for i, horizon in enumerate(horizons):
        actual = results[f'actual_{horizon}']
        predicted = results[f'predicted_{horizon}']
        
        axes[i].scatter(actual, predicted, alpha=0.5)
        max_val = max(actual.max(), predicted.max())
        axes[i].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[i].set_xlabel('Actual GHI (W/m²)')
        axes[i].set_ylabel('Predicted GHI (W/m²)')
        axes[i].set_title(f'{horizon} Horizon - {location}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/scatter_plots.png", dpi=150)
    plt.close()
    
    # Create time series plots for each horizon
    fig, axes = plt.subplots(len(horizons), 1, figsize=(15, 5*len(horizons)))
    if len(horizons) == 1:
        axes = [axes]
    
    for i, horizon in enumerate(horizons):
        actual = results[f'actual_{horizon}']
        predicted = results[f'predicted_{horizon}']
        
        axes[i].plot(actual, label='Actual', alpha=0.7)
        axes[i].plot(predicted, label='Predicted', alpha=0.7)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('GHI (W/m²)')
        axes[i].set_title(f'{horizon} Horizon - {location}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/time_series.png", dpi=150)
    plt.close()
    
    # Create combined comparison plot
    plt.figure(figsize=(15, 8))
    for i, horizon in enumerate(horizons):
        actual = results[f'actual_{horizon}']
        predicted = results[f'predicted_{horizon}']
        mae = results[f'mae_{horizon}']
        plt.plot(actual[:500], label=f'Actual {horizon}', alpha=0.5, linestyle='-')
        plt.plot(predicted[:500], label=f'Predicted {horizon} (MAE: {mae:.1f})', alpha=0.7, linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.title(f'GHI Forecast Comparison - {location}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_dir}/comparison.png", dpi=150)
    plt.close()

def main(skip_training=False, debug_data_loading=False):
    """
    Main execution function for individual training.
    
    Args:
        skip_training (bool): If True, load pre-trained models instead of training new ones
        debug_data_loading (bool): If True, stop after data loading for debugging
    """
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    is_colombia = args.is_colombia
    country = country_name(is_colombia)
    
    # Load and process data
    print("\nLoading and processing data...")
    try:
        if is_colombia:
            print("Using Colombia data locations")
            df = load_all_data(CONFIG["data_locations_colombia"])
        else:
            print("Using India data locations")
            df = load_all_data(CONFIG["data_locations"])
        print("✓ Data loaded successfully")
        
        if debug_data_loading:
            print("\nDebug mode: Stopping after data loading")
            return
            
    except Exception as e:
        print(f"× Error loading data: {str(e)}")
        raise

    # Get unique locations
    locations = sorted(df['location'].unique().tolist())
    print(f"\nLocations: {locations}")

    # Process each location separately
    all_results = []
    failed_locations = []
    
    for location in locations:
        print(f"\n{'='*80}")
        print(f"Processing {location}")
        print(f"{'='*80}")
        
        try:
            # Filter data for current location
            df_loc = df[df['location'] == location].copy()
            
            # Create features
            print("\nCreating features...")
            df_features = create_features(df_loc)
            
            # Split and scale data
            print("\nSplitting and scaling data...")
            target_columns = ["target_GHI_1d", "target_GHI_3d", "target_GHI_7d"]
            (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler) = split_and_scale_data(
                df_features, sequence_length=24, target_columns=target_columns
            )
            
            # Create and train model
            print("\nCreating model...")
            n_horizons = len(target_columns)
            model = create_model((X_train.shape[1], X_train.shape[2]), n_horizons=n_horizons)
            model.summary()
            
            # Print data statistics before training
            print(f"\nData statistics for {location}:")
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"X_train range: [{np.min(X_train):.4f}, {np.max(X_train):.4f}]")
            print(f"y_train range: [{np.min(y_train):.4f}, {np.max(y_train):.4f}]")
            print(f"y_val range: [{np.min(y_val):.4f}, {np.max(y_val):.4f}]")
            print(f"y_test range: [{np.min(y_test):.4f}, {np.max(y_test):.4f}]")
            
            model_file = model_path(country, "individual", f"{normalize_name_for_path(location)}.h5")
            
            if skip_training:
                if os.path.exists(model_file):
                    print(f"\nLoading pre-trained model...")
                    model = tf.keras.models.load_model(model_file, custom_objects={'custom_ghi_loss': custom_ghi_loss})
                    history = None
                else:
                    print("× No pre-trained model found")
                    failed_locations.append(location)
                    continue
            else:
                print("\nTraining model...")
                history, model = train_model(model, X_train, y_train, X_val, y_val, location, country)
                
                # Save model
                model.save(model_file)
                print(f"✓ Model saved to {model_file}")
            
            # Test the model on validation data before evaluation
            print("\nTesting model on validation data...")
            y_val_pred = model.predict(X_val[:10], verbose=0)
            print(f"Validation predictions shape: {y_val_pred.shape}")
            print("Validation predictions (1d, 3d, 7d) - should be in [0, 1]:")
            print(y_val_pred)
            print("Validation actual values (1d, 3d, 7d):")
            print(y_val[:10])
            
            # Evaluate model
            print("\nEvaluating model...")
            horizons = ["1d", "3d", "7d"]
            results = evaluate_model(model, X_test, y_test, target_scaler, location, country, horizons=horizons)
            all_results.append(results)
            
        except Exception as e:
            print(f"× Error processing {location}: {str(e)}")
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            failed_locations.append(location)
            continue
    
    # Create summary of results
    if all_results:
        # Create summary with metrics for each horizon
        summary_rows = []
        for r in all_results:
            row = {'Location': r['location']}
            for horizon in r['horizons']:
                row[f'MAE_{horizon}'] = r[f'mae_{horizon}']
                row[f'RMSE_{horizon}'] = r[f'rmse_{horizon}']
                row[f'R²_{horizon}'] = r[f'r2_{horizon}']
                row[f'Correlation_{horizon}'] = r[f'correlation_{horizon}']
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = results_path(country, "training", "individual", "summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary of results saved to {summary_file}")
    
    print("\nAll locations have been processed")
    if failed_locations:
        # Non-zero exit so a pipeline running this script stops
        sys.exit(f"× Failed locations: {failed_locations}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI Forecasting using Individual LSTM')
    parser.add_argument('--is-colombia', action='store_true',
                      help='Load data from Colombia locations')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained models')
    parser.add_argument('--debug-data', action='store_true',
                      help='Stop after data loading for debugging')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training, debug_data_loading=args.debug_data) 