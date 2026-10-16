# -*- coding: utf-8 -*-
"""
GHI forecasting with a joint LSTM trained on the pooled data of all sites
This script trains a single LSTM model using data from all locations with proper scaling.
"""

import os
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
    
    # Print summary statistics for each location
    print("\nSummary statistics by location:")
    for location in combined_df['location'].unique():
        loc_df = combined_df[combined_df['location'] == location]
        print(f"\n{location}:")
        print(f"Number of rows: {len(loc_df)}")
        print(f"Date range: {loc_df['datetime'].min()} to {loc_df['datetime'].max()}")
        print(f"Zero GHI values: {(loc_df['GHI'] == 0).sum()} ({(loc_df['GHI'] == 0).sum()/len(loc_df)*100:.2f}%)")
        print(f"Missing values: {loc_df.isnull().sum().sum()}")
    
    return combined_df

def create_joint_features(df):
    """
    Create features for joint training, including location-specific features.
    
    Args:
        df: Combined DataFrame with data from all locations
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("\nCreating joint features...")
    print(f"Initial shape: {df.shape}")
    print(f"Initial locations: {df['location'].unique()}")
    print(f"Initial date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Initial rows per location:\n{df['location'].value_counts()}")
    print(f"Initial memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Create time-based features
    print("\nCreating time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    print("✓ Time-based features created")
    print(f"Shape after time features: {df.shape}")
    print(f"Rows per location after time features:\n{df['location'].value_counts()}")
    
    # Targets and lags are shifted within each location: df stacks the sites one after
    # another, so shifting the whole frame would give each site's last week the targets
    # of the next site's first days (and its first rows the previous site's lags)
    by_location = df.groupby("location", sort=False)

    # Create targets: GHI for 1, 3, and 7 days ahead at the same hour
    print("\nCreating target variables for multiple horizons...")
    df["target_GHI_1d"] = by_location["GHI"].shift(-24)   # 1 day ahead (24 hours)
    df["target_GHI_3d"] = by_location["GHI"].shift(-72)   # 3 days ahead (72 hours)
    df["target_GHI_7d"] = by_location["GHI"].shift(-168)  # 7 days ahead (168 hours)
    print("✓ Target variables created for 1, 3, and 7 day horizons")
    print(f"Shape after target creation: {df.shape}")
    print(f"Rows per location after target creation:\n{df['location'].value_counts()}")
    
    # Create lag features for previous 24 hours of GHI
    print("\nCreating GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = by_location["GHI"].shift(lag)
    print("✓ GHI lag features created")
    print(f"Shape after GHI lag features: {df.shape}")
    print(f"Rows per location after GHI lag features:\n{df['location'].value_counts()}")
    
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
        df[f"{feature}_lag_24"] = by_location[feature].shift(24)
    print("✓ Meteorological lag features created")
    print(f"Shape after meteorological features: {df.shape}")
    print(f"Rows per location after meteorological features:\n{df['location'].value_counts()}")
    
    # Create location-specific features
    print("\nCreating location features...")
    # One-hot encode location - handle both old and new scikit-learn versions
    try:
        # Try new scikit-learn version first
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        # Fall back to old scikit-learn version
        encoder = OneHotEncoder(sparse=False)
    
    location_encoded = encoder.fit_transform(df[['location']])
    location_df = pd.DataFrame(location_encoded, 
                             columns=[f"location_{loc}" for loc in encoder.categories_[0]])
    
    # Combine with original data
    df = pd.concat([df, location_df], axis=1)
    print("✓ Location features created")
    print(f"Shape after location features: {df.shape}")
    print(f"Rows per location after location features:\n{df['location'].value_counts()}")
    
    # Clean data - fill NaN in lag features, then drop rows where target is NaN
    print("\nCleaning data...")
    print(f"Shape before cleaning: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    print(f"Rows per location before cleaning:\n{df['location'].value_counts()}")
    
    # Fill NaN in lag features (but not target columns) using forward fill, backward fill, then 0
    lag_columns = [col for col in df.columns if '_lag_' in col]
    print(f"Filling NaN in {len(lag_columns)} lag feature columns...")
    for col in lag_columns:
        # Forward fill, then backward fill, then fill remaining with 0
        df[col] = df.groupby("location", sort=False)[col].transform(lambda s: s.ffill().bfill()).fillna(0.0)
    
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
    
    # Check data per location
    location_counts = df_clean['location'].value_counts()
    locations_with_insufficient_data = location_counts[location_counts < 100]
    if len(locations_with_insufficient_data) > 0:
        print(f"WARNING: Some locations have very few rows after cleaning:")
        for loc, count in locations_with_insufficient_data.items():
            print(f"  {loc}: {count} rows (may not be enough for training)")
    
    print(f"\nShape after cleaning: {df_clean.shape}")
    print(f"Rows per location after cleaning:\n{df_clean['location'].value_counts()}")
    print(f"Date range after cleaning: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
    missing_after = df_clean.isnull().sum()
    if missing_after.sum() > 0:
        print(f"Missing values after cleaning:\n{missing_after[missing_after > 0]}")
    else:
        print("No missing values after cleaning")
    print(f"Memory usage after cleaning: {df_clean.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nFeatures created: {df_clean.columns.tolist()}")
    return df_clean

def split_and_scale_joint_data(df, locations, sequence_length=24, target_columns=["target_GHI_1d", "target_GHI_3d", "target_GHI_7d"], country="india"):
    """
    Split and scale the data for joint model training with multiple horizons.
    
    Args:
        df: DataFrame with all data
        locations: List of locations
        sequence_length: Length of input sequences
        target_columns: List of target column names (for multiple horizons)
        country: "india" or "colombia" (output directory)
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler)
    """
    print("\nSplitting and scaling data:")
    print(f"Total samples: {len(df)}")
    print(f"Locations: {locations}")
    
    # Create a more balanced split that ensures each location has data in all splits
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for location in locations:
        df_loc = df[df["location"] == location].copy()
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Calculate split indices
        n_samples = len(df_loc)
        train_end = int(n_samples * 0.7)  # 70% for training
        val_end = int(n_samples * 0.85)   # 15% for validation
        
        # Split the data
        train_dfs.append(df_loc.iloc[:train_end])
        val_dfs.append(df_loc.iloc[train_end:val_end])
        test_dfs.append(df_loc.iloc[val_end:])
    
    # Combine splits
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    print("\nSplit sizes:")
    print(f"Train: {len(df_train)} samples")
    print(f"Validation: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Print location distribution in each split
    for split_name, split_df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        print(f"\n{split_name} split location distribution:")
        for location in locations:
            count = len(split_df[split_df["location"] == location])
            print(f"{location}: {count} samples")
    
    # IMPORTANT FIX: Scale target values using the same approach as individual models
    # Fit scaler on all target columns from all data to get the full range
    target_scaler = MinMaxScaler()
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
    
    # Save the target scaler
    scaler_path = model_path(country, "joint", "target_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"\nTarget scaler saved to {scaler_path}")
    
    # Create sequences for each location
    X_train, y_train = create_sequences_joint(df_train, locations, sequence_length, target_columns)
    X_val, y_val = create_sequences_joint(df_val, locations, sequence_length, target_columns)
    X_test, y_test = create_sequences_joint(df_test, locations, sequence_length, target_columns)
    
    print("\nFinal sequence shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def create_joint_model(input_shape, n_horizons=3):
    """Create and compile a more complex joint LSTM model with attention mechanism for multiple horizons."""
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First LSTM block with residual connection
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second LSTM block with residual connection
    lstm2 = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm2 = tf.keras.layers.Dropout(0.3)(lstm2)
    x = tf.keras.layers.Add()([x, lstm2])  # Residual connection
    
    # Third LSTM block
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(64)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    
    # Apply attention weights
    x = tf.keras.layers.Multiply()([x, attention_weights])
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(x)
    
    # Dense layers with residual connection
    dense1 = tf.keras.layers.Dense(32, activation='relu')(x)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.3)(dense1)
    
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    # Final residual connection
    x = tf.keras.layers.Add()([dense1, dense2])
    
    # Output layer with sigmoid activation to constrain output to [0, 1]
    # Output shape: (batch_size, n_horizons) for 1d, 3d, 7d predictions
    outputs = tf.keras.layers.Dense(n_horizons, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    # Compile model with Huber loss for robustness
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )
    
    return model

def create_sequences_joint(df, locations, sequence_length, target_columns):
    """
    Create sequences for joint model training with multiple horizons.
    
    Args:
        df: DataFrame with data
        locations: List of locations
        sequence_length: Length of input sequences
        target_columns: List of target column names (for multiple horizons)
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values (n_samples, n_horizons)
    """
    print(f"\nCreating sequences for {len(df)} samples")
    print(f"Locations: {locations}")
    
    # Initialize lists to store sequences
    all_X_sequences = []
    all_y_sequences = []
    
    # Process each location separately
    for location in locations:
        print(f"\nProcessing {location}...")
        df_loc = df[df['location'] == location].copy()
        
        if len(df_loc) == 0:
            print(f"No data found for {location}")
            continue
            
        # Create one-hot encoded location vector for the current location
        location_vector = np.zeros(len(locations))
        location_vector[locations.index(location)] = 1
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Get all required features at once - filter to only include columns that exist
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
        feature_columns = [col for col in base_feature_columns if col in df_loc.columns]
        missing_columns = [col for col in base_feature_columns if col not in df_loc.columns]
        
        if missing_columns:
            print(f"  WARNING: Missing feature columns for {location}: {missing_columns}")
            print(f"  Using {len(feature_columns)} out of {len(base_feature_columns)} columns")
        
        # Validate we have enough features
        if len(feature_columns) == 0:
            print(f"  ERROR: No feature columns found for {location}. Skipping.")
            continue
        
        # Validate target columns exist
        for col in target_columns:
            if col not in df_loc.columns:
                print(f"  ERROR: Target column {col} not found for {location}. Skipping.")
                continue
        
        # Get all data at once
        data = df_loc[feature_columns].values
        targets = df_loc[target_columns].values  # Shape: (n_samples, n_horizons)
        
        # Calculate number of sequences
        n_sequences = len(df_loc) - sequence_length
        
        # Check if we have enough data to create sequences
        if n_sequences <= 0:
            print(f"  WARNING: Not enough data for {location}. Need at least {sequence_length + 1} samples, got {len(df_loc)}. Skipping.")
            continue
        
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
        
        # Check if we have any valid sequences after filtering
        if len(X) == 0:
            print(f"  WARNING: No valid sequences for {location} after filtering (all targets were zero/night time). Skipping.")
            continue
        
        # Add location features
        location_features = np.tile(location_vector, (len(X), sequence_length, 1))
        X = np.concatenate([X, location_features], axis=2)
        
        all_X_sequences.append(X)
        all_y_sequences.append(y)
        
        print(f"Created {len(X)} valid sequences for {location} with {len(target_columns)} horizons")
    
    if not all_X_sequences:
        raise ValueError("No valid sequences were created for any location")
    
    # Combine sequences from all locations
    X = np.concatenate(all_X_sequences, axis=0)
    y = np.concatenate(all_y_sequences, axis=0)
    
    print(f"\nFinal sequence shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    
    # Print sequence distribution by location
    print("\nSequence distribution by location:")
    for i, location in enumerate(locations):
        location_mask = X[:, 0, -len(locations):][:, i] == 1
        print(f"{location}: {np.sum(location_mask)} sequences")
    
    return X, y

def train_joint_model(model, X_train, y_train, X_val, y_val, country, batch_size=32, epochs=50):
    """Train the joint model with early stopping and model checkpointing."""
    # Print training data statistics
    print("\nTraining data statistics:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_train min: {np.min(X_train):.4f}")
    print(f"X_train max: {np.max(X_train):.4f}")
    print(f"X_train mean: {np.mean(X_train):.4f}")
    print(f"X_train std: {np.std(X_train):.4f}")
    print(f"y_train min: {np.min(y_train):.4f}")
    print(f"y_train max: {np.max(y_train):.4f}")
    print(f"y_train mean: {np.mean(y_train):.4f}")
    print(f"y_train std: {np.std(y_train):.4f}")
    
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
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    
    # Create model checkpoint callback
    checkpoint_path = model_path(country, "joint", "checkpoint.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Train the model with all callbacks
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

def evaluate_joint_model(model, X_test, y_test, target_scaler, locations, country, horizons=["1d", "3d", "7d"]):
    """
    Evaluate the joint model on test data for multiple horizons.
    
    Args:
        model: Trained model
        X_test: Test input sequences
        y_test: Test target values (n_samples, n_horizons)
        target_scaler: Scaler used for target values
        locations: List of location names
        country: "india" or "colombia" (output directory)
        horizons: List of horizon names
    
    Returns:
        dict: Dictionary with evaluation metrics for each horizon
    """
    print(f"\nEvaluating joint model")
    
    # Make predictions (shape: n_samples, n_horizons)
    y_pred = model.predict(X_test, verbose=0)
    
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
        'horizons': horizons
    }
    
    print(f"\nOverall Metrics:")
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
    
    # Calculate metrics per location
    print(f"\nMetrics by Location:")
    location_results = {}
    for location in locations:
        # Find indices for this location in test set
        # Location features are at the end of the feature vector
        location_indices = []
        for idx in range(len(X_test)):
            # Location one-hot encoding is at the end of each timestep
            # We need to check the location features in the first timestep
            location_features = X_test[idx, 0, -len(locations):]
            loc_idx = np.argmax(location_features)
            if locations[loc_idx] == location:
                location_indices.append(idx)
        
        if len(location_indices) == 0:
            print(f"  {location}: No test samples found")
            continue
        
        location_indices = np.array(location_indices)
        y_true_loc = y_test_original[location_indices]
        y_pred_loc = y_pred_original[location_indices]
        
        print(f"\n  {location}:")
        print(f"    {'Horizon':<10} {'MAE (W/m²)':<15} {'RMSE (W/m²)':<15} {'R²':<10} {'Correlation':<12}")
        print(f"    {'-' * 70}")
        
        loc_results = {}
        for i, horizon in enumerate(horizons):
            y_true_h = y_true_loc[:, i]
            y_pred_h = y_pred_loc[:, i]
            
            mae = mean_absolute_error(y_true_h, y_pred_h)
            rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
            r2 = r2_score(y_true_h, y_pred_h)
            correlation = np.corrcoef(y_true_h, y_pred_h)[0, 1]
            
            loc_results[f'mae_{horizon}'] = mae
            loc_results[f'rmse_{horizon}'] = rmse
            loc_results[f'r2_{horizon}'] = r2
            loc_results[f'correlation_{horizon}'] = correlation
            
            print(f"    {horizon:<10} {mae:<15.2f} {rmse:<15.2f} {r2:<10.4f} {correlation:<12.4f}")
        
        location_results[location] = loc_results
    
    results['location_results'] = location_results
    
    # Create and save plots
    plot_joint_results(results, country, horizons)
    
    # Save summary CSV
    summary_rows = []
    summary_rows.append({
        'Location': 'Overall',
        **{f'MAE_{h}': results[f'mae_{h}'] for h in horizons},
        **{f'RMSE_{h}': results[f'rmse_{h}'] for h in horizons},
        **{f'R²_{h}': results[f'r2_{h}'] for h in horizons},
        **{f'Correlation_{h}': results[f'correlation_{h}'] for h in horizons}
    })
    for location, loc_results in location_results.items():
        summary_rows.append({
            'Location': location,
            **{f'MAE_{h}': loc_results[f'mae_{h}'] for h in horizons},
            **{f'RMSE_{h}': loc_results[f'rmse_{h}'] for h in horizons},
            **{f'R²_{h}': loc_results[f'r2_{h}'] for h in horizons},
            **{f'Correlation_{h}': loc_results[f'correlation_{h}'] for h in horizons}
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_path(country, "training", "joint", "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    
    return results

def plot_joint_results(results, country, horizons=["1d", "3d", "7d"]):
    """
    Create and save plots for joint model evaluation with multiple horizons.
    
    Args:
        results: Dictionary with evaluation results
        country: "india" or "colombia" (output directory)
        horizons: List of horizon names
    """
    # Create directory for plots
    plot_dir = results_path(country, "training", "joint", "plots")
    
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
        axes[i].set_title(f'{horizon} Horizon - Joint Model')
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
        
        axes[i].plot(actual[:1000], label='Actual', alpha=0.7)
        axes[i].plot(predicted[:1000], label='Predicted', alpha=0.7)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('GHI (W/m²)')
        axes[i].set_title(f'{horizon} Horizon - Joint Model')
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
    plt.title('GHI Forecast Comparison - Joint Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_dir}/comparison.png", dpi=150)
    plt.close()

def main(skip_training=False, debug_data_loading=False, is_colombia=False):
    """
    Main execution function for joint training.
    
    Args:
        skip_training (bool): If True, load pre-trained model instead of training new one
        debug_data_loading (bool): If True, stop after data loading for debugging
        is_colombia (bool): If True, load data from Colombia locations
    """
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
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
    
    try:
        # Create features
        print("\nCreating features...")
        df_features = create_joint_features(df)
        
        # Split and scale data
        print("\nSplitting and scaling data...")
        target_columns = ["target_GHI_1d", "target_GHI_3d", "target_GHI_7d"]
        (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler) = split_and_scale_joint_data(
            df_features, locations, sequence_length=24, target_columns=target_columns, country=country
        )
        
        # Create and train model
        print("\nCreating model...")
        n_horizons = len(target_columns)
        model = create_joint_model((X_train.shape[1], X_train.shape[2]), n_horizons=n_horizons)
        model.summary()
        
        # Print data statistics before training
        print(f"\nData statistics:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_train range: [{np.min(X_train):.4f}, {np.max(X_train):.4f}]")
        print(f"y_train range: [{np.min(y_train):.4f}, {np.max(y_train):.4f}]")
        print(f"y_val range: [{np.min(y_val):.4f}, {np.max(y_val):.4f}]")
        print(f"y_test range: [{np.min(y_test):.4f}, {np.max(y_test):.4f}]")
        
        model_file = model_path(country, "joint", "model.h5")
        
        if skip_training:
            if os.path.exists(model_file):
                print(f"\nLoading pre-trained model...")
                model = tf.keras.models.load_model(model_file)
                history = None
            else:
                raise FileNotFoundError(f"No pre-trained model found at {model_file}")
        else:
            print("\nTraining model...")
            history, model = train_joint_model(model, X_train, y_train, X_val, y_val, country)
            
            # Save model
            model.save(model_file)
            print(f"✓ Model saved to {model_file}")
        
        # Test the model on validation data
        print("\nTesting model on validation data...")
        y_val_pred = model.predict(X_val[:10], verbose=0)
        print(f"Validation predictions shape: {y_val_pred.shape}")
        print("Validation predictions (1d, 3d, 7d) - should be in [0, 1]:")
        print(y_val_pred)
        print("Validation actual values (1d, 3d, 7d):")
        print(y_val[:10])
        
        # Test inverse transform
        y_val_pred_clipped = np.clip(y_val_pred, 0, 1)
        y_val_pred_original = target_scaler.inverse_transform(y_val_pred_clipped)
        y_val_original = target_scaler.inverse_transform(y_val[:10])
        print("Validation predictions (original scale):")
        print(y_val_pred_original)
        print("Validation actual values (original scale):")
        print(y_val_original)
        
        # Evaluate model on test data
        print("\nEvaluating model on test data...")
        horizons = ["1d", "3d", "7d"]
        results = evaluate_joint_model(model, X_test, y_test, target_scaler, locations, country, horizons=horizons)
        
        print(f"\n✓ Joint model training completed successfully")
        
    except Exception as e:
        print(f"× Error in joint training: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI forecasting with a joint LSTM')
    parser.add_argument('--is-colombia', action='store_true',
                      help='Load data from Colombia locations')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained models')
    parser.add_argument('--debug-data', action='store_true',
                      help='Stop after data loading for debugging')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training, debug_data_loading=args.debug_data, is_colombia=args.is_colombia) 