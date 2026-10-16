# train_gnn.py
# Multi-site GHI forecasting with a graph neural network (GNN)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv

# Keras 3 passes mask=[None, None] for multi-input layers; Spektral's GCNConv then does
# "output *= mask[0]" which raises "None values not supported". Subclass to skip mask when mask[0] is None.
class GCNConvSafe(GCNConv):
    """GCNConv that only applies mask when mask[0] is not None (Keras 3 compatibility)."""
    def call(self, inputs, mask=None):
        from tensorflow.keras import backend as K
        from spektral.layers import ops
        x, a = inputs
        output = K.dot(x, self.kernel)
        output = ops.modal_dot(a, output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if mask is not None and len(mask) > 0 and mask[0] is not None:
            output = output * mask[0]
        if self.activation is not None:
            output = self.activation(output)
        return output

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import datetime
import gc
import json
import shutil
import time
import warnings
from math import radians, cos, sin, asin, sqrt
from utils import CONFIG, load_data, get_city_coordinates, country_name, results_path, model_path

def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points, in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r

# Suppress sklearn feature name warnings (we're using .values to avoid them)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set global parameters
SEQUENCE_LENGTH = 24  # Use 24 hours for sequence
# Three time horizons: 1d (24h), 3d (72h), 7d (168h) at same hour
HORIZON_HOURS = [24, 72, 168]  # 1d, 3d, 7d
TARGET_COLUMNS_NAMES = ["target_GHI_1d", "target_GHI_3d", "target_GHI_7d"]
N_HORIZONS = 3

# Same curated 35-feature whitelist as the Individual and Joint LSTM models
# (see train_individual.py / train_joint.py), rather than all available
# columns, so the GNN never sees raw NSRDB variables (e.g., Cloud Type,
# Clearsky GHI/DNI/DHI, Solar Zenith Angle) that the LSTM models lack.
FEATURE_COLUMNS = (
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

# Default to India locations, will be updated in main() if needed
CITIES = list(CONFIG["data_locations"].keys())
CITY_IDX = {city: i for i, city in enumerate(CITIES)}

# ----------------------------------------------
# Data loading and sequence preparation
# ----------------------------------------------
def create_features(df):
    """Create features with three time-horizon targets: 1d, 3d, 7d (same hour)."""
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    # Debug: Check GHI values before processing
    print(f"      GHI stats before processing:")
    print(f"        Range: [{df['GHI'].min():.2f}, {df['GHI'].max():.2f}]")
    print(f"        Mean: {df['GHI'].mean():.2f}")
    print(f"        Non-zero: {(df['GHI'] > 0).sum()} out of {len(df)}")
    print(f"        Sample GHI values: {df['GHI'].head(20).tolist()}")
    
    print(f"      Adding time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    print(f"      Adding GHI lag features...")
    # Add lag features for previous 24 hours
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)

    print(f"      Adding meteorological lag features...")
    # Add meteorological features - only use columns that exist in the dataframe
    met_vars = ["Temperature", "Relative Humidity", "Pressure", 
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    
    # Filter to only include variables that exist in the dataframe
    available_met_vars = [var for var in met_vars if var in df.columns]
    missing_vars = [var for var in met_vars if var not in df.columns]
    
    if missing_vars:
        print(f"      WARNING: Missing meteorological variables: {missing_vars}")
        print(f"      Using available variables: {available_met_vars}")
    
    for var in available_met_vars:
        df[f"{var}_lag_24"] = df[var].shift(24)

    print(f"      Adding target variables for 1d, 3d, 7d horizons...")
    # Three horizons: same hour at 1 day, 3 days, 7 days ahead
    target_columns = []
    for col_name, hours_ahead in zip(TARGET_COLUMNS_NAMES, HORIZON_HOURS):
        df[col_name] = df["GHI"].shift(-hours_ahead)
        target_columns.append(col_name)
    
    # Debug: Check target values after creation
    print(f"      Target GHI stats after creation:")
    for col in target_columns:
        print(f"        {col}: range=[{df[col].min():.2f}, {df[col].max():.2f}], mean={df[col].mean():.2f}")
    
    print(f"      Dropping NaN values...")
    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    final_len = len(df)
    print(f"      Dropped {original_len - final_len} rows with NaN values")
    print(f"      Final shape: {df.shape}")
    
    # Debug: Check targets after dropping NaN
    print(f"      Target GHI stats after dropping NaN:")
    for col in target_columns:
        print(f"        {col}: range=[{df[col].min():.2f}, {df[col].max():.2f}], mean={df[col].mean():.2f}")
    
    return df, target_columns

def compute_weighted_adjacency_with_coordinates(df_all, city_coordinates=None, alpha=0.5):
    """
    Compute adjacency matrix using actual geographical coordinates if available.
    
    Args:
        df_all: DataFrame with location data
        city_coordinates: Dict mapping city names to (lat, lon) tuples
        alpha: Weight for combining geographical and correlation weights
    
    Returns:
        np.array: Weighted adjacency matrix
    """
    print("    Computing weighted adjacency matrix (with coordinates)...")
    
    # Get the actual cities present in the data
    actual_cities = df_all['location'].unique()
    num_cities = len(actual_cities)
    print(f"    Found {num_cities} cities in data: {actual_cities}")
    
    # Initialize with identity matrix (self-connections)
    geo_weights = np.eye(num_cities, dtype=np.float32)
    corr_weights = np.eye(num_cities, dtype=np.float32)
    
    if num_cities > 1:
        if city_coordinates is not None:
            print("    Using provided geographical coordinates...")
            # Use actual geographical distances
            for i, city_i in enumerate(actual_cities):
                for j, city_j in enumerate(actual_cities):
                    if i != j and city_i in city_coordinates and city_j in city_coordinates:
                        # Calculate actual geographical distance
                        lat1, lon1 = city_coordinates[city_i]
                        lat2, lon2 = city_coordinates[city_j]
                        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
                        
                        # Convert distance to weight (closer cities have higher weights)
                        # Use exponential decay: weight = exp(-distance/scale)
                        scale = 500  # Scale factor in km
                        geo_weight = np.exp(-distance_km / scale)
                        geo_weight = max(0.1, min(0.9, geo_weight))  # Clamp to reasonable range
                        
                        geo_weights[i, j] = geo_weight
                        corr_weights[i, j] = geo_weight * 0.8  # Correlation slightly lower
        else:
            print("    No coordinates provided, using simplified distance-based weights...")
            # Fall back to simplified approach
            for i in range(num_cities):
                for j in range(i + 1, num_cities):
                    distance_factor = 1.0 / (abs(i - j) + 1)
                    geo_weight = max(0.3, min(0.9, distance_factor))
                    corr_weight = max(0.2, min(0.8, distance_factor * 0.8))
                    
                    geo_weights[i, j] = geo_weight
                    geo_weights[j, i] = geo_weight
                    corr_weights[i, j] = corr_weight
                    corr_weights[j, i] = corr_weight
        
        print(f"    Adjacency matrix pattern:")
        print(f"      Geographic weights range: [{geo_weights.min():.2f}, {geo_weights.max():.2f}]")
        print(f"      Correlation weights range: [{corr_weights.min():.2f}, {corr_weights.max():.2f}]")
    
    # Combine weights
    weighted_adj = alpha * geo_weights + (1 - alpha) * corr_weights
    print(f"    Adjacency matrix computed: {weighted_adj.shape}")
    
    return weighted_adj

def split_dates(unique_dates, train_ratio=0.7, val_ratio=0.15):
    """Chronological split by day, identical to split_data_by_days() in
    evaluate.py, so the GNN is tested on the same days
    as the Individual and Joint LSTM models."""
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    return unique_dates[:train_end], unique_dates[train_end:val_end], unique_dates[val_end:]

def build_split_arrays(df_all, actual_cities, split_date_list, target_columns, split_name):
    """Build hourly samples for one split, aligned across cities.

    Mirrors the LSTM sample construction (create_sequences_individual_multi_horizon):
    within each split, sample k uses the features of row k-1 (the last step of the
    LSTM's 24-hour window) and the targets of row k, starting at k = SEQUENCE_LENGTH.
    Samples whose three targets are all zero (night) are masked out, as the LSTMs
    drop them.

    Returns:
        X: (T, C, F) raw features, Y: (T, C, 3) raw targets (W/m²),
        M: (T, C) bool daytime mask, timestamps: DatetimeIndex (T,)
    """
    split_date_set = set(split_date_list)
    per_city = []
    for city in actual_cities:
        sub = df_all[(df_all['location'] == city) & (df_all['date'].isin(split_date_set))]
        sub = sub.sort_values('datetime')
        n = len(sub)
        if n <= SEQUENCE_LENGTH:
            raise ValueError(f"{split_name}: only {n} rows for {city}, need more than {SEQUENCE_LENGTH}")
        feats = sub[FEATURE_COLUMNS].values[SEQUENCE_LENGTH - 1:n - 1].astype(np.float32)
        targs = sub[target_columns].values[SEQUENCE_LENGTH:].astype(np.float32)
        # Align sites on the hour: most NSRDB files are stamped at :30, but some
        # (e.g., Taminango 2018-2019) are stamped at :00
        ts = pd.DatetimeIndex(pd.to_datetime(sub['datetime'].values[SEQUENCE_LENGTH:])).floor('h')
        if not ts.is_unique:
            raise ValueError(f"{split_name}: duplicate timestamps for {city}")
        per_city.append((feats, targs, ts))

    # One graph per timestamp shared by all sites
    common = per_city[0][2]
    for _, _, ts in per_city[1:]:
        common = common.intersection(ts)
    common = common.sort_values()
    for city, (_, _, ts) in zip(actual_cities, per_city):
        if len(ts) != len(common):
            print(f"  WARNING ({split_name}): {city} has {len(ts)} timestamps, {len(ts) - len(common)} not shared by all sites and dropped")

    X = np.stack([f[ts.get_indexer(common)] for f, _, ts in per_city], axis=1)
    Y = np.stack([t[ts.get_indexer(common)] for _, t, ts in per_city], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    M = np.any(Y > 0, axis=2)
    keep = M.any(axis=1)  # drop timestamps that are night at every site
    X, Y, M, common = X[keep], Y[keep], M[keep], common[keep]

    print(f"  {split_name}: {len(common)} hourly graphs, "
          f"{int(M.sum())} daytime node samples ({M.sum(axis=0).min()}-{M.sum(axis=0).max()} per site)")
    return X, Y, M, common

def make_graphs(X_scaled, Y_scaled, M, adj_matrix):
    """One Spektral graph per timestamp. y packs the flattened (C*3) scaled targets
    followed by the matching (C*3) daytime mask, split apart again in masked_mse()."""
    a = adj_matrix.astype(np.float32)
    mask = np.repeat(M.astype(np.float32), N_HORIZONS, axis=1)  # (T, C*3), city-major like the targets
    y_flat = Y_scaled.reshape(len(Y_scaled), -1).astype(np.float32)
    return [Graph(x=X_scaled[t], a=a, y=np.concatenate([y_flat[t], mask[t]]))
            for t in range(len(X_scaled))]

def build_hourly_graphs(df_all, adj_matrix, actual_cities, target_columns):
    """Build hourly train/val/test graphs for three-horizon (1d, 3d, 7d) forecasting."""
    print("Building hourly graphs for three-horizon (1d, 3d, 7d) forecasting...")

    df_all['date'] = pd.to_datetime(df_all['datetime']).dt.date
    unique_dates = sorted(df_all['date'].unique())
    train_dates, val_dates, test_dates = split_dates(unique_dates)
    print(f"Found {len(unique_dates)} unique dates: train {len(train_dates)}, val {len(val_dates)}, test {len(test_dates)}")
    print(f"  Test period: {test_dates[0]} to {test_dates[-1]}")

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing expected feature columns: {missing_columns}")

    splits = {}
    for name, dates in [("train", train_dates), ("val", val_dates), ("test", test_dates)]:
        splits[name] = build_split_arrays(df_all, actual_cities, dates, target_columns, name)

    # Fit scalers on the training split only
    X_train, Y_train, M_train, _ = splits["train"]
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    zero_var_mask = feature_scaler.scale_ < 1e-8
    if np.any(zero_var_mask):
        print(f"WARNING: {zero_var_mask.sum()} features have zero variance, setting scale to 1.0")
        feature_scaler.scale_[zero_var_mask] = 1.0

    # MinMaxScaler to [0, 1] per horizon (same target scaling as individual/joint models)
    target_scaler = MinMaxScaler()
    target_scaler.fit(Y_train.reshape(-1, Y_train.shape[-1]))
    print(f"Target scaler range: [{target_scaler.data_min_.min():.2f}, {target_scaler.data_max_.max():.2f}] W/m²")

    datasets = {}
    for name, (X, Y, M, ts) in splits.items():
        C, F = X.shape[1], X.shape[2]
        X_scaled = feature_scaler.transform(X.reshape(-1, F)).reshape(-1, C, F).astype(np.float32)
        Y_scaled = target_scaler.transform(Y.reshape(-1, N_HORIZONS)).reshape(-1, C, N_HORIZONS)
        Y_scaled = np.clip(Y_scaled, 0.0, 1.0)
        datasets[name] = {
            "graphs": make_graphs(X_scaled, Y_scaled, M, adj_matrix),
            "Y_raw": Y,
            "mask": M,
            "timestamps": ts,
        }

    return datasets, feature_scaler, target_scaler

class GHIDataset(Dataset):
    def __init__(self, graphs, y, **kwargs):
        self.graphs = graphs
        self.y = y
        super().__init__(**kwargs)

    def read(self):
        return self.graphs

def build_gnn_model(input_shape, output_units, num_cities, n_horizons=N_HORIZONS):  # Dynamic number of cities
    """Build a state-of-the-art GNN model with three-horizon (1d, 3d, 7d) prediction per city."""
    print(f"Building advanced GNN model with three-horizon predictions for {num_cities} cities...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output units: {output_units} ({num_cities} cities × {n_horizons} horizons)")
    print(f"  Expected output per city: {n_horizons} (1d, 3d, 7d)")
    
    # Input layers
    x_in = layers.Input(shape=input_shape, name='x_in')
    a_in = layers.Input(shape=(None,), sparse=True, name='a_in')
    i_in = layers.Input(shape=(), dtype=tf.int64, name='i_in')
    
    # Simplified GNN layers for stability (reduced complexity to prevent NaN)
    # Use GCNConvSafe to avoid Keras 3 mask=[None, None] error in Spektral GCNConv
    x = GCNConvSafe(128, activation='relu')([x_in, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Residual connection 1
    x_res1 = x
    
    x = GCNConvSafe(128, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, x_res1])  # Residual connection
    
    # Layer normalization for stability
    x = layers.LayerNormalization()(x)
    
    x = GCNConvSafe(64, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Residual connection 2
    x_res2 = x
    
    x = GCNConvSafe(64, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, x_res2])  # Residual connection
    
    x = GCNConvSafe(32, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Custom aggregation layer to group nodes by graph (using custom Layer for Keras 3 compatibility)
    class AggregateNodesLayer(layers.Layer):
        """Custom layer to aggregate node features by graph using batch indices."""
        def call(self, inputs):
            node_features, batch_indices = inputs
            # Use tf.math.unsorted_segment_mean to aggregate features per graph
            aggregated_features = tf.math.unsorted_segment_mean(
                node_features, 
                batch_indices, 
                num_segments=tf.reduce_max(batch_indices) + 1
            )
            return aggregated_features
        
        def compute_output_shape(self, input_shapes):
            # input_shapes: [node_features_shape, batch_indices_shape]
            # Output: (batch_size, feature_dim) where batch_size is dynamic
            node_features_shape = input_shapes[0]
            return (None, node_features_shape[-1])
    
    # Apply the aggregation
    x = AggregateNodesLayer()([x, i_in])
    
    # Multi-task learning: Separate heads for each city (simplified for stability)
    city_heads = []
    for city_idx in range(num_cities):  # Dynamic number of cities
        # City-specific dense layers (reduced size for stability)
        city_x = layers.Dense(128, activation='relu')(x)
        city_x = layers.BatchNormalization()(city_x)
        city_x = layers.Dropout(0.3)(city_x)
        
        city_x = layers.Dense(64, activation='relu')(city_x)
        city_x = layers.BatchNormalization()(city_x)
        city_x = layers.Dropout(0.2)(city_x)
        
        # Output layer: 3 values per city (1d, 3d, 7d); sigmoid to match [0,1] scaled targets
        city_output = layers.Dense(n_horizons, activation='sigmoid', name=f'output_{city_idx}')(city_x)
        
        city_heads.append(city_output)
    
    # Concatenate all city predictions
    outputs = layers.Concatenate(name='output')(city_heads)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    return model

# ----------------------------------------------
# Evaluation functions
# ----------------------------------------------
def masked_mse(y_packed, y_pred):
    """MSE over daytime node outputs only; y_packed = [scaled targets | mask] (see make_graphs)."""
    n_out = tf.shape(y_pred)[1]
    y_true = y_packed[:, :n_out]
    mask = y_packed[:, n_out:]
    return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.maximum(tf.reduce_sum(mask), 1.0)

def _to_model_inputs(inputs):
    x, a, i = inputs
    # Keras 3 requires all inputs to be tensors; 'a' is already a SparseTensor from DisjointLoader
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if not isinstance(i, tf.Tensor):
        i = tf.convert_to_tensor(i, dtype=tf.int64)
    return [x, a, i]

def predict_graphs(model, graphs, batch_size=64):
    """Predict scaled outputs for a list of graphs, in order. Returns (T, C*3)."""
    loader = DisjointLoader(GHIDataset(graphs, None), batch_size=batch_size, epochs=1, shuffle=False)
    preds = [model(_to_model_inputs(inputs), training=False).numpy() for inputs, _ in loader]
    return np.concatenate(preds, axis=0)

def evaluate_loss(model, graphs, batch_size=64):
    """Masked MSE over a list of graphs, weighted by the number of daytime outputs."""
    loader = DisjointLoader(GHIDataset(graphs, None), batch_size=batch_size, epochs=1, shuffle=False)
    total, weight = 0.0, 0.0
    for inputs, y_packed in loader:
        y_packed = tf.cast(y_packed, tf.float32)
        y_pred = model(_to_model_inputs(inputs), training=False)
        n_mask = float(tf.reduce_sum(y_packed[:, y_pred.shape[1]:]).numpy())
        total += float(masked_mse(y_packed, y_pred).numpy()) * n_mask
        weight += n_mask
    return total / max(weight, 1.0)

def evaluate_gnn_model(model, test_data, target_scaler, actual_cities, expected_cities, country):
    """Evaluate the GNN on the held-out test graphs, per city and per horizon (1d, 3d, 7d).

    Only daytime samples are scored, matching the Individual and Joint LSTM evaluation."""
    horizons = ["1d", "3d", "7d"]
    print(f"\nEvaluating GNN model on {len(test_data['graphs'])} test graphs...")
    num_cities = len(actual_cities)
    y_pred = predict_graphs(model, test_data["graphs"])  # (T, C*3) scaled
    y_pred = np.clip(y_pred, 0.0, 1.0).reshape(-1, N_HORIZONS)
    y_pred = target_scaler.inverse_transform(y_pred).reshape(-1, num_cities, N_HORIZONS)
    # GHI cannot be negative; 1200 W/m² is a realistic maximum
    y_pred = np.clip(y_pred, 0.0, 1200.0)

    y_true = test_data["Y_raw"]
    mask = test_data["mask"]
    timestamps = test_data["timestamps"]

    all_metrics = {}
    overall_predictions = []
    overall_actuals = []

    for city_idx, city in enumerate(actual_cities):
        if city not in expected_cities:
            continue
        m = mask[:, city_idx]
        if m.sum() <= 5:
            print(f"\n{city}: Not enough daytime test samples ({m.sum()}) - skipping")
            continue
        actuals = y_true[m, city_idx, :]
        predictions = y_pred[m, city_idx, :]
        dates = timestamps[m]

        city_metrics = {'horizons': horizons, 'actual': actuals, 'predicted': predictions, 'dates': dates}
        print(f"\n{city} Performance:")
        print(f"  Test samples: {len(predictions)}")
        print(f"  {'Horizon':<8} {'MAE (W/m²)':<12} {'RMSE (W/m²)':<12} {'R²':<10} {'Correlation':<10}")
        print("  " + "-" * 55)

        for h_idx, h in enumerate(horizons):
            a_h = actuals[:, h_idx]
            p_h = predictions[:, h_idx]
            mae_h = mean_absolute_error(a_h, p_h)
            rmse_h = np.sqrt(mean_squared_error(a_h, p_h))
            r2_h = r2_score(a_h, p_h)
            corr_h = np.corrcoef(a_h, p_h)[0, 1] if len(a_h) > 1 else 0
            city_metrics[f'mae_{h}'] = mae_h
            city_metrics[f'rmse_{h}'] = rmse_h
            city_metrics[f'r2_{h}'] = r2_h
            city_metrics[f'correlation_{h}'] = corr_h
            city_metrics[f'actual_{h}'] = a_h
            city_metrics[f'predicted_{h}'] = p_h
            print(f"  {h:<8} {mae_h:<12.2f} {rmse_h:<12.2f} {r2_h:<10.4f} {corr_h:<10.4f}")

        # Overall (all horizons pooled)
        a_all = actuals.ravel()
        p_all = predictions.ravel()
        city_metrics['mae'] = mean_absolute_error(a_all, p_all)
        city_metrics['rmse'] = np.sqrt(mean_squared_error(a_all, p_all))
        city_metrics['r2'] = r2_score(a_all, p_all)
        city_metrics['correlation'] = np.corrcoef(a_all, p_all)[0, 1] if len(a_all) > 1 else 0
        print(f"  {'Overall':<8} {city_metrics['mae']:<12.2f} {city_metrics['rmse']:<12.2f} {city_metrics['r2']:<10.4f} {city_metrics['correlation']:<10.4f}")

        all_metrics[city] = city_metrics
        overall_predictions.extend(p_all)
        overall_actuals.extend(a_all)

        # Save test predictions for later analysis
        city_dir = results_path(country, "gnn", city)
        pred_df = pd.DataFrame({'datetime': dates})
        for h_idx, h in enumerate(horizons):
            pred_df[f'actual_{h}'] = actuals[:, h_idx]
            pred_df[f'predicted_{h}'] = predictions[:, h_idx]
        pred_df.to_csv(f"{city_dir}/test_predictions.csv", index=False)

    for city in expected_cities:
        if city not in actual_cities:
            print(f"\n{city}: No data was loaded - skipping")

    if len(overall_predictions) > 0:
        print(f"\nOverall GNN Performance (all cities, all horizons, test set):")
        print(f"  Total predictions: {len(overall_predictions)}")
        print(f"  MAE: {mean_absolute_error(overall_actuals, overall_predictions):.2f} W/m²")
        print(f"  RMSE: {np.sqrt(mean_squared_error(overall_actuals, overall_predictions)):.2f} W/m²")
        print(f"  R²: {r2_score(overall_actuals, overall_predictions):.4f}")
        print(f"  Correlation: {np.corrcoef(overall_actuals, overall_predictions)[0, 1]:.4f}")

    for city in expected_cities:
        if city in all_metrics:
            plot_gnn_results(all_metrics[city], city, country)

    create_gnn_summary_table(all_metrics, country, horizons=horizons)
    return all_metrics

def plot_gnn_results(results, city, country):
    """Create plots for GNN results (three horizons: 1d, 3d, 7d)."""
    try:
        city_dir = results_path(country, "gnn", city)
        horizons = results.get('horizons', ['1d', '3d', '7d'])
        
        # Scatter plots: one per horizon
        fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 6))
        if len(horizons) == 1:
            axes = [axes]
        for i, h in enumerate(horizons):
            actual_h = results[f'actual_{h}']
            pred_h = results[f'predicted_{h}']
            axes[i].scatter(actual_h, pred_h, alpha=0.6)
            max_val = max(actual_h.max(), pred_h.max())
            axes[i].plot([0, max_val], [0, max_val], 'r--', lw=2)
            axes[i].set_xlabel('Actual GHI (W/m²)')
            axes[i].set_ylabel('Predicted GHI (W/m²)')
            axes[i].set_title(f'{h} Horizon - {city}')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{city_dir}/scatter_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Time series: one subplot per horizon
        fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 5 * len(horizons)))
        if len(horizons) == 1:
            axes = [axes]
        dates = results['dates']
        for i, h in enumerate(horizons):
            axes[i].plot(dates, results[f'actual_{h}'], label='Actual', alpha=0.7)
            axes[i].plot(dates, results[f'predicted_{h}'], label='Predicted', alpha=0.7)
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('GHI (W/m²)')
            axes[i].set_title(f'{h} Horizon - {city}')
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"{city_dir}/time_series.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Combined comparison (first 500 samples)
        n_show = min(500, len(dates))
        plt.figure(figsize=(14, 6))
        for i, h in enumerate(horizons):
            mae_h = results.get(f'mae_{h}', 0)
            plt.plot(results[f'actual_{h}'][:n_show], alpha=0.6, label=f'Actual {h}')
            plt.plot(results[f'predicted_{h}'][:n_show], alpha=0.8, linestyle='--', label=f'Pred {h} (MAE: {mae_h:.1f})')
        plt.xlabel('Sample')
        plt.ylabel('GHI (W/m²)')
        plt.title(f'GNN Three-Horizon Comparison - {city}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{city_dir}/comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plots created for {city}")
        
    except Exception as e:
        print(f"  Error creating plots for {city}: {e}")

def calculate_daily_metrics_gnn(dates, actual, predicted):
    """Calculate daily metrics for GNN results with 24-hour forecasting."""
    # For GNN with 24-hour forecasting, we now have multiple predictions per day
    # Group by date and calculate daily metrics
    # Convert dates to datetime and extract date part
    dates_pd = pd.to_datetime(dates)
    
    # Handle both DatetimeIndex and Series - convert to Series to use .dt accessor
    if isinstance(dates_pd, pd.DatetimeIndex):
        dates_series = pd.Series(dates_pd)
    else:
        dates_series = dates_pd
    
    df = pd.DataFrame({
        'date': dates_series.dt.date,
        'actual': actual,
        'predicted': predicted
    })
    
    # Group by date and calculate daily metrics
    daily_stats = []
    for date, group in df.groupby('date'):
        # Filter out zero values
        non_zero_mask = group['actual'] > 0
        if non_zero_mask.sum() > 0:
            actual_nonzero = group.loc[non_zero_mask, 'actual']
            predicted_nonzero = group.loc[non_zero_mask, 'predicted']
            
            # Calculate correlation
            correlation = actual_nonzero.corr(predicted_nonzero)
            if pd.isna(correlation):
                correlation = 0
            
            # Calculate metrics directly on original values (consistent with main evaluation)
            mae = mean_absolute_error(actual_nonzero, predicted_nonzero)
            rmse = np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero))
            r2 = r2_score(actual_nonzero, predicted_nonzero)
            
            # Clamp R² to [0, 1]
            r2 = max(0, min(1, r2))
            
            daily_stats.append({
                'date': date,
                'correlation': correlation,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'num_points': len(actual_nonzero),
                'actual_max': actual_nonzero.max(),
                'actual_mean': actual_nonzero.mean(),
                'actual_std': actual_nonzero.std(),
                'predicted_max': predicted_nonzero.max(),
                'predicted_mean': predicted_nonzero.mean(),
                'predicted_std': predicted_nonzero.std()
            })
    
    return pd.DataFrame(daily_stats).sort_values('date')

def create_gnn_summary_table(all_metrics, country, horizons=None):
    """Create a summary table for all GNN results (per horizon and overall)."""
    horizons = horizons or ['1d', '3d', '7d']
    summary_data = []
    
    for city, metrics in all_metrics.items():
        row = {'City': city}
        for h in horizons:
            row[f'MAE_{h}'] = metrics.get(f'mae_{h}', np.nan)
            row[f'RMSE_{h}'] = metrics.get(f'rmse_{h}', np.nan)
            row[f'R²_{h}'] = metrics.get(f'r2_{h}', np.nan)
            row[f'Corr_{h}'] = metrics.get(f'correlation_{h}', np.nan)
        row['MAE_overall'] = metrics.get('mae', np.nan)
        row['RMSE_overall'] = metrics.get('rmse', np.nan)
        row['R²_overall'] = metrics.get('r2', np.nan)
        row['Corr_overall'] = metrics.get('correlation', np.nan)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = results_path(country, "gnn", "summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    print(summary_df)

def main(is_colombia=False, fresh=False):
    """Train and evaluate the GNN for one country. The graph's adjacency matrix is
    built from the Haversine distances between site coordinates."""
    global CITIES, CITY_IDX

    start_time = time.time()

    country = country_name(is_colombia)

    # Update CITIES and CITY_IDX based on data selection
    if is_colombia:
        print("Using Colombia data locations")
        CITIES = list(CONFIG["data_locations_colombia"].keys())
    else:
        print("Using India data locations")
        CITIES = list(CONFIG["data_locations"].keys())
    
    CITY_IDX = {city: i for i, city in enumerate(CITIES)}
    print(f"Cities: {CITIES}")
    
    try:
        print("\nLoading and preparing data...")
        
        # Load data with memory management
        all_dfs = []
        target_columns = None
        for i, city in enumerate(CITIES):
            print(f"Loading data for city {i+1}/{len(CITIES)}: {city}")
            if is_colombia:
                df = load_data(CONFIG["data_locations_colombia"], city)
            else:
                df = load_data(CONFIG["data_locations"], city)
            print(f"  Raw data shape: {df.shape}")
            
            print(f"  Creating features for {city}...")
            df, city_target_columns = create_features(df)
            if target_columns is None:
                target_columns = city_target_columns
            print(f"  After feature creation: {df.shape}")
            
            df['location'] = city
            all_dfs.append(df)
            print(f"  ✓ {city} data processed")
            
            # Clear memory
            gc.collect()
        
        print("Concatenating all city data...")
        df_all = pd.concat(all_dfs).sort_values("datetime")
        print(f"Combined data shape: {df_all.shape}")
        
        # Clear individual dataframes
        del all_dfs
        gc.collect()

        # Get the actual cities present in the data
        actual_cities = df_all['location'].unique()
        print(f"\nActual cities in data: {actual_cities}")
        print(f"Number of cities: {len(actual_cities)}")

        print("\nComputing adjacency matrix...")
        locations_dict = CONFIG["data_locations_colombia"] if is_colombia else CONFIG["data_locations"]
        city_coordinates = get_city_coordinates(locations_dict)
        missing = [c for c in actual_cities if c not in city_coordinates]
        if missing:
            print(f"WARNING: no coordinates found for {missing}; these cities will have zero geographic adjacency weight to all other sites")
        adj_matrix = compute_weighted_adjacency_with_coordinates(df_all, city_coordinates=city_coordinates, alpha=0.5)
        print(f"Adjacency matrix computed: {adj_matrix.shape}")

        print("\nBuilding hourly graphs for three-horizon (1d, 3d, 7d) forecasting...")
        datasets, feature_scaler, target_scaler = build_hourly_graphs(df_all, adj_matrix, actual_cities, target_columns)

        # Clear main dataframe
        del df_all
        gc.collect()

        train_graphs = datasets["train"]["graphs"]
        val_graphs = datasets["val"]["graphs"]
        if len(train_graphs) == 0 or len(val_graphs) == 0 or len(datasets["test"]["graphs"]) == 0:
            raise ValueError("Empty train, validation, or test split")

        print(f"\nCreating dataset...")
        train_dataset = GHIDataset(train_graphs, None)
        print(f"Graphs: train {len(train_graphs)}, val {len(val_graphs)}, test {len(datasets['test']['graphs'])}")
        print(f"First graph x shape: {train_dataset[0].x.shape}, y shape: {train_dataset[0].y.shape}")

        print("\nBuilding model...")
        num_features = train_dataset[0].x.shape[1]
        num_cities = len(actual_cities)
        print(f"Number of features per node: {num_features}")
        print(f"Number of cities: {num_cities}")
        print(f"Output units: {len(target_columns) * num_cities}")  # 3 horizons × num_cities

        model = build_gnn_model(input_shape=(num_features,), output_units=len(target_columns) * num_cities, num_cities=num_cities)
        model.summary()

        # Stable learning rate with gradient clipping
        initial_lr = 0.0001
        optimizer = tf.keras.optimizers.Adam(initial_lr, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=masked_mse)

        print("\nTraining model with early stopping on validation loss...")
        batch_size = 16
        epochs = 200
        patience = 25
        print(f"Batch size: {batch_size}, max epochs: {epochs}, early stopping patience: {patience}")

        best_val_loss = float('inf')
        best_weights = model.get_weights()
        patience_counter = 0

        # Checkpointing: after every epoch, save the model and optimizer state plus the
        # early-stopping state, so an interrupted run resumes from the last completed epoch
        # instead of starting over. The checkpoint is removed once evaluation has finished.
        ckpt_dir = model_path(country, "gnn", "training_checkpoint")
        state_path = os.path.join(ckpt_dir, "state.json")
        best_path = os.path.join(ckpt_dir, "best_weights.npz")
        fingerprint = {
            "country": country,
            "cities": [str(c) for c in actual_cities],
            "graphs": [len(train_graphs), len(val_graphs), len(datasets["test"]["graphs"])],
            "num_features": int(num_features),
            "batch_size": batch_size, "epochs": epochs, "patience": patience, "initial_lr": initial_lr,
        }
        state = None
        if fresh and os.path.isdir(ckpt_dir):
            print(f"--fresh: removing existing checkpoint {ckpt_dir}")
            shutil.rmtree(ckpt_dir)
        elif os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("fingerprint") != fingerprint:
                print(f"WARNING: checkpoint in {ckpt_dir} is from a different setup; starting from scratch")
                shutil.rmtree(ckpt_dir)
                state = None

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2)
        start_epoch = 0
        finished = False
        if state is not None:
            ckpt.restore(state["checkpoint"])
            with np.load(best_path) as z:
                best_weights = [z[f"arr_{k}"] for k in range(len(z.files))]
            best_val_loss = state["best_val_loss"]
            patience_counter = state["patience_counter"]
            start_epoch = state["epochs_done"]
            finished = state["finished"]
            print(f"Resuming from checkpoint: {start_epoch} epoch(s) done, best val loss {best_val_loss:.5f}, "
                  f"{patience_counter} epoch(s) without improvement"
                  + (" (training already finished)" if finished else ""))

        def save_checkpoint(epochs_done, improved, done):
            ckpt_path = manager.save(checkpoint_number=epochs_done)
            if improved or not os.path.exists(best_path):
                tmp = best_path + ".tmp.npz"
                np.savez(tmp, *best_weights)
                os.replace(tmp, best_path)
            tmp = state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"fingerprint": fingerprint, "checkpoint": ckpt_path, "epochs_done": epochs_done,
                           "best_val_loss": best_val_loss, "patience_counter": patience_counter,
                           "finished": done}, f, indent=2)
            os.replace(tmp, state_path)

        for epoch in range(start_epoch, epochs):
            if finished:
                break
            epoch_start_time = time.time()

            # Step learning rate schedule
            if epoch < 50:
                current_lr = initial_lr
            elif epoch < 100:
                current_lr = initial_lr * 0.5
            else:
                current_lr = initial_lr * 0.1
            model.optimizer.learning_rate.assign(float(current_lr))

            epoch_loss = 0.0
            num_batches = 0
            loader = DisjointLoader(train_dataset, batch_size=batch_size, epochs=1, shuffle=True)
            for inputs, y_packed in loader:
                y_packed = tf.cast(y_packed, tf.float32)
                with tf.GradientTape() as tape:
                    y_pred = model(_to_model_inputs(inputs), training=True)
                    loss = masked_mse(y_packed, y_pred)
                if tf.math.is_nan(loss):
                    print(f"      WARNING: NaN loss at batch {num_batches}, skipping...")
                    continue
                gradients = tape.gradient(loss, model.trainable_variables)
                gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_loss += float(loss.numpy())
                num_batches += 1

            if num_batches == 0:
                print(f"Epoch {epoch+1}/{epochs} - No valid batches processed")
                continue
            epoch_loss /= num_batches
            val_loss = evaluate_loss(model, val_graphs)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.5f} - Val loss: {val_loss:.5f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s")

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                patience_counter = 0
                print(f"  ✓ New best val loss: {best_val_loss:.5f}")
            else:
                patience_counter += 1
            stop = patience_counter >= patience or epoch + 1 == epochs
            save_checkpoint(epoch + 1, improved, stop)
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                break

            gc.collect()

        model.set_weights(best_weights)
        print(f"Training completed! Best val loss: {best_val_loss:.5f}")

        print("\nSaving model and scalers...")
        model.save(model_path(country, "gnn", "model.h5"))
        with open(model_path(country, "gnn", "feature_scaler.pkl"), "wb") as f:
            pickle.dump(feature_scaler, f)
        with open(model_path(country, "gnn", "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)
        print("✓ Model and scalers saved.")

        # Evaluate the model on the held-out test split
        print("\nEvaluating model...")
        all_metrics = evaluate_gnn_model(model, datasets["test"], target_scaler, actual_cities, CITIES, country)

        # Training and evaluation are done, so the resume checkpoint is no longer needed
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        total_time = time.time() - start_time
        print(f"\nGNN three-horizon (1d, 3d, 7d) training and evaluation completed in {total_time:.1f} seconds!")

    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI forecasting with a graph neural network - three horizons (1d, 3d, 7d)')
    parser.add_argument('--is-colombia', action='store_true',
                      help='Load data from Colombia locations')
    parser.add_argument('--fresh', action='store_true',
                      help="Ignore any saved training checkpoint and start from epoch 1 "
                           "(by default an interrupted run resumes from its last completed epoch)")
    args = parser.parse_args()

    main(is_colombia=args.is_colombia, fresh=args.fresh)