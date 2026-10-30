"""
Data Preparation and Baseline Evaluation for Traffic Volume Forecasting.

Task Definition:
----------------
Forecast hourly traffic volume (vehicles per hour) on a highway link at a chosen future hour and day.

Operational Constraint & Feature Boundary:
- When planning future routes or predicting future congestion, weather conditions for future hours
  are unknown at prediction time.
- Therefore, the PRIMARY forecasting model must use deterministic calendar features only:
  - hour_sin, hour_cos (cyclical representation of hour of day)
  - day_of_week (0 = Monday, 6 = Sunday)
  - is_weekend (binary flag: Saturday/Sunday)
  - is_holiday (binary flag: recognized US/regional holidays)
- Weather features (temp, rain_1h, snow_1h, clouds_all, weather_main) are preserved in a separate
  optional feature list for "nowcasting" experiments (evaluating traffic when contemporaneous weather
  readings are available in real time).

Split Protocol:
---------------
Chronological time-series split (70% train / 15% validation / 15% test).
No shuffling to prevent temporal data leakage.
"""

import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Feature groups
CALENDAR_FEATURES: List[str] = [
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
]

WEATHER_FEATURES: List[str] = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
]

TARGET_COL: str = "traffic_volume"


def load_and_clean_data(csv_path: str = "data/Metro_Interstate_Traffic_Volume.csv") -> pd.DataFrame:
    """
    Loads raw CSV, parses datetime, sorts chronologically, and removes duplicate timestamps.
    Duplicate timestamps occur in the raw dataset when multiple weather descriptions
    were logged for the same hour. The traffic_volume target is identical across duplicates.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please place the file in data/ first.")

    df = pd.read_csv(csv_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    
    # Sort chronologically
    df = df.sort_values("date_time").reset_index(drop=True)
    
    # Deduplicate by timestamp (keeping first occurrence)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["date_time"], keep="first").reset_index(drop=True)
    print(f"Loaded {initial_len} raw rows. Deduplicated by date_time: {len(df)} unique hourly records.")
    
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Constructs cyclical calendar features and flags.
    Returns the enriched DataFrame, calendar feature names, and optional weather feature names.
    """
    df = df.copy()
    hour = df["date_time"].dt.hour
    
    # Cyclical hour features
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    
    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["date_time"].dt.dayofweek
    
    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Holiday flag (non-'None' string indicates a holiday)
    df["is_holiday"] = (df["holiday"].fillna("None") != "None").astype(int)
    
    return df, CALENDAR_FEATURES, WEATHER_FEATURES


def split_data_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame strictly chronologically into Train, Validation, and Test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)
    
    print("\n--- Chronological Split Summary ---")
    print(f"Train set: {len(train_df):>6} rows | {train_df['date_time'].min()} to {train_df['date_time'].max()}")
    print(f"Val set:   {len(val_df):>6} rows | {val_df['date_time'].min()} to {val_df['date_time'].max()}")
    print(f"Test set:  {len(test_df):>6} rows | {test_df['date_time'].min()} to {test_df['date_time'].max()}")
    
    return train_df, val_df, test_df


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }


def evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Computes three reproducible non-learning baselines evaluated on the test set:
    1. Global Mean (from train only)
    2. Historical Mean by (hour, day_of_week) (from train only)
    3. Seasonal-Naive: traffic volume at same hour last week (t - 7 days).
       If historical sensor gap at t - 7 days, falls back to historical mean.
    """
    y_test = test_df[TARGET_COL].values
    
    # 1. Global Mean Baseline (Train only)
    global_mean = float(train_df[TARGET_COL].mean())
    pred_global = np.full_like(y_test, fill_value=global_mean, dtype=float)
    metrics_global = _calc_metrics(y_test, pred_global)
    
    # 2. Historical Mean by (hour, day_of_week) Baseline (Train only)
    train_hours = train_df["date_time"].dt.hour
    train_dows = train_df["date_time"].dt.dayofweek
    hist_lookup = (
        train_df.assign(h=train_hours, dow=train_dows)
        .groupby(["dow", "h"])[TARGET_COL]
        .mean()
        .to_dict()
    )
    
    test_hours = test_df["date_time"].dt.hour
    test_dows = test_df["date_time"].dt.dayofweek
    pred_hist = np.array([
        hist_lookup.get((dow, h), global_mean)
        for dow, h in zip(test_dows, test_hours)
    ], dtype=float)
    metrics_hist = _calc_metrics(y_test, pred_hist)
    
    # 3. Seasonal-Naive Baseline (same hour last week: t - 7 days)
    # Map full timeline for lag lookup
    dt_to_vol = dict(zip(full_df["date_time"], full_df[TARGET_COL]))
    matched_count = 0
    pred_seasonal = []
    
    for t, dow, h in zip(test_df["date_time"], test_dows, test_hours):
        t_prev_week = t - pd.Timedelta(days=7)
        if t_prev_week in dt_to_vol:
            pred_seasonal.append(float(dt_to_vol[t_prev_week]))
            matched_count += 1
        else:
            # Fallback to historical mean when sensor data was missing 7 days prior
            pred_seasonal.append(float(hist_lookup.get((dow, h), global_mean)))
            
    pred_seasonal = np.array(pred_seasonal, dtype=float)
    metrics_seasonal = _calc_metrics(y_test, pred_seasonal)
    metrics_seasonal["exact_prev_week_matches"] = f"{matched_count}/{len(test_df)} ({matched_count/len(test_df)*100:.2f}%)"
    
    results = {
        "global_mean": {
            "value": round(global_mean, 2),
            **metrics_global,
        },
        "historical_mean_by_hour_dow": metrics_hist,
        "seasonal_naive_7d": metrics_seasonal,
    }
    
    return results


def main():
    print("=" * 60)
    print("Step 2: Preparing Traffic Data & Evaluating Baselines")
    print("=" * 60)
    
    # 1. Load and Clean
    df = load_and_clean_data("data/Metro_Interstate_Traffic_Volume.csv")
    
    # 2. Build Features
    df, cal_feats, weather_feats = build_features(df)
    print(f"Calendar features (Primary for forecasting): {cal_feats}")
    print(f"Weather features (Optional for nowcasting):   {weather_feats}")
    
    # 3. Chronological Split
    train_df, val_df, test_df = split_data_chronological(df, 0.70, 0.15, 0.15)
    
    # 4. Baselines Evaluation
    baselines = evaluate_baselines(train_df, val_df, test_df, df)
    
    print("\n--- Test Set Baseline Results ---")
    print(f"1. Global Mean Baseline:       MAE = {baselines['global_mean']['mae']:>7.2f} | RMSE = {baselines['global_mean']['rmse']:>7.2f} | R^2 = {baselines['global_mean']['r2']:>7.4f}")
    print(f"2. Historical Mean (Hour, DOW): MAE = {baselines['historical_mean_by_hour_dow']['mae']:>7.2f} | RMSE = {baselines['historical_mean_by_hour_dow']['rmse']:>7.2f} | R^2 = {baselines['historical_mean_by_hour_dow']['r2']:>7.4f}")
    print(f"3. Seasonal-Naive (t - 7d):     MAE = {baselines['seasonal_naive_7d']['mae']:>7.2f} | RMSE = {baselines['seasonal_naive_7d']['rmse']:>7.2f} | R^2 = {baselines['seasonal_naive_7d']['r2']:>7.4f}")
    print(f"   (Exact 7-day prior matches: {baselines['seasonal_naive_7d']['exact_prev_week_matches']})")
    
    # 5. Save results to results/baselines.json
    os.makedirs("results", exist_ok=True)
    out_payload = {
        "task": "hourly_traffic_volume_forecasting",
        "dataset": "UCI Metro Interstate Traffic Volume",
        "target": TARGET_COL,
        "date_range": {
            "start": str(df["date_time"].min()),
            "end": str(df["date_time"].max()),
            "total_records": len(df),
        },
        "splits": {
            "train": {
                "count": len(train_df),
                "start": str(train_df["date_time"].min()),
                "end": str(train_df["date_time"].max()),
            },
            "val": {
                "count": len(val_df),
                "start": str(val_df["date_time"].min()),
                "end": str(val_df["date_time"].max()),
            },
            "test": {
                "count": len(test_df),
                "start": str(test_df["date_time"].min()),
                "end": str(test_df["date_time"].max()),
            },
        },
        "feature_sets": {
            "calendar_primary": cal_feats,
            "weather_nowcast_optional": weather_feats,
        },
        "baselines_test_set": baselines,
    }
    
    results_path = os.path.join("results", "baselines.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved baseline metrics to {results_path}")


if __name__ == "__main__":
    main()
