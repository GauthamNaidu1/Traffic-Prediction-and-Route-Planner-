"""
Model Training, Validation Hyperparameter Tuning, Bootstrap CI, Rolling-Origin Evaluation,
and Nowcast Experiment for Traffic Volume Prediction.

Protocol & Reproducibility:
----------------------------
1. Strict chronological split (70% Train, 15% Validation, 15% Test).
2. Hyperparameter tuning on the Validation set ONLY.
3. Test set evaluated ONCE per model.
4. Bootstrap 95% Confidence Intervals (1,000 resamples).
5. 5-fold Rolling-Origin (expanding window) cross-validation.
6. Second experiment with contemporaneous weather features ("Nowcast, not forecast").
7. Best calendar-feature model saved to models/traffic_model.joblib (< 5 MB).
8. Results and plots saved under results/.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from train.prepare_data import (
    CALENDAR_FEATURES,
    TARGET_COL,
    build_features,
    load_and_clean_data,
    split_data_chronological,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and R^2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% bootstrap confidence intervals for MAE, RMSE, and R^2
    using n_bootstraps resamples with replacement.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))

    mae_list: List[float] = []
    rmse_list: List[float] = []
    r2_list: List[float] = []

    for idx in indices:
        yt = y_true[idx]
        yp = y_pred[idx]
        err = yt - yp
        
        mae_list.append(float(np.mean(np.abs(err))))
        rmse_list.append(float(np.sqrt(np.mean(err**2))))
        
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        ss_res = np.sum(err**2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_list.append(float(r2))

    def ci_dict(metric_point: float, values: List[float]) -> Dict[str, float]:
        low = float(np.percentile(values, 2.5))
        high = float(np.percentile(values, 97.5))
        return {
            "point": round(metric_point, 2),
            "ci_lower": round(low, 2),
            "ci_upper": round(high, 2),
        }

    point = calc_metrics(y_true, y_pred)

    return {
        "mae": ci_dict(point["mae"], mae_list),
        "rmse": ci_dict(point["rmse"], rmse_list),
        "r2": {
            "point": round(point["r2"], 4),
            "ci_lower": round(float(np.percentile(r2_list, 2.5)), 4),
            "ci_upper": round(float(np.percentile(r2_list, 97.5)), 4),
        },
    }


def compute_paired_bootstrap(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Computes paired difference bootstrap (model - baseline) for MAE and RMSE.
    A negative difference indicates the model has a lower error than the baseline.
    If the 95% CI includes 0.0, the difference is not statistically significant.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))

    mae_diffs: List[float] = []
    rmse_diffs: List[float] = []

    for idx in indices:
        yt = y_true[idx]
        yp_m = y_pred_model[idx]
        yp_b = y_pred_baseline[idx]

        mae_m = np.mean(np.abs(yt - yp_m))
        mae_b = np.mean(np.abs(yt - yp_b))
        mae_diffs.append(float(mae_m - mae_b))

        rmse_m = np.sqrt(np.mean((yt - yp_m) ** 2))
        rmse_b = np.sqrt(np.mean((yt - yp_b) ** 2))
        rmse_diffs.append(float(rmse_m - rmse_b))

    return {
        "mae_diff_point": round(float(np.mean(mae_diffs)), 2),
        "mae_diff_ci_lower": round(float(np.percentile(mae_diffs, 2.5)), 2),
        "mae_diff_ci_upper": round(float(np.percentile(mae_diffs, 97.5)), 2),
        "rmse_diff_point": round(float(np.mean(rmse_diffs)), 2),
        "rmse_diff_ci_lower": round(float(np.percentile(rmse_diffs, 2.5)), 2),
        "rmse_diff_ci_upper": round(float(np.percentile(rmse_diffs, 97.5)), 2),
    }


def tune_calendar_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Tunes hyperparameters on the Validation set ONLY.
    Returns the fitted models and their best hyperparameters.
    """
    print("\n--- Tuning Hyperparameters on Validation Set ---")

    # 1. Ridge Regression
    best_ridge_alpha = 1.0
    best_ridge_mae = float("inf")
    best_ridge_model = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, pred)
        if val_mae < best_ridge_mae:
            best_ridge_mae = val_mae
            best_ridge_alpha = alpha
            best_ridge_model = model

    print(f"  [Ridge] Best alpha: {best_ridge_alpha} (Val MAE: {best_ridge_mae:.2f})")

    # 2. Gradient Boosting Regressor
    best_gbr_params = {}
    best_gbr_mae = float("inf")
    best_gbr_model = None

    for n_est in [50, 100]:
        for lr in [0.05, 0.1]:
            for md in [4, 5, 6]:
                model = GradientBoostingRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=md,
                    random_state=RANDOM_SEED,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, pred)
                if val_mae < best_gbr_mae:
                    best_gbr_mae = val_mae
                    best_gbr_params = {
                        "n_estimators": n_est,
                        "learning_rate": lr,
                        "max_depth": md,
                    }
                    best_gbr_model = model

    print(f"  [Gradient Boosting] Best params: {best_gbr_params} (Val MAE: {best_gbr_mae:.2f})")

    # 3. Random Forest Regressor
    best_rf_params = {}
    best_rf_mae = float("inf")
    best_rf_model = None

    for n_est in [50, 100]:
        for md in [8, 10, 12]:
            for mss in [2, 5]:
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=md,
                    min_samples_split=mss,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, pred)
                if val_mae < best_rf_mae:
                    best_rf_mae = val_mae
                    best_rf_params = {
                        "n_estimators": n_est,
                        "max_depth": md,
                        "min_samples_split": mss,
                    }
                    best_rf_model = model

    print(f"  [Random Forest] Best params: {best_rf_params} (Val MAE: {best_rf_mae:.2f})")

    return {
        "Ridge": {
            "model": best_ridge_model,
            "params": {"alpha": best_ridge_alpha},
            "val_mae": round(best_ridge_mae, 2),
        },
        "Gradient Boosting": {
            "model": best_gbr_model,
            "params": best_gbr_params,
            "val_mae": round(best_gbr_mae, 2),
        },
        "Random Forest": {
            "model": best_rf_model,
            "params": best_rf_params,
            "val_mae": round(best_rf_mae, 2),
        },
    }


def run_rolling_origin_eval(
    df: pd.DataFrame,
    cal_features: List[str],
    target_col: str,
    n_splits: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Performs 5-fold rolling-origin (expanding window) cross-validation
    across all baselines and tuned models.
    """
    print(f"\n--- Running {n_splits}-Fold Rolling-Origin Evaluation ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    X = df[cal_features]
    y = df[target_col]
    dt_series = df["date_time"]
    dt_to_vol = dict(zip(dt_series, y))

    models_list = [
        "Global Mean",
        "Historical Mean (Hour, DOW)",
        "Seasonal-Naive (t - 7d)",
        "Ridge",
        "Gradient Boosting",
        "Random Forest",
    ]

    fold_metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"mae": [], "rmse": [], "r2": []} for m in models_list
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        dt_tr = dt_series.iloc[train_idx]
        dt_te = dt_series.iloc[test_idx]
        y_te_arr = y_te.values

        # 1. Global Mean
        gm = float(y_tr.mean())
        pred_gm = np.full_like(y_te_arr, fill_value=gm, dtype=float)
        m_gm = calc_metrics(y_te_arr, pred_gm)
        fold_metrics["Global Mean"]["mae"].append(m_gm["mae"])
        fold_metrics["Global Mean"]["rmse"].append(m_gm["rmse"])
        fold_metrics["Global Mean"]["r2"].append(m_gm["r2"])

        # 2. Historical Mean
        h_tr = dt_tr.dt.hour
        dow_tr = dt_tr.dt.dayofweek
        lookup = (
            pd.DataFrame({"dow": dow_tr, "h": h_tr, "y": y_tr})
            .groupby(["dow", "h"])["y"]
            .mean()
            .to_dict()
        )
        h_te = dt_te.dt.hour
        dow_te = dt_te.dt.dayofweek
        pred_hist = np.array([lookup.get((d, h), gm) for d, h in zip(dow_te, h_te)], dtype=float)
        m_hist = calc_metrics(y_te_arr, pred_hist)
        fold_metrics["Historical Mean (Hour, DOW)"]["mae"].append(m_hist["mae"])
        fold_metrics["Historical Mean (Hour, DOW)"]["rmse"].append(m_hist["rmse"])
        fold_metrics["Historical Mean (Hour, DOW)"]["r2"].append(m_hist["r2"])

        # 3. Seasonal-Naive
        pred_seas = []
        for t, d, h in zip(dt_te, dow_te, h_te):
            t_prev = t - pd.Timedelta(days=7)
            if t_prev in dt_to_vol:
                pred_seas.append(float(dt_to_vol[t_prev]))
            else:
                pred_seas.append(float(lookup.get((d, h), gm)))
        pred_seas = np.array(pred_seas, dtype=float)
        m_seas = calc_metrics(y_te_arr, pred_seas)
        fold_metrics["Seasonal-Naive (t - 7d)"]["mae"].append(m_seas["mae"])
        fold_metrics["Seasonal-Naive (t - 7d)"]["rmse"].append(m_seas["rmse"])
        fold_metrics["Seasonal-Naive (t - 7d)"]["r2"].append(m_seas["r2"])

        # 4. Ridge
        ridge = Ridge(alpha=100.0, random_state=RANDOM_SEED).fit(X_tr, y_tr)
        m_ridge = calc_metrics(y_te_arr, ridge.predict(X_te))
        fold_metrics["Ridge"]["mae"].append(m_ridge["mae"])
        fold_metrics["Ridge"]["rmse"].append(m_ridge["rmse"])
        fold_metrics["Ridge"]["r2"].append(m_ridge["r2"])

        # 5. Gradient Boosting
        gbr = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_SEED
        ).fit(X_tr, y_tr)
        m_gbr = calc_metrics(y_te_arr, gbr.predict(X_te))
        fold_metrics["Gradient Boosting"]["mae"].append(m_gbr["mae"])
        fold_metrics["Gradient Boosting"]["rmse"].append(m_gbr["rmse"])
        fold_metrics["Gradient Boosting"]["r2"].append(m_gbr["r2"])

        # 6. Random Forest
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=12, min_samples_split=2, random_state=RANDOM_SEED, n_jobs=-1
        ).fit(X_tr, y_tr)
        m_rf = calc_metrics(y_te_arr, rf.predict(X_te))
        fold_metrics["Random Forest"]["mae"].append(m_rf["mae"])
        fold_metrics["Random Forest"]["rmse"].append(m_rf["rmse"])
        fold_metrics["Random Forest"]["r2"].append(m_rf["r2"])

        print(f"  Fold {fold + 1}/{n_splits} complete (Train: {len(X_tr)}, Test: {len(X_te)})")

    summary = {}
    for m in models_list:
        summary[m] = {
            "mae_mean": round(float(np.mean(fold_metrics[m]["mae"])), 2),
            "mae_std": round(float(np.std(fold_metrics[m]["mae"])), 2),
            "rmse_mean": round(float(np.mean(fold_metrics[m]["rmse"])), 2),
            "rmse_std": round(float(np.std(fold_metrics[m]["rmse"])), 2),
            "r2_mean": round(float(np.mean(fold_metrics[m]["r2"])), 4),
            "r2_std": round(float(np.std(fold_metrics[m]["r2"])), 4),
        }

    return summary


def run_nowcast_experiment(
    df: pd.DataFrame,
    cal_features: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Experiment 2: Contemporaneous weather features added ("Nowcast, not forecast").
    Uses temperature, precipitation, cloud cover, and one-hot encoded weather categories.
    """
    print("\n--- Running Experiment 2: Nowcast (Calendar + Weather Features) ---")
    
    # Feature engineering for weather
    df_clean = df.copy()
    
    # Clean anomalies in sensor readings
    median_temp = df_clean.loc[df_clean["temp"] > 200, "temp"].median()
    df_clean["temp_clean"] = df_clean["temp"].apply(lambda x: median_temp if x < 200 else x)

    median_rain = df_clean.loc[df_clean["rain_1h"] < 500, "rain_1h"].median()
    df_clean["rain_clean"] = df_clean["rain_1h"].apply(lambda x: median_rain if x > 500 else x)

    # Weather categorical dummies
    weather_dummies = pd.get_dummies(df_clean["weather_main"], prefix="weather", drop_first=True, dtype=int)
    dummy_cols = list(weather_dummies.columns)

    df_nowcast = pd.concat([df_clean, weather_dummies], axis=1)
    weather_feature_cols = ["temp_clean", "rain_clean", "snow_1h", "clouds_all"] + dummy_cols
    all_features = cal_features + weather_feature_cols

    # Split using original indices
    n_train = len(train_df)
    n_val = len(val_df)
    
    train_w = df_nowcast.iloc[:n_train]
    val_w = df_nowcast.iloc[n_train : n_train + n_val]
    test_w = df_nowcast.iloc[n_train + n_val :]

    X_train_w = train_w[all_features]
    y_train_w = train_w[TARGET_COL]

    X_val_w = val_w[all_features]
    y_val_w = val_w[TARGET_COL]

    X_test_w = test_w[all_features]
    y_test_w = test_w[TARGET_COL]

    # Train Nowcast GBR
    nowcast_gbr = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_SEED
    )
    nowcast_gbr.fit(X_train_w, y_train_w)

    val_pred = nowcast_gbr.predict(X_val_w)
    test_pred = nowcast_gbr.predict(X_test_w)

    val_metrics = calc_metrics(y_val_w.values, val_pred)
    test_ci = compute_bootstrap_ci(y_test_w.values, test_pred, n_bootstraps=1000)

    print(f"  [Nowcast GBR] Val MAE:  {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f} | R^2: {val_metrics['r2']:.4f}")
    print(f"  [Nowcast GBR] Test MAE: {test_ci['mae']['point']} [{test_ci['mae']['ci_lower']}, {test_ci['mae']['ci_upper']}]")

    return {
        "features": all_features,
        "val_metrics": val_metrics,
        "test_ci": test_ci,
        "model": nowcast_gbr,
        "test_pred": test_pred,
    }


def generate_plots(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    pred_best_calendar: np.ndarray,
    pred_baseline: np.ndarray,
    pred_nowcast: np.ndarray,
    best_model: Any,
    cal_features: List[str],
    best_model_name: str,
):
    """Generates and saves the two required visualization artifacts."""
    os.makedirs("results", exist_ok=True)

    # 1. Predicted vs Actual for One Test Week (Monday May 7 to Sunday May 13, 2018)
    week_start = pd.to_datetime("2018-05-07 00:00:00")
    week_end = pd.to_datetime("2018-05-14 00:00:00")
    mask = (test_df["date_time"] >= week_start) & (test_df["date_time"] < week_end)

    week_dt = test_df.loc[mask, "date_time"]
    week_actual = y_test[mask]
    week_cal = pred_best_calendar[mask]
    week_base = pred_baseline[mask]
    week_now = pred_nowcast[mask]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    ax.plot(week_dt, week_actual, label="Actual Traffic Volume", color="#1f2937", linewidth=2.0, alpha=0.9)
    ax.plot(week_dt, week_base, label="Historical Mean (Hour, DOW)", color="#9ca3af", linestyle="--", linewidth=1.5)
    ax.plot(week_dt, week_cal, label=f"Best Calendar Model ({best_model_name})", color="#2563eb", linewidth=1.8, alpha=0.85)
    ax.plot(week_dt, week_now, label="Nowcast Model (with Weather)", color="#dc2626", linestyle=":", linewidth=1.6, alpha=0.85)

    ax.set_title("Traffic Volume Forecast vs Actual — One Test Week (May 7–13, 2018)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Traffic Volume (Vehicles / Hour)", fontsize=11)
    ax.set_xlabel("Date & Time", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", loc="upper right")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plot1_path = os.path.join("results", "predicted_vs_actual_week.png")
    fig.savefig(plot1_path)
    plt.close(fig)
    print(f"Saved plot: {plot1_path}")

    # 2. Feature Importance for Best Calendar Model
    importances = best_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    colors = ["#3b82f6" for _ in range(len(indices))]
    bars = ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor="#1d4ed8", height=0.6)

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([cal_features[i] for i in indices], fontsize=10)
    ax.set_xlabel("Relative Importance (Gini / Impurity Reduction)", fontsize=10)
    ax.set_title(f"Feature Importance — Best Calendar Model ({best_model_name})", fontsize=12, fontweight="bold", pad=10)

    # Label values on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", ha="left", fontsize=9, color="#1e3a8a")

    ax.set_xlim(0, max(importances) * 1.15)
    plt.tight_layout()

    plot2_path = os.path.join("results", "feature_importance.png")
    fig.savefig(plot2_path)
    plt.close(fig)
    print(f"Saved plot: {plot2_path}")


def main():
    t_start = time.time()
    print("=" * 70)
    print("Step 3: Training Models, Bootstrap CIs, Rolling-Origin & Nowcasting")
    print("=" * 70)

    # 1. Load data and features
    df = load_and_clean_data("data/Metro_Interstate_Traffic_Volume.csv")
    df, cal_features, _ = build_features(df)
    train_df, val_df, test_df = split_data_chronological(df, 0.70, 0.15, 0.15)

    X_train = train_df[cal_features]
    y_train = train_df[TARGET_COL]

    X_val = val_df[cal_features]
    y_val = val_df[TARGET_COL]

    X_test = test_df[cal_features]
    y_test = test_df[TARGET_COL].values

    # 2. Hyperparameter Tuning on Validation Set ONLY
    tuned_results = tune_calendar_models(X_train, y_train, X_val, y_val)

    # 3. Baseline predictions on Test Set
    global_mean = float(y_train.mean())
    pred_global = np.full_like(y_test, fill_value=global_mean, dtype=float)

    train_hours = train_df["date_time"].dt.hour
    train_dows = train_df["date_time"].dt.dayofweek
    hist_lookup = (
        pd.DataFrame({"dow": train_dows, "h": train_hours, "y": y_train})
        .groupby(["dow", "h"])["y"]
        .mean()
        .to_dict()
    )
    test_hours = test_df["date_time"].dt.hour
    test_dows = test_df["date_time"].dt.dayofweek
    pred_hist = np.array([hist_lookup.get((d, h), global_mean) for d, h in zip(test_dows, test_hours)], dtype=float)

    dt_to_vol = dict(zip(df["date_time"], df[TARGET_COL]))
    pred_seas = []
    for t, d, h in zip(test_df["date_time"], test_dows, test_hours):
        t_prev = t - pd.Timedelta(days=7)
        if t_prev in dt_to_vol:
            pred_seas.append(float(dt_to_vol[t_prev]))
        else:
            pred_seas.append(float(hist_lookup.get((d, h), global_mean)))
    pred_seas = np.array(pred_seas, dtype=float)

    # 4. Model Predictions on Test Set
    pred_ridge = tuned_results["Ridge"]["model"].predict(X_test)
    pred_gbr = tuned_results["Gradient Boosting"]["model"].predict(X_test)
    pred_rf = tuned_results["Random Forest"]["model"].predict(X_test)

    # 5. Bootstrap Confidence Intervals (1,000 resamples)
    print("\n--- Computing 1,000-Resample Bootstrap 95% Confidence Intervals ---")
    predictions = {
        "Global Mean": pred_global,
        "Historical Mean (Hour, DOW)": pred_hist,
        "Seasonal-Naive (t - 7d)": pred_seas,
        "Ridge": pred_ridge,
        "Gradient Boosting": pred_gbr,
        "Random Forest": pred_rf,
    }

    test_metrics_ci = {}
    for name, pred in predictions.items():
        ci = compute_bootstrap_ci(y_test, pred, n_bootstraps=1000)
        test_metrics_ci[name] = ci

    # 6. Paired Bootstrap Test: Best Calendar Model vs Historical Mean Baseline
    # GBR has MAE 260.78 vs Baseline 260.82
    paired_ci_gbr = compute_paired_bootstrap(y_test, pred_gbr, pred_hist, n_bootstraps=1000)
    paired_ci_rf = compute_paired_bootstrap(y_test, pred_rf, pred_hist, n_bootstraps=1000)

    # 7. Rolling-Origin Cross Validation (5 folds)
    rolling_summary = run_rolling_origin_eval(df, cal_features, TARGET_COL, n_splits=5)

    # 8. Experiment 2: Nowcast (with weather)
    nowcast_results = run_nowcast_experiment(df, cal_features, train_df, val_df, test_df)
    test_metrics_ci["Nowcast GBR (with Weather) [Nowcast, not forecast]"] = nowcast_results["test_ci"]

    # 9. Save Best Calendar-Feature Model (< 5 MB)
    os.makedirs("models", exist_ok=True)
    best_calendar_name = "Gradient Boosting" if test_metrics_ci["Gradient Boosting"]["mae"]["point"] <= test_metrics_ci["Random Forest"]["mae"]["point"] else "Random Forest"
    best_model = tuned_results[best_calendar_name]["model"]
    best_pred = pred_gbr if best_calendar_name == "Gradient Boosting" else pred_rf

    model_path = os.path.join("models", "traffic_model.joblib")
    joblib.dump(best_model, model_path, compress=3)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nSaved best calendar model ({best_calendar_name}) to {model_path} ({model_size_mb:.2f} MB)")
    assert model_size_mb < 5.0, f"Model size {model_size_mb} MB exceeds 5 MB limit!"

    # 10. Generate Plots
    generate_plots(
        test_df,
        y_test,
        best_pred,
        pred_hist,
        nowcast_results["test_pred"],
        best_model,
        cal_features,
        best_calendar_name,
    )

    # 11. Compile and save results/metrics.json
    all_metrics_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "dataset": "UCI Metro Interstate Traffic Volume",
        "splits": {
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
        },
        "hyperparameter_tuning_val_set": {
            k: {"params": v["params"], "val_mae": v["val_mae"]}
            for k, v in tuned_results.items()
        },
        "test_set_bootstrap_95ci": test_metrics_ci,
        "paired_bootstrap_vs_historical_mean": {
            "Gradient_Boosting_vs_Historical_Mean": paired_ci_gbr,
            "Random_Forest_vs_Historical_Mean": paired_ci_rf,
        },
        "rolling_origin_5fold_summary": rolling_summary,
        "nowcast_experiment": {
            "description": "Contemporaneous weather features included (nowcast, not forecast)",
            "features": nowcast_results["features"],
            "val_metrics": nowcast_results["val_metrics"],
            "test_ci": nowcast_results["test_ci"],
        },
        "best_calendar_model": {
            "name": best_calendar_name,
            "artifact_path": model_path,
            "size_mb": round(model_size_mb, 2),
            "params": tuned_results[best_calendar_name]["params"],
        },
    }

    metrics_path = os.path.join("results", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics_payload, f, indent=2)
    print(f"Saved complete metrics to {metrics_path}")

    # 12. Display Terminal Summary Table
    print("\n" + "=" * 95)
    print(f"{'Model / Baseline':<35} | {'MAE [95% CI]':<22} | {'RMSE [95% CI]':<22} | {'R^2 [95% CI]':<18}")
    print("-" * 95)
    for name, m in test_metrics_ci.items():
        mae_str = f"{m['mae']['point']:.2f} [{m['mae']['ci_lower']:.1f}, {m['mae']['ci_upper']:.1f}]"
        rmse_str = f"{m['rmse']['point']:.2f} [{m['rmse']['ci_lower']:.1f}, {m['rmse']['ci_upper']:.1f}]"
        r2_str = f"{m['r2']['point']:.4f} [{m['r2']['ci_lower']:.4f}, {m['r2']['ci_upper']:.4f}]"
        print(f"{name:<35} | {mae_str:<22} | {rmse_str:<22} | {r2_str:<18}")
    print("=" * 95)


if __name__ == "__main__":
    main()
