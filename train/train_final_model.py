"""
Retrain Recommended Configuration on Train+Val and Final Test Set Evaluation.

Protocol & Integrity Rules:
----------------------------
1. Recommended Configuration: Config 3 (Calendar + US Holiday Proximity with Gradient Boosting).
2. Training Data: Train + Validation combined (2012-10-02 to 2018-01-19, N=34,488).
3. Test Set Evaluation: Evaluated strictly ONCE on the out-of-sample test split
   (2018-01-19 15:00:00 to 2018-09-30 23:00:00, N=6,087).
4. Bootstrap Confidence Intervals: 1,000 resamples for test MAE, RMSE, R^2,
   and paired error differences vs the Historical Mean baseline.
5. Saves results to results/final_test.json and updates models/traffic_model.joblib.
6. NO FURTHER ITERATION: Single final evaluation as mandated.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

# Suppress loky CPU detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train.experiments import US_FEDERAL_HOLIDAYS, build_experimental_features
from train.prepare_data import (
    TARGET_COL,
    build_features,
    load_and_clean_data,
    split_data_chronological,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }


def compute_test_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))

    # Model metrics
    mae_list, rmse_list, r2_list = [], [], []
    # Baseline metrics
    mae_base_list, rmse_base_list, r2_base_list = [], [], []
    # Paired differences
    diff_mae_list, diff_rmse_list = [], []

    err_model = np.abs(y_true - y_pred_model)
    err_base = np.abs(y_true - y_pred_baseline)
    diff_point = err_model - err_base

    for idx in indices:
        yt = y_true[idx]
        ym = y_pred_model[idx]
        yb = y_pred_baseline[idx]

        m_mae = float(np.mean(np.abs(yt - ym)))
        m_rmse = float(np.sqrt(np.mean((yt - ym) ** 2)))
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        m_r2 = float(1.0 - np.sum((yt - ym) ** 2) / ss_tot) if ss_tot > 0 else 0.0

        b_mae = float(np.mean(np.abs(yt - yb)))
        b_rmse = float(np.sqrt(np.mean((yt - yb) ** 2)))
        b_r2 = float(1.0 - np.sum((yt - yb) ** 2) / ss_tot) if ss_tot > 0 else 0.0

        mae_list.append(m_mae)
        rmse_list.append(m_rmse)
        r2_list.append(m_r2)

        mae_base_list.append(b_mae)
        rmse_base_list.append(b_rmse)
        r2_base_list.append(b_r2)

        diff_mae_list.append(float(np.mean(diff_point[idx])))
        diff_rmse_list.append(m_rmse - b_rmse)

    point_model = calc_metrics(y_true, y_pred_model)
    point_base = calc_metrics(y_true, y_pred_baseline)

    return {
        "model_metrics": {
            "mae": {
                "point": point_model["mae"],
                "ci_lower": round(float(np.percentile(mae_list, 2.5)), 2),
                "ci_upper": round(float(np.percentile(mae_list, 97.5)), 2),
            },
            "rmse": {
                "point": point_model["rmse"],
                "ci_lower": round(float(np.percentile(rmse_list, 2.5)), 2),
                "ci_upper": round(float(np.percentile(rmse_list, 97.5)), 2),
            },
            "r2": {
                "point": point_model["r2"],
                "ci_lower": round(float(np.percentile(r2_list, 2.5)), 4),
                "ci_upper": round(float(np.percentile(r2_list, 97.5)), 4),
            },
        },
        "historical_mean_metrics": {
            "mae": {
                "point": point_base["mae"],
                "ci_lower": round(float(np.percentile(mae_base_list, 2.5)), 2),
                "ci_upper": round(float(np.percentile(mae_base_list, 97.5)), 2),
            },
            "rmse": {
                "point": point_base["rmse"],
                "ci_lower": round(float(np.percentile(rmse_base_list, 2.5)), 2),
                "ci_upper": round(float(np.percentile(rmse_base_list, 97.5)), 2),
            },
            "r2": {
                "point": point_base["r2"],
                "ci_lower": round(float(np.percentile(r2_base_list, 2.5)), 4),
                "ci_upper": round(float(np.percentile(r2_base_list, 97.5)), 4),
            },
        },
        "paired_difference_vs_historical_mean": {
            "delta_mae_point": round(float(np.mean(diff_point)), 2),
            "delta_mae_ci_lower": round(float(np.percentile(diff_mae_list, 2.5)), 2),
            "delta_mae_ci_upper": round(float(np.percentile(diff_mae_list, 97.5)), 2),
            "delta_rmse_point": round(point_model["rmse"] - point_base["rmse"], 2),
            "delta_rmse_ci_lower": round(float(np.percentile(diff_rmse_list, 2.5)), 2),
            "delta_rmse_ci_upper": round(float(np.percentile(diff_rmse_list, 97.5)), 2),
            "significant_improvement": bool(np.percentile(diff_mae_list, 97.5) < 0.0),
        },
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("Retraining Recommended Model (Config 3) on Train+Val and Evaluating on Test")
    print("=" * 80)

    # 1. Load Clean Data
    data_path = os.path.join(REPO_ROOT, "data", "Metro_Interstate_Traffic_Volume.csv")
    df = load_and_clean_data(data_path)
    df, _, _ = build_features(df)

    # 2. Enrich with Experimental Holiday Proximity Features
    df, feature_sets = build_experimental_features(df)
    features = feature_sets["Config_3_Holiday_Proximity"]
    print(f"\n[Model Features] {features}")

    # 3. Chronological Splits
    train_df, val_df, test_df = split_data_chronological(df, 0.70, 0.15, 0.15)

    # Combine Train + Val
    df_train_val = pd.concat([train_df, val_df], ignore_index=True)
    del train_df, val_df

    print(f"\n[Training Pool] Train+Val: {len(df_train_val):,} rows ({df_train_val['date_time'].min()} to {df_train_val['date_time'].max()})")
    print(f"[Test Set] Single Final Evaluation: {len(test_df):,} rows ({test_df['date_time'].min()} to {test_df['date_time'].max()})")

    X_train_val = df_train_val[features]
    y_train_val = df_train_val[TARGET_COL].values

    X_test = test_df[features]
    y_test = test_df[TARGET_COL].values

    # 4. Fit Recommended Model (Config 3: Calendar + Holiday Proximity GBR)
    print("\nFitting GradientBoostingRegressor on Train+Val (34,488 rows)...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_SEED,
    )
    t_fit = time.time()
    model.fit(X_train_val, y_train_val)
    print(f"Fit completed in {time.time() - t_fit:.1f} seconds.")

    # 5. Predict on Test Set
    pred_test = model.predict(X_test)

    # 6. Historical Mean Baseline on Test Set
    h_tr = df_train_val["date_time"].dt.hour
    dow_tr = df_train_val["date_time"].dt.dayofweek
    lookup = (
        pd.DataFrame({"dow": dow_tr, "h": h_tr, "y": y_train_val})
        .groupby(["dow", "h"])["y"]
        .mean()
        .to_dict()
    )
    gm = float(y_train_val.mean())
    h_te = test_df["date_time"].dt.hour
    dow_te = test_df["date_time"].dt.dayofweek
    pred_hist = np.array([lookup.get((d, h), gm) for d, h in zip(dow_te, h_te)], dtype=float)

    # 7. Compute Bootstrap Confidence Intervals and Paired Differences
    print("\nComputing 1,000 bootstrap resamples on test set...")
    eval_results = compute_test_bootstrap_ci(y_test, pred_test, pred_hist, n_bootstraps=1000)

    mod_m = eval_results["model_metrics"]
    hist_m = eval_results["historical_mean_metrics"]
    diff_m = eval_results["paired_difference_vs_historical_mean"]

    print("\n" + "=" * 90)
    print(f"{'Evaluation on Test Split (N=6,087)':<35} | {'Point Estimate':<15} | {'95% Bootstrap CI':<25}")
    print("-" * 90)
    print(f"{'Model Test MAE':<35} | {mod_m['mae']['point']:<15.2f} | [{mod_m['mae']['ci_lower']:.2f}, {mod_m['mae']['ci_upper']:.2f}] veh/h")
    print(f"{'Model Test RMSE':<35} | {mod_m['rmse']['point']:<15.2f} | [{mod_m['rmse']['ci_lower']:.2f}, {mod_m['rmse']['ci_upper']:.2f}] veh/h")
    print(f"{'Model Test R^2':<35} | {mod_m['r2']['point']:<15.4f} | [{mod_m['r2']['ci_lower']:.4f}, {mod_m['r2']['ci_upper']:.4f}]")
    print("-" * 90)
    print(f"{'Hist Mean Test MAE':<35} | {hist_m['mae']['point']:<15.2f} | [{hist_m['mae']['ci_lower']:.2f}, {hist_m['mae']['ci_upper']:.2f}] veh/h")
    print(f"{'Hist Mean Test RMSE':<35} | {hist_m['rmse']['point']:<15.2f} | [{hist_m['rmse']['ci_lower']:.2f}, {hist_m['rmse']['ci_upper']:.2f}] veh/h")
    print(f"{'Hist Mean Test R^2':<35} | {hist_m['r2']['point']:<15.4f} | [{hist_m['r2']['ci_lower']:.4f}, {hist_m['r2']['ci_upper']:.4f}]")
    print("-" * 90)
    print(f"{'Delta MAE (Model - Hist)':<35} | {diff_m['delta_mae_point']:<+15.2f} | [{diff_m['delta_mae_ci_lower']:+.2f}, {diff_m['delta_mae_ci_upper']:+.2f}] veh/h")
    print(f"{'Delta RMSE (Model - Hist)':<35} | {diff_m['delta_rmse_point']:<+15.2f} | [{diff_m['delta_rmse_ci_lower']:+.2f}, {diff_m['delta_rmse_ci_upper']:+.2f}] veh/h")
    print(f"{'Statistically Significant?':<35} | {'YES' if diff_m['significant_improvement'] else 'NO':<15} | {'CI strictly < 0' if diff_m['significant_improvement'] else 'CI includes 0'}")
    print("=" * 90)

    # 8. Save results/final_test.json
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "final_test.json")

    payload = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_architecture": "GradientBoostingRegressor",
            "features": features,
            "train_val_count": len(df_train_val),
            "train_val_date_range": [str(df_train_val["date_time"].min()), str(df_train_val["date_time"].max())],
            "test_count": len(test_df),
            "test_date_range": [str(test_df["date_time"].min()), str(test_df["date_time"].max())],
            "note_on_test_split": (
                "The test split (2018-01-19 to 2018-09-30) was previously viewed during earlier baseline auditing. "
                "This evaluation was executed strictly once on the final frozen model without iteration or tuning."
            ),
        },
        "test_evaluation": eval_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Artifact Saved] Final test metrics -> {json_path}")

    # 9. Update models/traffic_model.joblib
    models_dir = os.path.join(REPO_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "traffic_model.joblib")
    joblib.dump(model, model_path, compress=3)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[Model Updated] Saved updated traffic_model.joblib ({model_size_mb:.2f} MB)")
    assert model_size_mb < 5.0, f"Model size {model_size_mb} MB exceeds 5 MB limit!"

    print(f"Total script runtime: {time.time() - t_start:.1f} seconds")


if __name__ == "__main__":
    main()
