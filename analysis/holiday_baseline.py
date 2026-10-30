"""
Evaluation of Holiday-Aware Historical Mean Baseline:
Grouping by (Hour, Day-of-Week, is_us_holiday_all_day) on Train+Val,
and Evaluating Once on the Test Split vs the Final Config 3 Gradient Boosting Model.

Protocol:
---------
1. Baseline lookup table: Group df_train_val (2012-10-02 to 2018-01-19, N=34,488)
   by (hour, day_of_week, is_us_holiday) and take the mean traffic_volume.
   Fallback hierarchy:
     Level 1: (hour, day_of_week, is_us_holiday)
     Level 2: (hour, day_of_week) [standard historical mean]
     Level 3: global mean of train+val
2. Test split: Exactly out-of-sample test split (2018-01-19 to 2018-09-30, N=6,087).
3. Final Model: Trained Config 3 GradientBoostingRegressor loaded from models/traffic_model.joblib.
4. Bootstrap 95% Confidence Intervals: 1,000 resamples for MAE, RMSE, R^2,
   and paired error differences (|y - y_model| - |y - y_holiday_baseline|).
5. Output: Save detailed results to results/holiday_baseline.json and print summary.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Suppress loky CPU detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train.experiments import US_FEDERAL_HOLIDAYS, build_experimental_features
from train.prepare_data import (
    TARGET_COL,
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


def compute_bootstrap_evaluation(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_holiday_base: np.ndarray,
    y_pred_std_hist: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Computes 95% bootstrap confidence intervals for:
      1. Model (Config 3 GBR)
      2. Holiday-aware baseline (Hour, DOW, is_us_holiday)
      3. Standard historical mean baseline (Hour, DOW)
      4. Paired differences: Model vs Holiday Baseline
      5. Paired differences: Holiday Baseline vs Standard Historical Mean
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = rng.randint(0, n, size=(n_bootstraps, n))

    # Metric collections
    m_mae_list, m_rmse_list, m_r2_list = [], [], []
    hb_mae_list, hb_rmse_list, hb_r2_list = [], [], []
    sh_mae_list, sh_rmse_list, sh_r2_list = [], [], []

    # Paired differences: Model minus Holiday Baseline (negative = model is better)
    err_model = np.abs(y_true - y_pred_model)
    err_hb = np.abs(y_true - y_pred_holiday_base)
    err_sh = np.abs(y_true - y_pred_std_hist)

    diff_model_vs_hb = err_model - err_hb
    diff_hb_vs_sh = err_hb - err_sh

    diff_m_hb_mae_list, diff_m_hb_rmse_list = [], []
    diff_hb_sh_mae_list, diff_hb_sh_rmse_list = [], []

    for idx in indices:
        yt = y_true[idx]
        ym = y_pred_model[idx]
        yhb = y_pred_holiday_base[idx]
        ysh = y_pred_std_hist[idx]

        ss_tot = np.sum((yt - np.mean(yt)) ** 2)

        # Model
        m_mae = float(np.mean(np.abs(yt - ym)))
        m_rmse = float(np.sqrt(np.mean((yt - ym) ** 2)))
        m_r2 = float(1.0 - np.sum((yt - ym) ** 2) / ss_tot) if ss_tot > 0 else 0.0
        m_mae_list.append(m_mae)
        m_rmse_list.append(m_rmse)
        m_r2_list.append(m_r2)

        # Holiday Baseline
        hb_mae = float(np.mean(np.abs(yt - yhb)))
        hb_rmse = float(np.sqrt(np.mean((yt - yhb) ** 2)))
        hb_r2 = float(1.0 - np.sum((yt - yhb) ** 2) / ss_tot) if ss_tot > 0 else 0.0
        hb_mae_list.append(hb_mae)
        hb_rmse_list.append(hb_rmse)
        hb_r2_list.append(hb_r2)

        # Standard Historical Mean
        sh_mae = float(np.mean(np.abs(yt - ysh)))
        sh_rmse = float(np.sqrt(np.mean((yt - ysh) ** 2)))
        sh_r2 = float(1.0 - np.sum((yt - ysh) ** 2) / ss_tot) if ss_tot > 0 else 0.0
        sh_mae_list.append(sh_mae)
        sh_rmse_list.append(sh_rmse)
        sh_r2_list.append(sh_r2)

        # Paired diffs
        diff_m_hb_mae_list.append(float(np.mean(diff_model_vs_hb[idx])))
        diff_m_hb_rmse_list.append(m_rmse - hb_rmse)

        diff_hb_sh_mae_list.append(float(np.mean(diff_hb_vs_sh[idx])))
        diff_hb_sh_rmse_list.append(hb_rmse - sh_rmse)

    point_m = calc_metrics(y_true, y_pred_model)
    point_hb = calc_metrics(y_true, y_pred_holiday_base)
    point_sh = calc_metrics(y_true, y_pred_std_hist)

    def ci_block(point_dict: Dict[str, float], maes: List[float], rmses: List[float], r2s: List[float]) -> Dict[str, Any]:
        return {
            "mae": {
                "point": point_dict["mae"],
                "ci_lower": round(float(np.percentile(maes, 2.5)), 2),
                "ci_upper": round(float(np.percentile(maes, 97.5)), 2),
            },
            "rmse": {
                "point": point_dict["rmse"],
                "ci_lower": round(float(np.percentile(rmses, 2.5)), 2),
                "ci_upper": round(float(np.percentile(rmses, 97.5)), 2),
            },
            "r2": {
                "point": point_dict["r2"],
                "ci_lower": round(float(np.percentile(r2s, 2.5)), 4),
                "ci_upper": round(float(np.percentile(r2s, 97.5)), 4),
            },
        }

    delta_m_hb_mae_low = round(float(np.percentile(diff_m_hb_mae_list, 2.5)), 2)
    delta_m_hb_mae_high = round(float(np.percentile(diff_m_hb_mae_list, 97.5)), 2)
    delta_m_hb_mae_pt = round(float(np.mean(diff_model_vs_hb)), 2)

    delta_hb_sh_mae_low = round(float(np.percentile(diff_hb_sh_mae_list, 2.5)), 2)
    delta_hb_sh_mae_high = round(float(np.percentile(diff_hb_sh_mae_list, 97.5)), 2)
    delta_hb_sh_mae_pt = round(float(np.mean(diff_hb_vs_sh)), 2)

    model_beats_holiday_baseline = bool(delta_m_hb_mae_high < 0.0)

    return {
        "model_config3_metrics": ci_block(point_m, m_mae_list, m_rmse_list, m_r2_list),
        "holiday_baseline_metrics": ci_block(point_hb, hb_mae_list, hb_rmse_list, hb_r2_list),
        "standard_historical_mean_metrics": ci_block(point_sh, sh_mae_list, sh_rmse_list, sh_r2_list),
        "paired_difference_model_vs_holiday_baseline": {
            "delta_mae_point": delta_m_hb_mae_pt,
            "delta_mae_ci_lower": delta_m_hb_mae_low,
            "delta_mae_ci_upper": delta_m_hb_mae_high,
            "delta_rmse_point": round(point_m["rmse"] - point_hb["rmse"], 2),
            "delta_rmse_ci_lower": round(float(np.percentile(diff_m_hb_rmse_list, 2.5)), 2),
            "delta_rmse_ci_upper": round(float(np.percentile(diff_m_hb_rmse_list, 97.5)), 2),
            "model_beats_holiday_baseline": model_beats_holiday_baseline,
            "excludes_zero": bool(delta_m_hb_mae_high < 0.0 or delta_m_hb_mae_low > 0.0),
        },
        "paired_difference_holiday_baseline_vs_standard_hist_mean": {
            "delta_mae_point": delta_hb_sh_mae_pt,
            "delta_mae_ci_lower": delta_hb_sh_mae_low,
            "delta_mae_ci_upper": delta_hb_sh_mae_high,
            "delta_rmse_point": round(point_hb["rmse"] - point_sh["rmse"], 2),
            "delta_rmse_ci_lower": round(float(np.percentile(diff_hb_sh_rmse_list, 2.5)), 2),
            "delta_rmse_ci_upper": round(float(np.percentile(diff_hb_sh_rmse_list, 97.5)), 2),
            "holiday_baseline_beats_standard_hist_mean": bool(delta_hb_sh_mae_high < 0.0),
            "excludes_zero": bool(delta_hb_sh_mae_high < 0.0 or delta_hb_sh_mae_low > 0.0),
        },
    }


def main():
    t_start = time.time()
    print("=" * 85)
    print("Holiday-Aware Historical Mean Baseline vs Config 3 Gradient Boosting Model")
    print("=" * 85)

    # 1. Load and process features
    df = load_and_clean_data()
    df_feat, feature_sets = build_experimental_features(df)
    features = feature_sets["Config_3_Holiday_Proximity"]

    # 2. Chronological split
    train_df, val_df, test_df = split_data_chronological(df_feat)
    df_train_val = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Train+Val pool: {len(df_train_val):,} rows ({df_train_val['date_time'].min()} to {df_train_val['date_time'].max()})")
    print(f"Test split:     {len(test_df):,} rows ({test_df['date_time'].min()} to {test_df['date_time'].max()})")

    y_train_val = df_train_val[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 3. Fit Holiday-Aware Historical Mean Baseline on Train+Val
    # Keys: (hour, day_of_week, is_us_holiday)
    h_tr = df_train_val["date_time"].dt.hour
    dow_tr = df_train_val["date_time"].dt.dayofweek
    hol_tr = df_train_val["is_us_holiday"].values

    lookup_df = pd.DataFrame({
        "h": h_tr,
        "dow": dow_tr,
        "hol": hol_tr,
        "y": y_train_val,
    })

    # Groupby lookup tables with fallbacks
    table_holiday = lookup_df.groupby(["h", "dow", "hol"])["y"].mean().to_dict()
    table_dow_hour = lookup_df.groupby(["h", "dow"])["y"].mean().to_dict()
    global_mean = float(y_train_val.mean())

    print(f"\n[Baseline Fitted] Created holiday lookup table with {len(table_holiday)} (hour, dow, holiday) cells.")
    print(f"[Baseline Fitted] Created fallback lookup table with {len(table_dow_hour)} (hour, dow) cells.")
    print(f"[Baseline Fitted] Global train+val mean: {global_mean:.2f} veh/h")

    # 4. Predict Holiday Baseline on Test Split
    h_te = test_df["date_time"].dt.hour
    dow_te = test_df["date_time"].dt.dayofweek
    hol_te = test_df["is_us_holiday"].values

    pred_holiday_base = []
    pred_std_hist = []

    fallback_count = 0
    for h, d, hol in zip(h_te, dow_te, hol_te):
        # Holiday baseline prediction
        if (h, d, hol) in table_holiday:
            pred_holiday_base.append(table_holiday[(h, d, hol)])
        elif (h, d) in table_dow_hour:
            pred_holiday_base.append(table_dow_hour[(h, d)])
            fallback_count += 1
        else:
            pred_holiday_base.append(global_mean)
            fallback_count += 1

        # Standard historical mean baseline (hour, dow only)
        pred_std_hist.append(table_dow_hour.get((h, d), global_mean))

    pred_holiday_base = np.array(pred_holiday_base, dtype=float)
    pred_std_hist = np.array(pred_std_hist, dtype=float)

    if fallback_count > 0:
        print(f"[Notice] {fallback_count}/{len(test_df)} test records fell back to (hour, dow) table.")
    else:
        print(f"[Verification] All {len(test_df):,} test records matched an exact (hour, dow, holiday) cell.")

    # 5. Load Trained Config 3 Model
    model_path = os.path.join(REPO_ROOT, "models", "traffic_model.joblib")
    assert os.path.exists(model_path), f"Model file not found at {model_path}!"
    model = joblib.load(model_path)
    print(f"\nLoaded trained model from {model_path}")
    print(f"Model feature count: {model.n_features_in_} | Features: {features}")

    X_test = test_df[features]
    pred_model = model.predict(X_test)

    # 6. Evaluate Bootstrap CIs & Paired Hypothesis Tests
    print("\nComputing 1,000 bootstrap resamples on test set...")
    eval_results = compute_bootstrap_evaluation(
        y_true=y_test,
        y_pred_model=pred_model,
        y_pred_holiday_base=pred_holiday_base,
        y_pred_std_hist=pred_std_hist,
        n_bootstraps=1000,
        seed=RANDOM_SEED,
    )

    m_m = eval_results["model_config3_metrics"]
    hb_m = eval_results["holiday_baseline_metrics"]
    sh_m = eval_results["standard_historical_mean_metrics"]
    diff_m_hb = eval_results["paired_difference_model_vs_holiday_baseline"]
    diff_hb_sh = eval_results["paired_difference_holiday_baseline_vs_standard_hist_mean"]

    # 7. Print Comparative Table
    print("\n" + "=" * 95)
    print(f"{'Evaluation on Test Split (N=6,087)':<40} | {'Point Estimate':<15} | {'95% Bootstrap CI':<25}")
    print("-" * 95)
    print(f"{'1. Standard Hist Mean (Hour, DOW) MAE':<40} | {sh_m['mae']['point']:<15.2f} | [{sh_m['mae']['ci_lower']:.2f}, {sh_m['mae']['ci_upper']:.2f}] veh/h")
    print(f"{'   Standard Hist Mean RMSE':<40} | {sh_m['rmse']['point']:<15.2f} | [{sh_m['rmse']['ci_lower']:.2f}, {sh_m['rmse']['ci_upper']:.2f}] veh/h")
    print(f"{'   Standard Hist Mean R^2':<40} | {sh_m['r2']['point']:<15.4f} | [{sh_m['r2']['ci_lower']:.4f}, {sh_m['r2']['ci_upper']:.4f}]")
    print("-" * 95)
    print(f"{'2. Holiday Baseline (Hour, DOW, Hol) MAE':<40} | {hb_m['mae']['point']:<15.2f} | [{hb_m['mae']['ci_lower']:.2f}, {hb_m['mae']['ci_upper']:.2f}] veh/h")
    print(f"{'   Holiday Baseline RMSE':<40} | {hb_m['rmse']['point']:<15.2f} | [{hb_m['rmse']['ci_lower']:.2f}, {hb_m['rmse']['ci_upper']:.2f}] veh/h")
    print(f"{'   Holiday Baseline R^2':<40} | {hb_m['r2']['point']:<15.4f} | [{hb_m['r2']['ci_lower']:.4f}, {hb_m['r2']['ci_upper']:.4f}]")
    print("-" * 95)
    print(f"{'3. Final Model (Config 3 GBR) MAE':<40} | {m_m['mae']['point']:<15.2f} | [{m_m['mae']['ci_lower']:.2f}, {m_m['mae']['ci_upper']:.2f}] veh/h")
    print(f"{'   Final Model RMSE':<40} | {m_m['rmse']['point']:<15.2f} | [{m_m['rmse']['ci_lower']:.2f}, {m_m['rmse']['ci_upper']:.2f}] veh/h")
    print(f"{'   Final Model R^2':<40} | {m_m['r2']['point']:<15.4f} | [{m_m['r2']['ci_lower']:.4f}, {m_m['r2']['ci_upper']:.4f}]")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("PAIRED HYPOTHESIS TESTING SUMMARY")
    print("=" * 95)
    print(f"A. Holiday Baseline vs Standard Historical Mean:")
    print(f"   Delta MAE:  {diff_hb_sh['delta_mae_point']:+.2f} veh/h [{diff_hb_sh['delta_mae_ci_lower']:+.2f}, {diff_hb_sh['delta_mae_ci_upper']:+.2f}]")
    print(f"   Delta RMSE: {diff_hb_sh['delta_rmse_point']:+.2f} veh/h [{diff_hb_sh['delta_rmse_ci_lower']:+.2f}, {diff_hb_sh['delta_rmse_ci_upper']:+.2f}]")
    print(f"   Statistically Significant Improvement? {'YES' if diff_hb_sh['holiday_baseline_beats_standard_hist_mean'] else 'NO'}")
    print("-" * 95)
    print(f"B. Final Config 3 Model vs Holiday-Aware Baseline:")
    print(f"   Delta MAE:  {diff_m_hb['delta_mae_point']:+.2f} veh/h [{diff_m_hb['delta_mae_ci_lower']:+.2f}, {diff_m_hb['delta_mae_ci_upper']:+.2f}]")
    print(f"   Delta RMSE: {diff_m_hb['delta_rmse_point']:+.2f} veh/h [{diff_m_hb['delta_rmse_ci_lower']:+.2f}, {diff_m_hb['delta_rmse_ci_upper']:+.2f}]")
    print(f"   Does Model Still Beat the Holiday Baseline? {'YES' if diff_m_hb['model_beats_holiday_baseline'] else 'NO'}")
    print(f"   Statistically Significant (CI < 0)? {'YES' if diff_m_hb['model_beats_holiday_baseline'] else 'NO'}")
    print("=" * 95)

    # 8. Plain Statement
    if diff_m_hb["model_beats_holiday_baseline"]:
        verdict_str = (
            f"The final Config 3 Gradient Boosting model STILL BEATS the holiday-aware historical mean baseline. "
            f"Model Test MAE is {m_m['mae']['point']:.2f} veh/h vs Holiday Baseline MAE of {hb_m['mae']['point']:.2f} veh/h "
            f"(paired difference: {diff_m_hb['delta_mae_point']:+.2f} veh/h, 95% CI [{diff_m_hb['delta_mae_ci_lower']:+.2f}, {diff_m_hb['delta_mae_ci_upper']:+.2f}] veh/h, strictly excluding zero)."
        )
    else:
        verdict_str = (
            f"The final Config 3 Gradient Boosting model DOES NOT statistically beat the holiday-aware historical mean baseline. "
            f"Model Test MAE is {m_m['mae']['point']:.2f} veh/h vs Holiday Baseline MAE of {hb_m['mae']['point']:.2f} veh/h "
            f"(paired difference: {diff_m_hb['delta_mae_point']:+.2f} veh/h, 95% CI [{diff_m_hb['delta_mae_ci_lower']:+.2f}, {diff_m_hb['delta_mae_ci_upper']:+.2f}] veh/h)."
        )
    print(f"\n>>> VERDICT: {verdict_str}\n")

    # 9. Save results/holiday_baseline.json
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "holiday_baseline.json")

    payload = {
        "metadata": {
            "evaluation_title": "Evaluation of Holiday-Aware Historical Mean Baseline on Test Split",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime_seconds": round(time.time() - t_start, 2),
            "train_val_count": len(df_train_val),
            "train_val_date_range": [str(df_train_val["date_time"].min()), str(df_train_val["date_time"].max())],
            "test_count": len(test_df),
            "test_date_range": [str(test_df["date_time"].min()), str(test_df["date_time"].max())],
            "baseline_definition": "Historical mean traffic_volume grouped by (hour, day_of_week, is_us_holiday_all_day) on train+val",
            "holiday_flags_definition": "All-day US Federal Holiday flag (1 for all 24 hours of federal holidays, 0 otherwise)",
            "verdict": verdict_str,
        },
        "evaluation_metrics": eval_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Artifact Saved] Results saved to {out_path}")
    print(f"Total script execution time: {time.time() - t_start:.1f} seconds")


if __name__ == "__main__":
    main()
