"""
Residual and Error Pattern Analysis on the Validation Split.

Task:
-----
Evaluate the current best calendar-feature traffic prediction model (models/traffic_model.joblib)
strictly on the validation split (2017-05-10 05:00:00 to 2018-01-19 14:00:00).

Strict Data Integrity Constraints:
----------------------------------
- NO TEST SET DATA IS ACCESSED OR USED HERE.
- All metrics are calculated exclusively on the validation slice.
- Reports MAE by hour of day, day of week, month, holiday vs non-holiday,
  and identifies the 20 worst-predicted hours with their meteorological and calendar context.
- Exports structured metrics and markdown tables to results/, and generates analytical plots.
"""

import json
import os
import sys
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train.prepare_data import (
    CALENDAR_FEATURES,
    TARGET_COL,
    build_features,
    load_and_clean_data,
    split_data_chronological,
)


def run_residual_analysis():
    print("=" * 80)
    print("Residual and Error Pattern Analysis on Validation Split")
    print("=" * 80)

    # 1. Load Data and Construct Features
    data_path = os.path.join(REPO_ROOT, "data", "Metro_Interstate_Traffic_Volume.csv")
    df = load_and_clean_data(data_path)
    df, cal_features, weather_features = build_features(df)

    # Strict chronological split: 70% Train, 15% Validation, 15% Test
    train_df, val_df, test_df = split_data_chronological(df, 0.70, 0.15, 0.15)
    del train_df, test_df  # Explicitly discard train and test sets to guarantee separation

    print(f"\n[Validation Set] Rows: {len(val_df):,}")
    print(f"[Validation Range] Start: {val_df['date_time'].min()} | End: {val_df['date_time'].max()}")

    # 2. Load Current Best Model
    model_path = os.path.join(REPO_ROOT, "models", "traffic_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    model = joblib.load(model_path)
    print(f"[Loaded Model] {type(model).__name__} from {model_path}")

    # 3. Predict on Validation Split Only
    X_val = val_df[cal_features]
    y_val = val_df[TARGET_COL].values
    val_preds = model.predict(X_val)

    # Enrich validation dataframe with predictions and error metrics
    val_df = val_df.copy()
    val_df["pred_volume"] = val_preds
    val_df["signed_error"] = val_df[TARGET_COL] - val_preds  # Positive = underprediction, Negative = overprediction
    val_df["abs_error"] = np.abs(val_df["signed_error"])
    val_df["hour"] = val_df["date_time"].dt.hour
    val_df["day_of_week"] = val_df["date_time"].dt.dayofweek
    val_df["day_name"] = val_df["date_time"].dt.day_name()
    val_df["month"] = val_df["date_time"].dt.month
    val_df["month_name"] = val_df["date_time"].dt.month_name()
    val_df["year"] = val_df["date_time"].dt.year

    overall_mae = mean_absolute_error(y_val, val_preds)
    overall_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    overall_r2 = r2_score(y_val, val_preds)
    print(f"\n[Overall Validation Performance] MAE: {overall_mae:.2f} veh/h | RMSE: {overall_rmse:.2f} veh/h | R^2: {overall_r2:.4f}")

    # -------------------------------------------------------------------------
    # Breakdown 1: MAE by Hour of Day
    # -------------------------------------------------------------------------
    hour_stats: List[Dict[str, Any]] = []
    for h in range(24):
        sub = val_df[val_df["hour"] == h]
        h_mae = mean_absolute_error(sub[TARGET_COL], sub["pred_volume"])
        h_rmse = np.sqrt(mean_squared_error(sub[TARGET_COL], sub["pred_volume"]))
        h_signed = float(sub["signed_error"].mean())
        hour_stats.append({
            "hour": int(h),
            "count": int(len(sub)),
            "mae": round(float(h_mae), 2),
            "rmse": round(float(h_rmse), 2),
            "mean_actual": round(float(sub[TARGET_COL].mean()), 2),
            "mean_pred": round(float(sub["pred_volume"].mean()), 2),
            "mean_signed_error": round(h_signed, 2),
        })

    # -------------------------------------------------------------------------
    # Breakdown 2: MAE by Day of Week
    # -------------------------------------------------------------------------
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats: List[Dict[str, Any]] = []
    for dow_idx, day_name in enumerate(day_order):
        sub = val_df[val_df["day_of_week"] == dow_idx]
        d_mae = mean_absolute_error(sub[TARGET_COL], sub["pred_volume"])
        d_rmse = np.sqrt(mean_squared_error(sub[TARGET_COL], sub["pred_volume"]))
        d_signed = float(sub["signed_error"].mean())
        dow_stats.append({
            "day_of_week": int(dow_idx),
            "day_name": day_name,
            "count": int(len(sub)),
            "mae": round(float(d_mae), 2),
            "rmse": round(float(d_rmse), 2),
            "mean_actual": round(float(sub[TARGET_COL].mean()), 2),
            "mean_pred": round(float(sub["pred_volume"].mean()), 2),
            "mean_signed_error": round(d_signed, 2),
        })

    # -------------------------------------------------------------------------
    # Breakdown 3: MAE by Month
    # -------------------------------------------------------------------------
    month_stats: List[Dict[str, Any]] = []
    for m in sorted(val_df["month"].unique()):
        sub = val_df[val_df["month"] == m]
        m_mae = mean_absolute_error(sub[TARGET_COL], sub["pred_volume"])
        m_rmse = np.sqrt(mean_squared_error(sub[TARGET_COL], sub["pred_volume"]))
        m_signed = float(sub["signed_error"].mean())
        m_name = sub["month_name"].iloc[0]
        month_stats.append({
            "month": int(m),
            "month_name": m_name,
            "count": int(len(sub)),
            "mae": round(float(m_mae), 2),
            "rmse": round(float(m_rmse), 2),
            "mean_actual": round(float(sub[TARGET_COL].mean()), 2),
            "mean_pred": round(float(sub["pred_volume"].mean()), 2),
            "mean_signed_error": round(m_signed, 2),
        })

    # -------------------------------------------------------------------------
    # Breakdown 4: Holiday vs Non-Holiday
    # -------------------------------------------------------------------------
    non_hol = val_df[val_df["is_holiday"] == 0]
    hol = val_df[val_df["is_holiday"] == 1]

    holiday_comparison = {
        "non_holiday": {
            "count": int(len(non_hol)),
            "mae": round(float(mean_absolute_error(non_hol[TARGET_COL], non_hol["pred_volume"])), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(non_hol[TARGET_COL], non_hol["pred_volume"]))), 2),
            "mean_actual": round(float(non_hol[TARGET_COL].mean()), 2),
            "mean_pred": round(float(non_hol["pred_volume"].mean()), 2),
            "mean_signed_error": round(float(non_hol["signed_error"].mean()), 2),
        },
        "holiday": {
            "count": int(len(hol)),
            "mae": round(float(mean_absolute_error(hol[TARGET_COL], hol["pred_volume"])), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(hol[TARGET_COL], hol["pred_volume"]))), 2),
            "mean_actual": round(float(hol[TARGET_COL].mean()), 2),
            "mean_pred": round(float(hol["pred_volume"].mean()), 2),
            "mean_signed_error": round(float(hol["signed_error"].mean()), 2),
        },
    }

    # Named holiday breakdown in validation split
    named_holiday_rows = val_df[val_df["holiday"].fillna("None") != "None"]
    named_holidays: List[Dict[str, Any]] = []
    for hname, grp in named_holiday_rows.groupby("holiday"):
        named_holidays.append({
            "holiday_name": str(hname),
            "count": int(len(grp)),
            "mae": round(float(mean_absolute_error(grp[TARGET_COL], grp["pred_volume"])), 2),
            "mean_actual": round(float(grp[TARGET_COL].mean()), 2),
            "mean_pred": round(float(grp["pred_volume"].mean()), 2),
            "mean_signed_error": round(float(grp["signed_error"].mean()), 2),
        })

    # -------------------------------------------------------------------------
    # Breakdown 5: 20 Worst-Predicted Hours
    # -------------------------------------------------------------------------
    worst_20_df = val_df.sort_values("abs_error", ascending=False).head(20)
    worst_20: List[Dict[str, Any]] = []
    for rank, (_, row) in enumerate(worst_20_df.iterrows(), start=1):
        temp_c = round(float(row["temp"]) - 273.15, 1) if "temp" in row and pd.notnull(row["temp"]) else None
        worst_20.append({
            "rank": rank,
            "date_time": str(row["date_time"]),
            "actual_volume": int(row[TARGET_COL]),
            "predicted_volume": round(float(row["pred_volume"]), 1),
            "abs_error": round(float(row["abs_error"]), 1),
            "signed_error": round(float(row["signed_error"]), 1),
            "direction": "Underpredicted (Actual > Pred)" if row["signed_error"] > 0 else "Overpredicted (Pred > Actual)",
            "day_name": str(row["day_name"]),
            "hour": int(row["hour"]),
            "holiday": str(row["holiday"]) if pd.notnull(row.get("holiday")) else "None",
            "weather_main": str(row.get("weather_main", "Unknown")),
            "weather_description": str(row.get("weather_description", "Unknown")),
            "temp_celsius": temp_c,
            "rain_1h_mm": round(float(row.get("rain_1h", 0.0)), 2),
            "snow_1h_mm": round(float(row.get("snow_1h", 0.0)), 2),
            "clouds_all": int(row.get("clouds_all", 0)),
        })

    # -------------------------------------------------------------------------
    # 4. Save Structured Results JSON
    # -------------------------------------------------------------------------
    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_payload = {
        "metadata": {
            "model_type": type(model).__name__,
            "split": "Validation Split Only (Strictly No Test Data)",
            "validation_rows": len(val_df),
            "validation_date_start": str(val_df["date_time"].min()),
            "validation_date_end": str(val_df["date_time"].max()),
            "overall_metrics": {
                "mae": round(overall_mae, 2),
                "rmse": round(overall_rmse, 2),
                "r2": round(overall_r2, 4),
            },
        },
        "mae_by_hour": hour_stats,
        "mae_by_day_of_week": dow_stats,
        "mae_by_month": month_stats,
        "holiday_vs_non_holiday": holiday_comparison,
        "named_holidays_in_val": named_holidays,
        "worst_20_hours": worst_20,
    }

    json_path = os.path.join(results_dir, "residual_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[Artifact Saved] JSON metrics -> {json_path}")

    # -------------------------------------------------------------------------
    # 5. Save Markdown Summary Table
    # -------------------------------------------------------------------------
    md_path = os.path.join(results_dir, "residual_analysis_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Residual Analysis on Validation Split (2017-05-10 to 2018-01-19)\n\n")
        f.write(f"- **Model**: `{type(model).__name__}` (Gradient Boosting)\n")
        f.write(f"- **Validation Records**: {len(val_df):,} hourly timestamps\n")
        f.write(f"- **Overall Validation Metrics**: MAE = **{overall_mae:.2f}** veh/h | RMSE = **{overall_rmse:.2f}** veh/h | $R^2$ = **{overall_r2:.4f}**\n\n")

        f.write("## 1. MAE by Hour of Day\n\n")
        f.write("| Hour | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias (y - y_hat) | MAE (veh/h) | RMSE (veh/h) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for h in hour_stats:
            f.write(f"| {h['hour']:02d}:00 | {h['mean_actual']:.1f} | {h['mean_pred']:.1f} | {h['mean_signed_error']:+.1f} | **{h['mae']:.2f}** | {h['rmse']:.2f} |\n")

        f.write("\n## 2. MAE by Day of Week\n\n")
        f.write("| Day | Count | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias | MAE (veh/h) | RMSE (veh/h) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for d in dow_stats:
            f.write(f"| {d['day_name']} | {d['count']} | {d['mean_actual']:.1f} | {d['mean_pred']:.1f} | {d['mean_signed_error']:+.1f} | **{d['mae']:.2f}** | {d['rmse']:.2f} |\n")

        f.write("\n## 3. MAE by Month\n\n")
        f.write("| Month | Count | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias | MAE (veh/h) | RMSE (veh/h) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in month_stats:
            f.write(f"| {m['month_name']} | {m['count']} | {m['mean_actual']:.1f} | {m['mean_pred']:.1f} | {m['mean_signed_error']:+.1f} | **{m['mae']:.2f}** | {m['rmse']:.2f} |\n")

        f.write("\n## 4. Holiday vs Non-Holiday\n\n")
        f.write("| Category | Count | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias | MAE (veh/h) | RMSE (veh/h) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| Non-Holiday | {holiday_comparison['non_holiday']['count']:,} | {holiday_comparison['non_holiday']['mean_actual']:.1f} | {holiday_comparison['non_holiday']['mean_pred']:.1f} | {holiday_comparison['non_holiday']['mean_signed_error']:+.1f} | **{holiday_comparison['non_holiday']['mae']:.2f}** | {holiday_comparison['non_holiday']['rmse']:.2f} |\n")
        f.write(f"| Holiday | {holiday_comparison['holiday']['count']} | {holiday_comparison['holiday']['mean_actual']:.1f} | {holiday_comparison['holiday']['mean_pred']:.1f} | {holiday_comparison['holiday']['mean_signed_error']:+.1f} | **{holiday_comparison['holiday']['mae']:.2f}** | {holiday_comparison['holiday']['rmse']:.2f} |\n")

        f.write("\n## 5. 20 Worst-Predicted Hours on Validation Split\n\n")
        f.write("| Rank | Date & Hour | Day | Actual | Pred | Abs Error | Direction | Weather | Rain/Snow | Holiday |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :--- |\n")
        for w in worst_20:
            rain_snow = f"{w['rain_1h_mm']}mm rain" if w['rain_1h_mm'] > 0 else (f"{w['snow_1h_mm']}mm snow" if w['snow_1h_mm'] > 0 else "None")
            f.write(f"| {w['rank']} | {w['date_time']} | {w['day_name'][:3]} | {w['actual_volume']} | {w['predicted_volume']:.0f} | **{w['abs_error']:.0f}** | {w['direction'][:14]} | {w['weather_main']} ({w['weather_description']}) | {rain_snow} | {w['holiday']} |\n")

    print(f"[Artifact Saved] Markdown summary -> {md_path}")

    # -------------------------------------------------------------------------
    # 6. Generate Analytical Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel A: MAE by Hour of Day
    hours = [h["hour"] for h in hour_stats]
    maes_h = [h["mae"] for h in hour_stats]
    mean_act_h = [h["mean_actual"] for h in hour_stats]

    ax1 = axes[0, 0]
    bars1 = ax1.bar(hours, maes_h, color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8", label="MAE (veh/h)")
    ax1.set_xlabel("Hour of Day (0 - 23)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MAE (veh/h)", fontsize=11, fontweight="bold", color="#1d4ed8")
    ax1.set_title("Panel A: MAE by Hour of Day (Validation Split)", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Highlight peak rush hours
    for h_idx in [7, 8, 16, 17]:
        bars1[h_idx].set_color("#ef4444")
        bars1[h_idx].set_edgecolor("#b91c1c")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(hours, mean_act_h, color="#f59e0b", marker="o", linewidth=2, label="Mean Actual Volume")
    ax1_twin.set_ylabel("Mean Volume (veh/h)", fontsize=11, fontweight="bold", color="#f59e0b")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")

    # Panel B: MAE by Day of Week
    ax2 = axes[0, 1]
    d_names = [d["day_name"][:3] for d in dow_stats]
    maes_d = [d["mae"] for d in dow_stats]
    bar_colors = ["#2563eb" if i < 5 else "#10b981" for i in range(7)]
    ax2.bar(d_names, maes_d, color=bar_colors, alpha=0.85, edgecolor="#1e293b")
    ax2.set_xlabel("Day of Week", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MAE (veh/h)", fontsize=11, fontweight="bold")
    ax2.set_title("Panel B: MAE by Day of Week (Blue=Weekday, Green=Weekend)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(maes_d):
        ax2.text(i, v + 8, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Panel C: MAE by Month
    ax3 = axes[1, 0]
    m_names = [m["month_name"][:3] for m in month_stats]
    maes_m = [m["mae"] for m in month_stats]
    ax3.bar(m_names, maes_m, color="#8b5cf6", alpha=0.85, edgecolor="#6d28d9")
    ax3.set_xlabel("Month (May 2017 - Jan 2018)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("MAE (veh/h)", fontsize=11, fontweight="bold")
    ax3.set_title("Panel C: MAE by Month in Validation Period", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(maes_m):
        ax3.text(i, v + 6, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Panel D: Holiday vs Non-Holiday MAE
    ax4 = axes[1, 1]
    cats = ["Non-Holiday\n(N=6,078)", "Holiday\n(N=8)"]
    cat_maes = [holiday_comparison["non_holiday"]["mae"], holiday_comparison["holiday"]["mae"]]
    cat_colors = ["#3b82f6", "#dc2626"]
    bars4 = ax4.bar(cats, cat_maes, color=cat_colors, alpha=0.85, width=0.45, edgecolor="#0f172a")
    ax4.set_ylabel("MAE (veh/h)", fontsize=11, fontweight="bold")
    ax4.set_title("Panel D: Holiday vs Non-Holiday Error", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    for bar, val in zip(bars4, cat_maes):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 15, f"{val:.1f} veh/h", ha="center", fontsize=10, fontweight="bold")

    plot1_path = os.path.join(results_dir, "residuals_breakdown_panels.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifact Saved] Breakdown plot -> {plot1_path}")

    # Plot 2: Worst 20 Hours Breakdown Plot
    fig2, ax_w = plt.subplots(figsize=(14, 8))
    w_labels = [f"#{w['rank']} {w['date_time'][5:16]} ({w['day_name'][:3]})" for w in reversed(worst_20)]
    w_errors = [w["abs_error"] for w in reversed(worst_20)]
    w_dirs = [w["direction"] for w in reversed(worst_20)]
    colors_w = ["#ef4444" if "Under" in d else "#f59e0b" for d in w_dirs]

    bars_w = ax_w.barh(w_labels, w_errors, color=colors_w, alpha=0.85, edgecolor="#334155")
    ax_w.set_xlabel("Absolute Error |Actual - Predicted| (veh/h)", fontsize=11, fontweight="bold")
    ax_w.set_title("Top 20 Worst-Predicted Hours on Validation Split (Red=Underpredicted, Orange=Overpredicted)", fontsize=12, fontweight="bold")
    ax_w.grid(True, linestyle="--", alpha=0.5)

    for bar, err, w in zip(bars_w, w_errors, reversed(worst_20)):
        desc = f"{err:.0f} (Act: {w['actual_volume']}, Pred: {w['predicted_volume']:.0f} | {w['weather_main']})"
        ax_w.text(err + 25, bar.get_y() + bar.get_height() / 2, desc, va="center", fontsize=8.5)

    ax_w.set_xlim(0, max(w_errors) * 1.35)
    plot2_path = os.path.join(results_dir, "residuals_worst_20_hours.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifact Saved] Worst 20 plot -> {plot2_path}")

    print("\nResidual analysis complete successfully.")


if __name__ == "__main__":
    run_residual_analysis()
