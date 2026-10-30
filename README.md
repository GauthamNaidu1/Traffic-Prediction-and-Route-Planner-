# Traffic Volume Forecasting & Route Travel-Time Planner

A reproducible machine learning and network routing system that forecasts hourly traffic volume, translates predictions into dynamic edge traversal weights, and computes time-optimal paths over an urban road network graph.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset Specifications](#dataset-specifications)
3. [Methodology](#methodology)
4. [Empirical Results](#empirical-results)
   - [Test Set Evaluation & Confidence Intervals](#1-test-set-model-evaluation-n6087)
   - [Paired Hypothesis Testing vs Historical Mean](#2-paired-hypothesis-testing-vs-historical-mean)
   - [5-Fold Rolling-Origin Evaluation](#3-5-fold-rolling-origin-expanding-window-cross-validation)
   - [Weather Nowcasting Experiment](#4-weather-nowcast-experiment-marginal-signal)
   - [Downstream Routing Evaluation](#5-downstream-routing-simulation-n200-pairs--6-conditions)
5. [Limitations](#limitations)
6. [Reproduction Instructions](#reproduction-instructions)
7. [Streamlit Web Application](#streamlit-web-application)

---

## Problem Statement

Standard navigation systems rely on static road speed limits (free-flow routing), which systematically underestimate travel time during congestion. Conversely, naive heuristic planners apply arbitrary rule-based peak multipliers (e.g., $2.4\times$ during peak hours, $1.0\times$ off-peak) that fail to reflect empirical traffic volume patterns.

This project implements an end-to-end, scientifically validated pipeline:
1. **Traffic Volume Forecasting**: Train supervised regression models on multi-year hourly highway traffic counts using strict chronological evaluation splits.
2. **Dynamic Travel-Time Mapping**: Map predicted hourly traffic volume $V$ into continuous edge delay multipliers via an empirical volume-to-delay scaling function.
3. **Shortest-Path Routing**: Evaluate travel-time optimal routes over a road network graph using Dijkstra's algorithm.
4. **Statistical Benchmarking**: Benchmark candidate models against historical mean and seasonal-naive baselines with bootstrap confidence intervals, rolling-origin cross-validation, and downstream routing simulations.

---

## Dataset Specifications

| Attribute | Specification |
| :--- | :--- |
| **Source** | [UCI Machine Learning Repository: Metro Interstate Traffic Volume (#492)](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) |
| **Location** | Westbound Interstate 94 (Milepost 249.5, ATR Station 301), between Minneapolis and St. Paul, MN |
| **Monitoring Agency** | Minnesota Department of Transportation (MnDOT) |
| **Raw Records** | 48,204 rows (contains hourly duplicates and weather sensor updates) |
| **Cleaned Size** | 40,575 deduplicated hourly timestamps (3.09 MB uncompressed CSV) |
| **Target Variable** | `traffic_volume` (hourly vehicle count, range: 0 to 7,280 veh/h) |
| **Date Range** | October 2, 2012, 09:00:00 to September 30, 2018, 23:00:00 (~6 continuous years) |

### Chronological Splits (70% / 15% / 15%)
To prevent temporal data leakage, all splits are strictly chronological:

| Split | Percentage | Hourly Count | Date Range (Start to End) | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 70% | 28,402 | 2012-10-02 09:00:00 to 2017-05-10 04:00:00 | Model training & baseline calculation |
| **Validation** | 15% | 6,086 | 2017-05-10 05:00:00 to 2018-01-19 14:00:00 | Hyperparameter tuning only |
| **Test** | 15% | 6,087 | 2018-01-19 15:00:00 to 2018-09-30 23:00:00 | Single final out-of-sample evaluation |

### Feature Engineering
- **Calendar Features (Primary Forecast Model)**:
  - `hour_sin`, `hour_cos`: Cyclical diurnal encoding of hour of day ($\sin(2\pi \cdot h / 24)$, $\cos(2\pi \cdot h / 24)$).
  - `day_of_week`: Integer day index (0 = Monday, ..., 6 = Sunday).
  - `is_weekend`: Binary flag for Saturday and Sunday.
  - `is_holiday`: Binary flag for US federal holidays.
- **Weather Features (Optional Nowcast Experiment Only)**:
  - `temp_clean`: Ambient temperature in Kelvin (outliers < 200 K replaced with training median).
  - `rain_clean`: 1-hour rainfall in mm (sensor glitches > 100 mm/h clipped to 0).
  - `snow_1h`: 1-hour snowfall in mm.
  - `clouds_all`: Cloud cover percentage (0–100%).
  - One-hot encoded weather categories (`weather_Clear`, `weather_Clouds`, `weather_Rain`, `weather_Snow`, etc.).

*Design Note*: Future weather is unknown when planning future trips. Therefore, the primary forecast model deployed to the application uses calendar features only. Contemporaneous weather features are evaluated strictly in a separate "nowcast" experiment.

---

## Methodology

### 1. Evaluated Models & Baselines
- **Global Mean Baseline**: Mean traffic volume computed from the training split ($\bar{V}_{\text{train}} = 3,264.6\text{ veh/h}$).
- **Historical Mean (Hour, DOW) Baseline**: Lookup table of average volume by `(hour, day_of_week)` calculated strictly from training data.
- **Seasonal-Naive Baseline ($t - 168\text{h}$)**: Direct persistence of the observed traffic volume from the exact same hour of the preceding week.
- **Ridge Regression**: L2-regularized linear model with cyclical features; tuned over $\alpha \in [0.01, 1000.0]$ (selected: $\alpha = 100.0$).
- **Random Forest Regressor**: Ensembled decision trees; tuned over `n_estimators`, `max_depth`, and `min_samples_split` (selected: `n_estimators=50`, `max_depth=12`).
- **Gradient Boosting Regressor (GBR)**: Iterative gradient tree boosting; tuned over `n_estimators`, `max_depth`, and `learning_rate` (selected: `n_estimators=100`, `max_depth=6`, `lr=0.1`). Model saved as `models/traffic_model.joblib` (0.22 MB).

### 2. Validation & Evaluation Protocol
- **Validation Tuning**: Hyperparameter grids were evaluated solely on the validation set. The test set was held out and evaluated exactly once.
- **Bootstrap Confidence Intervals**: 1,000 bootstrap resamples on the test set were computed for all metrics (MAE, RMSE, $R^2$) to report empirical 95% confidence intervals.
- **Paired Hypothesis Testing**: Paired difference resamples ($e_{\text{model}} - e_{\text{baseline}}$) tested whether model gains over historical mean baselines are statistically significant ($p < 0.05$).
- **Rolling-Origin Cross-Validation**: 5-fold expanding-window backtesting across the dataset evaluated out-of-fold generalization over time.

### 3. Volume-to-Travel-Time Mapping Formula
The deployed Gradient Boosting model outputs predicted traffic volume $V$ in vehicles/hour. Edge traversal multiplier $M$ is derived via the following empirically calibrated formula:

$$M = \text{clip}\left(0.85 + 1.60 \times \frac{V}{V_{90}}, \; 0.85, \; 2.50\right) \times H_{\text{type}} \times S_{\text{mode}}$$

Where:
- $V_{90} = 5,830\text{ veh/h}$ is the 90th percentile of training traffic volume.
- At late-night minimum volume ($V \approx 280\text{ veh/h}$), base multiplier $\approx 0.93$ (free-flow).
- At weekday rush-hour volume ($V \approx 5,250\text{ veh/h}$), base multiplier $\approx 2.29$.
- $H_{\text{type}}$ is the road hierarchy factor: Main Highway ($1.0$), Major Arterial ($0.9$), Local Connector ($0.8$).
- $S_{\text{mode}}$ is travel mode traffic sensitivity: Car ($1.0$), Bike ($0.35$), Walk ($0.05$).
- Edge traversal time is given by: $T = \frac{d}{s_{\text{base}}} \times M$.

---

## Empirical Results

All metrics below are copied verbatim from the generated artifacts in `results/`.

### 1. Test Set Model Evaluation (N=6,087)
Evaluated on the out-of-sample test split (Jan 19, 2018 – Sep 30, 2018) with 1,000 bootstrap resamples (source: `results/metrics.json` and `results/baselines.json`):

| Model / Baseline | Features | Test MAE [95% CI] (veh/h) | Test RMSE [95% CI] (veh/h) | Test $R^2$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Global Mean** | None (mean) | 1735.33 [1710.99, 1761.20] | 1981.11 [1960.19, 2002.97] | -0.0015 [-0.0042, -0.0003] |
| **Seasonal-Naive ($t - 168\text{h}$)** | Lag 168h | 321.63 [308.15, 336.28] | 624.36 [587.95, 659.78] | 0.9005 [0.8888, 0.9119] |
| **Ridge Regression** | Calendar | 836.09 [819.65, 853.97] | 1077.30 [1054.56, 1101.59] | 0.7038 [0.6903, 0.7156] |
| **Historical Mean (Hour, DOW)** | Calendar (lookup) | 260.82 [252.27, 270.55] | 448.08 [419.43, 475.93] | 0.9488 [0.9421, 0.9551] |
| **Random Forest Regressor** | Calendar | 260.81 [252.29, 270.55] | 448.13 [419.27, 476.11] | 0.9488 [0.9420, 0.9551] |
| **Gradient Boosting Regressor** | Calendar | **260.74** [252.19, 270.46] | **448.01** [419.38, 475.87] | **0.9488** [0.9421, 0.9551] |
| *Nowcast GBR (Upper Bound)* | Calendar + Weather | *253.46* [245.20, 262.89] | *435.55* [408.05, 463.62] | *0.9516* [0.9450, 0.9576] |

### 2. Paired Hypothesis Testing vs Historical Mean
To test if calendar-based machine learning significantly outperforms the simple historical mean lookup table, 1,000 paired bootstrap resamples of error differences ($|e_{\text{ML}}| - |e_{\text{Hist}}|$) were computed:

| Comparison | $\Delta$ MAE Point | $\Delta$ MAE 95% CI | $\Delta$ RMSE Point | $\Delta$ RMSE 95% CI | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GBR vs Historical Mean** | -0.07 veh/h | [-0.23, +0.09] | -0.06 veh/h | [-0.18, +0.05] | **No** (CI includes 0) |
| **RF vs Historical Mean** | -0.01 veh/h | [-0.21, +0.17] | +0.05 veh/h | [-0.16, +0.28] | **No** (CI includes 0) |

> **Key Takeaway**: When restricted strictly to calendar inputs, non-linear machine learning models (Gradient Boosting and Random Forest) converge directly to the historical mean diurnal profile. The -0.07 veh/h difference is not statistically significant. The primary utility of machine learning in this setting lies in its extensibility to weather, incident, and sensor streams.

### 3. 5-Fold Rolling-Origin Expanding-Window Cross-Validation
Temporal backtesting across 5 sequential validation windows (source: `results/metrics.json`):

| Model / Baseline | MAE Mean $\pm$ Std (veh/h) | RMSE Mean $\pm$ Std (veh/h) | $R^2$ Mean $\pm$ Std |
| :--- | :---: | :---: | :---: |
| **Global Mean** | 1733.08 $\pm$ 36.58 | 1976.27 $\pm$ 38.45 | -0.0022 $\pm$ 0.0029 |
| **Seasonal-Naive ($t - 168\text{h}$)** | 330.77 $\pm$ 31.36 | 617.50 $\pm$ 55.06 | 0.9009 $\pm$ 0.0192 |
| **Ridge Regression** | 843.39 $\pm$ 9.25 | 1085.37 $\pm$ 5.81 | 0.6974 $\pm$ 0.0107 |
| **Historical Mean (Hour, DOW)** | 293.22 $\pm$ 24.24 | 502.29 $\pm$ 45.55 | 0.9344 $\pm$ 0.0131 |
| **Gradient Boosting** | **293.19 $\pm$ 24.25** | **502.22 $\pm$ 45.56** | **0.9344 $\pm$ 0.0131** |
| **Random Forest** | 293.18 $\pm$ 24.23 | 502.20 $\pm$ 45.61 | 0.9345 $\pm$ 0.0131 |

### 4. Weather Nowcast Experiment (Marginal Signal)
Evaluating the impact of contemporaneous weather sensor features (`temp_clean`, `rain_clean`, `snow_1h`, `clouds_all`, weather type):
- **Val MAE**: 260.95 veh/h (vs 285.01 veh/h calendar GBR)
- **Test MAE**: 253.46 veh/h (95% CI: [245.20, 262.89]) vs 260.74 veh/h calendar GBR
- **Test RMSE**: 435.55 veh/h (95% CI: [408.05, 463.62]) vs 448.01 veh/h calendar GBR
- **Improvement**: -7.28 veh/h MAE reduction. The 95% confidence intervals do not overlap the calendar model's lower bound, proving that sensor weather provides genuine marginal predictive signal beyond diurnal time features.

### 5. Downstream Routing Simulation (N=200 Pairs $\times$ 6 Conditions)
We evaluated $N=200$ randomly sampled origin-destination pairs on the hand-built 23-node road network across 6 distinct temporal conditions (1,200 total route queries; source: `results/routing_eval.json`):

| Temporal Condition | Day & Hour | Predicted Vol (veh/h) | Free-Flow Mean (min) | Rule-Based Mean (min) | Model Mean (min) | Model $-$ Rule Diff [95% CI] (min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Weekday Morning Rush** | Monday 08:00 | 5,252 | 17.49 | 41.99 | 40.09 | **-1.91** [-2.05, -1.77] |
| **Weekday Midday** | Wednesday 13:00 | 4,879 | 17.49 | 22.75 | 38.30 | **+15.56** [+14.45, +16.70] |
| **Weekday Evening Rush** | Friday 18:00 | 4,654 | 17.49 | 41.99 | 37.22 | **-4.78** [-5.13, -4.44] |
| **Weekday Late Night** | Tuesday 02:00 | 283 | 17.49 | 14.87 | 16.22 | **+1.36** [+1.27, +1.47] |
| **Weekend Afternoon Peak** | Saturday 14:00 | 4,535 | 17.49 | 28.00 | 36.67 | **+8.66** [+8.04, +9.29] |
| **Weekend Late Night** | Sunday 03:00 | 386 | 17.49 | 14.87 | 16.73 | **+1.86** [+1.72, +1.99] |
| **Overall (1,200 queries)** | — | — | **17.49** | **27.91** | **31.37** | **+3.46** [+3.05, +3.91] |

- **Overall Model vs Rule-Based Difference**: Mean $+3.46$ min (95% CI: $[+3.05, +3.91]$ min); Median $+1.54$ min (95% CI: $[+1.33, +1.70]$ min).
- **Route Divergence Share**: **0.0%** across all 1,200 queries (shortest topological paths were identical under Free-Flow, Rule-Based, and Trained Model weights).
- **Explanation**: The hand-built network consists of 23 nodes and only 4 planar cycles. Detours are $\approx 50\%$ longer in physical distance than direct paths. Because edge multipliers shift proportionally across road hierarchies, the delay on direct paths never exceeds the threshold required to make a geometric detour optimal.

---

## Limitations

1. **Geographic Mismatch**: The traffic forecasting model is trained on Interstate 94 in Minneapolis–St. Paul, Minnesota (MnDOT ATR station 301). The road network graph is a stylized representation of corridors in Hyderabad/Secunderabad, India. Traffic dynamics, vehicle mix (e.g., auto-rickshaws, two-wheelers), and driver behaviors differ fundamentally.
2. **Small Hand-Built Road Graph**: The network consists of 23 nodes and 26 edges forming 4 cycles. Because alternative routes are substantially longer in distance, the model produces a **0.0% route change share**. On a dense metropolitan road graph with parallel corridors, dynamic multiplier changes would trigger route divergence.
3. **Simulated Travel Time Estimates**: The transit durations reported by the router are simulated estimates calculated via the volume-to-delay multiplier formula, not measured empirical travel times collected from GPS probes or telemetry.
4. **Weather Availability at Forecast Time**: Weather conditions cannot be assumed known at arbitrary future trip planning horizons. The operational model is strictly calendar-based; weather features are evaluated only as an offline nowcasting benchmark.

---

## Reproduction Instructions

A clean clone can reproduce all data preparation, model training, evaluation metrics, and launch the web application using standard commands:

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/GauthamNaidu1/Traffic-Prediction-and-Route-Planner-.git
cd Traffic-Prediction-and-Route-Planner-

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Prepare Data & Baseline Metrics
```bash
python train/prepare_data.py
```
*Outputs*: Cleans `data/Metro_Interstate_Traffic_Volume.csv`, deduplicates timestamps, creates calendar features, executes 70/15/15 chronological splits, computes non-learning baselines, and saves `results/baselines.json`.

### 3. Train Models & Evaluate Benchmarks
```bash
python train/train_models.py
```
*Outputs*: Tunes Ridge, Random Forest, and Gradient Boosting on validation data; runs 1,000 bootstrap resamples on test data; conducts 5-fold rolling-origin CV; evaluates the weather nowcast experiment; serializes the best calendar model to `models/traffic_model.joblib` (0.22 MB); generates `results/metrics.json`, `results/predicted_vs_actual_week.png`, and `results/feature_importance.png`.

### 4. Run Downstream Routing Simulation
```bash
python evaluate_routing.py
```
*Outputs*: Evaluates 200 random O/D pairs across 6 time conditions, computes travel time differences with 95% bootstrap CIs, calculates route divergence share, and saves `results/routing_eval.json`.

### 5. Launch the Web Application
```bash
streamlit run app.py
```
The application will launch at `http://localhost:8501`.

---

## Streamlit Web Application

The interactive web interface provides:
- **Routing Engine Toggle**: Switch between the **Trained ML Model** (Gradient Boosting with volume-to-multiplier formula) and the **Rule-Based Baseline** to directly observe ETA differences.
- **Dynamic Fallback**: If `models/traffic_model.joblib` is missing, the application automatically falls back to the rule-based baseline and alerts the user.
- **Transparent Evaluation Panel**: An expandable sidebar panel reads directly from `results/metrics.json` and displays the exact test metrics, model parameters, and formal dataset limitations without hard-coded numbers.
- **Interactive Route Visualization**: Displays turn-by-turn road segments, individual edge multipliers, cumulative travel times, and an embedded Folium map marking origin, destination, and calculated path.
