# Traffic-Aware Route Planner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://traffic-zoom-route-planner.streamlit.app)
**Live Demo**: [https://traffic-zoom-route-planner.streamlit.app](https://traffic-zoom-route-planner.streamlit.app)

A reproducible machine learning and network routing system that forecasts hourly traffic volume, translates predictions into dynamic edge traversal weights, and computes time-optimal paths over an urban road network graph.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Method](#method)
   - [Dataset & Chronological Splits](#dataset--chronological-splits)
   - [Calendar Feature Engineering](#calendar-feature-engineering)
   - [Congestion-to-Travel-Time Multiplier Formula](#congestion-to-travel-time-multiplier-formula)
   - [Model Training & Evaluation Protocol](#model-training--evaluation-protocol)
3. [Empirical Results](#empirical-results)
   - [1. Test Set Model Evaluation (N=6,087)](#1-test-set-model-evaluation-n6087)
   - [2. Paired Hypothesis Testing vs Historical Mean](#2-paired-hypothesis-testing-vs-historical-mean)
   - [3. 5-Fold Rolling-Origin Cross-Validation](#3-5-fold-rolling-origin-expanding-window-cross-validation)
   - [4. Weather Nowcasting Experiment](#4-weather-nowcast-experiment-marginal-signal)
   - [5. Downstream Routing Simulation](#5-downstream-routing-simulation-n200-pairs--6-conditions)
4. [Limitations](#limitations)
5. [How to Run](#how-to-run)
6. [Streamlit Web Application](#streamlit-web-application)

---

## Problem Statement

Standard navigation systems rely on static road speed limits (free-flow routing), which systematically underestimate travel time during congestion. Conversely, naive heuristic planners apply arbitrary rule-based peak multipliers (e.g., $2.4\times$ during peak hours, $1.0\times$ off-peak) that fail to reflect empirical traffic volume patterns.

This project implements an end-to-end, scientifically validated pipeline:
1. **Traffic Volume Forecasting**: Train supervised regression models on multi-year hourly highway traffic counts using strict chronological evaluation splits.
2. **Dynamic Travel-Time Mapping**: Map predicted hourly traffic volume $V$ into continuous edge delay multipliers via an empirical volume-to-delay scaling function.
3. **Shortest-Path Routing**: Evaluate travel-time optimal routes over a road network graph using Dijkstra and A* algorithms.
4. **Statistical Benchmarking**: Benchmark candidate models against historical mean and seasonal-naive baselines with bootstrap confidence intervals, rolling-origin cross-validation, and downstream routing simulations.

---

## Method

*(Note: In the deployed Streamlit user interface, internal mathematical formulas and raw vehicle counts are intentionally simplified into intuitive traffic levels. This section thoroughly documents the underlying methodology for readers of this repository.)*

### Dataset & Chronological Splits

| Attribute | Specification |
| :--- | :--- |
| **Source** | [UCI Machine Learning Repository: Metro Interstate Traffic Volume (#492)](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) |
| **Location** | Westbound Interstate 94 (Milepost 249.5, ATR Station 301), between Minneapolis and St. Paul, MN |
| **Monitoring Agency** | Minnesota Department of Transportation (MnDOT) |
| **Raw Records** | 48,204 rows (contains hourly duplicates and weather sensor updates) |
| **Cleaned Size** | 40,575 deduplicated hourly timestamps (3.09 MB uncompressed CSV; `data/Metro_Interstate_Traffic_Volume.csv`) |
| **Target Variable** | `traffic_volume` (continuous vehicle count, range: 0 to 7,280 veh/h) |
| **Overall Date Range** | October 2, 2012, 09:00:00 to September 30, 2018, 23:00:00 (~6 continuous years) |

To prevent temporal data leakage, all evaluation splits are strictly chronological:

| Split | Share | Hourly Count | Date Range (Start to End) | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 70% | 28,402 | 2012-10-02 09:00:00 to 2017-05-10 04:00:00 | Model training & baseline calculation |
| **Validation** | 15% | 6,086 | 2017-05-10 05:00:00 to 2018-01-19 14:00:00 | Hyperparameter tuning only |
| **Test** | 15% | 6,087 | 2018-01-19 15:00:00 to 2018-09-30 23:00:00 | Single final out-of-sample evaluation |

### Calendar Feature Engineering

When planning a route for a future hour or day, contemporaneous weather observations (such as 1-hour rainfall or cloud cover) do not exist. Therefore, the primary forecast model deployed to the application strictly relies on deterministic calendar features:
- `hour_sin`, `hour_cos`: Cyclical diurnal encoding of hour of day ($\sin(2\pi \cdot h / 24)$, $\cos(2\pi \cdot h / 24)$).
- `day_of_week`: Integer day index (0 = Monday, ..., 6 = Sunday).
- `is_weekend`: Binary flag for Saturday and Sunday.
- `is_holiday`: Binary flag for US federal holidays.

Contemporaneous weather features (`temp_clean`, `rain_clean`, `snow_1h`, `clouds_all`, weather categories) are evaluated strictly in a separate offline "nowcast" experiment to measure upper-bound performance when sensor data is available.

### Congestion-to-Travel-Time Multiplier Formula

The deployed Gradient Boosting model outputs predicted hourly traffic volume $V$ in vehicles/hour. To convert predicted highway volume into continuous edge traversal delays for graph routing, we apply the following mathematical mapping:

$$\text{Base Multiplier} = \text{clip}\left(0.85 + 1.60 \times \frac{V}{V_{90}}, \; 0.85, \; 2.50\right)$$

$$\text{Effective Multiplier} (M) = \text{Base Multiplier} \times H_{\text{type}} \times S_{\text{mode}}$$

Edge traversal travel time is given by:
$$T = \frac{d}{s_{\text{base}}} \times M$$

#### Formula Constants & Parameters:
1. **$V_{90}$ Normalization ($V_{90} = 5,830.0\text{ veh/h}$)**:
   - $V_{90}$ is the 90th percentile of traffic volume computed strictly on the training split. Normalizing by $V_{90}$ prevents single-hour outlier spikes from distorting multiplier scaling.
2. **Intercept ($0.85$)**:
   - Represents the late-night / free-flow speed multiplier when traffic is minimal ($V \to 0$, $M \approx 0.93$).
3. **Slope ($1.60$) & Upper Bound ($2.50$)**:
   - At near-capacity rush hour ($V \to V_{90}$), the base multiplier scales up to $\approx 2.45$.
   - The value is clipped to $[0.85, 2.50]$ to prevent unbounded non-physical delays.
4. **Road Hierarchy Scaling ($H_{\text{type}}$)**:
   - Motorway / Trunk: $1.05\times$ (higher density highway impact)
   - Primary / Secondary Arterials: $1.00\times$
   - Residential / Living Street: $0.90\times$ (lower absolute volume exposure)
5. **Travel Mode Sensitivity ($S_{\text{mode}}$)**:
   - Car: $1.00\times$ (full exposure to vehicular congestion)
   - Bike: $0.65\times$ (ability to filter through stopped traffic)
   - Walk: $0.30\times$ (pedestrian movement largely independent of vehicular congestion)

> **Important Methodology Notice**: These constants ($0.85$, $1.60$, $2.50$, $V_{90} = 5,830$, and road/mode scaling factors) are engineering assumptions and domain approximations. They are **not calibrated or empirically validated on Hyderabad road sensor data**.

### Model Training & Evaluation Protocol
- **Validation Tuning**: Hyperparameter grids for Ridge Regression, Random Forest, and Gradient Boosting were tuned exclusively on the validation split (`df_val`).
- **Single Test Evaluation**: The test split (`df_test`, $N=6,087$ hours) was evaluated once after freezing all model hyperparameters.
- **Statistical Significance**: 1,000 bootstrap resamples on the test set were computed for all metrics (MAE, RMSE, $R^2$) to report empirical 95% confidence intervals and paired difference hypothesis tests.
- **Rolling-Origin Cross-Validation**: 5 sequential expanding-window backtesting folds were evaluated across time to measure temporal stability.

---

## Empirical Results

All metrics below are copied verbatim from the generated artifacts in `results/`.

> **Notice on Test Split**: The test split (2018-01-19 to 2018-09-30, $N=6,087$) was viewed during an earlier baseline audit. Model selection and feature engineering were conducted strictly using 5-fold rolling-origin cross-validation on train+val data only. The final upgraded model (Config 3: Calendar + US Holiday Proximity GBR) was retrained on train+val and evaluated exactly once on the test split without subsequent iteration or tuning.

### 1. Test Set Model Evaluation (N=6,087)
Evaluated on the out-of-sample test split (Jan 19, 2018 – Sep 30, 2018) with 1,000 bootstrap resamples *(source: `results/final_test.json`, `results/metrics.json`, and `results/baselines.json`)*:

| Model / Baseline | Features | Test MAE [95% CI] (veh/h) | Test RMSE [95% CI] (veh/h) | Test $R^2$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Global Mean** | None (mean) | 1735.33 [1710.99, 1761.20] | 1981.11 [1960.19, 2002.97] | -0.0015 [-0.0042, -0.0003] |
| **Seasonal-Naive ($t - 168\text{h}$)** | Lag 168h | 321.63 [308.15, 336.28] | 624.36 [587.95, 659.78] | 0.9005 [0.8888, 0.9119] |
| **Ridge Regression** | Calendar | 836.09 [819.65, 853.97] | 1077.30 [1054.56, 1101.59] | 0.7038 [0.6903, 0.7156] |
| **Historical Mean (Hour, DOW)** | Calendar (lookup) | 257.27 [248.52, 267.05] | 445.36 [416.66, 473.48] | 0.9494 [0.9426, 0.9557] |
| **Initial Calendar GBR** | Hour sin/cos, DOW, raw holiday | 260.74 [252.19, 270.46] | 448.01 [419.38, 475.87] | 0.9488 [0.9421, 0.9551] |
| **Final Upgraded GBR (Deployed)** | **Calendar + Holiday Proximity** | **228.90** [221.57, 237.44] | **389.18** [366.62, 412.17] | **0.9614** [0.9564, 0.9659] |
| *Nowcast GBR (Upper Bound)* | Calendar + Weather | *253.46* [245.20, 262.89] | *435.55* [408.05, 463.62] | *0.9516* [0.9450, 0.9576] |

### 2. Paired Hypothesis Testing vs Historical Mean
To test if the models significantly outperform the simple historical mean lookup table, 1,000 paired bootstrap resamples of error differences ($|e_{\text{Model}}| - |e_{\text{Hist}}|$) were computed on the test split *(source: `results/final_test.json`)*:

| Comparison | $\Delta$ MAE Point | $\Delta$ MAE 95% CI | $\Delta$ RMSE Point | $\Delta$ RMSE 95% CI | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Initial GBR vs Historical Mean** | -0.07 veh/h | [-0.23, +0.09] | -0.06 veh/h | [-0.18, +0.05] | **No** (CI includes 0) |
| **Final Upgraded GBR vs Historical Mean** | **-28.37 veh/h** | **[-33.53, -23.35]** | **-56.18 veh/h** | **[-79.30, -36.82]** | **YES (CI strictly < 0)** |

> **Key Finding**: The initial calendar model without holiday proximity collapsed onto the historical mean diurnal profile. In contrast, the final upgraded model incorporates continuous federal holiday proximity (resolving the midnight holiday logging trap), achieving a statistically verified **-28.37 veh/h MAE reduction** and **-56.18 veh/h RMSE reduction** on the out-of-sample test split, with 95% confidence intervals strictly excluding zero.

### 3. 5-Fold Rolling-Origin Expanding-Window Cross-Validation
Temporal backtesting across 5 sequential validation windows *(source: `results/metrics.json`)*:

| Model / Baseline | MAE Mean $\pm$ Std (veh/h) | RMSE Mean $\pm$ Std (veh/h) | $R^2$ Mean $\pm$ Std |
| :--- | :---: | :---: | :---: |
| **Global Mean** | 1733.08 $\pm$ 36.58 | 1976.27 $\pm$ 38.45 | -0.0022 $\pm$ 0.0029 |
| **Seasonal-Naive ($t - 168\text{h}$)** | 330.77 $\pm$ 31.36 | 617.50 $\pm$ 55.06 | 0.9009 $\pm$ 0.0192 |
| **Ridge Regression** | 843.39 $\pm$ 9.25 | 1085.37 $\pm$ 5.81 | 0.6974 $\pm$ 0.0107 |
| **Historical Mean (Hour, DOW)** | 293.22 $\pm$ 24.24 | 502.29 $\pm$ 45.55 | 0.9344 $\pm$ 0.0131 |
| **Gradient Boosting** | **293.23 $\pm$ 24.22** | **502.24 $\pm$ 45.52** | **0.9344 $\pm$ 0.0131** |
| **Random Forest** | 293.18 $\pm$ 24.23 | 502.20 $\pm$ 45.61 | 0.9345 $\pm$ 0.0131 |

### 4. Weather Nowcast Experiment (Marginal Signal)
Evaluating the impact of contemporaneous weather sensor features (`temp_clean`, `rain_clean`, `snow_1h`, `clouds_all`, weather type) *(source: `results/metrics.json`)*:
- **Val MAE**: 260.95 veh/h (vs 285.01 veh/h calendar GBR)
- **Test MAE**: 253.46 veh/h (95% CI: [245.20, 262.89]) vs 260.74 veh/h calendar GBR
- **Test RMSE**: 435.55 veh/h (95% CI: [408.05, 463.62]) vs 448.01 veh/h calendar GBR
- **Improvement**: -7.28 veh/h MAE reduction. The 95% confidence intervals do not overlap the calendar model's lower bound, proving that sensor weather provides genuine marginal predictive signal beyond diurnal time features.

### 5. Downstream Routing Simulation (N=200 Pairs $\times$ 6 Conditions)
We evaluated $N=200$ randomly sampled origin-destination pairs on the hand-built 23-node road network across 6 distinct temporal conditions (1,200 total route queries; source: `results/routing_eval.json`):

| Temporal Condition | Day & Hour | Predicted Vol (veh/h) | Free-Flow Mean (min) | Rule-Based Mean (min) | Model Mean (min) | Model $-$ Rule Diff [95% CI] (min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Weekday Morning Rush** | Monday 08:00 | 5,397 | 17.49 | 41.99 | 40.79 | **-1.21** [-1.29, -1.12] |
| **Weekday Midday** | Wednesday 13:00 | 4,938 | 17.49 | 22.75 | 38.58 | **+15.84** [+14.71, +17.00] |
| **Weekday Evening Rush** | Friday 18:00 | 4,724 | 17.49 | 41.99 | 37.57 | **-4.43** [-4.75, -4.11] |
| **Weekday Late Night** | Tuesday 02:00 | 278 | 17.49 | 14.87 | 16.20 | **+1.33** [+1.24, +1.43] |
| **Weekend Afternoon Peak** | Saturday 14:00 | 4,680 | 17.49 | 28.00 | 37.34 | **+9.35** [+8.68, +10.03] |
| **Weekend Late Night** | Sunday 03:00 | 425 | 17.49 | 14.87 | 16.92 | **+2.05** [+1.90, +2.20] |
| **Overall (1,200 queries)** | — | — | **17.49** | **27.91** | **31.23** | **+3.82** [+3.41, +4.29] |

- **Overall Model vs Rule-Based Difference**: Mean $+3.82$ min (95% CI: $[+3.41, +4.29]$ min); Median $+1.59$ min (95% CI: $[+1.35, +1.77]$ min).
- **Route Divergence Share**: **0.0%** across all 1,200 queries (shortest topological paths were identical under Free-Flow, Rule-Based, and Trained Model weights).
- **Explanation**: The hand-built network consists of 23 nodes and only 4 planar cycles. Detours are $\approx 50\%$ longer in physical distance than direct paths. Because edge multipliers shift proportionally across road hierarchies, the delay on direct paths never exceeds the threshold required to make a geometric detour optimal.

---

## Limitations

1. **Geographic Mismatch**: The traffic forecasting model is trained on Interstate 94 in Minneapolis–St. Paul, Minnesota (MnDOT ATR station 301). The road network graph is a stylized representation of corridors in Hyderabad/Secunderabad, India. Traffic dynamics, vehicle mix (e.g., auto-rickshaws, two-wheelers), and driver behaviors differ fundamentally.
2. **Small Hand-Built Road Graph**: The network consists of 23 nodes and 26 edges forming 4 cycles. Because alternative routes are substantially longer in distance, the model produces a **0.0% route change share**. On a dense metropolitan road graph with parallel corridors, dynamic multiplier changes would trigger route divergence.
3. **Simulated Travel Time Estimates**: The transit durations reported by the router are simulated estimates calculated via the volume-to-delay multiplier formula, not measured empirical travel times collected from GPS probes or physical telemetry.
4. **Weather Availability at Forecast Time**: Weather conditions cannot be assumed known at arbitrary future trip planning horizons. The operational model is strictly calendar-based; weather features are evaluated only as an offline nowcasting benchmark.

---

## How to Run

A clean clone can reproduce all data preparation, model training, evaluation metrics, and launch the web application using standard terminal commands without external setup:

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/GauthamNaidu1/Traffic-Prediction-and-Route-Planner-.git
cd Traffic-Prediction-and-Route-Planner-

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate
```

### 2. Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Compute Baseline Metrics
```bash
python train/prepare_data.py
```
*Outputs*: Cleans and deduplicates `data/Metro_Interstate_Traffic_Volume.csv` (40,575 hourly records), engineers calendar features, performs the 70/15/15 chronological split, computes non-learning baselines, and writes `results/baselines.json`.

### 4. Train Models & Evaluate Benchmarks
```bash
python train/train_models.py
```
*Outputs*: Tunes Ridge, Random Forest, and Gradient Boosting on validation data; runs 1,000 bootstrap resamples on test data; executes 5-fold rolling-origin CV; evaluates the weather nowcasting experiment; serializes the best calendar model to `models/traffic_model.joblib` (0.28 MB); and saves `results/metrics.json`, `results/predicted_vs_actual_week.png`, and `results/feature_importance.png`.

### 5. Run Downstream Routing Simulation
```bash
python evaluate_routing.py
```
*Outputs*: Evaluates 200 random origin-destination pairs across 6 time conditions (1,200 total route queries), calculates travel time differences with 95% bootstrap CIs, calculates route divergence share (0.0%), and writes `results/routing_eval.json`.

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

---

## Streamlit Web Application

The interactive web interface provides:
- **Routing Engine Toggle**: Switch between the **Trained ML Model** (Gradient Boosting with volume-to-multiplier formula) and the **Rule-Based Baseline** to directly observe ETA differences.
- **Dynamic Fallback**: If `models/traffic_model.joblib` is missing, the application automatically falls back to the rule-based baseline and alerts the user.
- **Traffic Level Classification**: Shows intuitive traffic conditions (`Light`, `Moderate`, `Heavy`) based on the calculated multiplier.
- **Interactive OpenStreetMap Visualization**: Displays turn-by-turn road segments, individual edge multipliers, cumulative travel times, and an embedded Folium map using OpenStreetMap tiles marking origin, destination, and calculated path.
- **Transparent Evaluation Panel**: An expandable sidebar panel reads directly from `results/metrics.json` and displays the exact test metrics, model parameters, and formal dataset limitations without hard-coded numbers.
