# Residual Analysis on Validation Split (2017-05-10 to 2018-01-19)

- **Model**: `GradientBoostingRegressor` (Gradient Boosting)
- **Validation Records**: 6,086 hourly timestamps
- **Overall Validation Metrics**: MAE = **285.01** veh/h | RMSE = **501.25** veh/h | $R^2$ = **0.9350**

## 1. MAE by Hour of Day

| Hour | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias (y - y_hat) | MAE (veh/h) | RMSE (veh/h) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 00:00 | 917.4 | 834.0 | +83.4 | **151.46** | 248.27 |
| 01:00 | 549.6 | 514.5 | +35.1 | **85.32** | 136.71 |
| 02:00 | 402.6 | 393.5 | +9.2 | **58.99** | 92.04 |
| 03:00 | 381.2 | 372.0 | +9.2 | **35.70** | 57.55 |
| 04:00 | 724.2 | 689.4 | +34.8 | **76.14** | 114.45 |
| 05:00 | 2205.3 | 2086.1 | +119.2 | **273.05** | 422.16 |
| 06:00 | 4214.7 | 4169.5 | +45.2 | **434.01** | 824.94 |
| 07:00 | 4768.6 | 4812.7 | -44.1 | **519.56** | 980.80 |
| 08:00 | 4623.0 | 4616.4 | +6.6 | **520.92** | 870.59 |
| 09:00 | 4455.7 | 4374.7 | +80.9 | **410.81** | 653.90 |
| 10:00 | 4265.1 | 4183.9 | +81.2 | **295.11** | 461.05 |
| 11:00 | 4550.8 | 4479.2 | +71.6 | **278.25** | 401.24 |
| 12:00 | 4778.2 | 4726.9 | +51.3 | **250.76** | 373.02 |
| 13:00 | 4779.7 | 4741.3 | +38.4 | **238.83** | 371.21 |
| 14:00 | 4954.9 | 4966.7 | -11.8 | **247.15** | 395.56 |
| 15:00 | 5284.6 | 5257.6 | +27.1 | **306.88** | 493.80 |
| 16:00 | 5764.3 | 5669.5 | +94.8 | **428.25** | 624.08 |
| 17:00 | 5415.2 | 5349.9 | +65.3 | **395.57** | 589.32 |
| 18:00 | 4369.7 | 4287.7 | +82.0 | **316.61** | 442.16 |
| 19:00 | 3419.8 | 3272.2 | +147.6 | **270.90** | 344.94 |
| 20:00 | 2991.0 | 2823.5 | +167.4 | **284.26** | 380.40 |
| 21:00 | 2813.0 | 2667.3 | +145.6 | **310.76** | 416.82 |
| 22:00 | 2339.4 | 2196.1 | +143.3 | **340.19** | 475.07 |
| 23:00 | 1619.7 | 1453.5 | +166.1 | **305.62** | 516.33 |

## 2. MAE by Day of Week

| Day | Count | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias | MAE (veh/h) | RMSE (veh/h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Monday | 862 | 3318.6 | 3330.5 | -12.0 | **399.99** | 759.62 |
| Tuesday | 861 | 3566.7 | 3545.3 | +21.4 | **277.59** | 544.15 |
| Wednesday | 879 | 3703.6 | 3630.7 | +72.9 | **207.46** | 316.89 |
| Thursday | 883 | 3690.9 | 3645.5 | +45.4 | **289.51** | 535.36 |
| Friday | 879 | 3734.7 | 3659.5 | +75.2 | **271.49** | 424.13 |
| Saturday | 862 | 2923.2 | 2808.5 | +114.7 | **262.78** | 380.74 |
| Sunday | 860 | 2562.7 | 2398.4 | +164.3 | **287.91** | 420.90 |

## 3. MAE by Month

| Month | Count | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias | MAE (veh/h) | RMSE (veh/h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| January | 446 | 3073.6 | 3350.1 | -276.5 | **380.03** | 707.25 |
| May | 523 | 3383.8 | 3325.0 | +58.8 | **260.84** | 492.83 |
| June | 720 | 3446.9 | 3308.2 | +138.7 | **217.28** | 314.89 |
| July | 738 | 3299.4 | 3249.2 | +50.3 | **274.84** | 520.47 |
| August | 743 | 3517.1 | 3317.7 | +199.4 | **233.45** | 339.59 |
| September | 716 | 3455.8 | 3277.3 | +178.5 | **292.53** | 468.73 |
| October | 744 | 3472.1 | 3265.4 | +206.7 | **262.23** | 376.54 |
| November | 716 | 3368.5 | 3323.1 | +45.4 | **303.13** | 574.07 |
| December | 740 | 3125.2 | 3244.4 | -119.1 | **370.71** | 660.92 |

## 4. Holiday vs Non-Holiday

| Category | Count | Actual Mean (veh/h) | Pred Mean (veh/h) | Bias | MAE (veh/h) | RMSE (veh/h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Non-Holiday | 6,076 | 3364.4 | 3295.9 | +68.5 | **284.91** | 501.38 |
| Holiday | 10 | 1030.1 | 825.1 | +205.0 | **343.09** | 414.34 |

## 5. 20 Worst-Predicted Hours on Validation Split

| Rank | Date & Hour | Day | Actual | Pred | Abs Error | Direction | Weather | Rain/Snow | Holiday |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :--- |
| 1 | 2018-01-01 07:00:00 | Mon | 677 | 5869 | **5192** | Overpredicted  | Clear (sky is clear) | None | None |
| 2 | 2017-11-23 07:00:00 | Thu | 991 | 6153 | **5162** | Overpredicted  | Clear (sky is clear) | None | None |
| 3 | 2017-07-04 07:00:00 | Tue | 1091 | 6148 | **5057** | Overpredicted  | Clear (sky is clear) | None | None |
| 4 | 2017-12-25 07:00:00 | Mon | 942 | 5869 | **4927** | Overpredicted  | Mist (mist) | None | None |
| 5 | 2017-05-29 07:00:00 | Mon | 1222 | 5869 | **4647** | Overpredicted  | Clouds (broken clouds) | None | None |
| 6 | 2017-11-23 06:00:00 | Thu | 834 | 5430 | **4596** | Overpredicted  | Clear (sky is clear) | None | None |
| 7 | 2018-01-01 06:00:00 | Mon | 686 | 5169 | **4483** | Overpredicted  | Clear (sky is clear) | None | None |
| 8 | 2018-01-01 08:00:00 | Mon | 811 | 5252 | **4440** | Overpredicted  | Clear (sky is clear) | None | None |
| 9 | 2017-07-04 06:00:00 | Tue | 1125 | 5512 | **4387** | Overpredicted  | Clear (sky is clear) | None | None |
| 10 | 2017-12-25 06:00:00 | Mon | 798 | 5169 | **4371** | Overpredicted  | Snow (light snow) | None | None |
| 11 | 2017-11-23 08:00:00 | Thu | 1267 | 5568 | **4301** | Overpredicted  | Clear (sky is clear) | None | None |
| 12 | 2017-09-04 07:00:00 | Mon | 1589 | 5869 | **4280** | Overpredicted  | Clouds (broken clouds) | None | None |
| 13 | 2017-07-04 08:00:00 | Tue | 1333 | 5602 | **4269** | Overpredicted  | Clear (sky is clear) | None | None |
| 14 | 2017-05-29 06:00:00 | Mon | 930 | 5169 | **4239** | Overpredicted  | Clouds (overcast clouds) | None | None |
| 15 | 2017-09-04 06:00:00 | Mon | 1128 | 5169 | **4041** | Overpredicted  | Clear (sky is clear) | None | None |
| 16 | 2017-12-25 08:00:00 | Mon | 1260 | 5252 | **3992** | Overpredicted  | Haze (haze) | None | None |
| 17 | 2017-12-05 07:00:00 | Tue | 2334 | 6148 | **3814** | Overpredicted  | Clear (sky is clear) | None | None |
| 18 | 2017-11-24 07:00:00 | Fri | 2419 | 5977 | **3558** | Overpredicted  | Clear (sky is clear) | None | None |
| 19 | 2017-05-29 08:00:00 | Mon | 1735 | 5252 | **3516** | Overpredicted  | Clouds (overcast clouds) | None | None |
| 20 | 2018-01-01 09:00:00 | Mon | 1122 | 4572 | **3450** | Overpredicted  | Clear (sky is clear) | None | None |
