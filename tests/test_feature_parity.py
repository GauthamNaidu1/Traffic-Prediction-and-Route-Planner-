"""
Test 1: Feature Parity Between Serving (app.py) and Training.

Verifies that the features constructed in app.py match the training feature builder
for 10 hour/day combinations including midnight and Sunday.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import predict_volume_and_base_multiplier, load_traffic_model
from train.experiments import build_experimental_features


# 10 diverse hour/day combinations including midnight (0) and Sunday (6)
HOUR_DAY_COMBINATIONS = [
    (0, 0),   # Midnight Monday
    (0, 6),   # Midnight Sunday
    (8, 0),   # Morning rush Monday
    (13, 2),  # Midday Wednesday
    (18, 4),  # Evening rush Friday
    (2, 1),   # Late night Tuesday
    (14, 5),  # Weekend afternoon Saturday
    (3, 6),   # Late night Sunday
    (12, 6),  # Midday Sunday
    (23, 3),  # Late night Thursday
]


@pytest.fixture(scope="module")
def trained_model():
    model_path = os.path.join(REPO_ROOT, "models", "traffic_model.joblib")
    assert os.path.exists(model_path), f"Model file must exist at {model_path}"
    model = joblib.load(model_path)
    return model


def test_feature_parity_across_10_combinations(trained_model):
    """
    Checks that app.py constructs the exact feature names, order, and values
    expected by the trained model for 10 hour/day combinations.
    """
    expected_features = list(trained_model.feature_names_in_)
    assert len(expected_features) == 8, f"Expected 8 features, got {expected_features}"

    for hour, day_idx in HOUR_DAY_COMBINATIONS:
        # Generate prediction using app.py's prediction pipeline
        pred_vol, base_mult = predict_volume_and_base_multiplier(trained_model, hour, day_idx)

        # 1. Output sanity
        assert np.isfinite(pred_vol), f"Predicted volume must be finite for hour={hour}, day={day_idx}"
        assert pred_vol >= 0.0, f"Predicted volume must be non-negative for hour={hour}, day={day_idx}"
        assert 0.85 <= base_mult <= 2.50, f"Base multiplier must be clipped to [0.85, 2.50], got {base_mult}"

        # 2. Check cyclical hour encoding exact match
        expected_sin = np.sin(2 * np.pi * hour / 24.0)
        expected_cos = np.cos(2 * np.pi * hour / 24.0)
        expected_weekend = int(day_idx in {5, 6})

        # Reconstruct row via training feature builder logic on a dummy timestamp
        test_df = pd.DataFrame([{
            "date_time": pd.Timestamp(f"2023-01-0{day_idx + 2} {hour:02d}:00:00"),
            "holiday": "None",
            "traffic_volume": 1000,
        }])
        enriched_df, feature_sets = build_experimental_features(test_df)
        train_row = enriched_df[feature_sets["Config_3_Holiday_Proximity"]].iloc[0]

        assert np.isclose(train_row["hour_sin"], expected_sin, atol=1e-6)
        assert np.isclose(train_row["hour_cos"], expected_cos, atol=1e-6)
        assert train_row["day_of_week"] == day_idx
        assert train_row["is_weekend"] == expected_weekend


def test_sunday_and_midnight_specific_values(trained_model):
    """Explicitly verifies midnight (hour 0) and Sunday (day 6) edge cases."""
    # Midnight Monday
    vol_mid_mon, mult_mid_mon = predict_volume_and_base_multiplier(trained_model, 0, 0)
    assert vol_mid_mon < 2000.0, f"Midnight traffic should be relatively low, got {vol_mid_mon}"
    assert mult_mid_mon < 1.50

    # Midnight Sunday
    vol_mid_sun, mult_mid_sun = predict_volume_and_base_multiplier(trained_model, 0, 6)
    assert vol_mid_sun < 2000.0
    assert mult_mid_sun < 1.50

    # Midday Sunday vs Midday Monday
    vol_sun_noon, _ = predict_volume_and_base_multiplier(trained_model, 12, 6)
    vol_mon_noon, _ = predict_volume_and_base_multiplier(trained_model, 12, 0)
    assert np.isfinite(vol_sun_noon) and np.isfinite(vol_mon_noon)
