"""
Test 4: Clean Fallback When Model File is Missing.

Verifies that:
1. load_traffic_model returns None safely when the model file is missing or invalid.
2. traffic_aware_astar_path falls back cleanly to the rule-based baseline without raising an error.
3. The routing engine continues to compute valid shortest paths, travel times, and segments.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import (
    LANDMARKS,
    load_graph,
    load_traffic_model,
    find_nearest_node,
    traffic_aware_astar_path,
    get_traffic_multiplier,
)


@pytest.fixture(scope="module")
def road_graph():
    return load_graph()


def test_missing_model_loader_returns_none():
    """Verifies that attempting to load a non-existent model returns None cleanly."""
    missing_path = os.path.join(REPO_ROOT, "models", "non_existent_model_12345.joblib")
    loaded = load_traffic_model(missing_path)
    assert loaded is None, "Missing model file must return None instead of throwing an unhandled exception"


def test_routing_fallback_to_rule_based(road_graph):
    """
    Verifies that when _model is None (simulating a missing model file),
    traffic_aware_astar_path falls back seamlessly to the deterministic rule-based baseline.
    """
    start_coord = LANDMARKS["Secunderabad Railway Station"]
    end_coord = LANDMARKS["Hitech City"]

    source_node = find_nearest_node(road_graph, start_coord[0], start_coord[1])
    target_node = find_nearest_node(road_graph, end_coord[0], end_coord[1])

    # Call with predictor_type='Trained model' but _model=None
    path_nodes, total_time, total_dist, segments = traffic_aware_astar_path(
        road_graph,
        source_node,
        target_node,
        hour=8,
        day_idx=0,
        mode="Car",
        predictor_type="Trained model",
        _model=None,  # Missing model simulation
    )

    assert path_nodes is not None, "Fallback must return a valid path"
    assert len(path_nodes) >= 2, "Path must contain at least 2 nodes"
    assert total_time > 0.0, "Fallback travel time must be positive"
    assert total_dist > 0.0, "Fallback distance must be positive"
    assert len(segments) >= 1, "Fallback segments must be populated"

    # Compare against explicit Rule-based baseline call
    path_rule, time_rule, dist_rule, _ = traffic_aware_astar_path(
        road_graph,
        source_node,
        target_node,
        hour=8,
        day_idx=0,
        mode="Car",
        predictor_type="Rule-based baseline",
        _model=None,
    )

    assert path_nodes == path_rule, "Fallback path must match Rule-based baseline path"
    assert abs(total_time - time_rule) < 1e-4, "Fallback time must match Rule-based baseline time"
    assert abs(total_dist - dist_rule) < 1e-4, "Fallback distance must match Rule-based baseline distance"


def test_rule_based_multiplier_valid_range():
    """Verifies that the heuristic rule-based multiplier function returns bounded values."""
    for h in range(24):
        for d in range(7):
            mult = get_traffic_multiplier(h, d, "motorway", "Car")
            assert 0.70 <= mult <= 2.60, f"Rule multiplier {mult} out of expected range for hour={h}, day={d}"
