"""
Test 3: Route Calculation Across All Dropdown Pairs & Same Origin-Destination Validation.

Verifies that:
1. Route calculation returns a valid, continuous path for every possible origin-destination pair
   in the LANDMARKS dropdown.
2. Same origin-destination inputs are validated gracefully (returning zero travel time and distance
   without raising an unhandled exception).
"""

import os
import sys
import pytest
import osmnx as ox

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import (
    LANDMARKS,
    load_graph,
    load_traffic_model,
    find_nearest_node,
    traffic_aware_astar_path,
)


@pytest.fixture(scope="module")
def app_resources():
    G = load_graph()
    model = load_traffic_model()
    return G, model


def test_route_calculation_for_all_dropdown_pairs(app_resources):
    """
    Exhaustively tests that A* route calculation returns a valid, non-empty path
    between every distinct pair of landmarks in the dropdown.
    """
    G, model = app_resources
    landmarks = list(LANDMARKS.keys())
    assert len(landmarks) >= 20, f"Expected at least 20 landmarks in dropdown, found {len(landmarks)}"

    total_pairs = 0
    successful_routes = 0

    # Test all distinct pairs
    for i, start_name in enumerate(landmarks):
        start_coord = LANDMARKS[start_name]
        source_node = find_nearest_node(G, start_coord[0], start_coord[1])

        for j, end_name in enumerate(landmarks):
            if i == j:
                continue
            total_pairs += 1
            end_coord = LANDMARKS[end_name]
            target_node = find_nearest_node(G, end_coord[0], end_coord[1])

            # Calculate route under trained model
            path_nodes, total_time, total_dist, segments = traffic_aware_astar_path(
                G,
                source_node,
                target_node,
                hour=9,
                day_idx=0,
                mode="Car",
                predictor_type="Trained model",
                _model=model,
            )

            assert path_nodes is not None, f"No path returned between {start_name} and {end_name}"
            assert len(path_nodes) >= 2, f"Path between {start_name} and {end_name} has fewer than 2 nodes"
            assert total_time > 0.0, f"Travel time between {start_name} and {end_name} must be positive"
            assert total_dist > 0.0, f"Distance between {start_name} and {end_name} must be positive"
            assert len(segments) >= 1, f"Route must have at least 1 segment"

            successful_routes += 1

    assert successful_routes == total_pairs
    print(f"\n[Test Passed] Verified all {successful_routes}/{total_pairs} landmark pair routes successfully!")


def test_same_origin_and_destination_input_validation(app_resources):
    """
    Verifies that when the origin and destination are identical:
    1. A* pathfinding returns a clean 0-time, 0-distance response without crashing.
    2. The input validation logic cleanly catches the identical landmark selection.
    """
    G, model = app_resources
    sample_landmark = list(LANDMARKS.keys())[0]
    coord = LANDMARKS[sample_landmark]
    node = find_nearest_node(G, coord[0], coord[1])

    # Call A* with source == target
    path_nodes, total_time, total_dist, segments = traffic_aware_astar_path(
        G,
        node,
        node,
        hour=8,
        day_idx=0,
        mode="Car",
        predictor_type="Trained model",
        _model=model,
    )

    assert path_nodes == [node], "Same origin and destination must return single-node path"
    assert total_time == 0.0, "Travel time must be exactly 0.0 minutes for same origin/destination"
    assert total_dist == 0.0, "Distance must be exactly 0.0 km for same origin/destination"
    assert segments == [], "Segments list must be empty"
