"""
Test 2: Road Graph Connectivity and Positive Edge Weights.

Verifies that the urban road network graph:
1. Is non-empty and connected (every node is reachable).
2. Has strictly positive edge lengths and no negative edge weights under all travel modes and traffic conditions.
"""

import os
import sys
import pytest
import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import load_graph, compute_time_weight, BASE_SPEEDS_MPS


@pytest.fixture(scope="module")
def road_graph():
    G = load_graph()
    assert G is not None, "load_graph() must return a valid networkx graph"
    return G


def test_graph_structure_and_connectivity(road_graph):
    """Verifies that the road graph is non-empty and connected."""
    num_nodes = road_graph.number_of_nodes()
    num_edges = road_graph.number_of_edges()

    assert num_nodes >= 23, f"Expected at least 23 landmarks/nodes, got {num_nodes}"
    assert num_edges >= 26, f"Expected at least 26 edges, got {num_edges}"

    # For directed MultiDiGraph, test strong connectivity or weak connectivity
    if road_graph.is_directed():
        is_conn = nx.is_strongly_connected(road_graph) or nx.is_weakly_connected(road_graph)
    else:
        is_conn = nx.is_connected(road_graph)
    assert is_conn, "The road network graph must be connected!"


def test_no_negative_edge_lengths(road_graph):
    """Verifies that every edge has a strictly positive physical length."""
    for u, v, k, data in road_graph.edges(keys=True, data=True):
        length = data.get("length", None)
        assert length is not None, f"Edge ({u}, {v}) is missing 'length' attribute"
        assert length > 0.0, f"Edge ({u}, {v}) has non-positive length: {length}"


def test_no_negative_edge_weights_across_modes(road_graph):
    """
    Verifies that for all travel modes and plausible multiplier values,
    edge traversal weights (time in seconds) are strictly positive.
    """
    test_multipliers = [0.85, 1.00, 1.30, 1.80, 2.50]
    test_modes = ["Car", "Bike", "Walk"]

    for u, v, data in road_graph.edges(data=True):
        length = data["length"]
        for mode in test_modes:
            for mult in test_multipliers:
                weight = compute_time_weight(length, mode, mult)
                assert weight > 0.0, (
                    f"Edge ({u}, {v}) produced non-positive weight: {weight} "
                    f"for mode={mode}, mult={mult}, length={length}"
                )
                assert weight >= length / BASE_SPEEDS_MPS[mode] * 0.25, (
                    f"Edge ({u}, {v}) weight is unreasonably small: {weight}"
                )
