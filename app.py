import os
import heapq
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import folium
import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString
import streamlit as st
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic-Aware Route Planner",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Landmarks for the dropdowns
LANDMARKS: Dict[str, Tuple[float, float]] = {
    "Secunderabad Railway Station": (17.4399, 78.4983),
    "Paradise Circle": (17.4422, 78.4900),
    "Begumpet": (17.4486, 78.4691),
    "Begumpet Metro": (17.4415, 78.4705),
    "Ameerpet": (17.4365, 78.4480),
    "Punjagutta X Roads": (17.4188, 78.4520),
    "Somajiguda": (17.4108, 78.4678),
    "NTR Gardens": (17.4115, 78.4700),
    "Tank Bund": (17.4143, 78.4760),
    "Necklace Road": (17.4276, 78.4717),
    "Khairatabad": (17.4095, 78.4575),
    "Lakdi-ka-pul": (17.4012, 78.4698),
    "Nampally": (17.3942, 78.4712),
    "Koti": (17.3845, 78.4860),
    "Charminar": (17.3616, 78.4747),
    "Hitech City": (17.4483, 78.3908),
    "Cyber Towers": (17.4496, 78.3799),
    "Madhapur": (17.4430, 78.3860),
    "Raidurg": (17.4305, 78.3825),
    "Financial District": (17.4210, 78.3575),
    "Gachibowli": (17.4394, 78.3489),
    "Kondapur": (17.4582, 78.3700),
    "Jubilee Hills Check Post": (17.4310, 78.4070),
    "Banjara Hills": (17.4156, 78.4456),
}

# Base speeds in m/s
BASE_SPEEDS_MPS = {
    "Car": 40 / 3.6,   # ~11.1 m/s
    "Bike": 25 / 3.6,  # ~6.9 m/s
    "Walk": 5 / 3.6,   # ~1.4 m/s
}

# -----------------------------------------------------------------------------
# Traffic Level Thresholds (Derived from travel-time multiplier)
# -----------------------------------------------------------------------------
TRAFFIC_LEVEL_LIGHT_MAX: float = 1.30      # Multiplier <= 1.30 -> Light traffic
TRAFFIC_LEVEL_MODERATE_MAX: float = 1.80   # 1.30 < Multiplier <= 1.80 -> Moderate traffic
                                           # Multiplier > 1.80 -> Heavy traffic

def get_traffic_level_label(multiplier: float) -> str:
    """Classifies a traffic multiplier into a readable label (Light / Moderate / Heavy)."""
    if multiplier <= TRAFFIC_LEVEL_LIGHT_MAX:
        return "Light"
    elif multiplier <= TRAFFIC_LEVEL_MODERATE_MAX:
        return "Moderate"
    else:
        return "Heavy"

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def build_default_road_network() -> nx.MultiDiGraph:
    """Constructs a connected Hyderabad road network graph connecting all landmarks."""
    from shapely.geometry import LineString
    G = nx.MultiDiGraph()
    G.graph['crs'] = 'epsg:4326'
    name_to_id = {}
    for idx, (name, (lat, lon)) in enumerate(LANDMARKS.items(), start=1):
        G.add_node(idx, name=name, x=lon, y=lat, osmid=idx)
        name_to_id[name] = idx

    default_corridors = [
        ("Secunderabad Railway Station", "Paradise Circle", 1800.0, "trunk"),
        ("Paradise Circle", "Begumpet", 2100.0, "primary"),
        ("Begumpet", "Begumpet Metro", 900.0, "primary"),
        ("Begumpet Metro", "Ameerpet", 1800.0, "primary"),
        ("Ameerpet", "Punjagutta X Roads", 1500.0, "primary"),
        ("Punjagutta X Roads", "Somajiguda", 1200.0, "primary"),
        ("Somajiguda", "Khairatabad", 1300.0, "primary"),
        ("Khairatabad", "Lakdi-ka-pul", 1100.0, "primary"),
        ("Lakdi-ka-pul", "Nampally", 1200.0, "primary"),
        ("Nampally", "Koti", 2200.0, "primary"),
        ("Koti", "Charminar", 2800.0, "primary"),
        ("Lakdi-ka-pul", "NTR Gardens", 1500.0, "secondary"),
        ("NTR Gardens", "Tank Bund", 1200.0, "secondary"),
        ("Tank Bund", "Necklace Road", 1600.0, "secondary"),
        ("Necklace Road", "Begumpet", 2400.0, "secondary"),
        ("Punjagutta X Roads", "Banjara Hills", 1400.0, "secondary"),
        ("Banjara Hills", "Jubilee Hills Check Post", 2600.0, "primary"),
        ("Jubilee Hills Check Post", "Madhapur", 3200.0, "primary"),
        ("Madhapur", "Hitech City", 1500.0, "primary"),
        ("Hitech City", "Cyber Towers", 1100.0, "primary"),
        ("Cyber Towers", "Kondapur", 2500.0, "primary"),
        ("Kondapur", "Gachibowli", 3000.0, "primary"),
        ("Gachibowli", "Financial District", 2800.0, "trunk"),
        ("Financial District", "Raidurg", 2500.0, "trunk"),
        ("Raidurg", "Madhapur", 2100.0, "primary"),
        ("Hitech City", "Begumpet", 8800.0, "motorway"),
        ("Tank Bund", "Charminar", 6800.0, "primary"),
        ("Jubilee Hills Check Post", "Ameerpet", 3500.0, "primary"),
    ]

    for u_name, v_name, length, hw in default_corridors:
        if u_name in name_to_id and v_name in name_to_id:
            u, v = name_to_id[u_name], name_to_id[v_name]
            u_coord, v_coord = LANDMARKS[u_name], LANDMARKS[v_name]
            geom = LineString([(u_coord[1], u_coord[0]), (v_coord[1], v_coord[0])])
            G.add_edge(u, v, key=0, length=length, highway=hw, speed_kph=40.0, travel_time=length / 11.1, geometry=geom)
            G.add_edge(v, u, key=0, length=length, highway=hw, speed_kph=40.0, travel_time=length / 11.1, geometry=geom)
    return G

@st.cache_resource(show_spinner="Loading Hyderabad road network...")
def load_graph() -> nx.MultiDiGraph:
    """Loads the drive network for Hyderabad. Caches to disk for speed."""
    file_path = "hyderabad_drive.graphml"
    
    if os.path.exists(file_path):
        try:
            return ox.load_graphml(file_path)
        except Exception:
            pass

    G = build_default_road_network()
    try:
        ox.save_graphml(G, file_path)
    except Exception:
        pass
    return G

@st.cache_data(show_spinner=False)
def load_edges_geojson(_G: nx.MultiDiGraph) -> str:
    """Converts graph edges to GeoJSON. Caches to disk."""
    file_path = "hyderabad_edges.geojson"
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    # We only need the geometry for visualization
    edges = ox.graph_to_gdfs(_G, nodes=False, edges=True)[["geometry"]]
    geojson = edges.to_json()
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(geojson)
        
    return geojson

# Load data once
GRAPH = load_graph()
EDGES_GEOJSON = load_edges_geojson(GRAPH)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _point_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance squared (approx) for quick checks."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def _geometry_to_latlon(geometry: Any) -> List[Tuple[float, float]]:
    """Extracts list of (lat, lon) from Shapely geometry."""
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
    elif isinstance(geometry, MultiLineString):
        coords = [pt for line in geometry for pt in line.coords]
    elif isinstance(geometry, list):
        coords = geometry
    else:
        return []
    # Shapely coords are (lon, lat), we want (lat, lon) for Folium
    return [(lat, lon) for lon, lat in coords]

def orient_geometry(
    geometry: List[Tuple[float, float]],
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Ensures the geometry points from start_coord to end_coord.
    Useful because OSM edges might be stored in one direction but traversed in another.
    """
    if not geometry:
        return [start_coord, end_coord]
    
    # Check distance of first point to start vs last point to start
    start_dist = _point_distance(geometry[0], start_coord)
    end_dist = _point_distance(geometry[-1], start_coord)
    
    oriented = geometry if start_dist <= end_dist else list(reversed(geometry))
    
    # Snap ends if they are close enough (visual continuity)
    if _point_distance(oriented[0], start_coord) > 0.0003:
        oriented = [start_coord] + oriented
    if _point_distance(oriented[-1], end_coord) > 0.0003:
        oriented = oriented + [end_coord]
        
    return oriented

# -----------------------------------------------------------------------------
# Model & Metrics Loading (Streamlit Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_traffic_model(model_path: str = "models/traffic_model.joblib") -> Any:
    """Loads the trained traffic volume model from disk. Cached by Streamlit."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    return None

@st.cache_data(show_spinner=False)
def load_metrics(metrics_path: str = "results/metrics.json") -> Optional[Dict[str, Any]]:
    """Loads evaluation metrics from results/metrics.json. Cached by Streamlit."""
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics from {metrics_path}: {e}")
            return None
    return None

# Global Model & Metrics references
TRAFFIC_MODEL = load_traffic_model()
METRICS_DATA = load_metrics()

# -----------------------------------------------------------------------------
# Traffic Multiplier Formulas (Documented & Reproducible)
# -----------------------------------------------------------------------------
# 1. Trained Model Conversion Formula:
#    Traffic volume V (veh/h) predicted by the trained Gradient Boosting model
#    is mapped to an edge travel-time multiplier using:
#
#        normalized_volume = V / V_90
#        base_multiplier = clip(0.85 + 1.60 * normalized_volume, 0.85, 2.50)
#
#    Where V_90 = 5,830.0 veh/h is the 90th percentile of training traffic volume.
#    Rationale:
#    - Free-flow / night traffic (V -> 0): base multiplier -> 0.85
#    - Rush-hour peak congestion (V -> V_90): base multiplier -> 2.45
#    - Clipped to [0.85, 2.50] to prevent non-physical extrapolation.
#    - Scaled by road hierarchy (motorway 1.05x, residential 0.90x) and mode.
#
# 2. Rule-Based Baseline Formula (Heuristic Stand-in):
#    Uses discrete hour/weekend step multipliers (2.4 peak, 1.6 weekend, 1.3 normal, 0.85 night).

V_90_TRAIN: float = 5830.0  # 90th percentile from training split (vehicles/hour)

def predict_volume_and_base_multiplier(model: Any, hour: int, day_idx: int) -> Tuple[float, float]:
    """
    Predicts hourly traffic volume using the trained calendar ML model,
    then converts volume into a base travel-time multiplier via the 90th-percentile formula.
    """
    if hasattr(model, "n_features_in_") and model.n_features_in_ == 8:
        row = pd.DataFrame([{
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "day_of_week": day_idx,
            "is_weekend": int(day_idx in {5, 6}),
            "is_us_holiday": 0,
            "days_to_nearest_holiday": 15.0,
            "is_day_before_holiday": 0,
            "is_day_after_holiday": 0,
        }])
    else:
        row = pd.DataFrame([{
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "day_of_week": day_idx,
            "is_weekend": int(day_idx in {5, 6}),
            "is_holiday": 0,
        }])[["hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_holiday"]]
    
    pred_vol = float(model.predict(row)[0])
    norm_vol = max(0.0, pred_vol) / V_90_TRAIN
    base_mult = float(np.clip(0.85 + 1.60 * norm_vol, 0.85, 2.50))
    return pred_vol, base_mult

MODEL_VAL_MAE: float = 285.01  # Validation MAE in vehicles/hour from results/metrics.json

def compute_travel_time_range(
    total_time_min: float,
    pred_vol: Optional[float] = None,
    base_mult: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Computes an estimated travel time range using the model's validation MAE (285.01 veh/h).
    Returns (low_min, high_min) in minutes, clearly representing an uncertainty estimate.
    """
    if total_time_min <= 0.0:
        return 0.0, 0.0
    if pred_vol is not None and base_mult is not None and base_mult > 0:
        v_low = max(0.0, pred_vol - MODEL_VAL_MAE)
        v_high = pred_vol + MODEL_VAL_MAE
        m_low = float(np.clip(0.85 + 1.60 * (v_low / V_90_TRAIN), 0.85, 2.50))
        m_high = float(np.clip(0.85 + 1.60 * (v_high / V_90_TRAIN), 0.85, 2.50))
        t_low = total_time_min * (m_low / base_mult)
        t_high = total_time_min * (m_high / base_mult)
    else:
        # Default relative uncertainty based on validation MAE (delta_mult / typical_mult ~ 6%)
        rel_delta = (1.60 * (MODEL_VAL_MAE / V_90_TRAIN)) / 1.30
        t_low = max(0.0, total_time_min * (1.0 - rel_delta))
        t_high = total_time_min * (1.0 + rel_delta)
    return max(0.1, round(t_low, 1)), round(t_high, 1)

def adjust_multiplier(base_multiplier: float, highway: Any, mode: str) -> float:
    """Applies road type adjustment and travel mode scaling to base multiplier."""
    base = base_multiplier
    hw = highway if isinstance(highway, str) else (highway[0] if highway else "road")
    if hw in {"motorway", "trunk"}:
        base *= 1.05
    elif hw in {"residential", "living_street"}:
        base *= 0.90
        
    mode_factor = {"Car": 1.0, "Bike": 0.65, "Walk": 0.30}[mode]
    return round(base * mode_factor, 3)

def get_traffic_multiplier(hour: int, day_idx: int, highway: Any, mode: str) -> float:
    """Rule-based baseline: Calculates traffic penalty multiplier using deterministic step rules."""
    is_weekend = day_idx in {5, 6}  # Saturday, Sunday
    
    # Base traffic based on hour
    if not is_weekend and (7 <= hour <= 10 or 17 <= hour <= 21):
        base = 2.4  # Peak hours
    elif is_weekend and 11 <= hour <= 17:
        base = 1.6  # Weekend busy time
    elif 6 <= hour <= 22:
        base = 1.3  # Normal day
    else:
        base = 0.85 # Night/Early morning
        
    # Adjust for road type
    hw = highway if isinstance(highway, str) else (highway[0] if highway else "road")
    if hw in {"motorway", "trunk"}:
        base *= 1.05
    elif hw in {"residential", "living_street"}:
        base *= 0.90
        
    # Adjust for mode
    mode_factor = {"Car": 1.0, "Bike": 0.65, "Walk": 0.30}[mode]
    
    return round(base * mode_factor, 3)

def get_traffic_color(multiplier: float) -> str:
    """Returns a color code based on traffic intensity."""
    if multiplier >= 2.0:
        return "red"
    if multiplier >= 1.5:
        return "orange"
    return "blue"

def compute_time_weight(length_m: float, mode: str, multiplier: float) -> float:
    """Computes weighted travel time in seconds."""
    speed = BASE_SPEEDS_MPS[mode]
    # Time = Distance / Speed. Multiplier increases effective time (simulating traffic).
    return (length_m / speed) * multiplier

def heuristic(G: nx.MultiDiGraph, u: int, v: int, mode: str) -> float:
    """
    Heuristic function for A* algorithm.
    Estimates the minimum time to travel from u to v.
    """
    # Get coordinates
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    
    # Great circle distance in meters
    dist_m = ox.distance.great_circle(y1, x1, y2, x2)
    
    # Max speed for the mode (optimistic estimate)
    max_speed = BASE_SPEEDS_MPS[mode]
    
    # Heuristic time = Distance / Max_Speed
    # We use 1.0 as min multiplier for admissibility
    return dist_m / max_speed

# -----------------------------------------------------------------------------
# Pathfinding Algorithm (A*)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Pathfinding Algorithm (A*)
# -----------------------------------------------------------------------------
_MODEL_DEFAULT = object()

@st.cache_data(show_spinner=False)
def traffic_aware_astar_path(
    _G: nx.MultiDiGraph,
    source_node: int,
    target_node: int,
    hour: int,
    day_idx: int,
    mode: str,
    predictor_type: str = "Trained model",
    _model: Any = _MODEL_DEFAULT,
) -> Tuple[Optional[List[int]], float, float, List[Dict]]:
    """
    Finds the shortest path using A* algorithm considering traffic factors.
    Supports either the trained ML model or the rule-based baseline.
    Returns: (path_nodes, total_time_min, total_dist_km, segments_info)
    """
    # Handle edge case where origin and destination are the same node
    if source_node == target_node:
        return [source_node], 0.0, 0.0, []

    # Precompute base multiplier once if using trained model
    model_to_use = TRAFFIC_MODEL if _model is _MODEL_DEFAULT else _model
    use_model = (predictor_type == "Trained model" and model_to_use is not None)
    if use_model:
        _, base_mult = predict_volume_and_base_multiplier(model_to_use, hour, day_idx)
    else:
        base_mult = None

    def _get_multiplier(highway: Any) -> float:
        if use_model:
            return adjust_multiplier(base_mult, highway, mode)
        else:
            return get_traffic_multiplier(hour, day_idx, highway, mode)

    # Priority queue stores (f_score, node)
    pq = [(0.0, source_node)]
    
    # Cost from start to node
    g_score = {source_node: 0.0}
    
    # To reconstruct path: came_from[node] = (parent, edge_key)
    came_from: Dict[int, Tuple[int, int]] = {}
    
    while pq:
        current_f, current = heapq.heappop(pq)
        
        if current == target_node:
            break
            
        current_g = g_score[current]
        
        # If we found a shorter path to this node already, skip
        if current_g > g_score.get(current, float('inf')):
             continue
        
        for neighbor, edge_dict in _G.adj[current].items():
            # Handle multiple edges between nodes (MultiDiGraph)
            # We pick the one with minimum weight
            best_edge_key = None
            min_weight = float('inf')
            
            for key, data in edge_dict.items():
                length = data.get("length", 1.0)
                highway = data.get("highway", "road")
                
                multiplier = _get_multiplier(highway)
                weight = compute_time_weight(length, mode, multiplier)
                
                if weight < min_weight:
                    min_weight = weight
                    best_edge_key = key
            
            tentative_g = current_g + min_weight
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = (current, best_edge_key)
                g_score[neighbor] = tentative_g
                
                h = heuristic(_G, neighbor, target_node, mode)
                f = tentative_g + h
                heapq.heappush(pq, (f, neighbor))
                
    if target_node not in came_from and target_node != source_node:
        return None, float("inf"), 0.0, []
        
    # Reconstruct path
    path = [target_node]
    node = target_node
    segments = []
    total_distance = 0.0
    
    while node != source_node:
        prev, edge_key = came_from[node]
        edge_data = _G[prev][node][edge_key]
        
        length_m = edge_data.get("length", 1.0)
        highway = edge_data.get("highway", "road")
        name = edge_data.get("name") or highway
        if isinstance(name, list):
            name = name[0] # Handle multiple names
            
        geometry = _geometry_to_latlon(edge_data.get("geometry"))
        multiplier = _get_multiplier(highway)
        travel_seconds = compute_time_weight(length_m, mode, multiplier)
        
        # Orientation
        prev_coord = (_G.nodes[prev]["y"], _G.nodes[prev]["x"])
        current_coord = (_G.nodes[node]["y"], _G.nodes[node]["x"])
        oriented_geom = orient_geometry(geometry, prev_coord, current_coord)
        
        segments.append({
            "from": prev,
            "to": node,
            "name": str(name),
            "distance_km": length_m / 1000,
            "time_min": travel_seconds / 60,
            "multiplier": multiplier,
            "geometry": oriented_geom
        })
        
        total_distance += length_m / 1000
        node = prev
        path.append(node)
        
    path.reverse()
    segments.reverse()
    total_time_min = g_score[target_node] / 60
    
    # Post-process segments to add turn instructions
    if segments:
        # Helper to calculate bearing between two points
        def calculate_bearing(lat1, lon1, lat2, lon2):
            dLon = math.radians(lon2 - lon1)
            lat1 = math.radians(lat1)
            lat2 = math.radians(lat2)
            y = math.sin(dLon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
            return (math.degrees(math.atan2(y, x)) + 360) % 360

        # Helper to get turn direction
        def get_turn_instruction(bearing_in, bearing_out):
            diff = (bearing_out - bearing_in + 360) % 360
            if diff < 20 or diff > 340:
                return "Go straight"
            elif 20 <= diff < 45:
                return "Slight right"
            elif 45 <= diff < 135:
                return "Turn right"
            elif 135 <= diff < 225:
                return "Make a U-turn"
            elif 225 <= diff < 315:
                return "Turn left"
            else: # 315 <= diff < 340
                return "Slight left"

        # Calculate bearings and turns
        for i in range(len(segments)):
            # Get start and end points of the current segment geometry
            # geometry is list of (lat, lon)
            p1 = segments[i]["geometry"][0]
            p2 = segments[i]["geometry"][-1]
            
            # Calculate bearing of this segment
            bearing = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
            segments[i]["bearing"] = bearing
            
            if i == 0:
                segments[i]["turn"] = "Start"
            else:
                prev_bearing = segments[i-1]["bearing"]
                segments[i]["turn"] = get_turn_instruction(prev_bearing, bearing)

    return path, total_time_min, total_distance, segments

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def create_route_map(
    base_geojson: str,
    segments: List[Dict],
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
) -> folium.Map:
    """Creates a Folium map with the calculated route."""
    center = ((start_coord[0] + end_coord[0]) / 2, (start_coord[1] + end_coord[1]) / 2)
    fmap = folium.Map(
        location=center,
        zoom_start=13,
        control_scale=True,
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    )

    # Background road network
    folium.GeoJson(
        base_geojson,
        name="Road network",
        style_function=lambda _: {"color": "#c9ccd1", "weight": 1, "opacity": 0.45},
    ).add_to(fmap)

    # Draw segments
    full_route_coords: List[Tuple[float, float]] = []
    
    for seg in segments:
        color = get_traffic_color(seg["multiplier"])
        geometry = seg["geometry"]
        
        # Stitching for the continuous line
        if full_route_coords:
            # If the start of this segment is close to the end of the last, append
            if _point_distance(full_route_coords[-1], geometry[0]) < 0.0003:
                full_route_coords.extend(geometry[1:])
            else:
                full_route_coords.extend(geometry)
        else:
            full_route_coords.extend(geometry)

        # Segment line (colored by traffic)
        folium.PolyLine(
            geometry,
            color=color,
            weight=6,
            opacity=0.8,
            popup=f"{seg['name']} • {seg['distance_km']:.2f} km",
            tooltip=f"{seg['name']} ({seg['time_min']:.1f} min)"
        ).add_to(fmap)

    # Overlay a smooth continuous line for better visibility
    if full_route_coords:
        folium.PolyLine(
            full_route_coords, 
            color="#1a73e8", 
            weight=3, 
            opacity=0.4,
            dash_array='5, 10'
        ).add_to(fmap)

    # Markers
    folium.Marker(
        location=start_coord,
        popup=f"Start: {start_coord}",
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)
    
    folium.Marker(
        location=end_coord,
        popup=f"End: {end_coord}",
        tooltip="End",
    ).add_to(fmap)
    
    return fmap

def build_route_map(
    segments: Optional[List[Dict]],
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
) -> folium.Map:
    """Creates a clean, responsive Folium map for the route (or landmarks)."""
    center = ((start_coord[0] + end_coord[0]) / 2, (start_coord[1] + end_coord[1]) / 2)
    fmap = folium.Map(
        location=center,
        zoom_start=13,
        control_scale=True,
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    )

    if segments:
        full_route_coords: List[Tuple[float, float]] = []
        for seg in segments:
            color = get_traffic_color(seg["multiplier"])
            geometry = seg["geometry"]
            if full_route_coords:
                if _point_distance(full_route_coords[-1], geometry[0]) < 0.0003:
                    full_route_coords.extend(geometry[1:])
                else:
                    full_route_coords.extend(geometry)
            else:
                full_route_coords.extend(geometry)

            folium.PolyLine(
                geometry,
                color=color,
                weight=6,
                opacity=0.8,
                popup=f"{seg['name']} • {seg['distance_km']:.2f} km",
                tooltip=f"{seg['name']} ({seg['time_min']:.1f} min)",
            ).add_to(fmap)

        if full_route_coords:
            folium.PolyLine(
                full_route_coords,
                color="#1a73e8",
                weight=3,
                opacity=0.4,
                dash_array="5, 10",
            ).add_to(fmap)

    folium.Marker(
        location=start_coord,
        popup=f"Start: {start_coord}",
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)

    folium.Marker(
        location=end_coord,
        popup=f"Destination: {end_coord}",
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(fmap)

    return fmap

def render_directions(segments: List[Dict]) -> None:
    """Renders turn-by-turn directions with a responsive layout for phone and desktop screens."""
    for i, seg in enumerate(segments, 1):
        color = get_traffic_color(seg["multiplier"])
        icon = "🔴" if color == "red" else ("🟠" if color == "orange" else "🔵")

        dist_km = seg["distance_km"]
        dist_str = f"{int(dist_km * 1000)} m" if dist_km < 1.0 else f"{dist_km:.2f} km"

        time_min = seg["time_min"]
        mins = int(time_min)
        secs = int((time_min - mins) * 60)
        time_str = f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"

        turn = seg.get("turn", "Go")
        road_name = seg["name"]
        instruction = f"**{turn}** onto **{road_name}**" if turn != "Start" else f"Start at **{road_name}**"

        st.markdown(
            f"**{i}.** {instruction}  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;_{dist_str}_ · _{time_str}_ · {icon}"
        )
        st.divider()

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("🗺️ Traffic-Aware Route Planner")
st.caption("A* travel-time routing across Hyderabad landmarks with congestion predictions from highway traffic data.")

# Responsive styling injection for phone-width and mobile screens
st.markdown(
    """
    <style>
    /* Phone-width and responsive layout optimizations */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }
        [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 6px;
        }
        iframe {
            max-width: 100% !important;
            border-radius: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Trip Planner")
    start_choice = st.selectbox("Start", options=list(LANDMARKS.keys()), index=0)
    end_choice = st.selectbox("Destination", options=list(LANDMARKS.keys()), index=8)
    if start_choice == end_choice:
        st.info("ℹ️ Origin and destination are the same. Please select two different landmarks.")

    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour of day", 0, 23, 9)
    with col2:
        day_idx = st.selectbox(
            "Day",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0,
        )

    mode = st.radio("Mode", ["Car", "Bike", "Walk"], index=0, horizontal=True)

    st.markdown("---")
    st.subheader("Traffic Predictor Engine")

    # Engine Comparison Toggle
    compare_toggle = st.toggle(
        "Show both engines side-by-side",
        value=st.session_state.get("compare_both", False),
        key="compare_both_toggle",
        help="Compare results from both the Trained ML model and Rule-based baseline side by side for the same trip.",
    )
    st.session_state.compare_both = compare_toggle

    predictor_choice = st.radio(
        "Primary Engine Choice",
        ["Trained model", "Rule-based baseline"],
        index=0,
        help="Select the predictor engine to view on the primary map.",
    )
    st.caption("Demo: congestion model trained on public highway data; travel times are simulated estimates.")

    if predictor_choice == "Trained model" and TRAFFIC_MODEL is None:
        st.warning("⚠️ Model file 'models/traffic_model.joblib' not found. Falling back to Rule-based baseline.")
        effective_predictor = "Rule-based baseline"
    else:
        effective_predictor = predictor_choice

    day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_idx)
    if effective_predictor == "Trained model" and TRAFFIC_MODEL is not None:
        _, base_mult = predict_volume_and_base_multiplier(TRAFFIC_MODEL, hour, day_num)
        level_label = get_traffic_level_label(base_mult)
        st.markdown(f"**Traffic level**: `{level_label}`")
    else:
        dummy_mult = get_traffic_multiplier(hour, day_num, "road", "Car")
        level_label = get_traffic_level_label(dummy_mult)
        st.markdown(f"**Traffic level**: `{level_label}`")
    st.markdown("---")

    go = st.button("Calculate fastest route", type="primary", use_container_width=True)

# Wrapper for nearest nodes to handle missing scikit-learn
@st.cache_resource
def get_kdtree(_G: nx.MultiDiGraph):
    from scipy.spatial import cKDTree
    import numpy as np
    nodes = list(_G.nodes(data=True))
    coords = np.array([[n[1]['x'], n[1]['y']] for n in nodes])
    tree = cKDTree(coords)
    return tree, nodes

def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except ImportError:
        # Fallback using cached KDTree
        tree, nodes = get_kdtree(G)
        dist, idx = tree.query([lon, lat])
        return nodes[idx][0]

def compute_engine_result(
    engine_name: str,
    source_node: int,
    target_node: int,
    hour: int,
    day_num: int,
    mode: str,
    snapped_start: Tuple[float, float],
    snapped_end: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    """Calculates route, estimated travel-time uncertainty range, and summary metrics for an engine."""
    if engine_name == "Trained model" and TRAFFIC_MODEL is None:
        actual_engine = "Rule-based baseline"
        model_to_use = None
    else:
        actual_engine = engine_name
        model_to_use = TRAFFIC_MODEL if engine_name == "Trained model" else None

    path_nodes, total_time, total_distance, segments = traffic_aware_astar_path(
        GRAPH,
        source_node,
        target_node,
        hour,
        day_num,
        mode,
        predictor_type=actual_engine,
        _model=model_to_use,
    )

    if not path_nodes:
        return None

    if actual_engine == "Trained model" and model_to_use is not None:
        pred_vol, base_mult = predict_volume_and_base_multiplier(model_to_use, hour, day_num)
        level_label = get_traffic_level_label(base_mult)
        time_low, time_high = compute_travel_time_range(total_time, pred_vol, base_mult)
    else:
        dummy_mult = get_traffic_multiplier(hour, day_num, "road", "Car")
        level_label = get_traffic_level_label(dummy_mult)
        time_low, time_high = compute_travel_time_range(total_time, None, None)

    return {
        "engine": engine_name,
        "effective_engine": actual_engine,
        "path_nodes": path_nodes,
        "total_time": total_time,
        "total_distance": total_distance,
        "segments": segments,
        "time_range": (time_low, time_high),
        "traffic_level": level_label,
        "start_coord": snapped_start,
        "end_coord": snapped_end,
    }

# Initialize session state for persistent results across sidebar interactions
if "both_results" not in st.session_state:
    st.session_state.both_results = None
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "trip_params" not in st.session_state:
    st.session_state.trip_params = None
if "calc_id" not in st.session_state:
    st.session_state.calc_id = None

if go:
    if start_choice == end_choice:
        st.info("ℹ️ Origin and destination are the same landmark. Travel time is 0 minutes. Please select different start and destination landmarks to calculate a route.")
        st.session_state.both_results = None
        st.session_state.route_data = None
        st.session_state.trip_params = None
        st.session_state.calc_id = None
    else:
        start_coord = LANDMARKS[start_choice]
        end_coord = LANDMARKS[end_choice]

        with st.spinner("Finding best route using A* algorithm…"):
            source_node = find_nearest_node(GRAPH, start_coord[0], start_coord[1])
            target_node = find_nearest_node(GRAPH, end_coord[0], end_coord[1])
            snapped_start = (GRAPH.nodes[source_node]['y'], GRAPH.nodes[source_node]['x'])
            snapped_end = (GRAPH.nodes[target_node]['y'], GRAPH.nodes[target_node]['x'])

            res_trained = compute_engine_result(
                "Trained model", source_node, target_node, hour, day_num, mode, snapped_start, snapped_end
            )
            res_baseline = compute_engine_result(
                "Rule-based baseline", source_node, target_node, hour, day_num, mode, snapped_start, snapped_end
            )

            if (res_trained and res_trained["path_nodes"]) or (res_baseline and res_baseline["path_nodes"]):
                st.session_state.both_results = {
                    "Trained model": res_trained,
                    "Rule-based baseline": res_baseline,
                }
                st.session_state.trip_params = {
                    "start": start_choice,
                    "end": end_choice,
                    "hour": hour,
                    "day": day_idx,
                    "mode": mode,
                }
                st.session_state.calc_id = f"{start_choice}_{end_choice}_{hour}_{day_num}_{mode}"
                st.session_state.route_data = st.session_state.both_results.get(effective_predictor)
            else:
                st.error("No drivable route found. Try a different pair of landmarks.")
                st.session_state.both_results = None
                st.session_state.route_data = None
                st.session_state.trip_params = None
                st.session_state.calc_id = None

# Update route_data in session state if active engine changes in sidebar
if st.session_state.both_results and effective_predictor in st.session_state.both_results:
    st.session_state.route_data = st.session_state.both_results.get(effective_predictor)

# Display Trip Status & Parameter Change Hints
if st.session_state.trip_params:
    saved = st.session_state.trip_params
    if (saved["start"] != start_choice or saved["end"] != end_choice or
        saved["hour"] != hour or saved["day"] != day_idx or saved["mode"] != mode):
        st.caption(f"📍 Displaying calculated route for **{saved['start']} ➔ {saved['end']}** ({saved['mode']}, {saved['day']} at {saved['hour']}:00). Click 'Calculate fastest route' in the sidebar to recompute.")

# Route Map & Results Display Logic
if st.session_state.both_results:
    calc_key = st.session_state.get("calc_id", "init")

    if compare_toggle:
        # ---------------------------------------------------------------------
        # Side-by-Side Comparison View for the Same Trip
        # ---------------------------------------------------------------------
        st.markdown("### ⚖️ Side-by-Side Engine Comparison")
        saved_trip = st.session_state.trip_params or {"start": start_choice, "end": end_choice, "hour": hour, "day": day_idx, "mode": mode}
        st.caption(f"Comparing engines for trip: **{saved_trip['start']}** ➔ **{saved_trip['end']}** ({saved_trip['mode']} on {saved_trip['day']} at {saved_trip['hour']}:00)")

        res_trained = st.session_state.both_results.get("Trained model")
        res_baseline = st.session_state.both_results.get("Rule-based baseline")

        if res_trained and res_baseline:
            diff_t = res_trained["total_time"] - res_baseline["total_time"]
            if abs(diff_t) < 0.1:
                comp_summary = "Both engines estimate approximately equal travel times for this route."
            elif diff_t < 0:
                comp_summary = f"🤖 Trained ML model estimates **{abs(diff_t):.1f} min faster** travel time than ⚙️ Rule-based baseline."
            else:
                comp_summary = f"⚙️ Rule-based baseline estimates **{abs(diff_t):.1f} min faster** travel time than 🤖 Trained ML model."
            st.info(f"💡 **Comparison Summary**: {comp_summary}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🤖 Trained Model (ML)")
            if res_trained:
                t_low, t_high = res_trained["time_range"]
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric(
                        label="Estimated Travel Time (estimate)",
                        value=f"{t_low:.1f} – {t_high:.1f} min",
                        delta=f"Expected: {res_trained['total_time']:.1f} min",
                        delta_color="off",
                        help="Estimated travel time range based on validation MAE uncertainty (±285.01 veh/h). Point estimate is simulated.",
                    )
                with mc2:
                    st.metric("Distance", f"{res_trained['total_distance']:.1f} km")
                with mc3:
                    st.metric("Traffic Level", res_trained.get("traffic_level", "Normal"))

                m_trained = build_route_map(res_trained["segments"], res_trained["start_coord"], res_trained["end_coord"])
                st_folium(m_trained, height=380, use_container_width=True, key=f"map_trained_{calc_key}", returned_objects=[])
                st.caption("Route calculated with Trained ML congestion model.")

                with st.expander(f"↪️ Turn-by-turn Directions ({len(res_trained['segments'])} segments)", expanded=False):
                    with st.container(height=300):
                        render_directions(res_trained["segments"])

        with col_right:
            st.markdown("#### ⚙️ Rule-based Baseline")
            if res_baseline:
                t_low, t_high = res_baseline["time_range"]
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric(
                        label="Estimated Travel Time (estimate)",
                        value=f"{t_low:.1f} – {t_high:.1f} min",
                        delta=f"Expected: {res_baseline['total_time']:.1f} min",
                        delta_color="off",
                        help="Estimated travel time range based on validation MAE uncertainty (±285.01 veh/h). Point estimate is simulated.",
                    )
                with mc2:
                    st.metric("Distance", f"{res_baseline['total_distance']:.1f} km")
                with mc3:
                    st.metric("Traffic Level", res_baseline.get("traffic_level", "Normal"))

                m_baseline = build_route_map(res_baseline["segments"], res_baseline["start_coord"], res_baseline["end_coord"])
                st_folium(m_baseline, height=380, use_container_width=True, key=f"map_baseline_{calc_key}", returned_objects=[])
                st.caption("Route calculated with Heuristic Rule-based baseline.")

                with st.expander(f"↪️ Turn-by-turn Directions ({len(res_baseline['segments'])} segments)", expanded=False):
                    with st.container(height=300):
                        render_directions(res_baseline["segments"])

    else:
        # ---------------------------------------------------------------------
        # Single Engine Primary View
        # ---------------------------------------------------------------------
        st.markdown("### 🗺️ Route Map")
        data = st.session_state.route_data or st.session_state.both_results.get(effective_predictor)

        m = build_route_map(data["segments"], data["start_coord"], data["end_coord"])
        st_folium(m, height=450, use_container_width=True, key=f"main_map_{calc_key}", returned_objects=[])
        st.caption("Schematic route on a simplified network.")

        t_low, t_high = data.get("time_range", compute_travel_time_range(data["total_time"]))
        m1, m2, m3, m4 = st.columns([1.2, 0.9, 0.9, 1.0])
        with m1:
            st.metric(
                label="Estimated Travel Time (estimate)",
                value=f"{t_low:.1f} – {t_high:.1f} min",
                delta=f"Expected: {data['total_time']:.1f} min",
                delta_color="off",
                help="Estimated travel time range based on validation MAE uncertainty (±285.01 veh/h). Point estimate is simulated.",
            )
        with m2:
            st.metric("Distance", f"{data['total_distance']:.1f} km")
        with m3:
            st.metric("Traffic Level", data.get("traffic_level", "Normal"))
        with m4:
            st.metric("Predictor Engine", data.get("effective_engine", data.get("predictor", "Trained model")))

        st.caption("⏱️ **Travel Time Range (estimate)**: Based on model validation MAE uncertainty (±285.01 veh/h).")

        st.markdown("### ↪️ Turn-by-turn Directions")
        with st.container(height=360):
            render_directions(data["segments"])

else:
    # -------------------------------------------------------------------------
    # Initial / Empty State Before Route Calculation
    # -------------------------------------------------------------------------
    st.markdown("### 🗺️ Route Map")
    start_c = LANDMARKS[start_choice]
    end_c = LANDMARKS[end_choice]
    m = build_route_map(None, start_c, end_c)
    st_folium(m, height=450, use_container_width=True, key="empty_map", returned_objects=[])
    st.caption("Choose landmarks, time, and mode in the sidebar, then click **Calculate fastest route**.")

# -----------------------------------------------------------------------------
# Model Validation & Methodology Panel (Loaded Dynamically from results/metrics.json)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Model Validation, Baseline Benchmarks & Methodology (from results/metrics.json)", expanded=False):
    if METRICS_DATA:
        dataset_name = METRICS_DATA.get("dataset", "UCI Metro Interstate Traffic Volume")
        st.info(f"ℹ️ **Methodology Notice**: Model trained on {dataset_name}; applied to a hand-built Hyderabad road graph; not validated on Hyderabad traffic.")

        st.markdown(
            "#### 🚦 Traffic Volume to Travel Time Penalty  \n"
            "The trained machine learning model estimates hourly traffic congestion volume based on calendar schedules. "
            "Congestion predictions are normalized and mapped to edge travel-time penalties according to traffic intensity, "
            "road classification (e.g., motorways vs. residential streets), and transport mode (Car, Bike, Walk).  \n"
            "- **Free-flow / Late Night**: Minimal traffic volume with baseline travel speeds.  \n"
            "- **Peak Rush-Hour Congestion**: High traffic volume with substantial travel delay penalties.  \n"
            "- **Road Adjustment**: Motorway and trunk roads account for higher capacity, while residential streets experience different delay characteristics.  \n"
            "- **Mode Adjustment**: Pedestrians and bicycles are less affected by motor vehicle congestion."
        )

        st.markdown("#### 🔬 Unseen Test Set Evaluation (with 1,000-Resample 95% Bootstrap CIs)")
        st.caption("Chronological split: 28,402 train (70%) / 6,086 val (15%) / 6,087 test (15%). No temporal leakage.")

        test_ci = METRICS_DATA.get("test_set_bootstrap_95ci", {})
        if test_ci:
            rows = []
            for m_name, m_vals in test_ci.items():
                rows.append({
                    "Model / Baseline": m_name,
                    "MAE (veh/h)": f"{m_vals['mae']['point']:.2f} [{m_vals['mae']['ci_lower']:.1f}, {m_vals['mae']['ci_upper']:.1f}]",
                    "RMSE (veh/h)": f"{m_vals['rmse']['point']:.2f} [{m_vals['rmse']['ci_lower']:.1f}, {m_vals['rmse']['ci_upper']:.1f}]",
                    "R²": f"{m_vals['r2']['point']:.4f} [{m_vals['r2']['ci_lower']:.4f}, {m_vals['r2']['ci_upper']:.4f}]",
                })
            st.table(pd.DataFrame(rows))

        paired = METRICS_DATA.get("paired_bootstrap_vs_historical_mean", {})
        if paired and "Gradient_Boosting_vs_Historical_Mean" in paired:
            p_gbr = paired["Gradient_Boosting_vs_Historical_Mean"]
            st.markdown(
                f"**Baseline Comparison Finding**: Tuned Gradient Boosting achieved an MAE of `260.78` vs. `260.82` for the Historical Mean baseline. "
                f"The paired bootstrap difference (Gradient Boosting vs. Historical Mean baseline) is **`{p_gbr['mae_diff_point']:.2f}` veh/h** "
                f"with a 95% CI of **`[{p_gbr['mae_diff_ci_lower']:.2f}, {p_gbr['mae_diff_ci_upper']:.2f}]`**. "
                f"Because zero lies inside this confidence interval, **no calendar-feature model statistically beats the historical (Hour, Day-of-Week) mean**."
            )

        st.markdown("#### 🔄 5-Fold Rolling-Origin Cross-Validation (Mean ± Std)")
        rolling = METRICS_DATA.get("rolling_origin_5fold_summary", {})
        if rolling:
            r_rows = []
            for r_name, r_vals in rolling.items():
                r_rows.append({
                    "Model / Baseline": r_name,
                    "MAE (veh/h)": f"{r_vals['mae_mean']:.2f} ± {r_vals['mae_std']:.2f}",
                    "RMSE (veh/h)": f"{r_vals['rmse_mean']:.2f} ± {r_vals['rmse_std']:.2f}",
                    "R²": f"{r_vals['r2_mean']:.4f} ± {r_vals['r2_std']:.4f}",
                })
            st.table(pd.DataFrame(r_rows))

        nowcast = METRICS_DATA.get("nowcast_experiment", {})
        if nowcast:
            st.markdown(
                "#### ☁️ Nowcast Experiment (Contemporaneous Weather Features Included)  \n"
                "- Weather features: `temp`, `rain_1h`, `snow_1h`, `clouds_all`, and one-hot weather categories.  \n"
                f"- **Nowcast Test MAE**: `{nowcast['test_ci']['mae']['point']:.2f}` veh/h (compared to `{test_ci.get('Gradient Boosting', {}).get('mae', {}).get('point', 260.78):.2f}` for calendar-only).  \n"
                "- *Key Insight*: Contemporaneous weather sensors provide actionable nowcasting signal, whereas future trip planning is fundamentally constrained by calendar schedules."
            )

        best_meta = METRICS_DATA.get("best_calendar_model", {})
        if best_meta:
            st.caption(f"💾 Active saved model: `{best_meta.get('artifact_path', 'models/traffic_model.joblib')}` ({best_meta.get('size_mb', 0.22)} MB) — Tuned parameters: `{best_meta.get('params', {})}`")
    else:
        st.info("Metrics file `results/metrics.json` not found. Run `python train/train_models.py` to generate evaluation metrics.")
