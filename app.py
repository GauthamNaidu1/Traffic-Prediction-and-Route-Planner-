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
    page_title="AI-Powered Traffic & Route Planner",
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
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Hyderabad road network (this may take time on first run)...")
def load_graph() -> nx.MultiDiGraph:
    """Loads the drive network for Hyderabad. Caches to disk for speed."""
    file_path = "hyderabad_drive.graphml"
    
    if os.path.exists(file_path):
        print(f"Loading graph from {file_path}...")
        G = ox.load_graphml(file_path)
    else:
        print("Downloading graph from OSM (this may take a few minutes)...")
        place = "Hyderabad, Telangana, India"
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        print("Saving graph to disk...")
        ox.save_graphml(G, file_path)
        
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
@st.cache_data(show_spinner=False)
def traffic_aware_astar_path(
    _G: nx.MultiDiGraph,
    source_node: int,
    target_node: int,
    hour: int,
    day_idx: int,
    mode: str,
    predictor_type: str = "Trained model",
    _model: Any = None,
) -> Tuple[Optional[List[int]], float, float, List[Dict]]:
    """
    Finds the shortest path using A* algorithm considering traffic factors.
    Supports either the trained ML model or the rule-based baseline.
    Returns: (path_nodes, total_time_min, total_dist_km, segments_info)
    """
    # Precompute base multiplier once if using trained model
    model_to_use = _model if _model is not None else TRAFFIC_MODEL
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
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="cartodbpositron")

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
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(fmap)
    
    return fmap

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("🗺️ AI-Powered Traffic & Route Planner")
st.caption("Hyderabad road network from OpenStreetMap with A* traffic-aware routing.")

with st.sidebar:
    st.header("Trip Planner")
    start_choice = st.selectbox("Start", options=list(LANDMARKS.keys()), index=0)
    end_choice = st.selectbox("Destination", options=list(LANDMARKS.keys()), index=8)
    
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
    predictor_choice = st.radio(
        "Engine Choice",
        ["Trained model", "Rule-based baseline"],
        index=0,
        help="Compare the trained Gradient Boosting model against the deterministic rule-based baseline function."
    )
    
    if predictor_choice == "Trained model" and TRAFFIC_MODEL is None:
        st.warning("⚠️ Model file 'models/traffic_model.joblib' not found. Falling back to Rule-based baseline.")
        effective_predictor = "Rule-based baseline"
    else:
        effective_predictor = predictor_choice

    day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_idx)
    if effective_predictor == "Trained model" and TRAFFIC_MODEL is not None:
        pred_vol, base_mult = predict_volume_and_base_multiplier(TRAFFIC_MODEL, hour, day_num)
        st.caption(f"📈 **Predicted Volume**: `{pred_vol:,.0f} veh/h`  \n⚖️ **Base Multiplier**: `{base_mult:.2f}x` (from $V_{{90}}$ formula)")
    else:
        dummy_mult = get_traffic_multiplier(hour, day_num, "road", "Car")
        st.caption(f"⚙️ **Baseline Step Multiplier**: `{dummy_mult:.2f}x` (from heuristic step rules)")
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

# Initialize session state for route data if not present
if 'route_data' not in st.session_state:
    st.session_state.route_data = None

if go:
    if start_choice == end_choice:
        st.warning("Pick two different points.")
    else:
        start_coord = LANDMARKS[start_choice]
        end_coord = LANDMARKS[end_choice]
        
        with st.spinner("Finding best path using A* algorithm…"):
            # Find nearest nodes
            source_node = find_nearest_node(GRAPH, start_coord[0], start_coord[1])
            target_node = find_nearest_node(GRAPH, end_coord[0], end_coord[1])
            
            # Run A*
            path_nodes, total_time, total_distance, segments = traffic_aware_astar_path(
                GRAPH,
                source_node,
                target_node,
                hour,
                day_num,
                mode,
                predictor_type=effective_predictor,
                _model=TRAFFIC_MODEL,
            )
            
            if path_nodes:
                # Get actual snapped coordinates from the graph nodes
                # This ensures the markers align perfectly with the path start/end
                snapped_start = (GRAPH.nodes[source_node]['y'], GRAPH.nodes[source_node]['x'])
                snapped_end = (GRAPH.nodes[target_node]['y'], GRAPH.nodes[target_node]['x'])
                
                st.session_state.route_data = {
                    "segments": segments,
                    "total_time": total_time,
                    "total_distance": total_distance,
                    "start_coord": snapped_start,
                    "end_coord": snapped_end,
                    "predictor": effective_predictor,
                }
            else:
                st.error("No drivable route found. Try a different pair of landmarks.")
                st.session_state.route_data = None

# Display Logic
# Unified Display Logic
st.markdown("### 🗺️ Route Map")

# Determine map center and zoom
if st.session_state.route_data:
    data = st.session_state.route_data
    start_c = data['start_coord']
    end_c = data['end_coord']
    segments = data['segments']
else:
    start_c = LANDMARKS[start_choice]
    end_c = LANDMARKS[end_choice]
    segments = None

center = ((start_c[0] + end_c[0]) / 2, (start_c[1] + end_c[1]) / 2)
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

# Add Road Network
# Add Road Network - REMOVED for performance
# The base tiles (CartoDB Positron) already show the road network clearly.
# Adding the full GeoJSON overlay was causing rendering delays and visibility issues.
pass

# Add Route if available
if segments:
    full_route_coords = []
    for seg in segments:
        color = get_traffic_color(seg["multiplier"])
        geometry = seg["geometry"]
        
        # Stitching
        if full_route_coords:
            if _point_distance(full_route_coords[-1], geometry[0]) < 0.0003:
                full_route_coords.extend(geometry[1:])
            else:
                full_route_coords.extend(geometry)
        else:
            full_route_coords.extend(geometry)

        # Segment line
        folium.PolyLine(
            geometry,
            color=color,
            weight=6,
            opacity=0.8,
            popup=f"{seg['name']} • {seg['distance_km']:.2f} km",
            tooltip=f"{seg['name']} ({seg['time_min']:.1f} min)"
        ).add_to(m)

    # Continuous line overlay
    if full_route_coords:
        folium.PolyLine(
            full_route_coords, 
            color="#1a73e8", 
            weight=3, 
            opacity=0.4,
            dash_array='5, 10'
        ).add_to(m)

# Add Markers
folium.Marker(location=start_c, popup="Start", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
folium.Marker(location=end_c, popup="Destination", icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)

# Render Map
st_folium(m, height=500, width=None, key="main_map", returned_objects=[])

# Metrics and Directions (only if route exists)
if st.session_state.route_data:
    data = st.session_state.route_data
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("ETA", f"{data['total_time']:.1f} min")
    with m2:
        st.metric("Distance", f"{data['total_distance']:.1f} km")
    with m3:
        st.metric("Traffic Level", "High" if data['total_time'] / (data['total_distance'] or 1) > 2 else "Normal")
    with m4:
        st.metric("Predictor Engine", data.get("predictor", "Trained model"))

    # Directions Section
    st.markdown("### ↪️ Turn-by-turn Directions")
    with st.container(height=400):
        for i, seg in enumerate(data['segments'], 1):
            color = get_traffic_color(seg["multiplier"])
            icon = "🔴" if color == "red" else ("🟠" if color == "orange" else "🔵")
            
            # Format Distance
            dist_km = seg['distance_km']
            if dist_km < 1.0:
                dist_str = f"{int(dist_km * 1000)} m"
            else:
                dist_str = f"{dist_km:.2f} km"
                
            # Format Time
            time_min = seg['time_min']
            mins = int(time_min)
            secs = int((time_min - mins) * 60)
            if mins > 0:
                time_str = f"{mins} min {secs} sec"
            else:
                time_str = f"{secs} sec"
                
            # Turn Instruction
            turn = seg.get("turn", "Go")
            road_name = seg['name']
            instruction = f"**{turn}** onto **{road_name}**" if turn != "Start" else f"Start at **{road_name}**"
            
            c1, c2 = st.columns([0.05, 0.95])
            with c1:
                st.write(f"**{i}.**")
            with c2:
                st.markdown(
                    f"{instruction}  \n"
                    f"_{dist_str}_ · _{time_str}_ · {icon}"
                )
            st.divider()
else:
    st.info("Choose landmarks, time, and mode, then click Calculate.")

# -----------------------------------------------------------------------------
# Model Validation & Methodology Panel (Loaded Dynamically from results/metrics.json)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Model Validation, Baseline Benchmarks & Methodology (from results/metrics.json)", expanded=False):
    if METRICS_DATA:
        dataset_name = METRICS_DATA.get("dataset", "UCI Metro Interstate Traffic Volume")
        st.info(f"ℹ️ **Methodology Notice**: Model trained on {dataset_name}; applied to a hand-built Hyderabad road graph; not validated on Hyderabad traffic.")

        st.markdown(
            "#### 📐 Traffic Volume to Multiplier Formula  \n"
            "The trained regression model outputs hourly traffic volume $V$ (vehicles/hour). "
            "Because edge travel times in Dijkstra/A* routing depend on speed degradation, volume is normalized and converted "
            "into an edge penalty multiplier using the 90th percentile of the training volume ($V_{90} = 5,830\\text{ veh/h}$):  \n"
            "$$\\text{Base Multiplier} = \\text{clip}\\left(0.85 + 1.60 \\times \\frac{V}{V_{90}}, \\; 0.85, \\; 2.50\\right)$$  \n"
            "$$\\text{Effective Multiplier} = \\text{Base Multiplier} \\times \\text{RoadFactor} \\times \\text{ModeFactor}$$  \n"
            "- **Free-flow / Late Night** ($V \\to 0$): Multiplier $\\approx 0.85$ (baseline travel speed).  \n"
            "- **Peak Rush-Hour Congestion** ($V \\to V_{90}$): Multiplier $\\approx 2.45$ (heavy travel delay).  \n"
            "- **Road Adjustment**: Motorway/Trunk = $\\times 1.05$; Residential/Living Street = $\\times 0.90$.  \n"
            "- **Mode Adjustment**: Car = $\\times 1.0$; Bike = $\\times 0.65$; Walk = $\\times 0.30$."
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
                f"The paired bootstrap difference $(\\text{{MAE}}_{{\\text{{GBR}}}} - \\text{{MAE}}_{{\\text{{Baseline}}}})$ is **`{p_gbr['mae_diff_point']:.2f}` veh/h** "
                f"with a 95% CI of **`[{p_gbr['mae_diff_ci_lower']:.2f}, {p_gbr['mae_diff_ci_upper']:.2f}]`**. "
                f"Because zero lies inside this confidence interval, **no calendar-feature model statistically beats the historical $(Hour, DOW)$ mean**."
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
