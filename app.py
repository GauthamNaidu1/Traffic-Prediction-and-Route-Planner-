import math
from typing import Dict, List, Tuple, Optional

import streamlit as st
import networkx as nx
import folium
from streamlit_folium import st_folium


# -------------------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------------------
st.set_page_config(page_title="AI-Powered Traffic Predictor & Route Planner", layout="wide")


# -------------------------------------------------------------
# Week 1: Problem Scope & Research (Documented in-code)
# - Scope: Small urban district in Hyderabad/Secunderabad
# - Data source: Hardcoded miniature network (conceptual stand-in for OSM)
# - Algorithm: Dijkstra's algorithm over a NetworkX graph
# -------------------------------------------------------------


# -------------------------------------------------------------
# Week 2: Build the Road Network Graph
# - Nodes: intersections/landmarks (lat, lon)
# - Edges: roads with distance_km and geometry (polyline)
# -------------------------------------------------------------

# Expanded network around Hyderabad/Secunderabad with realistic road connections
# Node IDs are readable names for clarity in the UI
CITY_NODES: Dict[str, Tuple[float, float]] = {
    # Secunderabad Area
    "Secunderabad Station": (17.4399, 78.4983),
    "Parade Ground": (17.4452, 78.4984),
    "Paradise Circle": (17.4422, 78.4900),
    "Patny Circle": (17.4391, 78.5082),
    "Jubilee Bus Station": (17.4566, 78.5028),
    
    # Central Hyderabad
    "Begumpet Station": (17.4486, 78.4691),
    "Begumpet Metro": (17.4486, 78.4691),
    "Necklace Road": (17.4276, 78.4717),
    "Tank Bund": (17.4143, 78.4760),
    "Hussain Sagar": (17.4239, 78.4738),
    "Lakdi-ka-pul": (17.4012, 78.4698),
    
    # IT Corridor
    "Hitech City": (17.4483, 78.3908),
    "Cyber Towers": (17.4496, 78.3799),
    "Kondapur": (17.4582, 78.3700),
    "Gachibowli": (17.4394, 78.3489),
    "Nanakramguda": (17.4256, 78.3654),
    
    # South Hyderabad
    "Jubilee Hills": (17.4310, 78.4070),
    "Jubilee Hills Check Post": (17.4285, 78.4354),
    "Banjara Hills": (17.4156, 78.4456),
    "Panjagutta": (17.4134, 78.4567),
    "Somajiguda": (17.4108, 78.4678),
    
    # Old City
    "Charminar": (17.3616, 78.4747),
    "Madina Circle": (17.3756, 78.4789),
    "MGBS": (17.3856, 78.4689),
}

# Helper to compute straight-line distance (fallback for edges without explicit distance)
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# Comprehensive road network with realistic geometries following actual roads
# Each edge contains road_id, name, distance_km, and detailed geometry
CITY_EDGES: List[Dict] = [
    # Secunderabad Area Connections
    {
        "u": "Secunderabad Station", "v": "Parade Ground", "road_id": "R001", "name": "Station Road",
        "distance_km": 1.2, "geometry": [(17.4399, 78.4983), (17.4415, 78.4980), (17.4430, 78.4982), (17.4452, 78.4984)]
    },
    {
        "u": "Parade Ground", "v": "Paradise Circle", "road_id": "R002", "name": "Parade Ground Road",
        "distance_km": 1.8, "geometry": [(17.4452, 78.4984), (17.4445, 78.4950), (17.4435, 78.4925), (17.4422, 78.4900)]
    },
    {
        "u": "Paradise Circle", "v": "Begumpet Station", "road_id": "R003", "name": "MG Road",
        "distance_km": 2.3, "geometry": [(17.4422, 78.4900), (17.4445, 78.4850), (17.4465, 78.4775), (17.4486, 78.4691)]
    },
    {
        "u": "Patny Circle", "v": "Parade Ground", "road_id": "R004", "name": "Patny Road",
        "distance_km": 1.1, "geometry": [(17.4391, 78.5082), (17.4405, 78.5050), (17.4420, 78.5020), (17.4452, 78.4984)]
    },
    {
        "u": "Jubilee Bus Station", "v": "Parade Ground", "road_id": "R005", "name": "Jubilee Road",
        "distance_km": 1.8, "geometry": [(17.4566, 78.5028), (17.4520, 78.5010), (17.4485, 78.5000), (17.4452, 78.4984)]
    },
    
    # Central Hyderabad Connections
    {
        "u": "Begumpet Station", "v": "Necklace Road", "road_id": "R006", "name": "Begumpet Road",
        "distance_km": 2.5, "geometry": [(17.4486, 78.4691), (17.4420, 78.4700), (17.4350, 78.4705), (17.4276, 78.4717)]
    },
    {
        "u": "Necklace Road", "v": "Hussain Sagar", "road_id": "R007", "name": "Necklace Road",
        "distance_km": 1.2, "geometry": [(17.4276, 78.4717), (17.4260, 78.4725), (17.4250, 78.4730), (17.4239, 78.4738)]
    },
    {
        "u": "Hussain Sagar", "v": "Tank Bund", "road_id": "R008", "name": "Tank Bund Road",
        "distance_km": 1.4, "geometry": [(17.4239, 78.4738), (17.4200, 78.4745), (17.4170, 78.4750), (17.4143, 78.4760)]
    },
    {
        "u": "Tank Bund", "v": "Lakdi-ka-pul", "road_id": "R009", "name": "Tank Bund Extension",
        "distance_km": 2.1, "geometry": [(17.4143, 78.4760), (17.4100, 78.4730), (17.4050, 78.4715), (17.4012, 78.4698)]
    },
    
    # IT Corridor Connections
    {
        "u": "Begumpet Station", "v": "Hitech City", "road_id": "R010", "name": "Begumpet-Hitech Expressway",
        "distance_km": 8.8, "geometry": [(17.4486, 78.4691), (17.4480, 78.4600), (17.4485, 78.4500), (17.4490, 78.4400), (17.4485, 78.4300), (17.4483, 78.4200), (17.4483, 78.4100), (17.4483, 78.4000), (17.4483, 78.3908)]
    },
    {
        "u": "Hitech City", "v": "Cyber Towers", "road_id": "R011", "name": "Hitech City Road",
        "distance_km": 1.2, "geometry": [(17.4483, 78.3908), (17.4490, 78.3850), (17.4496, 78.3799)]
    },
    {
        "u": "Cyber Towers", "v": "Kondapur", "road_id": "R012", "name": "Cyber Towers Road",
        "distance_km": 2.8, "geometry": [(17.4496, 78.3799), (17.4520, 78.3770), (17.4550, 78.3740), (17.4582, 78.3700)]
    },
    {
        "u": "Kondapur", "v": "Gachibowli", "road_id": "R013", "name": "Gachibowli Road",
        "distance_km": 3.2, "geometry": [(17.4582, 78.3700), (17.4520, 78.3650), (17.4450, 78.3600), (17.4394, 78.3489)]
    },
    {
        "u": "Hitech City", "v": "Nanakramguda", "road_id": "R014", "name": "Nanakramguda Road",
        "distance_km": 3.8, "geometry": [(17.4483, 78.3908), (17.4420, 78.3850), (17.4350, 78.3800), (17.4300, 78.3750), (17.4256, 78.3654)]
    },
    
    # South Hyderabad Connections
    {
        "u": "Hitech City", "v": "Jubilee Hills", "road_id": "R015", "name": "Road No. 36",
        "distance_km": 4.2, "geometry": [(17.4483, 78.3908), (17.4440, 78.3950), (17.4390, 78.4000), (17.4350, 78.4035), (17.4310, 78.4070)]
    },
    {
        "u": "Jubilee Hills", "v": "Jubilee Hills Check Post", "road_id": "R016", "name": "Jubilee Hills Road",
        "distance_km": 2.8, "geometry": [(17.4310, 78.4070), (17.4300, 78.4120), (17.4290, 78.4200), (17.4285, 78.4354)]
    },
    {
        "u": "Jubilee Hills Check Post", "v": "Banjara Hills", "road_id": "R017", "name": "Road No. 45",
        "distance_km": 1.5, "geometry": [(17.4285, 78.4354), (17.4250, 78.4380), (17.4200, 78.4420), (17.4156, 78.4456)]
    },
    {
        "u": "Banjara Hills", "v": "Panjagutta", "road_id": "R018", "name": "Road No. 12",
        "distance_km": 1.8, "geometry": [(17.4156, 78.4456), (17.4145, 78.4500), (17.4140, 78.4535), (17.4134, 78.4567)]
    },
    {
        "u": "Panjagutta", "v": "Somajiguda", "road_id": "R019", "name": "Panjagutta Road",
        "distance_km": 1.2, "geometry": [(17.4134, 78.4567), (17.4125, 78.4620), (17.4115, 78.4650), (17.4108, 78.4678)]
    },
    {
        "u": "Somajiguda", "v": "Necklace Road", "road_id": "R020", "name": "Somajiguda Road",
        "distance_km": 1.8, "geometry": [(17.4108, 78.4678), (17.4150, 78.4690), (17.4200, 78.4700), (17.4250, 78.4708), (17.4276, 78.4717)]
    },
    
    # Old City Connections
    {
        "u": "Tank Bund", "v": "Charminar", "road_id": "R021", "name": "Charminar Road",
        "distance_km": 6.8, "geometry": [(17.4143, 78.4760), (17.4000, 78.4750), (17.3850, 78.4740), (17.3700, 78.4730), (17.3616, 78.4747)]
    },
    {
        "u": "Charminar", "v": "Madina Circle", "road_id": "R022", "name": "Madina Road",
        "distance_km": 1.8, "geometry": [(17.3616, 78.4747), (17.3680, 78.4760), (17.3720, 78.4775), (17.3756, 78.4789)]
    },
    {
        "u": "Madina Circle", "v": "MGBS", "road_id": "R023", "name": "MGBS Road",
        "distance_km": 2.5, "geometry": [(17.3756, 78.4789), (17.3800, 78.4750), (17.3830, 78.4720), (17.3856, 78.4689)]
    },
    {
        "u": "MGBS", "v": "Lakdi-ka-pul", "road_id": "R024", "name": "MGBS-Lakdi Road",
        "distance_km": 3.2, "geometry": [(17.3856, 78.4689), (17.3950, 78.4690), (17.4050, 78.4695), (17.4012, 78.4698)]
    },
    
    # Cross-connections for better routing
    {
        "u": "Nanakramguda", "v": "Jubilee Hills", "road_id": "R025", "name": "Nanakramguda-Jubilee Link",
        "distance_km": 4.5, "geometry": [(17.4256, 78.3654), (17.4280, 78.3750), (17.4295, 78.3850), (17.4305, 78.3950), (17.4310, 78.4070)]
    },
    {
        "u": "Gachibowli", "v": "Nanakramguda", "road_id": "R026", "name": "Gachibowli-Nanakramguda Link",
        "distance_km": 2.8, "geometry": [(17.4394, 78.3489), (17.4350, 78.3520), (17.4300, 78.3580), (17.4256, 78.3654)]
    },
]


def build_graph(nodes: Dict[str, Tuple[float, float]], edges: List[Dict]) -> nx.Graph:
    """Create an undirected NetworkX graph from provided nodes and edges."""
    G = nx.Graph()
    for name, (lat, lon) in nodes.items():
        G.add_node(name, lat=lat, lon=lon)
    for e in edges:
        u = e["u"]
        v = e["v"]
        distance_km = e.get("distance_km")
        if distance_km is None:
            distance_km = haversine_km(nodes[u], nodes[v])
        G.add_edge(
            u,
            v,
            road_id=e["road_id"],
            name=e["name"],
            distance_km=float(distance_km),
            geometry=e.get("geometry", [nodes[u], nodes[v]]),
        )
    return G


GRAPH = build_graph(CITY_NODES, CITY_EDGES)


# -------------------------------------------------------------
# Week 3 & 4: Traffic Data Simulation and Deterministic "Model"
# - We implement a rule-based predictor that mimics a trained regressor.
# - Input: hour (0-23), road_id, travel_mode
# - Output: traffic multiplier for travel time
# -------------------------------------------------------------

def predict_traffic_multiplier(road_id: str, hour: int, day_of_week: int, travel_mode: str) -> float:
    """
    Deterministic rule-based traffic multiplier.
    Higher multipliers during rush hours (8-10, 17-20) for motor vehicles.
    Slight variance by road_id to reflect different capacities.
    Walking is less affected by traffic.
    """
    rush_hours_morning = {8, 9, 10}
    rush_hours_evening = {17, 18, 19, 20}
    base = 1.0

    # Road capacity factor (stable per road)
    road_hash = sum(ord(c) for c in road_id)
    road_factor = 0.9 + (road_hash % 7) * 0.02  # between ~0.9 and ~1.02

    # Mode sensitivity to traffic
    if travel_mode == "Car":
        mode_factor = 1.0
    elif travel_mode == "Bike":
        mode_factor = 0.9
    else:  # Walk
        mode_factor = 0.7

    # Weekday/weekend profile
    is_weekend = day_of_week in {5, 6}  # Saturday=5, Sunday=6
    if not is_weekend and (hour in rush_hours_morning or hour in rush_hours_evening):
        congestion = 1.5  # significant congestion on weekdays at rush hours
    elif is_weekend and (12 <= hour <= 16):
        congestion = 1.25  # weekend midday peak
    elif 7 <= hour <= 22:
        congestion = 1.15  # moderate daytime traffic
    else:
        congestion = 0.9  # off-peak/late night

    multiplier = base * road_factor * (1 + (congestion - 1) * mode_factor)
    return max(0.7, round(multiplier, 3))


# -------------------------------------------------------------
# Week 5: Integrate "AI" with Route Planner
# - Weighted Dijkstra using travel time as weight
# -------------------------------------------------------------

def estimate_travel_time_minutes(distance_km: float, travel_mode: str, multiplier: float) -> float:
    if travel_mode == "Car":
        base_speed_kmh = 35.0
    elif travel_mode == "Bike":
        base_speed_kmh = 22.0
    else:  # Walk
        base_speed_kmh = 5.0
    hours = distance_km / max(1e-6, base_speed_kmh)
    minutes = hours * 60.0 * multiplier
    return minutes


def find_shortest_path(
    G: nx.Graph,
    source: str,
    target: str,
    hour: int,
    day_of_week: int,
    travel_mode: str,
) -> Tuple[Optional[List[str]], float, float, List[Dict]]:
    """
    Custom Dijkstra's algorithm (heapq) to minimize travel time.
    Returns (path_nodes, total_minutes, total_distance_km, segments_meta).
    segments_meta contains per-edge details for turn-by-turn and mapping.
    """
    import heapq

    if source not in G or target not in G:
        return None, float("inf"), 0.0, []
    if source == target:
        return [source], 0.0, 0.0, []

    # Priority queue of (accumulated_minutes, current_node)
    pq = [(0.0, source)]
    best_time: Dict[str, float] = {source: 0.0}
    best_dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Tuple[str, Dict]] = {}

    visited = set()
    while pq:
        acc_min, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            break

        for neighbor in G.neighbors(node):
            edge = G.get_edge_data(node, neighbor)
            # Single-edge graph (no multi-edges). If Graph becomes MultiGraph, handle keys.
            if not edge:
                continue
            road_id = edge.get("road_id", f"{node}-{neighbor}")
            distance_km = float(edge.get("distance_km", haversine_km(
                (G.nodes[node]["lat"], G.nodes[node]["lon"]),
                (G.nodes[neighbor]["lat"], G.nodes[neighbor]["lon"]),
            )))
            multiplier = predict_traffic_multiplier(road_id, hour, day_of_week, travel_mode)
            edge_minutes = estimate_travel_time_minutes(distance_km, travel_mode, multiplier)
            new_time = acc_min + edge_minutes
            if neighbor not in best_time or new_time < best_time[neighbor]:
                best_time[neighbor] = new_time
                best_dist[neighbor] = best_dist.get(node, 0.0) + distance_km
                prev[neighbor] = (
                    node,
                    {
                        "from": node,
                        "to": neighbor,
                        "road_id": road_id,
                        "name": edge.get("name", road_id),
                        "distance_km": distance_km,
                        "multiplier": multiplier,
                        "minutes": edge_minutes,
                        "geometry": edge.get("geometry"),
                    },
                )
                heapq.heappush(pq, (new_time, neighbor))

    if target not in best_time:
        return None, float("inf"), 0.0, []

    # Reconstruct path and segments
    path_nodes: List[str] = [target]
    segments: List[Dict] = []
    curr = target
    while curr != source:
        if curr not in prev:
            break
        pnode, meta = prev[curr]
        segments.append(meta)
        path_nodes.append(pnode)
        curr = pnode
    path_nodes.reverse()
    segments.reverse()

    total_minutes = best_time[target]
    total_distance = best_dist[target]
    return path_nodes, total_minutes, total_distance, segments


# -------------------------------------------------------------
# Week 6: Map Visualization with Folium
# - Only display the final computed route and start/end markers
# -------------------------------------------------------------

def render_route_map(
    G: nx.Graph,
    segments: List[Dict],
    start_node: str,
    end_node: str,
) -> folium.Map:
    # Compute map center (midpoint between start and end)
    s_lat, s_lon = G.nodes[start_node]["lat"], G.nodes[start_node]["lon"]
    e_lat, e_lon = G.nodes[end_node]["lat"], G.nodes[end_node]["lon"]
    center = ((s_lat + e_lat) / 2.0, (s_lon + e_lon) / 2.0)

    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="cartodbpositron")

    # Construct the continuous route shape from all segments
    route_shape = []
    
    for seg in segments:
        from_node = seg["from"]
        to_node = seg["to"]
        
        # Get the edge data directly from the graph
        edge_data = G.get_edge_data(from_node, to_node)
        if edge_data and "geometry" in edge_data:
            original_geom = edge_data["geometry"]
            
            # Check if we need to reverse the geometry based on direction
            first_point = original_geom[0]
            from_coord = (G.nodes[from_node]["lat"], G.nodes[from_node]["lon"])
            
            # Calculate distance from first geometry point to from_node
            dist_from_start = ((first_point[0] - from_coord[0])**2 + (first_point[1] - from_coord[1])**2)**0.5
            
            if dist_from_start < 0.01:  # If first point is close to from_node, use as-is
                segment_geom = original_geom
            else:  # Otherwise, reverse the geometry
                segment_geom = list(reversed(original_geom))
            
            # Add this segment's geometry to the route
            if route_shape:
                # Skip the first point if it's the same as the last point
                if abs(segment_geom[0][0] - route_shape[-1][0]) < 0.0001 and abs(segment_geom[0][1] - route_shape[-1][1]) < 0.0001:
                    route_shape.extend(segment_geom[1:])
                else:
                    route_shape.extend(segment_geom)
            else:
                route_shape.extend(segment_geom)
        else:
            # Fallback to straight line if no geometry
            route_shape.append((G.nodes[from_node]["lat"], G.nodes[from_node]["lon"]))
            route_shape.append((G.nodes[to_node]["lat"], G.nodes[to_node]["lon"]))

    # Draw the single continuous route line
    if route_shape:
        folium.PolyLine(
            route_shape, 
            color="#1a73e8", 
            weight=6, 
            opacity=0.8,
            popup="Calculated Route"
        ).add_to(fmap)

    # Start and end markers
    folium.Marker(
        location=(s_lat, s_lon),
        popup=f"Start: {start_node}",
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)
    folium.Marker(
        location=(e_lat, e_lon),
        popup=f"End: {end_node}",
        tooltip="End",
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(fmap)

    return fmap


# -------------------------------------------------------------
# Week 7 & 8: Streamlit UI, Testing, and Final Touches
# -------------------------------------------------------------

st.title("AI-Powered Traffic Predictor and Route Planner")
st.caption("Miniature urban network in Hyderabad/Secunderabad with a rule-based traffic model")

# Initialize session state for persistent map
if 'route_map_html' not in st.session_state:
    st.session_state.route_map_html = None
if 'last_route_params' not in st.session_state:
    st.session_state.last_route_params = None

with st.sidebar:
    st.header("Plan Your Trip")
    start_location = st.selectbox("Start", options=list(CITY_NODES.keys()), index=0)
    end_location = st.selectbox("End", options=list(CITY_NODES.keys()), index=10)  # Hitech City
    hour = st.slider("Hour of day", min_value=0, max_value=23, value=9, step=1)
    day_of_week = st.selectbox("Day of week (Mon=0)", options=list(range(7)), index=0, help="0=Mon, 6=Sun")
    travel_mode = st.radio("Travel mode", options=["Car", "Bike", "Walk"], index=0, horizontal=True)
    go = st.button("Calculate Fastest Route", type="primary", use_container_width=True)


col1, col2 = st.columns([0.45, 0.55], gap="large")

# Check if route parameters have changed
current_route_params = (start_location, end_location, hour, day_of_week, travel_mode)
route_params_changed = st.session_state.last_route_params != current_route_params

if go:
    if start_location == end_location:
        with col1:
            st.info("Start and end are the same. Estimated time is 0 minutes.")
            st.metric(label="ETA (min)", value="0.0")
            st.metric(label="Distance (km)", value="0.0")
        with col2:
            # Show a small map centered at the point
            single_map = folium.Map(location=CITY_NODES[start_location], zoom_start=14, tiles="cartodbpositron")
            folium.Marker(CITY_NODES[start_location], tooltip="Start/End", icon=folium.Icon(color="blue")).add_to(single_map)
            st.session_state.last_route_params = current_route_params
            st_folium(single_map, height=520, width=None, key="same_location_map", returned_objects=[])
    else:
        # Compute best path based on predicted traffic-adjusted travel times
        path, total_minutes, total_km, segments = find_shortest_path(
            GRAPH, start_location, end_location, hour, day_of_week, travel_mode
        )

        if not path:
            with col1:
                st.error("No route found between the selected points.")
            with col2:
                # Show overview map if no route found
                all_lats = [lat for lat, _ in CITY_NODES.values()]
                all_lons = [lon for _, lon in CITY_NODES.values()]
                center = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons))
                overview_map = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
                for name, (lat, lon) in CITY_NODES.items():
                    folium.Marker((lat, lon), popup=name, tooltip=name, icon=folium.Icon(color="gray")).add_to(overview_map)
                st_folium(overview_map, height=520, width=None, key="no_route_map", returned_objects=[])
        else:
            with col1:
                st.subheader("Results")
                st.metric(label="ETA (min)", value=f"{total_minutes:.1f}")
                st.metric(label="Distance (km)", value=f"{total_km:.2f}")

                with st.expander("Turn-by-turn and segment details", expanded=True):
                    for i, seg in enumerate(segments, start=1):
                        st.write(
                            f"{i}. {seg['name']} — {seg['distance_km']:.2f} km, "
                            f"multiplier {seg['multiplier']:.2f}, {seg['minutes']:.1f} min"
                        )

                # Conceptual comparison (distance-only vs time-based). We show distance-only ETA as baseline.
                st.divider()
                st.caption("Comparison (conceptual): distance-only vs traffic-aware time")
                # Compute baseline: treat multiplier=1 for all edges and Car baseline speed
                def baseline_time_minutes(distance_km: float) -> float:
                    return estimate_travel_time_minutes(distance_km, travel_mode, 1.0)
                baseline_eta = baseline_time_minutes(total_km)
                st.write(
                    f"Baseline (no traffic): ~{baseline_eta:.1f} min vs Traffic-aware: ~{total_minutes:.1f} min"
                )

            with col2:
                # Render route map
                fmap = render_route_map(GRAPH, segments, start_location, end_location)
                st.session_state.last_route_params = current_route_params
                st.subheader("Map")
                st_folium(fmap, height=520, width=None, key="route_map", returned_objects=[])
else:
    with col1:
        st.info("Set your start/end, choose hour and mode, then click Calculate Fastest Route.")
    with col2:
        # Display a neutral city overview map
        all_lats = [lat for lat, _ in CITY_NODES.values()]
        all_lons = [lon for _, lon in CITY_NODES.values()]
        center = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons))
        overview_map = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
        
        # Add location markers
        for name, (lat, lon) in CITY_NODES.items():
            folium.Marker((lat, lon), popup=name, tooltip=name, icon=folium.Icon(color="gray")).add_to(overview_map)
        
        st_folium(overview_map, height=520, width=None, key="overview_map", returned_objects=[])


