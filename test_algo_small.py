import networkx as nx
import osmnx as ox
from app import traffic_aware_astar_path, LANDMARKS

# Mocking load_graph to use a smaller area
def load_small_graph():
    print("Loading small graph (Banjara Hills)...")
    place = "Banjara Hills, Hyderabad, India"
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G

def test_astar_small():
    G = load_small_graph()
    print("Graph loaded.")

    # Use coordinates within Banjara Hills
    # Banjara Hills: (17.4156, 78.4456)
    # Nearby: KBR Park (approx 17.424, 78.419) - might be outside small graph
    # Let's pick two nodes in the graph
    
    nodes = list(G.nodes(data=True))
    start_node = nodes[0][0]
    end_node = nodes[-1][0]
    
    print(f"Finding path from {start_node} to {end_node}...")
    
    path, time, dist, segments = traffic_aware_astar_path(
        G, start_node, end_node, hour=9, day_idx=0, mode="Car"
    )

    if path:
        print("Path found!")
        print(f"Total Time: {time:.2f} min")
        print(f"Total Distance: {dist:.2f} km")
        print(f"Number of segments: {len(segments)}")
    else:
        print("No path found (might be disconnected in small subgraph)!")

if __name__ == "__main__":
    test_astar_small()
