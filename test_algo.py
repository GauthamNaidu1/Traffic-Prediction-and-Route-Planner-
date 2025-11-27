import networkx as nx
import osmnx as ox
from app import traffic_aware_astar_path, load_graph, LANDMARKS

def test_astar():
    print("Loading graph...")
    G = load_graph()
    print("Graph loaded.")

    start_coord = LANDMARKS["Secunderabad Railway Station"]
    end_coord = LANDMARKS["Hitech City"]

    print(f"Finding path from {start_coord} to {end_coord}...")
    source_node = ox.distance.nearest_nodes(G, X=start_coord[1], Y=start_coord[0])
    target_node = ox.distance.nearest_nodes(G, X=end_coord[1], Y=end_coord[0])

    path, time, dist, segments = traffic_aware_astar_path(
        G, source_node, target_node, hour=9, day_idx=0, mode="Car"
    )

    if path:
        print("Path found!")
        print(f"Total Time: {time:.2f} min")
        print(f"Total Distance: {dist:.2f} km")
        print(f"Number of segments: {len(segments)}")
        assert len(path) > 0
        assert time > 0
        assert dist > 0
    else:
        print("No path found!")
        exit(1)

if __name__ == "__main__":
    test_astar()
