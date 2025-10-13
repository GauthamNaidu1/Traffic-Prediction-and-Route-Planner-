# AI-Powered Route Planner

A Streamlit web app that predicts traffic and computes the fastest route over a miniature urban road network (Hyderabad/Secunderabad sample). It uses a deterministic, rule-based function as a stand-in for a trained regression model to estimate traffic multipliers by hour and travel mode, then runs Dijkstra's algorithm over a NetworkX graph to find the time-optimal path. Folium renders the final route map with clear start/end markers.

## Features

- **Traffic-aware routing**: Predicts edge-wise traffic multipliers (hour, road, mode) and minimizes travel time, not just distance.
- **Weekday/weekend sensitivity**: Traffic predictor accounts for weekdays vs weekends (rush hours vs weekend peaks).
- **Multiple travel modes**: Car, Bike, Walk with different base speeds and traffic sensitivity.
- **Deterministic predictor**: Rule-based function simulates a trained regression model for reproducible results.
- **Clean map visualization**: Folium map shows only the final route and start/end markers to avoid network clutter.
- **Turn-by-turn details**: Displays segment-level distance, multiplier, and minutes.
- **Interactive UI**: Streamlit sidebar inputs for start, end, hour of day, day of week, and travel mode.

## Core Technologies

- **Streamlit**: UI and web app runner
- **NetworkX**: Graph modeling and Dijkstra’s algorithm
- **Folium**: Map rendering of the computed route
- **streamlit-folium**: Embedding Folium map inside Streamlit
- **Python**: End-to-end application logic

## How to Set Up and Run Locally

### Prerequisites

- Python 3.9+
- Git

### Steps

```bash
# 1) Clone the repository
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

# 2) (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run app.py
```

The app will open in your default browser (or visit the printed local URL). Select your start/end, choose an hour, day of week, and mode, and click "Calculate Fastest Route" to view ETA, distance, turn-by-turn information, and a clean map of the final route.

## 8-Week Plan Mapping

- **Week 1**: Scoped a city district, studied Dijkstra/NetworkX, created project structure (captured in comments and code).
- **Week 2**: Built a small road network graph with nodes/edges and geometry.
- **Week 3**: Simulated traffic data mapping by hour, road, and mode.
- **Week 4**: Implemented a deterministic predictor to mimic a trained model.
- **Week 5**: Integrated predictions into travel-time edge weights; used Dijkstra.
- **Week 6**: Rendered a clean Folium map with route and markers.
- **Week 7**: Tested UI inputs and handled edge cases (no route, start=end).
- **Week 8**: Finalized documentation and app structure.

## Notes

- The predictor is deterministic and rule-based for demo purposes. Replace `predict_traffic_multiplier` with a trained model as a future enhancement.
- The road network is a small, hand-crafted sample. You can swap it with OSM-derived data (e.g., via `osmnx`) and extend the app accordingly.
