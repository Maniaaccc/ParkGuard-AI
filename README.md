🚦 ParkGuard AI MVP

ParkGuard AI MVP is an advanced, AI-driven spatial and temporal intelligence dashboard built with Streamlit. Designed for urban traffic enforcement and city planning, it quantifies spatial parking externalities in Bengaluru to prioritize law enforcement operations, predict gridlock, and calculate the economic ROI of patrol deployments.

🌟 Key Features

The dashboard is divided into specialized data engines accessible via an interactive tabbed interface:

📍 Prioritized Hotspots (Spatial Intelligence): Uses HDBSCAN (or DBSCAN as a fallback) to cluster geospatial violation data. It isolates high-impact zones by combining violation volume, time decay (persistence), and junction proximity.

🔮 Temporal Forecast: Employs a RandomForestRegressor to capture historical periodic trends, predicting hourly violation spikes and warning of upcoming gridlock risks over the next 7 days.

⛓️ Spillover Network: Calculates a directional cascade matrix. It maps sequential congestion ripples within a 30-minute window to predict where gridlock will spill over next if a specific junction bottlenecks.

🚛 Fleet Recidivism: Profiles vehicle identities and classifications to separate localized personal parking errors from systemic, commercial repeat offenders.

👮 Patrol Efficiency: Maps actual officer logs against AI-generated cluster demand to analyze enforcement optimization and highlight tactical operational gaps.

💡 ROI & SUMO Simulator: Simulates "What-If" scenarios. Calculates the economic return on enforcement operations (civic time value saved vs. officer deployment cost) and provides a localized gauge of junction delay reductions.

👁️ Dual-View Authentication: Toggles between an anonymized Public Demo View and a secure Internal Police View that reveals sensitive data like license plates and officer IDs.

🛠️ Tech Stack

Framework: Streamlit

Data Processing: Pandas, NumPy, Python AST, Datetime

Machine Learning / AI: Scikit-Learn (RandomForestRegressor, DBSCAN), HDBSCAN (Optional but recommended for optimal clustering)

Data Visualization: Plotly Express, Plotly Graph Objects

🚀 Installation & Setup

1. Clone the repository:

git clone [https://github.com/Maniaaccc/ParkGuard-AI.git](https://github.com/Maniaaccc/ParkGuard-AI.git)
cd ParkGuard-AI


2. Create and activate a virtual environment (Optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


3. Install required dependencies:
Ensure you have scikit-learn >= 1.3.0 installed to utilize native HDBSCAN.

pip install streamlit pandas numpy plotly scikit-learn


(Note: If you are using an older version of scikit-learn, install the standalone hdbscan package via pip install hdbscan or the app will automatically fall back to standard DBSCAN.)

4. Run the Streamlit app:

streamlit run app.py


📊 Data Requirements

To run the application, you need to upload a CSV dataset containing parking violation records. The engine is optimized for Bengaluru coordinates by default.

Expected CSV Columns:

latitude (float)

longitude (float)

created_datetime (datetime string)

violation_type (string)

vehicle_number (string)

vehicle_type (string)

junction_name (string - use "No Junction" if not applicable)

police_station (string)

created_by_id (string/int representing the officer)

⚙️ Configuration & Tuning

You can adjust the engine's behavior directly from the Streamlit sidebar:

Filters: Isolate data by specific Police Stations or time of day (Hour Range).

Algorithm Params: Adjust the Min Violations for Hotspot slider to tune the strictness of the spatial clustering algorithm.

Built to bring intelligence to urban enforcement and alleviate city gridlock.
