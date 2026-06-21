ParkGuard AI: Parking Violation & Congestion Intelligence

Transforming reactive parking enforcement into proactive, quantifiable urban mobility optimization.

ParkGuard AI uses spatial intelligence and machine learning to analyze illegal parking hotspots, predict future gridlocks, and calculate the economic ROI of targeted law enforcement deployment. Designed specifically to map raw city parking data directly to downstream congestion degradation.

✨ Features

Spatiotemporal Hotspotting: Uses density-based algorithms (HDBSCAN/DBSCAN) to identify precise "Congestion Shadows" based on violation clustering and persistence.

7-Day Risk Forecasting: Random Forest ML engine predicts upcoming violation spikes using historical temporal features (hour, day of week, weekend status).

Economic ROI Simulator: Allows city planners to simulate officer deployment and calculate the civic time-value saved vs. enforcement operational costs.

Dual-View User Interface: Toggle between sanitized, anonymized public heatmaps and an Internal Police feed showing raw validation points and officer IDs.

Highly Optimized Engine: Uses vectorized Pandas functions and chunk sampling to smoothly handle hundreds of thousands of coordinates in an interactive browser layout.

🚀 How to Run Locally

Clone the repository:

git clone [https://github.com/yourusername/ParkGuard-AI.git](https://github.com/yourusername/ParkGuard-AI.git)
cd parkguard-ai


Install the dependencies:
Ensure you have Python 3.9+ installed, then run:

pip install -r requirements.txt


Provide the Dataset:
Place your urban parking violation dataset inside the root folder. The script natively looks for:

jan to may police violation_anonymized791b166.csv

Or compressed_data.csv

Note: If the dataset is too large for GitHub, it should be ignored via .gitignore and uploaded manually via the Streamlit sidebar during runtime.

Run the Application:

python -m streamlit run app.py


🛠️ Tech Stack

Frontend/Dashboard: Streamlit

Geospatial & Visualizations: Plotly Express, Plotly Graph Objects

Data Processing: Pandas, NumPy

Machine Learning: scikit-learn (HDBSCAN, RandomForestRegressor)
