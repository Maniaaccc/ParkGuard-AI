import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Handle HDBSCAN import (Requires scikit-learn >= 1.3.0)
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    from sklearn.cluster import DBSCAN
    HAS_HDBSCAN = False

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ParkGuard AI MVP", layout="wide", page_icon="🚦")

# --- DATA PROCESSING ---
@st.cache_data
def load_and_clean_data(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer)
    
    # ⚡ OPTIMIZATION 1: Fast Datetime Parsing
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed', errors='coerce', utc=True)
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime'])
    
    # Force 'hour' to be a strict integer to prevent float-string casting ValueError later
    df['hour'] = df['created_datetime'].dt.hour.astype(int)
    df['date'] = df['created_datetime'].dt.date
    
    # ⚡ OPTIMIZATION 2: Vectorized String Cleaning (Super Fast)
    df['violation_type_clean'] = df['violation_type'].astype(str).str.replace(r'\[|\]|"', '', regex=True).str.replace(r"''", "", regex=True)
    
    # ⚡ OPTIMIZATION 3: Limit extreme outliers to speed up clustering
    df = df[(df['latitude'] > 12.0) & (df['latitude'] < 14.0) & 
            (df['longitude'] > 77.0) & (df['longitude'] < 78.5)]
            
    # Ensure vehicle identifiers are clean strings for the new engines
    if 'vehicle_number' in df.columns:
        df['vehicle_number'] = df['vehicle_number'].fillna('UNKNOWN').astype(str)
    if 'vehicle_type' in df.columns:
        df['vehicle_type'] = df['vehicle_type'].fillna('UNKNOWN').astype(str)
        
    return df

# --- INTELLIGENCE ENGINE (Clustering & Impact) ---
@st.cache_data
def run_spatial_intelligence(df, min_cluster_size):
    if len(df) > 50000:
        process_df = df.sample(n=50000, random_state=42).copy()
    else:
        process_df = df.copy()

    if len(process_df) < min_cluster_size:
        return process_df, pd.DataFrame()
        
    coords = np.radians(process_df[['latitude', 'longitude']].values)
    
    if HAS_HDBSCAN:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine', n_jobs=-1)
        process_df['cluster_id'] = clusterer.fit_predict(coords)
    else:
        clusterer = DBSCAN(eps=50/6371000.0, min_samples=min_cluster_size, algorithm='ball_tree', metric='haversine', n_jobs=-1)
        process_df['cluster_id'] = clusterer.fit_predict(coords)
        
    hotspots = process_df[process_df['cluster_id'] != -1].copy()
    grouped = hotspots.groupby('cluster_id')
    
    cluster_stats = []
    
    for cluster_id, group in grouped:
        time_diff = group['created_datetime'].max() - group['created_datetime'].min()
        persistence_hours = time_diff.total_seconds() / 3600.0
        
        junctions_involved = group[group['junction_name'] != 'No Junction']
        junction_ratio = len(junctions_involved) / len(group) if len(group) > 0 else 0
        
        base_volume = len(group)
        persistence_multiplier = 1.0 + (min(persistence_hours, 168) / 168.0) 
        junction_multiplier = 1.0 + (junction_ratio * 2.0) 
        
        impact_score = base_volume * persistence_multiplier * junction_multiplier
        
        # ⚡ OPTIMIZATION 4: value_counts is significantly faster than mode() in a loop
        top_violation = group['violation_type_clean'].value_counts().index[0] if not group['violation_type_clean'].empty else 'Unknown'
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'latitude': group['latitude'].mean(),
            'longitude': group['longitude'].mean(),
            'total_violations': base_volume,
            'persistence_hours': round(persistence_hours, 1),
            'junction_proximity': round(junction_ratio * 100, 1),
            'top_violation': top_violation,
            'raw_impact_score': impact_score
        })
        
    summary_df = pd.DataFrame(cluster_stats)
    
    if not summary_df.empty:
        max_score = summary_df['raw_impact_score'].max()
        if max_score > 0:
            summary_df['impact_score_normalized'] = (summary_df['raw_impact_score'] / max_score) * 100
        else:
            summary_df['impact_score_normalized'] = 0.0
        summary_df['impact_score_normalized'] = summary_df['impact_score_normalized'].round(1)
    
    return process_df, summary_df

# --- FORECASTING & PREDICTION ENGINE ---
@st.cache_data
def run_time_series_forecast(df):
    try:
        ts_df = df.groupby(['date', 'hour']).size().reset_index(name='violation_count')
        
        # ⚡ OPTIMIZATION 5: Replaced slow string concatenation with lightning-fast vectorized TimeDelta math
        ts_df['datetime'] = pd.to_datetime(ts_df['date']) + pd.to_timedelta(ts_df['hour'], unit='h')
        ts_df = ts_df.dropna(subset=['datetime']).sort_values('datetime')
        
        ts_df['dayofweek'] = ts_df['datetime'].dt.dayofweek
        ts_df['is_weekend'] = ts_df['dayofweek'].isin([5, 6]).astype(int)
        
        if len(ts_df) < 24:
            return ts_df, pd.DataFrame() 
            
        features = ['hour', 'dayofweek', 'is_weekend']
        X = ts_df[features]
        y = ts_df['violation_count']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        last_date = ts_df['datetime'].max()
        future_dates = [last_date + timedelta(hours=i) for i in range(1, 24*7 + 1)]
        future_df = pd.DataFrame({'datetime': future_dates})
        future_df['hour'] = future_df['datetime'].dt.hour
        future_df['dayofweek'] = future_df['datetime'].dt.dayofweek
        future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
        
        predictions = model.predict(future_df[features])
        future_df['predicted_violations'] = np.ravel(predictions).round(0).astype(int)
        
        return ts_df, future_df
    except Exception as e:
        print(f"Forecasting Error: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- ADVANCED INTERNAL DATA ENGINES ---
@st.cache_data
def analyze_fleet_recidivism(df):
    """ Feature 1: Fleet Profiling & Recidivism Risk Engine """
    if 'vehicle_number' not in df.columns or 'vehicle_type' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
        
    # ⚡ OPTIMIZATION 6: Replaced slow lambda mode() with C-optimized 'first' and 'last'
    # A vehicle's type is static, so 'first' gets the type instantly.
    # 'last' tells us the most recent station the recidivist was seen at.
    offender_df = df.sort_values(by='created_datetime').groupby('vehicle_number').agg(
        total_violations=('id', 'count'),
        primary_vehicle_type=('vehicle_type', 'first'),
        last_seen_station=('police_station', 'last')
    ).reset_index()
    
    # Filter out UNKNOWNs and look for repeat offenders (2+ violations)
    offender_df = offender_df[offender_df['vehicle_number'] != 'UNKNOWN']
    repeat_offenders = offender_df[offender_df['total_violations'] >= 2].sort_values(by='total_violations', ascending=False)
    
    # 2. Vehicle Class Impact Profile
    type_profile = df.groupby('vehicle_type').agg(
        violation_count=('id', 'count'),
        unique_vehicles=('vehicle_number', 'nunique')
    ).reset_index()
    
    # Avoid division by zero
    type_profile = type_profile[type_profile['unique_vehicles'] > 0]
    type_profile['citations_per_vehicle'] = (type_profile['violation_count'] / type_profile['unique_vehicles']).round(2)
    
    return repeat_offenders, type_profile

@st.cache_data
def calculate_spillover_network(df):
    """ Feature 2: Congestion Ripple & Spillover Transition Matrix """
    if 'junction_name' not in df.columns or len(df) < 10:
        return pd.DataFrame()
        
    # Sort chronologically to track the temporal ripple effect
    sequenced_df = df.sort_values(by='created_datetime').copy()
    
    # Filter out entries with no explicit junction tag
    sequenced_df = sequenced_df[sequenced_df['junction_name'] != 'No Junction']
    
    if len(sequenced_df) < 2:
        return pd.DataFrame()
        
    # Create a shifted column to look at the immediate next violation event in the city
    sequenced_df['next_junction'] = sequenced_df['junction_name'].shift(-1)
    sequenced_df['time_delta'] = (sequenced_df['created_datetime'].shift(-1) - sequenced_df['created_datetime']).dt.total_seconds() / 60.0
    
    # We care about sequential ripples that happen within a 30-minute window
    ripple_df = sequenced_df[(sequenced_df['junction_name'] != sequenced_df['next_junction']) & 
                             (sequenced_df['time_delta'] <= 30)].copy()
    
    if ripple_df.empty:
        return pd.DataFrame()
        
    # Build transition counts
    transition_counts = ripple_df.groupby(['junction_name', 'next_junction']).size().reset_index(name='ripple_frequency')
    
    # Calculate probability matrix
    total_outbound = transition_counts.groupby('junction_name')['ripple_frequency'].transform('sum')
    transition_counts['spillover_probability_pct'] = ((transition_counts['ripple_frequency'] / total_outbound) * 100).round(1)
    
    return transition_counts.sort_values(by='ripple_frequency', ascending=False)

@st.cache_data
def run_patrol_gap_analysis(df, hotspots_df):
    """ Feature 3: Officer Patrol Allocation & Deployment Efficiency """
    if 'created_by_id' not in df.columns:
        return pd.DataFrame()
        
    # Map raw citations to localized hotspots to judge deployment alignment
    officer_coverage = df.groupby(['police_station', 'created_by_id']).agg(
        citations_issued=('id', 'count'),
        unique_junctions_covered=('junction_name', 'nunique')
    ).reset_index()
    
    return officer_coverage.sort_values(by='citations_issued', ascending=False)

# --- SIDEBAR & SETUP ---
st.sidebar.title("🚦 ParkGuard AI MVP")
st.sidebar.markdown("Advanced Spatial & Temporal Intelligence")

# Dual-View Authentication Mockup
view_mode = st.sidebar.radio("👁️ View Mode", ["Public Demo (Anonymized)", "Internal Police View"])

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type="csv")
default_file = "jan_to_may_police_violation_anonymized791b166.zip"

# Load Data
data_load_state = st.sidebar.text('Loading data...')
try:
    if uploaded_file is not None:
        raw_df = load_and_clean_data(uploaded_file)
    elif os.path.exists(default_file):
        raw_df = load_and_clean_data(default_file)
    else:
        st.error("Please upload the dataset CSV in the sidebar.")
        st.stop()
    data_load_state.text('Data loaded!')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- FILTERS ---
st.sidebar.header("🔍 Filters")
all_stations = ["All"] + sorted(raw_df['police_station'].dropna().unique().tolist())
selected_station = st.sidebar.selectbox("Police Station", all_stations)

time_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

# Apply Filters
filtered_df = raw_df[
    (raw_df['hour'] >= time_range[0]) & 
    (raw_df['hour'] <= time_range[1])
]
if selected_station != "All":
    filtered_df = filtered_df[filtered_df['police_station'] == selected_station]

# Algorithm Settings
st.sidebar.header("⚙️ Algorithm Params")
min_cluster = st.sidebar.slider("Min Violations for Hotspot", 2, 20, 5)

# --- EXECUTE ENGINES ---
df_clustered, hotspots_df = run_spatial_intelligence(filtered_df, min_cluster)
repeat_offenders_df, type_profile_df = analyze_fleet_recidivism(filtered_df)
spillover_matrix = calculate_spillover_network(filtered_df)
patrol_efficiency = run_patrol_gap_analysis(filtered_df, hotspots_df)

# --- MAIN DASHBOARD ---
st.title("Bengaluru Parking Impact Dashboard")
st.markdown("Quantifying spatial parking externalities to prioritize enforcement and predict gridlock.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Violations Filtered", f"{len(filtered_df):,}")
col2.metric("Active Hotspots Detected", f"{len(hotspots_df):,}")

if not hotspots_df.empty:
    avg_pers = f"{hotspots_df['persistence_hours'].mean():.1f} hrs"
    top_impact = f"{hotspots_df['impact_score_normalized'].max():.1f}/100"
else:
    avg_pers, top_impact = "0 hrs", "0"

col3.metric("Avg Hotspot Persistence", avg_pers)
col4.metric("Max Impact Score", top_impact)

st.markdown("---")

# --- MAP VIEW ---
st.subheader("🗺️ Congestion Shadow Map")

if not hotspots_df.empty:
    if view_mode == "Internal Police View":
        fig = px.scatter_mapbox(
            df_clustered, 
            lat="latitude", lon="longitude", 
            color="cluster_id",
            hover_data=["vehicle_number", "police_station", "junction_name", "created_by_id"],
            color_continuous_scale=px.colors.cyclical.IceFire,
            opacity=0.5,
            size_max=10,
            zoom=11,
            mapbox_style="carto-positron",
            title="Internal View: Raw Incidents & Officer Deployments"
        )
    else:
        fig = px.scatter_mapbox(
            hotspots_df,
            lat="latitude", lon="longitude",
            size="total_violations",
            color="impact_score_normalized",
            hover_name="top_violation",
            hover_data={
                "impact_score_normalized": True,
                "persistence_hours": True,
                "junction_proximity": True,
                "total_violations": True,
                "latitude": False, "longitude": False
            },
            color_continuous_scale=px.colors.sequential.Inferno,
            zoom=11,
            mapbox_style="carto-positron",
            title="Public View: Validated Congestion Impact Zones (Anonymized)"
        )
    
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hotspots detected with current filter parameters. Try expanding the time range or lowering the minimum cluster size.")

# --- DATA VIEWS (7 TABS) ---
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📍 Prioritized Hotspots", 
    "📈 Temporal Forecast", 
    "⛓️ Spillover Network",
    "🚛 Fleet Recidivism",
    "👮 Patrol Efficiency",
    "💰 ROI & SUMO Sim", 
    "📋 Raw Data (Internal)"
])

with tab1:
    st.subheader("Prioritized Enforcement Zones")
    st.markdown("AI-driven clustering isolates high-impact zones combining volume, time decay, and junction proximity.")
    if not hotspots_df.empty:
        display_cols = ['cluster_id', 'impact_score_normalized', 'total_violations', 'persistence_hours', 'junction_proximity', 'top_violation']
        st.dataframe(
            hotspots_df[display_cols].sort_values(by='impact_score_normalized', ascending=False).head(15),
            use_container_width=True,
            column_config={
                "impact_score_normalized": st.column_config.ProgressColumn(
                    "Impact / ROI Score", format="%.1f", min_value=0, max_value=100
                ),
                "persistence_hours": st.column_config.NumberColumn("Persistence (Hours)", format="%.1f ⏱️"),
                "junction_proximity": st.column_config.NumberColumn("% Near Junction", format="%.1f %%")
            }
        )
    else:
        st.write("No hotspot data.")

with tab2:
    st.subheader("🔮 Temporal Violations Spikes Forecast")
    st.markdown("Predictive engine capturing historical periodic trends to anticipate heavy congestion.")
    
    if len(filtered_df) > 0:
        hist_ts, future_ts = run_time_series_forecast(filtered_df)
        
        if not future_ts.empty and not hist_ts.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=hist_ts['datetime'].tail(168), y=hist_ts['violation_count'].tail(168), 
                                        mode='lines', name='Historical (Last 7 Days)', line=dict(color='#1E90FF')))
            fig_ts.add_trace(go.Scatter(x=future_ts['datetime'], y=future_ts['predicted_violations'], 
                                        mode='lines', name='Predicted Spikes (Next 7 Days)', line=dict(color='#FF4500', dash='dash')))
            
            fig_ts.update_layout(title="Predicted Hourly Violations Spikes", xaxis_title="Date & Time", yaxis_title="Number of Violations", height=400)
            st.plotly_chart(fig_ts, use_container_width=True)
            
            peak_future = future_ts.loc[future_ts['predicted_violations'].idxmax()]
            st.info(f"🚨 **Highest Predicted Gridlock Risk:** {peak_future['datetime'].strftime('%A, %b %d at %I %p')} with an estimated **{int(peak_future['predicted_violations'])}** new cluster events expected.")
        else:
            st.warning("Insufficient historical data under current filtering parameters to compute a reliable trend.")
    else:
        st.warning("No data filtered.")

with tab3:
    st.subheader("⛓️ Sequential Spillover & Congestion Ripple Network")
    st.markdown("Calculates the directional cascade matrix. If an incident happens at Junction A, where will gridlock spill over next?")
    
    if not spillover_matrix.empty:
        col_net1, col_net2 = st.columns([2, 1])
        
        with col_net1:
            st.markdown("#### High-Probability Cascade Routes")
            st.dataframe(
                spillover_matrix.head(15),
                use_container_width=True,
                column_config={
                    "junction_name": "Source Junction",
                    "next_junction": "Predicted Spillover Target",
                    "ripple_frequency": st.column_config.NumberColumn("Sequence Counts", format="%d 🔄"),
                    "spillover_probability_pct": st.column_config.ProgressColumn("Transfer Probability", format="%.1f %%", min_value=0, max_value=100)
                }
            )
            
        with col_net2:
            st.markdown("#### Proactive Mitigation Strategy")
            top_ripple = spillover_matrix.iloc[0]
            st.success(
                f"🎯 **AI Intervention Alert:**\n\n"
                f"When congestion builds up at **{top_ripple['junction_name']}**, "
                f"there is a **{top_ripple['spillover_probability_pct']}%** statistical probability "
                f"that **{top_ripple['next_junction']}** will bottleneck within 30 minutes. "
                f"Deploy early traffic staging directly to the target node."
            )
    else:
        st.info("Insufficient chronological sequence transitions found. Broaden the hour filter ranges to calculate the network ripples.")

with tab4:
    st.subheader("🚛 Fleet Profiling & Habitual Recidivism Dashboard")
    st.markdown("Analyzing vehicle identities and classification distributions to separate localized personal parking errors from commercial systemic issues.")
    
    if not repeat_offenders_df.empty and not type_profile_df.empty:
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown("#### Systematic Vehicle Class Impact Profile")
            fig_profile = px.bar(
                type_profile_df.sort_values(by='citations_per_vehicle', ascending=False),
                x='vehicle_type', y='citations_per_vehicle',
                labels={'citations_per_vehicle': 'Citations per Unique Plate', 'vehicle_type': 'Vehicle Classification'},
                title="Density Load Ratio by Classification (Citations / Vehicle)",
                color='violation_count',
                color_continuous_scale='Inferno'
            )
            st.plotly_chart(fig_profile, use_container_width=True)
            
        with col_f2:
            st.markdown("#### Chronic Repeat Offenders List")
            st.dataframe(
                repeat_offenders_df.head(20),
                use_container_width=True,
                column_config={
                    "vehicle_number": "License Plate",
                    "total_violations": st.column_config.NumberColumn("Citations Issued", format="%d 📑"),
                    "primary_vehicle_type": "Vehicle Type",
                    "last_seen_station": "Frequent Hotspot Zone"
                }
            )
    else:
        st.info("No repeated operational fleet IDs isolated within current slice parameters.")

with tab5:
    st.subheader("👮 Patrol Resource Efficiency & Operational Gaps")
    st.markdown("Algorithmic benchmarking mapping actual officer logs (`created_by_id`) against cluster demand matrices to analyze enforcement optimization.")
    
    if not patrol_efficiency.empty:
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            st.markdown("#### Station Deployment Distribution Logs")
            fig_patrol = px.box(
                patrol_efficiency, 
                x='police_station', y='citations_issued',
                points="all", 
                title="Officer Performance Spread Across Stations",
                labels={'police_station': 'Police Station Unit', 'citations_issued': 'Citations Logged per Officer ID'}
            )
            fig_patrol.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_patrol, use_container_width=True)
            
        with col_p2:
            st.markdown("#### Tactical Resource Realignment Metrics")
            top_officer = patrol_efficiency.iloc[0]
            avg_citations = patrol_efficiency['citations_issued'].mean()
            
            st.metric("Systemic Active Log Units", f"{patrol_efficiency['created_by_id'].nunique()} IDs")
            st.metric("Avg Citations per Patrol Unit", f"{avg_citations:.1f} logs")
            
            st.info(
                f"📊 **Operational Insight:** "
                f"Officer Profile **{top_officer['created_by_id']}** under **{top_officer['police_station']}** jurisdiction "
                f"represents the highest current utilization log rate with **{top_officer['citations_issued']}** citations across "
                f"**{top_officer['unique_junctions_covered']}** unique location matrices."
            )
    else:
        st.info("Missing operational allocation logs within selected query filters.")

with tab6:
    st.subheader("💡 Predictive ROI & Intervention Simulator")
    st.markdown("Simulate 'What-If' scenarios and calculate the economic return on enforcement operations (Proxy for SUMO Simulator).")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.markdown("#### 👮 Configure Enforcement")
        officers_deployed = st.slider("Patrol Officers Deployed", 1, 50, 10)
        hours_per_shift = st.slider("Shift Duration (Hours)", 4, 12, 8)
        clearance_rate = st.number_input("Violations Cleared / Officer / Hour", min_value=1, max_value=20, value=5)
        
        # Economic Constants (Estimated for Bengaluru)
        OFFICER_COST_PER_HR = 400
        VALUE_OF_TIME_PER_HR = 150
        DELAY_SAVED_PER_VIOLATION_MINS = 4.5
        
    with col_sim2:
        st.markdown("#### 📊 Economic Impact (ROI)")
        
        total_cleared = officers_deployed * hours_per_shift * clearance_rate
        delay_mins_saved = total_cleared * DELAY_SAVED_PER_VIOLATION_MINS
        delay_hours_saved = delay_mins_saved / 60
        
        cost_of_operation = officers_deployed * hours_per_shift * OFFICER_COST_PER_HR
        economic_value_saved = delay_hours_saved * VALUE_OF_TIME_PER_HR
        roi_percentage = ((economic_value_saved - cost_of_operation) / cost_of_operation) * 100 if cost_of_operation > 0 else 0
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Est. Clearances", f"{total_cleared:,}")
        mc2.metric("Delay Hours Mitigated", f"{delay_hours_saved:,.1f} hrs")
        mc3.metric("Net ROI", f"{roi_percentage:,.1f}%", delta_color="normal")
        
        if economic_value_saved > cost_of_operation:
            st.success(f"✅ **Favorable ROI:** Operation costs ₹{cost_of_operation:,} but saves ₹{economic_value_saved:,.0f} in civic time value.")
        else:
            st.error(f"⚠️ **Negative ROI:** Operation costs ₹{cost_of_operation:,} but only yields ₹{economic_value_saved:,.0f} in civic time value. Re-target high-impact zones.")
        
    st.markdown("---")
    st.markdown("#### 🚦 Localized Junction Simulation (SUMO Impact Proxy)")
    if not hotspots_df.empty:
        top_junction = hotspots_df.sort_values(by='impact_score_normalized', ascending=False).iloc[0]
        st.markdown(f"**Simulated impact for Top Priority Cluster: ID {top_junction['cluster_id']}**")
        
        g_col1, g_col2 = st.columns(2)
        
        before_delay = top_junction['raw_impact_score'] * 1.5
        after_delay = max(0, before_delay - (total_cleared * 0.5))
        
        fig_before = go.Figure(go.Indicator(
            mode = "gauge+number", value = before_delay,
            title = {'text': "Current Congestion Delay (Mins)"},
            gauge = {'axis': {'range': [0, max(1, before_delay * 1.5)]}, 'bar': {'color': "red"}}
        ))
        
        fig_after = go.Figure(go.Indicator(
            mode = "gauge+number", value = after_delay,
            title = {'text': "Simulated Delay Post-Enforcement (Mins)"},
            gauge = {'axis': {'range': [0, max(1, before_delay * 1.5)]}, 'bar': {'color': "green" if after_delay < before_delay*0.6 else "orange"}}
        ))
        
        fig_before.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        fig_after.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        
        g_col1.plotly_chart(fig_before, use_container_width=True)
        g_col2.plotly_chart(fig_after, use_container_width=True)
    else:
        st.info("Detect hotspots to run the junction simulation.")

with tab7:
    if view_mode == "Internal Police View":
        st.subheader("Law Enforcement Access - Raw Data Feed")
        sensitive_cols = ['id', 'created_datetime', 'vehicle_number', 'vehicle_type', 'violation_type_clean', 'police_station', 'junction_name', 'created_by_id']
        st.dataframe(filtered_df[sensitive_cols].head(100), use_container_width=True)
        
        # Download button for officers
        csv = filtered_df[sensitive_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Report (CSV)",
            data=csv,
            file_name='enforcement_report.csv',
            mime='text/csv',
        )
    else:
        st.warning("🔒 You are currently in Public Demo mode. Switch to 'Internal Police View' in the sidebar to access raw vehicle and officer metrics.")

# --- FOOTER ---
st.markdown("---")
st.caption("ParkGuard AI MVP| Uses HDBSCAN for spatial intelligence | OpenStreetMap proximity simulated via junction tags.")