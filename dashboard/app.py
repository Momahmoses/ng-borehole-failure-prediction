"""Streamlit RUWASA borehole health monitoring dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.models.lstm_autoencoder import generate_synthetic_pump_data, SEQ_LEN, N_FEATURES

st.set_page_config(page_title="Borehole Failure Prediction", page_icon="💧", layout="wide")
st.title("💧 Borehole & Water Point Failure Prediction — Nigeria")
st.caption("IoT-based predictive maintenance for rural water points | RUWASA operational dashboard")

HEALTH_COLOURS = {"HEALTHY": "#00CC00", "WARNING": "#FFA500", "CRITICAL": "#CC0000", "FAILED": "#8B0000"}


@st.cache_data
def generate_borehole_fleet(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    states = ["Jigawa", "Plateau", "Benue", "Nasarawa", "Katsina", "Niger", "Kebbi", "Borno"]
    health_statuses = np.random.choice(
        ["HEALTHY", "WARNING", "CRITICAL", "FAILED"],
        n, p=[0.55, 0.25, 0.12, 0.08]
    )
    anomaly_scores = {
        "HEALTHY": lambda: np.random.uniform(0.01, 0.08),
        "WARNING": lambda: np.random.uniform(0.10, 0.20),
        "CRITICAL": lambda: np.random.uniform(0.22, 0.38),
        "FAILED": lambda: np.random.uniform(0.40, 0.80),
    }
    lat_range = {"Jigawa": (11.8, 12.8), "Plateau": (8.5, 10.2), "Benue": (6.8, 8.5),
                  "Nasarawa": (7.8, 9.2), "Katsina": (11.5, 13.0), "Niger": (9.0, 11.5),
                  "Kebbi": (11.0, 13.0), "Borno": (10.0, 13.5)}
    lon_range = {"Jigawa": (9.0, 10.5), "Plateau": (8.5, 10.0), "Benue": (7.5, 10.5),
                  "Nasarawa": (7.5, 9.5), "Katsina": (7.0, 8.5), "Niger": (4.5, 7.5),
                  "Kebbi": (3.5, 5.5), "Borno": (12.0, 15.0)}
    state_col = np.random.choice(states, n)
    lats = [np.random.uniform(*lat_range.get(s, (7, 13))) for s in state_col]
    lons = [np.random.uniform(*lon_range.get(s, (4, 15))) for s in state_col]

    return pd.DataFrame({
        "borehole_id": [f"BH-{i:05d}" for i in range(n)],
        "state": state_col,
        "latitude": lats,
        "longitude": lons,
        "health_status": health_statuses,
        "anomaly_score": [anomaly_scores[s]() for s in health_statuses],
        "pump_age_years": np.random.uniform(0.5, 15, n).round(1),
        "last_maintenance_days": np.random.randint(1, 730, n),
        "water_table_depth_m": np.random.uniform(10, 60, n).round(1),
        "daily_users": np.random.randint(50, 500, n),
        "current_draw_A": np.random.uniform(3.5, 8.5, n).round(2),
        "vibration_rms": np.random.uniform(0.3, 2.8, n).round(3),
        "predicted_days_to_failure": np.where(
            health_statuses == "FAILED", 0,
            np.where(health_statuses == "CRITICAL", np.random.randint(1, 15, n),
            np.where(health_statuses == "WARNING", np.random.randint(15, 45, n),
                     np.random.randint(45, 365, n)))
        ),
    })


fleet = generate_borehole_fleet()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Boreholes Monitored", len(fleet))
with col2:
    failed = len(fleet[fleet["health_status"].isin(["FAILED", "CRITICAL"])])
    st.metric("Failed or Critical", failed, delta_color="inverse")
with col3:
    warning = len(fleet[fleet["health_status"] == "WARNING"])
    st.metric("Warning (30-day risk)", warning, delta_color="inverse")
with col4:
    affected_users = fleet[fleet["health_status"] == "FAILED"]["daily_users"].sum()
    st.metric("Users Without Water", f"{affected_users:,}", delta_color="inverse")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🗺️ Fleet Map", "📈 Sensor Trends", "🔧 Maintenance Queue"])

with tab1:
    colour_map = {"HEALTHY": "#00CC00", "WARNING": "#FFA500", "CRITICAL": "#FF4500", "FAILED": "#8B0000"}
    fig = px.scatter_mapbox(
        fleet,
        lat="latitude", lon="longitude",
        color="health_status",
        color_discrete_map=colour_map,
        size="anomaly_score",
        size_max=15,
        hover_name="borehole_id",
        hover_data={"state": True, "health_status": True, "predicted_days_to_failure": True,
                    "daily_users": True, "pump_age_years": True},
        mapbox_style="carto-positron",
        zoom=5, center={"lat": 10.5, "lon": 8.0},
        title="Borehole Fleet Health Status",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    state_health = (
        fleet.groupby(["state", "health_status"])
        .size().reset_index(name="count")
    )
    fig2 = px.bar(
        state_health, x="state", y="count", color="health_status",
        color_discrete_map=colour_map,
        title="Health Status by State",
        barmode="stack",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Real-time Sensor Readings — 24-Hour Window")
    borehole_sel = st.selectbox("Select Borehole", fleet["borehole_id"].tolist())
    bh_row = fleet[fleet["borehole_id"] == borehole_sel].iloc[0]
    health = bh_row["health_status"]
    hours = 24
    t = pd.date_range(end=pd.Timestamp.now(), periods=hours * 4, freq="15min")
    degradation_trend = np.linspace(0, 0.3 if health in ["CRITICAL", "FAILED"] else 0.05, hours * 4)
    current = bh_row["current_draw_A"] + degradation_trend * 2 + np.random.normal(0, 0.1, hours * 4)
    vibration = bh_row["vibration_rms"] + degradation_trend + np.random.normal(0, 0.05, hours * 4)

    fig_sensors = go.Figure()
    fig_sensors.add_trace(go.Scatter(x=t, y=current, name="Current (A)", line=dict(color="red")))
    fig_sensors.add_trace(go.Scatter(x=t, y=vibration * 10, name="Vibration ×10", line=dict(color="orange")))
    fig_sensors.add_hline(y=7.0, line_dash="dash", line_color="red", annotation_text="Current alarm (7A)")
    fig_sensors.update_layout(title=f"{borehole_sel} — 24h Sensor Readings | Status: {health}")
    st.plotly_chart(fig_sensors, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    status_colour = {"HEALTHY": "success", "WARNING": "warning", "CRITICAL": "error", "FAILED": "error"}
    with c1:
        st.metric("Health Status", health)
    with c2:
        st.metric("Anomaly Score", f"{bh_row['anomaly_score']:.4f}")
    with c3:
        st.metric("Days to Predicted Failure", int(bh_row["predicted_days_to_failure"]))

with tab3:
    st.subheader("Maintenance Work Queue — Priority Order")
    maintenance_needed = fleet[fleet["health_status"].isin(["CRITICAL", "WARNING", "FAILED"])].copy()
    maintenance_needed = maintenance_needed.sort_values(
        ["health_status", "predicted_days_to_failure"],
        key=lambda x: x.map({"FAILED": 0, "CRITICAL": 1, "WARNING": 2}) if x.name == "health_status" else x
    )
    st.dataframe(
        maintenance_needed[["borehole_id", "state", "health_status", "anomaly_score",
                              "predicted_days_to_failure", "daily_users",
                              "pump_age_years", "last_maintenance_days"]].head(50)
        .style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
        use_container_width=True, height=400,
    )
    st.download_button(
        "📥 Download Work Orders (CSV)",
        data=maintenance_needed.to_csv(index=False),
        file_name=f"borehole_maintenance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("MOMAH MOSES .C. · Geospatial AI Engineer & Data Scientist · github.com/Momahmoses")
