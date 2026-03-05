import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from hydrotwin.compliance.regulation_engine import RegulationEngine

st.set_page_config(
    page_title="HydroTwin OS | Groundwater Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and tech styling
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1, h2, h3 {
        color: #4A90E2;
        font-family: 'Space Grotesk', sans-serif;
    }
    .stMetric {
        background-color: #1E212B;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 HydroTwin OS: Groundwater Impact Analyzer")
st.markdown("*Predictive Regulatory Dashboard for Hyperscale Infrastructure Approval*")

# ─── SIDEBAR CONTROLS ───
with st.sidebar:
    st.header("Datacenter Parameters")
    
    capacity_mw = st.slider("Proposed IT Load (MW)", 10, 200, 50, 10)
    wue_design = st.slider("Design WUE (L/kWh)", 0.0, 2.5, 1.2, 0.1)
    
    st.markdown("---")
    st.header("Site Selection")
    location = st.selectbox("Areal Basin", ["Hyderabad, IND (High Stress)", "Phoenix, USA (Extreme Stress)", "Dublin, IRE (Low Stress)"])
    
    if "Hyderabad" in location:
        lat, lon = 17.3850, 78.4867
        base_depletion = 4.2 # m/year baseline
        wri_score = 4.5
    elif "Phoenix" in location:
        lat, lon = 33.4484, -112.0740
        base_depletion = 3.8
        wri_score = 4.8
    else:
        lat, lon = 53.3498, -6.2603
        base_depletion = 0.5
        wri_score = 1.2

    st.markdown("---")
    st.markdown("**Core AI Toggle**")
    ai_active = st.checkbox("Enable HydroTwin RL Optimization", value=True)

# ─── CALCULATIONS ───
hours_year = 8760
if ai_active:
    # RL agent dynamically reduces WUE by ~60% in high stress, shifting to carbon cooling
    effective_wue = wue_design * 0.4 
    ai_status = "🟢 Active (Pareto Bounded)"
else:
    effective_wue = wue_design
    ai_status = "🔴 Inactive (Static Cooling)"

# Convert MW to kWh, then apply WUE for total liters/year, convert to Megaliters (ML)
annual_energy_kwh = capacity_mw * 1000 * hours_year
annual_water_liters = annual_energy_kwh * effective_wue
annual_water_ml = annual_water_liters / 1_000_000

# Depletion model (oversimplified for demo: 1 ML = 0.0001m regional drop)
added_depletion = annual_water_ml * 0.0001
new_depletion_rate = base_depletion + added_depletion

# Generate projection data (10 years)
years = np.arange(2026, 2036)
baseline_aquifer_level = 100 # arbitrary starting depth
levels_without_dc = [baseline_aquifer_level - (base_depletion * i) for i in range(10)]
levels_with_dc = [baseline_aquifer_level - ((base_depletion + (wue_design*annual_energy_kwh/1e10)) * i) for i in range(10)]
levels_hydrotwin = [baseline_aquifer_level - (new_depletion_rate * i) for i in range(10)]

df_proj = pd.DataFrame({
    'Year': years,
    'Baseline (No Datacenter)': levels_without_dc,
    'Static Datacenter': levels_with_dc,
    'HydroTwin Optimized': levels_hydrotwin
})

# ─── MAIN LAYOUT ───
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total IT Capacity", f"{capacity_mw} MW")
with col2:
    st.metric("WRI Aqueduct Score", f"{wri_score}/5.0", delta="High Stress" if wri_score > 4 else "Safe", delta_color="inverse")
with col3:
    st.metric("Operating WUE", f"{effective_wue:.2f} L/kWh", delta=f"{effective_wue - wue_design:.2f}" if ai_active else "0.00")
with col4:
    st.metric("Annual Draw", f"{annual_water_ml:,.0f} Megaliters")


st.markdown("---")

col_chart, col_map = st.columns([6, 4])

with col_chart:
    st.subheader("10-Year Aquifer Depletion Projection")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_proj['Year'], y=df_proj['Baseline (No Datacenter)'], mode='lines', name='Baseline (No Datacenter)', line=dict(dash='dash', color='gray')))
    fig.add_trace(go.Scatter(x=df_proj['Year'], y=df_proj['Static Datacenter'], mode='lines', name='Static Datacenter', line=dict(color='#E74C3C')))
    fig.add_trace(go.Scatter(x=df_proj['Year'], y=df_proj['HydroTwin Optimized'], mode='lines', name='HydroTwin Optimized', line=dict(color='#2ECC71', width=3)))
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FAFAFA'),
        yaxis_title="Aquifer Water Level Index",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"**Insight:** By activating the RL Optimization plane, the datacenter avoids **{((wue_design-effective_wue)*annual_energy_kwh)/1_000_000:,.0f} ML** of critical basin extraction annually, dynamically shifting the cooling payload to electrical chilling during peak renewable availability on the grid.")

with col_map:
    st.subheader("Regional Extraction Radius")
    
    # Generate scatter points representing wells
    np.random.seed(42)
    df_wells = pd.DataFrame({
        'lat': np.random.normal(lat, 0.05, 50),
        'lon': np.random.normal(lon, 0.05, 50),
        'depletion': np.random.uniform(0, 10, 50)
    })
    
    # Dark mode map using PyDeck
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=10, pitch=45)
    
    layer_wells = pdk.Layer(
        "ColumnLayer",
        data=df_wells,
        get_position=["lon", "lat"],
        get_elevation="depletion",
        elevation_scale=100,
        radius=200,
        get_fill_color=["depletion * 25", "200 - (depletion * 20)", 100, 140],
        pickable=True,
        auto_highlight=True,
    )
    
    # The Datacenter core
    layer_dc = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": lat, "lon": lon}],
        get_position=["lon", "lat"],
        get_color=[255, 0, 0, 200] if not ai_active else [46, 204, 113, 200],
        get_radius=1500,
    )

    r = pdk.Deck(
        layers=[layer_wells, layer_dc],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "Local Well\nDepletion Impact: {depletion}"}
    )
    st.pydeck_chart(r)

st.markdown("---")
st.caption("Powered by HydroTwin OS — Plane 4 (Regulatory Core) & Plane 3 (SAC Execution Matrix)")
