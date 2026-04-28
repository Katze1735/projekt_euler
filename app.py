import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go

# Optional: If you want to use a heavy pre-trained model like TimesFM
# from timesfm import TimesFm 

st.set_page_config(page_title="AI Peak Analytics", layout="wide")

## --- UI Header ---
st.title("🏔️ AI-Powered Peak Detector")
st.markdown("Upload your CSV to automatically detect peaks and export timestamps.")

## --- File Handling ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    
    time_col = st.sidebar.selectbox("Timestamp Column", cols)
    val_col = st.sidebar.selectbox("Data/Signal Column", cols)
    
    # Ensure time is sorted
    df = df.sort_values(by=time_col)

    ## --- "AI" Tuning Parameters ---
    st.sidebar.header("Model Tuning")
    # In AI peak detection, 'prominence' is the relative height threshold
    prominence = st.sidebar.slider("Peak Sensitivity (Prominence)", 0.0, float(df[val_col].max() * 0.5), 1.0)
    # 'distance' prevents the model from double-counting the same peak
    min_dist = st.sidebar.number_input("Min Distance (Data Points)", value=5)

    ## --- The Detection Logic ---
    # Using find_peaks (the industry standard 'classic AI' approach)
    peaks, _ = find_peaks(df[val_col], prominence=prominence, distance=min_dist)
    
    peak_times = df.iloc[peaks][time_col]
    peak_values = df.iloc[peaks][val_col]

    ## --- Visualization ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[val_col], name="Raw Signal", line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=peak_times, y=peak_values, mode='markers', name='Detected Peaks',
                             marker=dict(color='red', size=10, symbol='x')))
    
    fig.update_layout(title="Signal Analysis", xaxis_title=time_col, yaxis_title=val_col)
    st.plotly_chart(fig, use_container_with_width=True)

    ## --- Table & Export ---
    st.subheader("📍 Detected Peak List")
    results = pd.DataFrame({
        "Timestamp": peak_times,
        "Magnitude": peak_values
    }).reset_index(drop=True)
    
    st.dataframe(results, use_container_with_width=True)
    
    csv_out = results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv_out, "detected_peaks.csv", "text/csv")

else:
    st.info("Waiting for CSV upload...")
