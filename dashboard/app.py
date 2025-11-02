"""Streamlit dashboard for model monitoring and visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from pathlib import Path
import json

# Configure page
st.set_page_config(
    page_title="RoadScene3D Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 RoadScene3D Model Dashboard")


@st.cache_data
def load_mlflow_runs():
    """Load MLflow experiment runs."""
    try:
        mlflow.set_tracking_uri("./mlruns")
        experiment = mlflow.get_experiment_by_name("roadscene3d")
        
        if experiment is None:
            return []
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return runs
    except Exception as e:
        st.error(f"Error loading MLflow runs: {e}")
        return pd.DataFrame()


def plot_map_progression(runs_df):
    """Plot mAP progression over iterations."""
    if runs_df.empty:
        st.warning("No data available")
        return
    
    # Extract mAP metrics (adjust based on actual metric names)
    metrics = ['mAP@0.5', 'mAP@0.7']
    
    fig = go.Figure()
    
    for metric in metrics:
        if metric in runs_df.columns:
            fig.add_trace(go.Scatter(
                x=runs_df.index,
                y=runs_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='mAP Progression Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='mAP',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_speed_accuracy_tradeoff(runs_df):
    """Plot speed-accuracy trade-off."""
    if runs_df.empty:
        st.warning("No data available")
        return
    
    fig = go.Figure()
    
    # Extract relevant metrics
    if 'mAP@0.7' in runs_df.columns and 'latency_mean_ms' in runs_df.columns:
        fig.add_trace(go.Scatter(
            x=runs_df['latency_mean_ms'],
            y=runs_df['mAP@0.7'],
            mode='markers',
            marker=dict(
                size=10,
                color=runs_df.index,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"Run {i}" for i in runs_df.index],
            name='Model Versions'
        ))
    
    fig.update_layout(
        title='Speed-Accuracy Trade-off',
        xaxis_title='Latency (ms)',
        yaxis_title='mAP@0.7',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_per_class_performance(runs_df):
    """Plot per-class performance."""
    if runs_df.empty:
        st.warning("No data available")
        return
    
    # Extract per-class metrics
    classes = ['vehicle', 'pedestrian', 'cyclist']
    metrics = ['AP_vehicle@0.7', 'AP_pedestrian@0.7', 'AP_cyclist@0.7']
    
    data = {}
    for cls, metric in zip(classes, metrics):
        if metric in runs_df.columns:
            data[cls] = runs_df[metric].values
    
    if data:
        df_per_class = pd.DataFrame(data)
        
        fig = px.bar(
            df_per_class,
            barmode='group',
            title='Per-Class Performance (AP@0.7)',
            labels={'value': 'Average Precision', 'index': 'Run'}
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    
    # Sidebar
    st.sidebar.title("Settings")
    
    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        value="./mlruns"
    )
    
    refresh = st.sidebar.button("🔄 Refresh Data")
    
    # Load data
    runs_df = load_mlflow_runs()
    
    if runs_df.empty:
        st.info("No experiment data found. Run some training experiments first.")
        return
    
    # Overview metrics
    st.header("📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_map = runs_df['mAP@0.7'].iloc[-1] if 'mAP@0.7' in runs_df.columns else 0.0
        st.metric("Latest mAP@0.7", f"{latest_map:.4f}")
    
    with col2:
        best_map = runs_df['mAP@0.7'].max() if 'mAP@0.7' in runs_df.columns else 0.0
        st.metric("Best mAP@0.7", f"{best_map:.4f}")
    
    with col3:
        latest_latency = runs_df['latency_mean_ms'].iloc[-1] if 'latency_mean_ms' in runs_df.columns else 0.0
        st.metric("Latest Latency", f"{latest_latency:.2f} ms")
    
    with col4:
        total_runs = len(runs_df)
        st.metric("Total Runs", total_runs)
    
    # Charts
    st.header("📈 Model Performance")
    
    tab1, tab2, tab3 = st.tabs(["mAP Progression", "Speed-Accuracy", "Per-Class"])
    
    with tab1:
        plot_map_progression(runs_df)
    
    with tab2:
        plot_speed_accuracy_tradeoff(runs_df)
    
    with tab3:
        plot_per_class_performance(runs_df)
    
    # Model registry
    st.header("📦 Model Registry")
    
    try:
        from src.models.registry import ModelRegistry
        registry = ModelRegistry(tracking_uri=tracking_uri)
        
        st.subheader("Model Versions")
        
        # List model versions
        # versions = registry.get_all_versions()
        # st.dataframe(versions)
        
        st.info("Model registry integration would display version history here")
    
    except Exception as e:
        st.warning(f"Could not load model registry: {e}")


if __name__ == '__main__':
    main()
