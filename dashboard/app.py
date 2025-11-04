"""Enhanced Streamlit dashboard for RoadScene3D with nuScenes metrics and 3D visualizations."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="RoadScene3D Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 RoadScene3D - 3D Object Detection Dashboard")
st.markdown("**Real-time monitoring and visualization of autonomous driving models**")


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_mlflow_runs(tracking_uri: str):
    """Load MLflow experiment runs."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name("roadscene3d")
        
        if experiment is None:
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        return runs
    except Exception as e:
        st.error(f"Error loading MLflow runs: {e}")
        return pd.DataFrame()


def extract_nuscenes_metrics(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract nuScenes-specific metrics from MLflow runs."""
    if runs_df.empty:
        return pd.DataFrame()
    
    # MLflow returns metrics with 'metrics.' prefix
    # Create a normalized copy with metrics prefix removed
    metrics_df = runs_df.copy()
    
    # Rename columns: remove 'metrics.' prefix
    rename_dict = {}
    for col in metrics_df.columns:
        if col.startswith('metrics.'):
            new_name = col.replace('metrics.', '')
            rename_dict[col] = new_name
    
    metrics_df = metrics_df.rename(columns=rename_dict)
    
    # Common nuScenes metric columns to keep
    metric_cols = [
        'mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE',
        'AP_car', 'AP_truck', 'AP_bus', 'AP_pedestrian', 'AP_motorcycle', 'AP_bicycle',
        'latency_mean_ms', 'throughput_fps'
    ]
    
    # Keep all metric columns that exist, plus metadata columns
    keep_cols = []
    for col in metric_cols:
        if col in metrics_df.columns:
            keep_cols.append(col)
    
    # Also keep any other AP_* columns
    ap_cols = [col for col in metrics_df.columns if col.startswith('AP_')]
    keep_cols.extend(ap_cols)
    
    # Add metadata columns
    metadata_cols = ['run_id', 'start_time', 'status', 'end_time']
    for col in metadata_cols:
        if col in metrics_df.columns:
            keep_cols.append(col)
    
    # Remove duplicates
    keep_cols = list(set(keep_cols))
    
    if keep_cols:
        return metrics_df[keep_cols]
    
    return metrics_df


def plot_map_progression(runs_df: pd.DataFrame):
    """Plot mAP and NDS progression over time."""
    if runs_df.empty or 'mAP' not in runs_df.columns:
        st.warning("No mAP data available")
        return
    
    fig = go.Figure()
    
    # Sort by start time
    runs_df_sorted = runs_df.sort_values('start_time')
    
    # Plot mAP
    if 'mAP' in runs_df.columns:
        fig.add_trace(go.Scatter(
            x=runs_df_sorted.index,
            y=runs_df_sorted['mAP'] * 100,  # Convert to percentage
            mode='lines+markers',
            name='mAP (%)',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8)
        ))
    
    # Plot NDS
    if 'NDS' in runs_df.columns:
        fig.add_trace(go.Scatter(
            x=runs_df_sorted.index,
            y=runs_df_sorted['NDS'] * 100,  # Convert to percentage
            mode='lines+markers',
            name='NDS (%)',
            line=dict(width=3, color='#ff7f0e'),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Model Performance Progression',
        xaxis_title='Training Run',
        yaxis_title='Score (%)',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_error_metrics(runs_df: pd.DataFrame):
    """Plot nuScenes error metrics."""
    if runs_df.empty:
        st.warning("No data available")
        return
    
    error_metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
    available_errors = [m for m in error_metrics if m in runs_df.columns]
    
    if not available_errors:
        st.info("Error metrics not available yet")
        return
    
    fig = go.Figure()
    
    runs_df_sorted = runs_df.sort_values('start_time')
    
    for metric in available_errors:
        fig.add_trace(go.Scatter(
            x=runs_df_sorted.index,
            y=runs_df_sorted[metric],
            mode='lines+markers',
            name=metric,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Error Metrics (Lower is Better)',
        xaxis_title='Training Run',
        yaxis_title='Error',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_per_class_performance(runs_df: pd.DataFrame):
    """Plot per-class Average Precision."""
    if runs_df.empty:
        st.warning("No data available")
        return
    
    # Extract per-class AP columns
    ap_cols = [col for col in runs_df.columns if col.startswith('AP_')]
    
    if not ap_cols:
        st.info("Per-class metrics not available yet")
        return
    
    # Get latest run
    latest_run = runs_df.iloc[0] if len(runs_df) > 0 else None
    
    if latest_run is None:
        return
    
    # Prepare data - filter out NaN values
    classes = []
    aps = []
    for col in ap_cols:
        ap_value = latest_run[col]
        if pd.notna(ap_value) and ap_value > 0:
            classes.append(col.replace('AP_', ''))
            aps.append(ap_value * 100)  # Convert to percentage
    
    if not classes:
        st.info("No per-class metrics with valid values")
        return
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=aps,
            marker_color=px.colors.qualitative.Set3[:len(classes)],
            text=[f'{ap:.1f}%' for ap in aps],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Per-Class Average Precision (Latest Run)',
        xaxis_title='Class',
        yaxis_title='Average Precision (%)',
        height=400,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_speed_accuracy_tradeoff(runs_df: pd.DataFrame):
    """Plot speed-accuracy trade-off."""
    if runs_df.empty:
        st.warning("No data available")
        return
    
    required_cols = ['mAP', 'latency_mean_ms']
    if not all(col in runs_df.columns for col in required_cols):
        st.info("Speed metrics not available yet")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=runs_df['latency_mean_ms'],
        y=runs_df['mAP'] * 100,
        mode='markers+text',
        marker=dict(
            size=12,
            color=runs_df.index,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Run Index")
        ),
        text=[f"Run {i}" for i in runs_df.index],
        textposition="top center",
        name='Model Versions'
    ))
    
    fig.update_layout(
        title='Speed-Accuracy Trade-off',
        xaxis_title='Latency (ms)',
        yaxis_title='mAP (%)',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_model_registry():
    """Display model registry information."""
    st.subheader("📦 Model Registry")
    
    try:
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.models.registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Get latest versions
        try:
            latest_prod = registry.get_latest_version(stage="Production")
            latest_staging = registry.get_latest_version(stage="Staging")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Production Version", latest_prod if latest_prod > 0 else "None")
            
            with col2:
                st.metric("Staging Version", latest_staging if latest_staging > 0 else "None")
            
        except Exception as e:
            st.info("No models registered yet")
    
    except Exception as e:
        st.info("Model registry not available (this is optional)")
        st.caption(f"Note: {str(e)[:100]}")


def main():
    """Main dashboard application."""
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        value="./mlruns",
        help="Path to MLflow tracking directory"
    )
    
    refresh = st.sidebar.button("🔄 Refresh Data", type="primary")
    
    if refresh:
        st.cache_data.clear()
    
    # Load data
    with st.spinner("Loading experiment data..."):
        runs_df = load_mlflow_runs(tracking_uri)
        metrics_df = extract_nuscenes_metrics(runs_df)
    
    if runs_df.empty:
        st.info("""
        📊 **No experiment data found yet.**
        
        To get started:
        1. Run training with MLflow logging:
           ```bash
           python scripts/train_with_mlflow.py configs/pointpillars_nuscenes_mini.py
           ```
        
        2. Or start MLflow server:
           ```bash
           mlflow ui --backend-store-uri ./mlruns
           ```
        """)
        return
    
    # Overview metrics
    st.header("📊 Overview Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Get latest run (first row, already sorted by start_time DESC)
    latest_run = metrics_df.iloc[0] if len(metrics_df) > 0 else None
    
    with col1:
        if latest_run is not None and 'mAP' in metrics_df.columns:
            latest_map = latest_run['mAP'] * 100 if pd.notna(latest_run['mAP']) else 0.0
        else:
            latest_map = 0.0
        st.metric("Latest mAP", f"{latest_map:.2f}%")
    
    with col2:
        if latest_run is not None and 'NDS' in metrics_df.columns:
            latest_nds = latest_run['NDS'] * 100 if pd.notna(latest_run['NDS']) else 0.0
        else:
            latest_nds = 0.0
        st.metric("Latest NDS", f"{latest_nds:.2f}%")
    
    with col3:
        if 'mAP' in metrics_df.columns:
            best_map = metrics_df['mAP'].max() * 100 if metrics_df['mAP'].notna().any() else 0.0
        else:
            best_map = 0.0
        st.metric("Best mAP", f"{best_map:.2f}%")
    
    with col4:
        if latest_run is not None and 'latency_mean_ms' in metrics_df.columns:
            latest_latency = latest_run['latency_mean_ms'] if pd.notna(latest_run['latency_mean_ms']) else 0.0
        else:
            latest_latency = 0.0
        st.metric("Latency", f"{latest_latency:.1f} ms" if latest_latency > 0 else "N/A")
    
    with col5:
        total_runs = len(runs_df)
        st.metric("Total Runs", total_runs)
    
    # Main charts
    st.header("📈 Model Performance")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Trends", "Error Metrics", "Per-Class AP", "Speed-Accuracy"])
    
    with tab1:
        plot_map_progression(metrics_df)
    
    with tab2:
        plot_error_metrics(metrics_df)
    
    with tab3:
        plot_per_class_performance(metrics_df)
    
    with tab4:
        plot_speed_accuracy_tradeoff(metrics_df)
    
    # Model Registry
    display_model_registry()
    
    # Raw data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(runs_df, use_container_width=True)


if __name__ == '__main__':
    main()
