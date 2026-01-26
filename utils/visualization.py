"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from config import VIZ_CONFIG

def generate_heatmap(heatmap_data, output_path=None, title="Interaction Heatmap"):
    """Generate heatmap visualization"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=VIZ_CONFIG["dpi"])
    
    sns.heatmap(
        heatmap_data,
        cmap=VIZ_CONFIG["heatmap_colormap"],
        cbar=True,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Screen Region X")
    ax.set_ylabel("Screen Region Y")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig

def generate_timeline(data, timestamps, title="Interaction Timeline", ylabel="Value"):
    """Generate timeline graph"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=data,
        mode='lines+markers',
        name=ylabel
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title=ylabel,
        height=VIZ_CONFIG["timeline_height"]
    )
    
    return fig

def generate_comparison_chart(video_names, values, metric_name):
    """Generate comparison bar chart"""
    fig = go.Figure(data=[
        go.Bar(x=video_names, y=values)
    ])
    
    fig.update_layout(
        title=f"{metric_name} Comparison",
        xaxis_title="Video",
        yaxis_title=metric_name,
        height=VIZ_CONFIG["comparison_chart_height"]
    )
    
    return fig

def generate_trajectory_plot(positions, output_path=None):
    """Generate cursor trajectory plot"""
    if not positions:
        return None
    
    x_coords = [p["x"] for p in positions]
    y_coords = [p["y"] for p in positions]
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=VIZ_CONFIG["dpi"])
    
    # Plot trajectory
    ax.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=0.5)
    ax.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='viridis', s=1)
    
    ax.set_title("Cursor Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.invert_yaxis()  # Screen coordinates
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig
