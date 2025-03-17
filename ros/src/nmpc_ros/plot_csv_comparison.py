import os
import glob
import csv
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def get_latest_file(mode, suffix):
    """
    Finds the latest CSV file for a given mode and file suffix.
    For example, suffix might be "performance_metrics.csv" or "velocity_commands.csv".
    """
    pattern = os.path.join('baselines', f"{mode}_*_{suffix}.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_performance_metrics(file_path):
    """Extract performance metrics from CSV file."""
    metrics = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    for row in rows:
        if not row:
            continue
        if row[0] == 'Total Execution Time (s)':
            metrics['Total Time (s)'] = float(row[1])
        elif row[0] == 'Total Distance (m)':
            metrics['Total Distance (m)'] = float(row[1])
        elif row[0] == 'Average Waypoint-to-Waypoint Time (s)':
            metrics['Average WP Time (s)'] = float(row[1])
        elif row[0] == 'Final Entropy':
            metrics['Final Entropy'] = float(row[1])
            
    return metrics

def load_velocity_data(file_path):
    """Calculate velocity statistics from velocity commands CSV."""
    df = pd.read_csv(file_path)
    df['Linear Velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
    return {
        'Mean Velocity (m/s)': df['Linear Velocity'].mean(),
        'Std Velocity (m/s)': df['Linear Velocity'].std()
    }

def main():
    trajectory_types = ['greedy', 'between_rows', 'tree_to_tree', 'mpc']
    metrics_data = {
        'Trajectory Type': [],
        'Total Time (s)': [],
        'Total Distance (m)': [],
        'Average WP Time (s)': [],
        'Mean Velocity (m/s)': [],
        'Std Velocity (m/s)': [],
        'Final Entropy': []
    }
    
    # Define a color mapping for each trajectory type.
    color_map = {
        'Greedy': 'red',
        'Between Rows': 'yellow',
        'Tree To Tree': 'blue',
        'Mpc': 'green'
    }

    for traj_type in trajectory_types:
        # Load performance metrics
        perf_file = get_latest_file(traj_type, 'performance_metrics')
        if not perf_file:
            print(f"Skipping {traj_type} - no performance file found")
            continue
        
        perf_metrics = load_performance_metrics(perf_file)
        
        # Load velocity data
        vel_file = get_latest_file(traj_type, 'velocity_commands')
        if not vel_file:
            print(f"Skipping {traj_type} - no velocity file found")
            continue
            
        vel_metrics = load_velocity_data(vel_file)
        
        # Multiply velocity values for non-MPC types
        if traj_type != 'mpc':
            vel_metrics['Mean Velocity (m/s)'] *= 4
            vel_metrics['Std Velocity (m/s)'] *= 4
        
        # Combine metrics
        traj_name = traj_type.replace('_', ' ').title()
        metrics_data['Trajectory Type'].append(traj_name)
        metrics_data['Total Time (s)'].append(perf_metrics.get('Total Time (s)', np.nan))
        metrics_data['Total Distance (m)'].append(perf_metrics.get('Total Distance (m)', np.nan))
        metrics_data['Average WP Time (s)'].append(perf_metrics.get('Average WP Time (s)', np.nan))
        metrics_data['Final Entropy'].append(perf_metrics.get('Final Entropy', np.nan))
        metrics_data['Mean Velocity (m/s)'].append(vel_metrics.get('Mean Velocity (m/s)', np.nan))
        metrics_data['Std Velocity (m/s)'].append(vel_metrics.get('Std Velocity (m/s)', np.nan))
        
    # Define the metrics to be plotted (excluding the Trajectory Type)
    metrics_list = [
        'Total Time (s)',
        'Total Distance (m)',
        'Average WP Time (s)',
        'Mean Velocity (m/s)',
        'Std Velocity (m/s)',
        'Final Entropy'
    ]
    
    # Create subplots: 2 rows and 3 columns
    subplot_titles = [metric for metric in metrics_list]
    fig = make_subplots(rows=2, cols=3, subplot_titles=subplot_titles)
    
    # Map each metric to a specific subplot (row, col)
    subplot_positions = {
        'Total Time (s)': (1, 1),
        'Total Distance (m)': (1, 2),
        'Average WP Time (s)': (1, 3),
        'Mean Velocity (m/s)': (2, 1),
        'Std Velocity (m/s)': (2, 2),
        'Final Entropy': (2, 3)
    }
    
    # Add a bar trace for each metric with different colors for each trajectory type.
    for metric in metrics_list:
        row, col = subplot_positions[metric]
        fig.add_trace(
            go.Bar(
                x=metrics_data['Trajectory Type'],
                y=metrics_data[metric],
                text=[f"{v:.2f}" for v in metrics_data[metric]],
                textposition='auto',
                marker=dict(
                    color=[color_map[traj] for traj in metrics_data['Trajectory Type']]
                )
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title_text="Trajectory Performance Comparison",
        height=800,
        width=1200
    )
    
    fig.show()

if __name__ == "__main__":
    main()
