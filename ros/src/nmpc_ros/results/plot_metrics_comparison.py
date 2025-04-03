import os
import glob
import csv
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def get_latest_file(mode, path, suffix):
    """
    Finds the latest CSV file for a given mode and file suffix.
    For example, suffix might be "performance_metrics", "velocity_commands", or "velocity_metrics".
    """
    pattern = os.path.join(path, f"{mode}_*_{suffix}.csv")
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
        'Std Velocity (m/s)': df['Linear Velocity'].std(),
        'Mean Angular Velocity (rad/s)': np.abs(df['yaw_velocity']).mean(),
        'Std Angular Velocity (rad/s)': np.abs(df['yaw_velocity']).std()
    }

def load_velocity_metrics(file_path):
    """
    Computes execution time statistics from the velocity metrics CSV.
    Skips the first line of data (after the header) and computes the average,
    minimum, maximum, and standard deviation of the "Time (s)" column.
    """
    df = pd.read_csv(file_path)
    # Skip the first data row.
    df = df.iloc[1:]
    # Ensure the Time column is float type.
    df['Time (s)'] = df['Time (s)'].astype(float)
    delta_t = np.array(df['Time (s)'][1:]) - np.array(df['Time (s)'][:-1])

    avg_time = np.median(delta_t)
    min_time = delta_t.min()
    max_time = delta_t.max()
    std_time = delta_t.std()
    return {
        'Average Execution Time (s)': avg_time,
        'Min Execution Time (s)': min_time,
        'Max Execution Time (s)': max_time,
        'Std Execution Time (s)': std_time
    }

def main(path):
    trajectory_types = ['greedy', 'between_rows', 'tree_to_tree', 'mpc']
    metrics_data = {
        'Trajectory Type': [],
        'Total Time (s)': [],
        'Total Distance (m)': [],
        'Average WP Time (s)': [],
        'Mean Velocity (m/s)': [],
        'Std Velocity (m/s)': [],
        'Mean Angular Velocity (rad/s)': [],
        'Std Angular Velocity (rad/s)': [],
        'Final Entropy': [],
        'Average Execution Time (s)': [],
        'Min Execution Time (s)': [],
        'Max Execution Time (s)': [],
        'Std Execution Time (s)': []
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
        perf_file = get_latest_file(traj_type, path, 'performance_metrics')
        if not perf_file:
            print(f"Skipping {traj_type} - no performance file found")
            continue

        perf_metrics = load_performance_metrics(perf_file)

        # Load velocity commands data
        vel_cmd_file = get_latest_file(traj_type, path, 'velocity_commands')
        if not vel_cmd_file:
            print(f"Skipping {traj_type} - no velocity commands file found")
            continue

        vel_cmd_stats = load_velocity_data(vel_cmd_file)

        # Load velocity metrics data for execution times
        vel_metrics_file = get_latest_file(traj_type, path, 'velocity_commands')  # Corrected suffix
        if not vel_metrics_file:
            print(f"Skipping {traj_type} - no velocity metrics file found")
            continue
        exec_time_stats = load_velocity_metrics(vel_metrics_file)

        # Combine metrics into the dictionary.
        traj_name = traj_type.replace('_', ' ').title()
        metrics_data['Trajectory Type'].append(traj_name)
        metrics_data['Total Time (s)'].append(perf_metrics.get('Total Time (s)', np.nan))
        metrics_data['Total Distance (m)'].append(perf_metrics.get('Total Distance (m)', np.nan))
        metrics_data['Average WP Time (s)'].append(perf_metrics.get('Average WP Time (s)', np.nan))
        metrics_data['Final Entropy'].append(perf_metrics.get('Final Entropy', np.nan))
        metrics_data['Mean Velocity (m/s)'].append(vel_cmd_stats.get('Mean Velocity (m/s)', np.nan))
        metrics_data['Std Velocity (m/s)'].append(vel_cmd_stats.get('Std Velocity (m/s)', np.nan))
        metrics_data['Mean Angular Velocity (rad/s)'].append(vel_cmd_stats.get('Mean Angular Velocity (rad/s)', np.nan))
        metrics_data['Std Angular Velocity (rad/s)'].append(vel_cmd_stats.get('Std Angular Velocity (rad/s)', np.nan))
        metrics_data['Average Execution Time (s)'].append(exec_time_stats.get('Average Execution Time (s)', np.nan))
        metrics_data['Min Execution Time (s)'].append(exec_time_stats.get('Min Execution Time (s)', np.nan))
        metrics_data['Max Execution Time (s)'].append(exec_time_stats.get('Max Execution Time (s)', np.nan))
        metrics_data['Std Execution Time (s)'].append(exec_time_stats.get('Std Execution Time (s)', np.nan))

    # Define the metrics to be plotted in the new order
    metrics_list = [
        'Final Entropy',
        'Total Distance (m)',
        'Total Time (s)',
        '',
        'Average Execution Time (s)',
        'Min Execution Time (s)',
        'Max Execution Time (s)',
        'Std Execution Time (s)',
        'Mean Velocity (m/s)',
        'Std Velocity (m/s)',
        '',
        '',
        'Mean Angular Velocity (rad/s)',
        'Std Angular Velocity (rad/s)',
        '',
        '',
    ]

    # Create subplots: using 4 rows and 3 columns for 11 metrics.
    rows = 4
    cols = 4
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics_list)

    # Add a bar trace for each metric with different colors for each trajectory type.
    for i, metric in enumerate(metrics_list):
        if metric == '': continue
        row = i // cols + 1
        col = i % cols + 1
        fig.add_trace(
            go.Bar(
                x=metrics_data['Trajectory Type'],
                y=metrics_data[metric],
                text=[f"{v:.2f}" for v in metrics_data[metric]],
                textposition='auto',
                marker=dict(
                    color=[color_map[traj] for traj in metrics_data['Trajectory Type']]
                ),
                showlegend=False  # Ensure individual traces do not show legend
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title_text="Trajectory Performance Comparison",
        height=1200,
        width=1200,
        showlegend=False  # Hide the legend
    )

    fig.show()

if __name__ == "__main__":
    main('baselines')