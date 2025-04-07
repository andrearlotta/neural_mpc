import os
import argparse
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_latest_csv(mode, directory, suffix="plot_data.csv"):
    """Find the latest CSV file for a given mode and suffix in the directory."""
    pattern = os.path.join(directory, f"{mode}_*_{suffix}")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_csv_data(csv_path):
    """
    Load the CSV file containing tree positions and time series data.
    The first row must contain tree positions in the format:
        tree_positions, x0, y0, x1, y1, ...
    The next row is the header for time series data:
        time, x, y, theta, entropy, lambda_0, lambda_1, ...
    """
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    parts = first_line.split(',')
    if parts[0] == "tree_positions":
        tree_positions_list = [float(x) for x in parts[1:]]
        num_trees = len(tree_positions_list) // 2
        tree_positions = np.array(tree_positions_list).reshape(num_trees, 2)
    else:
        tree_positions = None

    df = pd.read_csv(csv_path, skiprows=1)
    time_history = df["time"].values
    x_trajectory = df["x"].values
    y_trajectory = df["y"].values
    theta_trajectory = df["theta"].values
    entropy_history = df["entropy"].values

    lambda_columns = [col for col in df.columns if col.startswith("lambda_")]
    lambda_history = df[lambda_columns].values

    return time_history, x_trajectory, y_trajectory, theta_trajectory, entropy_history, lambda_history, tree_positions

def main(path):
    # Define baseline modes and directories
    modes = ["greedy", "mpc"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baselines_dir = os.path.join(script_dir, path)

    # Collect data for each mode
    all_data = []
    for mode in modes:
        csv_file = get_latest_csv(mode, baselines_dir)
        if not csv_file:
            print(f"No CSV found for {mode}")
            continue
        try:
            data = load_csv_data(csv_file)
            all_data.append((mode, data))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not all_data:
        print("No data to plot")
        return

    # Create a separate figure for each mode/test
    for mode, data in all_data:
        time, x, y, theta, entropy, lambda_hist, trees = data

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],  # Added column width ratio
            subplot_titles=["Drone Trajectory", "Entropy Over Time"],  # Updated titles
            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.1
        )

        # Add trajectory plot on left subplot (col=1)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines+markers",
                line=dict(color="orange", width=4),  # Increased line width
                marker=dict(size=5, color="orange"),
                name="Drone Trajectory",  # Updated trace name
                showlegend=False
            ),
            row=1, col=1
        )

        # Add trees with lambda coloring (if available)
        if trees is not None:
            num_trees = trees.shape[0]
            for i in range(num_trees):
                final_lambda = lambda_hist[-1, i] if lambda_hist.size > 0 else 0.5
                fig.add_trace(
                    go.Scatter(
                        x=[trees[i, 0]], y=[trees[i, 1]],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color=[final_lambda],
                            colorscale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                            cmin=0,
                            cmax=1,
                            showscale=False
                        ),
                        text=str(i),
                        textposition="top center",
                        textfont=dict(size=14),  # Added text font size
                        showlegend=False
                    ),
                    row=1, col=1
                )

        # Add orientation arrows on the trajectory subplot (col=1)
        arrow_length = 0.5
        annotations = []
        xref = "x1"
        yref = "y1"
        for x0, y0, t in zip(x, y, theta):
            x1 = x0 + arrow_length * np.cos(t)
            y1 = y0 + arrow_length * np.sin(t)
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref=xref, yref=yref,
                axref=xref, ayref=yref,
                showarrow=True, arrowhead=3,
                arrowwidth=1.5, arrowcolor="orange"
            ))
        fig.update_layout(annotations=annotations)

        # Add entropy plot on right subplot (col=2)
        fig.add_trace(
            go.Scatter(
                x=time, y=entropy, mode="lines+markers",
                line=dict(color="blue", width=4),  # Increased line width
                marker=dict(size=5, color="blue"),
                name="Entropy",
                showlegend=False
            ),
            row=1, col=2
        )

        # Set axis labels with units
        fig.update_xaxes(title_text="X Position (m)", row=1, col=1)  # Added unit
        fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)  # Added unit
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Entropy (bits)", row=1, col=2)  # Added unit

        # Final layout adjustments
        fig.update_layout(
            title_text="Drone Trajectory and Entropy",  # Simplified title
            height=600,
            template="plotly_white"
        )

        # Save each figure to a separate HTML file
        output_path = os.path.join(baselines_dir, f"{mode}_baseline_comparison.html")
        fig.write_html(output_path)
        print(f"Saved comparison plot for {mode} to: {output_path}")
        fig.show()

if __name__ == "__main__":
    main('random_field')
