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

def main():
    # Define baseline modes and directories
    modes = ["mpc_n1", "mpc_n2", "mpc_n3"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baselines_dir = os.path.join(script_dir, "src/baselines")

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

    # Create a single figure with each mpc
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],  # Rapporto tra le colonne
        subplot_titles=["Drone Trajectory", "Entropy Over Time"],  
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        horizontal_spacing=0.1
    )

    # Lista di colori per distinguere le diverse modalità
    colors = ["orange", "blue", "green"]

    # Iterazione su tutte le modalità
    for i, (mode, data) in enumerate(all_data):
        time, x, y, theta, entropy, lambda_hist, trees = data
        color = colors[i % len(colors)]  # Per assegnare un colore diverso a ciascuna modalità

        # Aggiunta della traiettoria alla subplot sinistra
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=5, color=color),
                name=f"Trajectory {mode}"
            ),
            row=1, col=1
        )

        # Aggiunta degli alberi con colorazione basata su lambda (Per ora lascia cosi poi lambda diventano quelli su cui fanno consenso)
        if trees is not None:
            num_trees = trees.shape[0]
            for j in range(num_trees):
                final_lambda = lambda_hist[-1, j] if lambda_hist.size > 0 else 0.5
                fig.add_trace(
                    go.Scatter(
                        x=[trees[j, 0]], y=[trees[j, 1]],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color=[final_lambda],
                            colorscale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                            cmin=0,
                            cmax=1,
                            showscale=False
                        ),
                        text=str(j),
                        textposition="top center",
                        textfont=dict(size=14),
                        showlegend=False
                    ),
                    row=1, col=1
                )

        # Aggiunta delle frecce di orientazione
        arrow_length = 0.5
        annotations = []
        for x0, y0, t in zip(x, y, theta):
            x1 = x0 + arrow_length * np.cos(t)
            y1 = y0 + arrow_length * np.sin(t)
            annotations.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x1", yref="y1",
                axref="x1", ayref="y1",
                showarrow=True, arrowhead=3,
                arrowwidth=1.5, arrowcolor=color
            ))

        # Aggiunta dell'entropia alla subplot destra
        fig.add_trace(
            go.Scatter(
                x=time, y=entropy, mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=5, color=color),
                name=f"Entropy {mode}"
            ),
            row=1, col=2
        )

    # Aggiunta delle annotazioni delle frecce di orientazione
    fig.update_layout(annotations=annotations)

    # Impostazione delle etichette degli assi
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Entropy (bits)", row=1, col=2)

    # Configurazione finale del layout
    fig.update_layout(
        title_text="Drone Trajectories and Entropy Comparison",
        height=600,
        template="plotly_white"
    )

    # Save plot
    output_path = os.path.join(baselines_dir, "multi_plot.html")
    fig.write_html(output_path)
    print(f"Saved combined comparison plot to: {output_path}")

    fig.show()

if __name__ == "__main__":
    main()
