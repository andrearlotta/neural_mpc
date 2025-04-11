import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

def get_latest_csv(mode, directory, suffix="plot_data.csv"):
    """Find the latest CSV file for a given mode and suffix in the directory."""
    pattern = os.path.join(directory, f"{mode}*{suffix}")
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
    modes = ["greedy", "trees_to_trees"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baselines_dir = os.path.join(script_dir, "test_for_creating_plot")  # Adjust directory path if needed

    # Create a custom colormap for tree lambda visualization:
    # Red for 0, Yellow for 0.5, and Green for 1.
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )

    for mode in modes:
        csv_file = get_latest_csv(mode, baselines_dir)
        if not csv_file:
            print(f"No CSV found for {mode}")
            continue

        try:
            (time_history, x, y, theta, entropy,
             lambda_history, trees) = load_csv_data(csv_file)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

        # Create a figure with two subplots using gridspec with width ratios.
        fig, (ax_traj, ax_entropy) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [0.7, 0.3]}, figsize=(12, 6)
        )

        # Plot the drone trajectory (line with markers)
        traj_line, = ax_traj.plot(x, y, marker='o', color='orange', linewidth=4,
                                  markersize=5, label="Drone Trajectory")

        # Plot initial and final positions with distinct markers
        ax_traj.scatter(x[0], y[0], color='green', s=100, marker='o', label="Initial Position", zorder=4)
        ax_traj.scatter(x[-1], y[-1], color='red', s=100, marker='o', label="Final Position", zorder=4)

        # Plot tree positions as squares
        if trees is not None:
            num_trees = trees.shape[0]
            for i in range(num_trees):
                final_lambda = lambda_history[-1, i] if lambda_history.size > 0 else 0.5
                tree_color = custom_cmap(final_lambda)
                ax_traj.scatter(trees[i, 0], trees[i, 1], color=tree_color,
                                s=100, marker='s', zorder=3)
                ax_traj.text(trees[i, 0], trees[i, 1], str(i),
                             fontsize=14, ha="center", va="bottom")
            # Add a proxy artist for the trees in the legend
            tree_proxy = Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                  markersize=10, label='Ripe Raw Trees')
            # Collect legend handles from trajectory, initial, and final markers, plus tree proxy
            handles, labels = ax_traj.get_legend_handles_labels()
            handles.append(tree_proxy)
            labels.append("Ripe Raw Trees")
            ax_traj.legend(handles, labels)

        else:
            # If no trees available, just show the legend for the trajectory and positions.
            ax_traj.legend()

        # Add orientation arrows along the trajectory, plotting them less frequently
        arrow_length = 1.5
        arrow_interval = len(x)  # adjust frequency: at most ~15 arrows
        for idx in range(0, len(x),10):
            x0 = x[idx]
            y0 = y[idx]
            t = theta[idx]
            x1 = x0 + arrow_length * np.cos(t)
            y1 = y0 + arrow_length * np.sin(t)
            ax_traj.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="orange", linewidth=1.0)
            )

        # Set trajectory plot labels and title
        ax_traj.set_xlabel("X Position (m)")
        ax_traj.set_ylabel("Y Position (m)")


        # Plot the entropy over time on the right subplot
        ax_entropy.plot(time_history, entropy, marker='o', color='blue',
                        linewidth=4, markersize=5, label="Entropy")
        ax_entropy.set_xlabel("Time (s)")
        ax_entropy.set_ylabel("Entropy (bits)")
        ax_entropy.set_title("Entropy Over Time")
        # Assuming ax_traj is your subplot for the trajectory
        handles, labels = ax_traj.get_legend_handles_labels()
        # Optionally add any proxy handles if needed here

        # Place the legend below the subplot in one row
        ax_traj.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

        algorithm_labels = {
            "trees_to_trees": "Linear Path",
            "greedy": "Greedy Approach",
            "between_rows" : "Mower Path",
            "mpc": "Model Predictive Control",
            # Add additional mappings as needed
        }
        label = algorithm_labels.get(mode, mode)  
        # Overall figure title and layout adjustment
        fig.suptitle(f"{label}: Drone Trajectory and Entropy Trend", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure with a filename that includes the algorithm (mode) name
        output_path = os.path.join(baselines_dir, f"{mode}_baseline_comparison.png")
        plt.savefig(output_path)
        print(f"Saved comparison plot for {mode} to: {output_path}")

        plt.show()

if __name__ == "__main__":
    main()
