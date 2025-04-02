#!/usr/bin/env python
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_performance_metrics(file_path):
    """
    Loads performance metrics from a CSV file.
    Expected rows (one per metric) with:
      - "Total Execution Time (s)"
      - "Total Distance (m)"
      - "Average Waypoint-to-Waypoint Time (s)" (used here as Average NMPC Step Execution Time)
      - "Final Entropy"
      - "Total Commands" (not used in the plot)
    """
    metrics = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                # Remove any extra spaces and convert value to float.
                key = row[0].strip()
                try:
                    value = float(row[1].strip())
                    metrics[key] = value
                except ValueError:
                    continue
    return metrics

def load_average_velocity(vel_file_path):
    """
    Loads the velocity commands CSV and computes the mean linear velocity.
    The CSV is expected to have columns:
      Time (s), Tag, x_velocity, y_velocity, yaw_velocity
    """
    try:
        df = pd.read_csv(vel_file_path)
        # Compute the linear (translational) velocity as the Euclidean norm of x and y velocities.
        df['Linear Velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
        return df['Linear Velocity'].median()
    except Exception as e:
        print(f"Error loading {vel_file_path}: {e}")
        return np.nan

def gather_metrics(base_dir):
    """
    Searches through subdirectories (run_*) in the given base directory,
    loads the performance metrics and velocity CSV files from each run folder,
    and returns a dictionary with lists for each metric.
    The five metrics are:
      - Total Execution Time (s)
      - Final Entropy (interpreted as final entropy reduction)
      - Total Distance (m)
      - Average Velocity (m/s)
      - Average NMPC Step Execution Time (s) (from Average Waypoint-to-Waypoint Time)
    """
    run_folders = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    total_exec_times = []
    final_entropies = []
    total_distances = []
    avg_velocities = []
    avg_nmpc_step_times = []
    print(len(run_folders))
    for run in run_folders:
        # Look for the performance metrics CSV in the current run folder.
        perf_files = glob.glob(os.path.join(run, "mpc_*_performance_metrics.csv"))
        if not perf_files:
            print(f"No performance metrics CSV found in {run}")
            continue
        perf_file = perf_files[0]
        perf = load_performance_metrics(perf_file)
        
        if "Total Execution Time (s)" in perf:
            total_exec_times.append(perf["Total Execution Time (s)"])
        if "Final Entropy" in perf:
            final_entropies.append(perf["Final Entropy"])
        if "Total Distance (m)" in perf:
            total_distances.append(perf["Total Distance (m)"])
        if "Average Waypoint-to-Waypoint Time (s)" in perf:
            avg_nmpc_step_times.append(perf["Average Waypoint-to-Waypoint Time (s)"])
        
        # Look for the velocity commands CSV in the run folder.
        vel_files = glob.glob(os.path.join(run, "mpc_*_velocity_commands.csv"))
        if not vel_files:
            print(f"No velocity commands CSV found in {run}")
            avg_velocities.append(np.nan)
        else:
            vel_file = vel_files[0]
            avg_velocities.append(load_average_velocity(vel_file))
    
    metrics = {
        "Total Execution Time (s)": total_exec_times,
        "Final Entropy": final_entropies,
        "Total Distance (m)": total_distances,
        "Average Velocity (m/s)": avg_velocities,
        "Average NMPC Step Execution Time (s)": avg_nmpc_step_times
    }
    return metrics

def compute_statistics(values):
    """
    Computes the mean, standard deviation, minimum, and maximum of a list of values.
    """
    arr = np.array(values)
    mean = np.median(arr)
    std = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    return mean, std, min_val, max_val

def plot_metrics(metrics_data):
    """
    Creates a bar chart with error bars for each metric.
    Each bar shows the mean value, with error bars indicating ± std.
    The min and max values are annotated on each bar.
    """
    metric_names = list(metrics_data.keys())
    means = []
    stds = []
    mins = []
    maxs = []
    
    for key in metric_names:
        mean, std, min_val, max_val = compute_statistics(metrics_data[key])
        means.append(mean)
        stds.append(std)
        mins.append(min_val)
        maxs.append(max_val)
    
    x = np.arange(len(metric_names))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Performance Statistics Across Runs")
    
    # Annotate each bar with its min and max values.
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"min: {mins[i]:.2f}\nmax: {maxs[i]:.2f}",
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set the base directory where the run folders (run_*) are located.
    base_dir = "test_runs"  # Change this if the run folders are in a different directory.
    
    metrics_data = gather_metrics(base_dir)
    
    # Print out statistics for each metric.
    for key, values in metrics_data.items():
        mean, std, min_val, max_val = compute_statistics(values)
        print(f"{key}: Mean={mean:.2f}, Std={std:.2f}, Min={min_val:.2f}, Max={max_val:.2f}")
    
    # Plot the bar chart with error bars.
    plot_metrics(metrics_data)

if __name__ == "__main__":
    main()
