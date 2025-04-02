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
                key = row[0].strip()
                try:
                    value = float(row[1].strip())
                    metrics[key] = value
                except ValueError:
                    continue
    return metrics

def load_average_velocity(vel_file_path):
    """
    Loads the velocity commands CSV and computes the median linear velocity.
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
    and returns a dictionary with lists for each metric along with the number of tests
    (i.e. runs that contained valid performance metrics).
    The five metrics are:
      - Total Execution Time (s)
      - Final Entropy
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
    
    tests_executed = 0
    print(f"Found {len(run_folders)} run folders.")
    for run in run_folders:
        # Look for the performance metrics CSV in the current run folder.
        perf_files = glob.glob(os.path.join(run, "mpc_*_performance_metrics.csv"))
        if not perf_files:
            print(f"No performance metrics CSV found in {run}")
            continue
        
        tests_executed += 1  # Count this run since it has performance metrics
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
    return metrics, tests_executed

def compute_statistics(values):
    """
    Removes outliers using the IQR method and then computes the median, standard deviation,
    minimum, and maximum of the filtered list of values.
    """
    arr = np.array(values)
    if len(arr) == 0:
        return 0, 0, 0, 0
    # Compute the 25th and 75th percentiles
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    # Determine bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Filter out outliers
    filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    mean = np.median(filtered_arr)
    std = np.std(filtered_arr)
    min_val = np.min(filtered_arr)
    max_val = np.max(filtered_arr)
    return mean, std, min_val, max_val

def plot_grouped_metrics(metrics_data, num_tests):
    """
    Creates a figure with two subplots.
      - Left subplot (Group 1): Average Velocity, Final Entropy, and Average NMPC Step Execution Time.
      - Right subplot (Group 2): Total Execution Time and Total Distance.
    Each bar shows the median (after outlier removal) with error bars indicating ± standard deviation.
    The y-axis in each subplot is shared among its metrics.
    The overall figure title includes the number of tests examined.
    """
    # Define groups
    group1_keys = ["Average Velocity (m/s)", "Final Entropy", "Average NMPC Step Execution Time (s)"]
    group2_keys = ["Total Execution Time (s)", "Total Distance (m)"]
    
    # Prepare statistics for each group.
    def get_stats(keys):
        medians, stds, mins, maxs = [], [], [], []
        for key in keys:
            mean, std, min_val, max_val = compute_statistics(metrics_data[key])
            medians.append(mean)
            stds.append(std)
            mins.append(min_val)
            maxs.append(max_val)
        return medians, stds, mins, maxs

    medians1, stds1, mins1, maxs1 = get_stats(group1_keys)
    medians2, stds2, mins2, maxs2 = get_stats(group2_keys)
    
    # Create two subplots side-by-side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap1 = plt.get_cmap('Set2')
    cmap2 = plt.get_cmap('Set3')
    
    # Group 1 plot
    x1 = np.arange(len(group1_keys))
    bars1 = ax1.bar(x1, medians1, yerr=stds1, align='center', alpha=0.8, color=[cmap1(i) for i in range(len(group1_keys))], 
                    ecolor='black', capsize=10)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(group1_keys, rotation=45, ha="right")
    ax1.set_ylabel("Value")
    ax1.set_title("Group 1: Velocity, Final Entropy & NMPC Step Exec Time")
    # Set common y-axis scale for group 1
    ax1.set_ylim(0, max([m + s for m, s in zip(medians1, stds1)]) * 1.2)
    # Annotate bars in group 1.
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()*3/4, height,
                 f"min: {mins1[i]:.2f}\nmax: {maxs1[i]:.2f}",
                 ha='center', va='bottom', fontsize=8)

    # Group 2 plot
    x2 = np.arange(len(group2_keys))
    bars2 = ax2.bar(x2, medians2, yerr=stds2, align='center', alpha=0.8, color=[cmap2(i) for i in range(len(group2_keys))],
                    ecolor='black', capsize=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(group2_keys, rotation=45, ha="right")
    ax2.set_ylabel("Value")
    ax2.set_title("Group 2: Total Time & Total Distance")
    # Set common y-axis scale for group 2
    ax2.set_ylim(0, max([m + s for m, s in zip(medians2, stds2)]) * 1.2)
    # Annotate bars in group 2.
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()*3/4, height,
                 f"min: {mins2[i]:.2f}\nmax: {maxs2[i]:.2f}",
                 ha='center', va='bottom', fontsize=8)
    
    # Add the number of tests examined in the overall title.
    fig.suptitle(f"Performance Statistics Across Runs (Number of tests: {num_tests + 1})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def main():
    # Set the base directory where the run folders (run_*) are located.
    base_dir = "test_runs"  # Change this if the run folders are in a different directory.
    
    metrics_data, num_tests = gather_metrics(base_dir)
    
    # Print out statistics for each metric after removing outliers.
    for key, values in metrics_data.items():
        mean, std, min_val, max_val = compute_statistics(values)
        print(f"{key}: Median={mean:.2f}, Std={std:.2f}, Min={min_val:.2f}, Max={max_val:.2f}")
    
    # Plot the grouped bar charts.
    plot_grouped_metrics(metrics_data, num_tests)

if __name__ == "__main__":
    main()
