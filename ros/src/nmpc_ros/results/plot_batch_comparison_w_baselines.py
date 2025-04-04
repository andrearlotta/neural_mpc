#!/usr/bin/env python
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting y-axis if needed

# Make plots look a bit nicer
plt.style.use('seaborn-v0_8')

# --- load_performance_metrics, load_average_velocity, gather_metrics ---
# (Keep these functions as they were in the previous corrected version)
# ... (functions omitted for brevity - assume they are the same as the previous answer) ...
def load_performance_metrics(file_path):
    """ Loads performance metrics from a CSV file. """
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip()
                    try:
                        # Handle potential whitespace or extra quotes
                        value_str = row[1].strip().strip('"').strip("'")
                        value = float(value_str)
                        metrics[key] = value
                    except ValueError:
                        print(f"Warning: Could not parse value '{row[1]}' for '{key}' in {file_path}. Skipping row.")
                        continue
    except FileNotFoundError:
        print(f"Error: Performance metrics file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading performance metrics file {file_path}: {e}")
        return None
    # Ensure required keys exist, even if they have to be NaN
    required_keys = [
        "Total Execution Time (s)", "Total Distance (m)",
        "Average Waypoint-to-Waypoint Time (s)", "Final Entropy",
        "Total Commands"
    ]
    for rkey in required_keys:
        if rkey not in metrics:
            # Keep Total Commands out of this example, maybe add it later if needed
             #metrics[rkey] = np.nan
             pass
    return metrics

def load_average_velocity(vel_file_path):
    """ Loads velocity CSV and computes median linear velocity. """
    try:
        df = pd.read_csv(vel_file_path)
        if df.empty or 'x_velocity' not in df.columns or 'y_velocity' not in df.columns:
            # Check specifically for non-numeric before warning? Better to just try calculation.
             # print(f"Warning: Velocity file {vel_file_path} is empty or missing required columns.")
             pass # Let the calculation attempt handle non-numeric? Or check dtype?
        
        # Ensure velocity columns are numeric, coercing errors to NaN
        df['x_velocity'] = pd.to_numeric(df['x_velocity'], errors='coerce')
        df['y_velocity'] = pd.to_numeric(df['y_velocity'], errors='coerce')

        # Drop rows where essential velocity data became NaN
        df.dropna(subset=['x_velocity', 'y_velocity'], inplace=True)

        if df.empty:
            print(f"Warning: Velocity file {vel_file_path} contains no valid numeric x/y velocity data.")
            return np.nan

        df['Linear Velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
        # Ensure Linear Velocity didn't somehow become NaN (shouldn't if x/y were valid)
        df.dropna(subset=['Linear Velocity'], inplace=True)
        if df.empty:
             print(f"Warning: No valid linear velocities computed for {vel_file_path}.")
             return np.nan
             
        return df['Linear Velocity'].median()
    except FileNotFoundError:
        # print(f"Warning: Velocity file not found: {vel_file_path}") # Reduces noise maybe
        return np.nan
    except Exception as e:
        print(f"Error loading or processing velocity file {vel_file_path}: {e}")
        return np.nan


def gather_metrics(base_dir):
    """ Searches subdirs, loads metrics/velocity, returns dict & test count. """
    run_folders = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not run_folders:
        print(f"Warning: No 'run_*' directories found in {base_dir}")

    metrics_lists = {
        "Total Execution Time (s)": [],
        "Final Entropy": [],
        "Total Distance (m)": [],
        "Average Velocity (m/s)": [],
        "Average NMPC Step Execution Time (s)": [] # From Waypoint-to-Waypoint time
    }

    tests_found_metrics = 0
    # print(f"Processing {len(run_folders)} run folders in {base_dir}...") # Less verbose default
    for run_idx, run in enumerate(run_folders):
        # Optional: print progress if many runs
        # if (run_idx + 1) % 10 == 0 or run_idx == 0 or run_idx == len(run_folders) - 1:
        #      print(f"  Processing run {run_idx+1}/{len(run_folders)} in {base_dir}...")

        perf_files = glob.glob(os.path.join(run, "*performance_metrics.csv"))
        if not perf_files:
            # print(f"  - No performance metrics CSV found in {run}") # Reduce noise
            continue

        perf_file = perf_files[0]
        perf = load_performance_metrics(perf_file)

        if perf is None:
             # print(f"  - Failed to load performance metrics from {perf_file}") # Reduce noise
             continue

        # --- This run has valid metrics, proceed ---
        tests_found_metrics += 1

        metrics_lists["Total Execution Time (s)"].append(perf.get("Total Execution Time (s)", np.nan))
        metrics_lists["Final Entropy"].append(perf.get("Final Entropy", np.nan))
        metrics_lists["Total Distance (m)"].append(perf.get("Total Distance (m)", np.nan))
        metrics_lists["Average NMPC Step Execution Time (s)"].append(perf.get("Average Waypoint-to-Waypoint Time (s)", np.nan))

        vel_files = glob.glob(os.path.join(run, "*_velocity_commands.csv"))
        if not vel_files:
            # print(f"  - No velocity commands CSV found in {run}. Storing NaN for Avg Velocity.") # Reduce noise
            metrics_lists["Average Velocity (m/s)"].append(np.nan)
        else:
             vel_file = vel_files[0]
             avg_vel = load_average_velocity(vel_file)
             metrics_lists["Average Velocity (m/s)"].append(avg_vel)

    if tests_found_metrics > 0 :
        print(f"Processed {base_dir}: Found metrics in {tests_found_metrics}/{len(run_folders)} runs.")
    # else: # Report if nothing found for an algorithm
         # print(f"Processed {base_dir}: No runs with valid metrics found.")

    return metrics_lists, tests_found_metrics

def compute_statistics(values):
    """ Removes NaNs/outliers (IQR), computes median, std, min, max. """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)] # Filter NaNs first!

    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Handle case with very few data points where IQR might be zero or percentiles ill-defined
    if arr.size < 4: # Need at least 4 points for robust IQR
        # Fallback to simple stats without outlier removal for small samples
         filtered_arr = arr
         # print(f"Warning: compute_statistics received {arr.size} valid points. Skipping IQR outlier removal.")
    else:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        # Handle IQR == 0 case (all data points in middle are same)
        if iqr == 0:
            # Keep all non-NaN data if IQR is zero
            lower_bound = q1 - 1e-9 # small tolerance
            upper_bound = q3 + 1e-9 # small tolerance
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

        filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]

    if filtered_arr.size == 0:
        # This can happen if extreme outliers skew IQR bounds
         print(f"Warning: Outlier filtering removed all {arr.size} data points. Returning NaN stats. Original data (first 5): {arr[:5]}")
         return np.nan, np.nan, np.nan, np.nan

    # Calculate stats on the (potentially filtered) array
    median_val = np.median(filtered_arr)
    std_val = np.std(filtered_arr)
    min_val = np.min(filtered_arr)
    max_val = np.max(filtered_arr)

    return median_val, std_val, min_val, max_val


# --- plot_grouped_metrics (for single algo) remains the same ---
# ... (function omitted for brevity) ...
def plot_grouped_metrics(metrics_data, num_tests, algo_name="Algorithm"):
    """ Creates a figure with two subplots for a SINGLE algorithm's metrics. """
    group1_keys = ["Average Velocity (m/s)", "Final Entropy", "Average NMPC Step Execution Time (s)"]
    group2_keys = ["Total Execution Time (s)", "Total Distance (m)"]

    def get_stats(keys): # Inner helper function
        medians, stds, mins, maxs = [], [], [], []
        for key in keys:
            if key in metrics_data and metrics_data[key]:
                # Use robust compute_statistics
                median, std, min_val, max_val = compute_statistics(metrics_data[key])
            else:
                # print(f"Warning: No data for metric '{key}' for {algo_name}. Plotting as NaN.")
                median, std, min_val, max_val = np.nan, np.nan, np.nan, np.nan
            medians.append(median)
            stds.append(std)
            mins.append(min_val)
            maxs.append(max_val)
        return medians, stds, mins, maxs

    medians1, stds1, mins1, maxs1 = get_stats(group1_keys)
    medians2, stds2, mins2, maxs2 = get_stats(group2_keys)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    cmap1 = plt.get_cmap('Set2')
    cmap2 = plt.get_cmap('Set3')

    # --- Group 1 Plot ---
    x1 = np.arange(len(group1_keys))
    # Use np arrays for easier handling of potential NaNs in yerr
    plot_medians1 = np.array(medians1, dtype=float)
    plot_stds1 = np.array(stds1, dtype=float)
    # Replace NaN std deviations with 0 for error bars (otherwise plotting might fail)
    plot_stds1_safe = np.nan_to_num(plot_stds1)

    bars1 = ax1.bar(x1, plot_medians1, yerr=plot_stds1_safe, align='center', alpha=0.85,
                    color=[cmap1(i) for i in range(len(group1_keys))],
                    error_kw={'ecolor': 'black', 'capsize': 8})
    ax1.set_xticks(x1)
    ax1.set_xticklabels(group1_keys, rotation=40, ha="right")
    ax1.set_ylabel("Median Value")
    ax1.set_title("Group 1: Velocity, Final Entropy & NMPC Step Time")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    valid_upper_bounds1 = [m + s for m, s in zip(plot_medians1, plot_stds1_safe) if not np.isnan(m)]
    if valid_upper_bounds1:
        ax1.set_ylim(0, max(valid_upper_bounds1) * 1.2)
    else:
        ax1.set_ylim(0, 1)

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        median_val = plot_medians1[i] # Get the potentially NaN value
        if not np.isnan(median_val):
             text_y_pos = height + (plot_stds1_safe[i] * 1.05) # Place above error bar
             ax1.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                     f"{median_val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
             # Add min/max below main value if height allows
             if height > 0: # Avoid cluttering baseline
                  ax1.text(bar.get_x() + bar.get_width()/2., height / 2,
                          f"Min: {mins1[i]:.2f}\nMax: {maxs1[i]:.2f}",
                          ha='center', va='center', fontsize=7, alpha=0.8)

    # --- Group 2 Plot ---
    x2 = np.arange(len(group2_keys))
    plot_medians2 = np.array(medians2, dtype=float)
    plot_stds2 = np.array(stds2, dtype=float)
    plot_stds2_safe = np.nan_to_num(plot_stds2)

    bars2 = ax2.bar(x2, plot_medians2, yerr=plot_stds2_safe, align='center', alpha=0.85,
                    color=[cmap2(i) for i in range(len(group2_keys))],
                    error_kw={'ecolor': 'black', 'capsize': 8})
    ax2.set_xticks(x2)
    ax2.set_xticklabels(group2_keys, rotation=40, ha="right")
    ax2.set_ylabel("Median Value")
    ax2.set_title("Group 2: Total Time & Total Distance")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    valid_upper_bounds2 = [m + s for m, s in zip(plot_medians2, plot_stds2_safe) if not np.isnan(m)]
    if valid_upper_bounds2:
        ax2.set_ylim(0, max(valid_upper_bounds2) * 1.2)
    else:
        ax2.set_ylim(0, 1)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        median_val = plot_medians2[i]
        if not np.isnan(median_val):
             text_y_pos = height + (plot_stds2_safe[i] * 1.05)
             ax2.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                     f"{median_val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
             if height > 0:
                 ax2.text(bar.get_x() + bar.get_width()/2., height / 2,
                          f"Min: {mins2[i]:.2f}\nMax: {maxs2[i]:.2f}",
                          ha='center', va='center', fontsize=7, alpha=0.8)

    fig.suptitle(f"Performance Statistics for {algo_name} (Tests: {num_tests})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"performance_summary_{algo_name}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Saved single algorithm plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig) # Close the figure to free memory


def run_single_analysis(algo_name, base_dir):
    """ Helper function to run analysis for one algorithm and plot it """
    print(f"\n--- Running Single Analysis for {algo_name} ---")
    metrics_data, num_tests = gather_metrics(base_dir)
    if num_tests > 0:
        plot_grouped_metrics(metrics_data, num_tests, algo_name)
    else:
        print(f"Skipping plot for {algo_name}: No valid test data.")
    return metrics_data, num_tests

def run_batch_analysis(base_dirs_dict):
    """ Analyzes multiple algorithms, generates comparison plots, exports summary. """
    all_metrics = {}
    test_counts = {}

    print("\n--- Starting Batch Analysis ---")
    for algo_name, base_dir in base_dirs_dict.items():
        print(f"\n--- Processing Algorithm: {algo_name} (Source: {base_dir}) ---")
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' not found. Skipping '{algo_name}'.")
            continue
        metrics_data, num_tests = gather_metrics(base_dir)
        if num_tests == 0:
            print(f"Warning: No valid test data found for '{algo_name}'. Excluding from comparison.")
            continue

        all_metrics[algo_name] = metrics_data
        test_counts[algo_name] = num_tests
        # Optional: Run single plot generation here if desired for each algo
        # run_single_analysis(algo_name, base_dir) # Calls plot_grouped_metrics

    if not all_metrics:
        print("\nError: No algorithms processed successfully with data. Exiting batch analysis.")
        return

    print("\n--- Generating Comparison Plots & Summary Table ---")
    plot_comparison_metrics(all_metrics, test_counts)
    export_summary_table(all_metrics, test_counts)
    print("\n--- Batch Analysis Complete ---")

# ============================================================================
# ============= MODIFIED PLOT_COMPARISON_METRICS =============================
# ============================================================================
def plot_comparison_metrics(all_metrics, test_counts):
    """ Plots grouped bar charts comparing algorithms, with bar clipping and fixed legend. """
    group1_metrics = ["Total Execution Time (s)", "Total Distance (m)"]
    group2_metrics = ["Final Entropy", "Average Velocity (m/s)"]

    algorithms = list(all_metrics.keys())
    num_algorithms = len(algorithms)
    if num_algorithms == 0:
        print("No algorithms have data for comparison plotting.")
        return

    # Determine bar width dynamically
    total_width = 0.8 # Total width allocated for bars within a group
    bar_width = total_width / num_algorithms
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7)) # Slightly wider

    legend_handles = [] # To store one handle per algorithm for the legend
    legend_labels = []

    # --- Helper function to plot one group ---
    def plot_group(ax, group_metrics, title, set_y_limit=None, is_first_plot=False):
        num_metrics = len(group_metrics)
        x_indices = np.arange(num_metrics)
        clip_threshold = set_y_limit if set_y_limit is not None else np.inf # Use limit for clipping if set

        all_plot_upper_bounds = [] # Track y-values for axis limits

        for i, algo in enumerate(algorithms):
            algo_medians = []
            algo_stds = []
            algo_raw_values = []

            for metric in group_metrics:
                 values = all_metrics.get(algo, {}).get(metric, [])
                 algo_raw_values.append(values)
                 median, std, _, _ = compute_statistics(values) # Get the TRUE median/std
                 algo_medians.append(median)
                 algo_stds.append(std)

            plot_medians = np.array(algo_medians, dtype=float)
            plot_stds = np.array(algo_stds, dtype=float)
            plot_stds_safe = np.nan_to_num(plot_stds) # Use 0 std for NaN error bars

            # Calculate positions
            offset = (i - num_algorithms / 2 + 0.5) * bar_width # Center group around tick
            bar_positions = x_indices + offset

            # --- Bar Clipping Logic ---
            plot_heights = plot_medians.copy() # Start with actual medians
            clipped_mask = plot_heights > clip_threshold # Find bars exceeding threshold
            plot_heights[clipped_mask] = clip_threshold # Cap their height

            bars = ax.bar(bar_positions, plot_heights, width=bar_width * 0.9, # Slightly thinner bars
                          label=f"{algo} (n={test_counts.get(algo, 0)})", # Label needed for initial handle collection
                          yerr=plot_stds_safe,
                          capsize=4,
                          alpha=0.9, color=cmap(i % cmap.N),
                          error_kw={'alpha': 0.6}) # Make error bars slightly transparent

            # Store legend info only from the first plot group
            if is_first_plot:
                if bars: # Ensure bars were actually created (data wasn't all NaN)
                    legend_handles.append(bars[0]) # Get handle from first bar of this algo
                    legend_labels.append(f"{algo} (n={test_counts.get(algo, 0)})")

            # Add hatching and text for clipped bars
            for j, bar in enumerate(bars):
                original_median = plot_medians[j] # The actual calculated median
                is_clipped = clipped_mask[j]

                if np.isnan(original_median): # Skip annotation for NaN bars
                    continue

                # Apply hatching if clipped
                if is_clipped:
                    bar.set_hatch('///')
                    bar.set_edgecolor('grey') # Make edge visible over hatch

                # --- Text Annotation ---
                text_y = plot_heights[j] # Start text position at (potentially clipped) bar top
                # Add error bar height to text position if std exists
                if not np.isnan(plot_stds[j]) and plot_stds[j] > 0:
                    # Make sure not to position based on error bar if the bar itself was clipped much lower
                    # If clipped, put text just above clip line. Otherwise, above error bar.
                     if is_clipped:
                         text_y = clip_threshold # Position text right above the clip line
                     else:
                         text_y += plot_stds_safe[j] # Position above error bar end

                # Add a small absolute offset based on current axis scale for clarity
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                text_y += y_offset

                # Use the ORIGINAL median for the text label!
                ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                        f"{original_median:.2f}", ha='center', va='bottom', fontsize=7, rotation=0, fontweight='bold')


            # Track max values for y-limit setting (use capped height + error bar for visible extent)
            valid_bounds = [h + s for h, s in zip(plot_heights, plot_stds_safe) if not np.isnan(h)]
            if valid_bounds:
                all_plot_upper_bounds.extend(valid_bounds)


        ax.set_xticks(x_indices)
        ax.set_xticklabels(group_metrics, rotation=30, ha='right')
        ax.set_ylabel("Median Value")
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Set Y limits
        if set_y_limit is not None:
            ax.set_ylim(0, set_y_limit * 1.1) # Add a little space above forced limit
        elif all_plot_upper_bounds:
             y_upper = max(all_plot_upper_bounds) * 1.15 # Increased space for text above bars
             ax.set_ylim(0, y_upper if not np.isnan(y_upper) and y_upper > 0 else 1)
        else:
             ax.set_ylim(0, 1) # Default if no valid data


    # --- Plot the two groups ---
    # Pass is_first_plot=True only to the first call to collect legend handles
    plot_group(ax1, group1_metrics, "Execution Time & Distance", is_first_plot=True)
    plot_group(ax2, group2_metrics, "Entropy & Velocity", set_y_limit=4.0) # Explicitly clip/limit at 4

    # --- Final Figure Touches ---
    fig.suptitle("Performance Comparison Across Algorithms", fontsize=18)

    # Use the collected handles and labels for a single, unified legend
    if legend_handles: # Check if we collected any handles
         fig.legend(handles=legend_handles, labels=legend_labels,
                   loc='upper center', ncol=min(num_algorithms, 4),
                   bbox_to_anchor=(0.5, 0.96), fontsize='medium')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust rect for suptitle and legend

    plot_filename = "comparison_performance_summary_clipped.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nSaved comparison plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving comparison plot {plot_filename}: {e}")

    plt.show() # Display the plot
    plt.close(fig) # Close figure after showing/saving

# ============================================================================
# --- export_summary_table remains the same ---
# ... (function omitted for brevity) ...
def export_summary_table(all_metrics, test_counts, output_path="summary_metrics_comparison.csv"):
    """ Creates a CSV summary table with median, std, min, max. """
    rows = []
    algorithms = list(all_metrics.keys())
    if not algorithms:
        print("Cannot export summary table: No algorithm data.")
        return

    all_metric_keys = set()
    for metrics in all_metrics.values():
        all_metric_keys.update(metrics.keys())
    sorted_metric_keys = sorted(list(all_metric_keys))

    for algo in algorithms:
        metrics = all_metrics.get(algo, {}) # Use get for safety
        t_count = test_counts.get(algo, 0)
        for metric in sorted_metric_keys:
            values = metrics.get(metric, [])
            # Always calculate true statistics for the table
            median, std, min_val, max_val = compute_statistics(values)
            rows.append({
                "Metric": metric,
                "Algorithm": algo,
                "Median": round(median, 4) if not np.isnan(median) else 'N/A',
                "Std Dev": round(std, 4) if not np.isnan(std) else 'N/A',
                "Min": round(min_val, 4) if not np.isnan(min_val) else 'N/A',
                "Max": round(max_val, 4) if not np.isnan(max_val) else 'N/A',
                "Test Count": t_count
            })

    if not rows:
         print("No data rows generated for the summary table.")
         return

    df = pd.DataFrame(rows)
    df = df[["Metric", "Algorithm", "Median", "Std Dev", "Min", "Max", "Test Count"]]
    df = df.sort_values(by=["Metric", "Algorithm"])

    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Summary table saved: {output_path}")
        print("\n--- Summary Statistics Table ---")
        # Use Pandas to_string for better console formatting, handle NaNs
        print(df.to_string(index=False, na_rep='N/A', float_format="%.4f"))
    except Exception as e:
        print(f"\n❌ Error saving summary table to {output_path}: {e}")


# --- __main__ block with updated dummy data ---
if __name__ == "__main__":
    base_dirs_to_analyze = {
        'mpc': 'mpc_test_runs_25_trees',
        'greedy': 'batch_test_25trees_greedy',
        'tree_to_tree': 'batch_test_25trees_tree_to_tree',
        'mower': 'batch_test_25trees_between_rows'
    }

    print("\n--- Setting up Test Environment (Dummy Files) ---")
    metrics_content = """Total Execution Time (s), {:.2f}
    Total Distance (m), {:.2f}
    Average Waypoint-to-Waypoint Time (s), {:.4f}
    Final Entropy, {:.3f}
    Total Commands, {:.0f}""" # Format string
    velocity_header = "Time (s),Tag,x_velocity,y_velocity,yaw_velocity\n" # Added missing commas

    np.random.seed(42) # Make dummy data repeatable

    for name, path in base_dirs_to_analyze.items():
        if not os.path.exists(path):
            print(f"Creating directory and dummy data for: {path}")
            os.makedirs(path, exist_ok=True)
            for i in range(1, 6): # Create 5 dummy runs
                run_path = os.path.join(path, f"run_{i:03d}")
                os.makedirs(run_path, exist_ok=True)

                # --- Generate dummy performance data ---
                exec_time = 100 + i*5 + np.random.randn()*10 * (1 if name != 'greedy' else 0.7) # Greedy faster
                distance = 50 + i*3 + np.random.randn()*5 * (1 if name != 'mower' else 1.5) # Mower maybe longer dist
                nmpc_time = 0.1 + np.random.rand()*0.05 * (1 if name != 'mpc' else 1.3) # MPC maybe slower steps
                # Make mower entropy high, others lower
                if name == 'mower':
                     final_entropy = 6.5 + np.random.randn() * 0.5 # << High entropy for mower
                else:
                     final_entropy = 2.5 - i*0.1 + np.random.randn()*0.2 # Lower for others
                commands = 300 + i*15 + np.random.randint(-20, 20)

                perf_file = os.path.join(run_path, f"{name}_run{i}_perf.csv") # Simpler name
                with open(perf_file, 'w') as f:
                    f.write(metrics_content.format(
                        max(10, exec_time), # Ensure non-negative
                        max(5, distance),
                        max(0.01, nmpc_time),
                        max(0.1, final_entropy),
                        max(50, commands)
                    ))

                # --- Generate dummy velocity data ---
                vel_file = os.path.join(run_path, f"{name}_run{i}_vel.csv") # Simpler name
                with open(vel_file, 'w') as f:
                    f.write(velocity_header)
                    # Make mower velocity higher/different for testing
                    base_vel = 0.5
                    if name == 'mower':
                         base_vel = 0.8
                    elif name == 'greedy':
                         base_vel = 0.6

                    for t in range(10): # More data points
                        vx = base_vel + np.random.randn()*0.1 * (1 if i%2==0 else 1.2) # Add variation
                        vy = np.random.randn()*0.05 * (1 if name != 'tree_to_tree' else 0.3) # tree_to_tree maybe wider turns
                        # Introduce occasional missing value or non-numeric value? Maybe later.
                        f.write(f"{t*0.2:.1f},cmd,{vx:.3f},{vy:.3f},{np.random.randn()*0.1:.3f}\n") # Note format string fixed

        else:
            print(f"Directory exists: {path}. Skipping dummy data creation.")


    # Run the full batch analysis
    run_batch_analysis(base_dirs_to_analyze)

    # Example: You could still run single analysis if needed
    # run_single_analysis('mower', base_dirs_to_analyze['mower'])