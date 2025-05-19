#!/usr/bin/env python
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 

plt.style.use('seaborn-v0_8')

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
    
    required_keys = [
        "Total Execution Time (s)", "Total Distance (m)",
        "Average Waypoint-to-Waypoint Time (s)", "Final Entropy",
        "Total Commands"
    ]
    for rkey in required_keys:
        if rkey not in metrics:
                         pass
    return metrics

def load_average_velocity(vel_file_path):
    """ Loads velocity CSV and computes median linear velocity. """
    try:
        df = pd.read_csv(vel_file_path)
        if df.empty or 'x_velocity' not in df.columns or 'y_velocity' not in df.columns:
            pass
        df['x_velocity'] = pd.to_numeric(df['x_velocity'], errors='coerce')
        df['y_velocity'] = pd.to_numeric(df['y_velocity'], errors='coerce')

        
        df.dropna(subset=['x_velocity', 'y_velocity'], inplace=True)

        if df.empty:
            print(f"Warning: Velocity file {vel_file_path} contains no valid numeric x/y velocity data.")
            return np.nan

        df['Linear Velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
        
        df.dropna(subset=['Linear Velocity'], inplace=True)
        if df.empty:
             print(f"Warning: No valid linear velocities computed for {vel_file_path}.")
             return np.nan
             
        return df['Linear Velocity'].median()
    except FileNotFoundError:
        
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
        "Average NMPC Step Execution Time (s)": [] 
    }

    tests_found_metrics = 0
    
    for run_idx, run in enumerate(run_folders):

        perf_files = glob.glob(os.path.join(run, "*performance_metrics.csv"))
        if not perf_files:
            print(f"  - No performance metrics CSV found in {run}")
            continue

        perf_file = perf_files[0]
        perf = load_performance_metrics(perf_file)

        if perf is None:
             print(f"  - Failed to load performance metrics from {perf_file}")
             continue

        
        tests_found_metrics += 1

        metrics_lists["Total Execution Time (s)"].append(perf.get("Total Execution Time (s)", np.nan))
        metrics_lists["Final Entropy"].append(perf.get("Final Entropy", np.nan))
        metrics_lists["Total Distance (m)"].append(perf.get("Total Distance (m)", np.nan))
        metrics_lists["Average NMPC Step Execution Time (s)"].append(perf.get("Average Waypoint-to-Waypoint Time (s)", np.nan))

        vel_files = glob.glob(os.path.join(run, "*_velocity_commands.csv"))
        if not vel_files:
            
            metrics_lists["Average Velocity (m/s)"].append(np.nan)
        else:
             vel_file = vel_files[0]
             avg_vel = load_average_velocity(vel_file)
             metrics_lists["Average Velocity (m/s)"].append(avg_vel)

    if tests_found_metrics > 0 :
        print(f"Processed {base_dir}: Found metrics in {tests_found_metrics}/{len(run_folders)} runs.")
    
         

    return metrics_lists, tests_found_metrics

def compute_statistics(values):
    """ Removes NaNs/outliers (IQR), computes median, std, min, max. """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)] 

    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    
    if arr.size < 4: 
        
         filtered_arr = arr
         
    else:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        
        if iqr == 0:
            
            lower_bound = q1 - 1e-9 
            upper_bound = q3 + 1e-9 
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

        filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]

    if filtered_arr.size == 0:
        
         print(f"Warning: Outlier filtering removed all {arr.size} data points. Returning NaN stats. Original data (first 5): {arr[:5]}")
         return np.nan, np.nan, np.nan, np.nan

    
    median_val = np.median(filtered_arr)
    std_val = np.std(filtered_arr)
    min_val = np.min(filtered_arr)
    max_val = np.max(filtered_arr)

    return median_val, std_val, min_val, max_val




def plot_grouped_metrics(metrics_data, num_tests, algo_name="Algorithm"):
    """ Creates a figure with two subplots for a SINGLE algorithm's metrics. """
    group1_keys = ["Average Velocity (m/s)", "Final Entropy", "Average NMPC Step Execution Time (s)"]
    group2_keys = ["Total Execution Time (s)", "Total Distance (m)"]

    def get_stats(keys): 
        medians, stds, mins, maxs = [], [], [], []
        for key in keys:
            if key in metrics_data and metrics_data[key]:
                
                median, std, min_val, max_val = compute_statistics(metrics_data[key])
            else:
                
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

    
    x1 = np.arange(len(group1_keys))
    
    plot_medians1 = np.array(medians1, dtype=float)
    plot_stds1 = np.array(stds1, dtype=float)
    
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
        median_val = plot_medians1[i] 
        if not np.isnan(median_val):
             text_y_pos = height + (plot_stds1_safe[i] * 1.05) 
             ax1.text(bar.get_x() + bar.get_width()/2., text_y_pos,
                     f"{median_val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
             
             if height > 0: 
                  ax1.text(bar.get_x() + bar.get_width()/2., height / 2,
                          f"Min: {mins1[i]:.2f}\nMax: {maxs1[i]:.2f}",
                          ha='center', va='center', fontsize=7, alpha=0.8)

    
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
    plt.close(fig) 


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
        
        

    if not all_metrics:
        print("\nError: No algorithms processed successfully with data. Exiting batch analysis.")
        return

    print("\n--- Generating Comparison Plots & Summary Table ---")
    plot_comparison_metrics(all_metrics, test_counts)
    export_summary_table(all_metrics, test_counts)
    print("\n--- Batch Analysis Complete ---")




def plot_comparison_metrics(all_metrics, test_counts):
    """ Plots grouped bar charts comparing algorithms, with bar clipping and fixed legend. """
    group1_metrics = ["Total Execution Time (s)", "Total Distance (m)"]
    group2_metrics = ["Final Entropy", "Average Velocity (m/s)"]

    algorithms = list(all_metrics.keys())
    num_algorithms = len(algorithms)
    if num_algorithms == 0:
        print("No algorithms have data for comparison plotting.")
        return

    
    total_width = 0.8 
    bar_width = total_width / num_algorithms
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7)) 

    legend_handles = [] 
    legend_labels = []

    
    def plot_group(ax, group_metrics, title, set_y_limit=None, is_first_plot=False):
        num_metrics = len(group_metrics)
        x_indices = np.arange(num_metrics)
        clip_threshold = set_y_limit if set_y_limit is not None else np.inf 

        all_plot_upper_bounds = [] 

        for i, algo in enumerate(algorithms):
            algo_medians = []
            algo_stds = []
            algo_raw_values = []

            for metric in group_metrics:
                 values = all_metrics.get(algo, {}).get(metric, [])
                 algo_raw_values.append(values)
                 median, std, _, _ = compute_statistics(values) 
                 algo_medians.append(median)
                 algo_stds.append(std)

            plot_medians = np.array(algo_medians, dtype=float)
            plot_stds = np.array(algo_stds, dtype=float)
            plot_stds_safe = np.nan_to_num(plot_stds) 

            
            offset = (i - num_algorithms / 2 + 0.5) * bar_width 
            bar_positions = x_indices + offset

            
            plot_heights = plot_medians.copy() 
            clipped_mask = plot_heights > clip_threshold 
            plot_heights[clipped_mask] = clip_threshold 

            bars = ax.bar(bar_positions, plot_heights, width=bar_width * 0.9, 
                          label=f"{algo} (n={test_counts.get(algo, 0)})", 
                          yerr=plot_stds_safe,
                          capsize=4,
                          alpha=0.9, color=cmap(i % cmap.N),
                          error_kw={'alpha': 0.6}) 

            
            if is_first_plot:
                if bars: 
                    legend_handles.append(bars[0]) 
                    legend_labels.append(f"{algo} (n={test_counts.get(algo, 0)})")

            
            for j, bar in enumerate(bars):
                original_median = plot_medians[j] 
                is_clipped = clipped_mask[j]

                if np.isnan(original_median): 
                    continue

                
                if is_clipped:
                    bar.set_hatch('///')
                    bar.set_edgecolor('grey') 

                
                text_y = plot_heights[j] 
                
                if not np.isnan(plot_stds[j]) and plot_stds[j] > 0:
                    
                    
                     if is_clipped:
                         text_y = clip_threshold 
                     else:
                         text_y += plot_stds_safe[j] 

                
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                text_y += y_offset

                
                ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                        f"{original_median:.2f}", ha='center', va='bottom', fontsize=7, rotation=0, fontweight='bold')


            
            valid_bounds = [h + s for h, s in zip(plot_heights, plot_stds_safe) if not np.isnan(h)]
            if valid_bounds:
                all_plot_upper_bounds.extend(valid_bounds)


        ax.set_xticks(x_indices)
        ax.set_xticklabels(group_metrics, rotation=30, ha='right')
        ax.set_ylabel("Median Value")
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        
        if set_y_limit is not None:
            ax.set_ylim(0, set_y_limit * 1.1) 
        elif all_plot_upper_bounds:
             y_upper = max(all_plot_upper_bounds) * 1.15 
             ax.set_ylim(0, y_upper if not np.isnan(y_upper) and y_upper > 0 else 1)
        else:
             ax.set_ylim(0, 1) 


    
    plot_group(ax1, group1_metrics, "Execution Time & Distance", is_first_plot=True)
    plot_group(ax2, group2_metrics, "Entropy & Velocity", set_y_limit=4.0)


    fig.suptitle("Performance Comparison Across Algorithms", fontsize=18)


    if legend_handles: 
         fig.legend(handles=legend_handles, labels=legend_labels,
                   loc='upper center', ncol=num_algorithms,
                   bbox_to_anchor=(0.5, 0.96), fontsize='medium')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 

    plot_filename = "comparison_performance_summary_clipped_random_trees.eps"
    try:
        plt.savefig(plot_filename)
        print(f"\nSaved comparison plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving comparison plot {plot_filename}: {e}")

    plt.show()
    plt.close(fig)


def export_summary_table(all_metrics, test_counts, output_path="summary_metrics_comparison_random_trees.csv"):
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
        metrics = all_metrics.get(algo, {})
        t_count = test_counts.get(algo, 0)
        for metric in sorted_metric_keys:
            values = metrics.get(metric, [])
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
        print(df.to_string(index=False, na_rep='N/A', float_format="%.4f"))
    except Exception as e:
        print(f"\n❌ Error saving summary table to {output_path}: {e}")


if __name__ == "__main__":
    base_dirs_to_analyze = {
        'neural_mpc': 'mpc_test_runs_25_trees',
        'greedy': 'batch_test_trees_greedy_gt',
        'tree_to_tree': 'batch_test_trees_linear_gt',
        'mower': 'batch_test_trees_mower_gt'
    }

    print("\n--- Setting up Test Environment (Dummy Files) ---")
    metrics_content = """Total Execution Time (s), {:.2f}
    Total Distance (m), {:.2f}
    Average Waypoint-to-Waypoint Time (s), {:.4f}
    Final Entropy, {:.3f}
    Total Commands, {:.0f}""" 
    velocity_header = "Time (s),Tag,x_velocity,y_velocity,yaw_velocity\n"
    run_batch_analysis(base_dirs_to_analyze)