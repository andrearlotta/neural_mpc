import os
import glob
import csv
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_performance_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
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

def extract_timestamp(file_path):
    """
    Extracts timestamp from filename.
    Expects pattern: mpc_YYYYMMDD_HHMMSS_label_performance_metrics.csv
    """
    base = os.path.basename(file_path)
    clean = base.replace("mpc_", "").replace("_performance_metrics.csv", "")
    return clean

def main():
    # Base directory containing the metrics files
    base_folder = "/home/pantheon/drea/neural_mpc/comp_10_25_nmpc_metrics"
    pattern = os.path.join(base_folder, "**", "mpc_*performance_metrics.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("No MPC test files found.")
        return

    # For each file, ask the user to name it
    file_to_label = {}
    for file in sorted(files):
        timestamp = extract_timestamp(file)
        print(f"\nFound run with timestamp: {timestamp}")
        user_label = input("Which name should I assign to this run? (press enter to keep timestamp)\n> ")
        file_to_label[file] = user_label.strip() if user_label.strip() else timestamp

    metrics_data = {
        "Test": [],
        "Total Time (s)": [],
        "Total Distance (m)": [],
        "Average WP Time (s)": [],
        "Final Entropy": []
    }

    for file, label in file_to_label.items():
        metrics = load_performance_metrics(file)
        metrics_data["Test"].append(label)
        metrics_data["Total Time (s)"].append(metrics.get("Total Time (s)", np.nan))
        metrics_data["Total Distance (m)"].append(metrics.get("Total Distance (m)", np.nan))
        metrics_data["Average WP Time (s)"].append(metrics.get("Average WP Time (s)", np.nan))
        metrics_data["Final Entropy"].append(metrics.get("Final Entropy", np.nan))
    
    metrics_list = ["Total Time (s)", "Total Distance (m)", "Average WP Time (s)", "Final Entropy"]
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=metrics_list)

    for i, metric in enumerate(metrics_list):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(
            go.Bar(
                x=metrics_data["Test"],
                y=metrics_data[metric],
                text=[f"{v:.2f}" if not np.isnan(v) else "NaN" for v in metrics_data[metric]],
                textposition='auto'
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title="MPC Test Performance Comparison",
        height=600,
        width=900,
        showlegend=False
    )
    
    fig.show()

if __name__ == "__main__":
    main()
