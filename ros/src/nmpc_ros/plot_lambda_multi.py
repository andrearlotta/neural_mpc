import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_latest_csv(mode, directory, suffix="plot_data.csv"):
    pattern = os.path.join(directory, f"{mode}_*_{suffix}")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_csv_data(csv_path):
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    parts = first_line.split(',')
    
    df = pd.read_csv(csv_path, skiprows=1)
    time_history = df["time"].values
    lambda_columns = [col for col in df.columns if col.startswith("lambda_")]
    lambda_history = df[lambda_columns].values
    
    return time_history, lambda_history, lambda_columns

def main():
    modes = ["mpc_n1", "mpc_n2", "mpc_n3"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baselines_dir = os.path.join(script_dir, "src/baselines")

    all_data = {}
    
    for mode in modes:
        csv_file = get_latest_csv(mode, baselines_dir)
        if not csv_file:
            print(f"No CSV found for {mode}")
            continue
        try:
            time, lambda_history, lambda_columns = load_csv_data(csv_file)
            all_data[mode] = (time, lambda_history)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not all_data:
        print("No data to plot")
        return
    
    num_lambdas = len(lambda_columns)
    colors = ["orange", "blue", "green"]

    # Calculate the number of rows needed for 4 columns per row
    num_columns = 4
    num_rows = (num_lambdas + num_columns - 1) // num_columns  # This is a ceiling division

    # Create subplots (4 columns per row)
    fig = make_subplots(
        rows=num_rows, 
        cols=num_columns, 
        shared_xaxes=True, 
        subplot_titles=[f"Lambda {i}" for i in range(num_lambdas)]
    )
    
    # Add data for each lambda
    for lambda_idx in range(num_lambdas):
        row = lambda_idx // num_columns + 1  # Calculate row (1-based index)
        col = lambda_idx % num_columns + 1  # Calculate column (1-based index)
        
        for i, mode in enumerate(modes):
            if mode in all_data:
                time, lambda_history = all_data[mode]
                fig.add_trace(
                    go.Scatter(
                        x=time, y=lambda_history[:, lambda_idx],
                        mode="lines+markers",
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=5, color=colors[i]),
                        name=f"{mode}"
                    ),
                    row=row, col=col  # Adding the trace to the correct subplot (row, column)
                )

    # Update layout for the figure
    fig.update_layout(
        title="Lambda Values Over Time for Each Tree",
        xaxis_title="Time (s)",
        yaxis_title="Lambda Value",
        template="plotly_white",
        showlegend=True
    )

    # Save and show the figure
    output_path = os.path.join(baselines_dir, "all_lambdas_plot.html")
    fig.write_html(output_path)
    print(f"Saved combined plot to: {output_path}")
    
    fig.show()

if __name__ == "__main__":
    main()
