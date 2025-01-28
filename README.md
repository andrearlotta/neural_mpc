# Neural MPC for Tree Monitoring Using a Surrogate Model

This repository contains a Python implementation of a Neural Model Predictive Control (MPC) system for a drone navigating a field of trees. The goal is to estimate tree maturity confidence using a surrogate model, optimize the drone's trajectory, and visualize the results.

---

## Features
- Train a neural network (NN) as a surrogate model for estimating confidence based on tree and drone positions.
- Perform Bayesian updates to refine tree confidence estimates.
- Implement Neural MPC for trajectory planning to maximize information gain (reduce entropy).
- Visualize results, including drone trajectory, entropy reduction, and computation durations.

---

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Train the Neural Network** (Optional):
    - Set `train = True` in the script to train the NN. Training data is synthetically generated based on drone and tree positions.
    - The trained model is saved in the `saved_models_*` directory.

2. **Run the MPC Script**:
    - Simply run the script using:
      ```bash
      python script_name.py
      ```
    - The script will load the pre-trained NN and simulate the drone navigating the environment.

3. **View Visualizations**:
    - The script generates an animated plot that includes:
      - Drone trajectory
      - Tree maturity confidence (`λ`) changes
      - Entropy reduction
      - Computation durations for MPC iterations
    - Results are also saved as an interactive HTML file: `neural_mpc_results_lambda_maximization.html`.

---

## Configuration

- The script allows for adjusting key parameters like:
  - Neural network architecture (hidden layers, neurons, etc.).
  - MPC parameters such as horizon, time step, and state/control constraints.
  - Tree grid size and spacing.

Modify these parameters directly in the script to experiment with different settings.

---

## Outputs

- **Animation**: An animated visualization of the drone's trajectory and the confidence (`λ`) updates.
- **HTML File**: Interactive visualization saved as `neural_mpc_results.html`.

---
