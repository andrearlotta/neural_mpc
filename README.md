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
