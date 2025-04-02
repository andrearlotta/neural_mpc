import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))  
src_path = os.path.join(script_dir, "src")  
sys.path.append(src_path)
from nmpc_multi import NeuralMPC
# from nmpc_multi_grid_test import NeuralMPC

if __name__ == '__main__':
    # Run the simulation with the default parameters
    mpc = NeuralMPC()
    all_trajectories, entropy_history, lambda_history, durations, g_nn_raw, g_nn_ripe, trees_pos, lb, ub =  mpc.run_simulation()
    entropy_mpc_pred = mpc.plot_animated_trajectory_and_entropy_2d(all_trajectories, entropy_history, lambda_history, trees_pos, lb, ub, durations)