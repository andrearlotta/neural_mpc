from nmpc_ros_package.nmpc  import NeuralMPC

if __name__ == '__main__':
    # Run the simulation with the default parameters
    mpc = NeuralMPC()
    all_trajectories, entropy_history, lambda_history, durations, g_nn_raw, g_nn_ripe, trees_pos, lb, ub =  mpc.run_simulation()
    entropy_mpc_pred = mpc.plot_animated_trajectory_and_entropy_2d(all_trajectories, entropy_history, lambda_history, trees_pos, lb, ub, durations)