from nmpc_ros_package.nmpc_decay  import run_simulation, plot_animated_trajectory_and_entropy_2d

if __name__ == '__main__':
    # Run the simulation with the default parameters
    all_trajectories, entropy_history, lambda_history, durations, g_nn, trees_pos, lb, ub = run_simulation()
    entropy_mpc_pred = plot_animated_trajectory_and_entropy_2d(all_trajectories, entropy_history, lambda_history, trees_pos, lb, ub, durations)