from nmpc_ros_package.nmpc  import NeuralMPC
import os
if __name__ == '__main__':
    
    # Run 100 tests consecutively.
    for test_num in range(9, 50):
        print(f"================== Starting Test Run {test_num} ==================")
        # Base folder to store all test run outputs.
        base_test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpc_test_runs_25_trees")
        os.makedirs(base_test_folder, exist_ok=True)
        # Generate a valid random initial state.
        # Get the field domain from the tree positions.
        # Create a dedicated folder for this run.
        run_folder = os.path.join(base_test_folder, f"run_{test_num}")
        os.makedirs(run_folder, exist_ok=True)
        mpc = NeuralMPC(run_dir=run_folder, initial_randomic=True)
        # Run the simulation with the specified initial state and output folder.
        sim_results = mpc.run_simulation()
        # Optionally, plot the entropy and tree lambda trends for this run.
        #mpc.plot_entropy_separately()
        #mpc.plot_tree_lambda_trends()