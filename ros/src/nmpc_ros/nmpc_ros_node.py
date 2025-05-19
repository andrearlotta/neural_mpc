from nmpc_ros_package.nmpc  import NeuralMPC
import os
import re


if __name__ == '__main__':
    N_tests = 1
    for test_num in range(0,N_tests):
        # Base folder to store all test run outputs.
        base_test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_test_100_trees_nmpc_05_25")
        os.makedirs(base_test_folder, exist_ok=True)

        # Find the next test number
        existing_runs = [
            int(match.group(1)) for d in os.listdir(base_test_folder)
            if (match := re.match(r'run_(\d+)', d)) and os.path.isdir(os.path.join(base_test_folder, d))
        ]
        next_test_num = max(existing_runs, default=0) + 1

        # Create run folder
        run_folder = os.path.join(base_test_folder, f"run_{next_test_num}")
        os.makedirs(run_folder, exist_ok=True)
        print(f"================== Starting Test Run {next_test_num} ==================")

        mpc = NeuralMPC(run_dir=run_folder, initial_randomic=True)
        mpc.run_simulation()
        # Optionally, plot the entropy and tree lambda trends for this run.
        #mpc.plot_entropy_separately()