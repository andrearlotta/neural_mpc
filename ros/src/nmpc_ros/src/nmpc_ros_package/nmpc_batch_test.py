#!/usr/bin/env python
import os
import re
import time
import csv
import math
import threading

import rospy
import casadi as ca
import numpy as np
import torch
from scipy.stats import norm

from nmpc_ros_package.ros_com_lib.bridge_class import BridgeClass
from nmpc_ros_package.ros_com_lib.sensors import SENSORS

import l4casadi as l4c
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Simple Neural Network
# ---------------------------
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        # If input_dim==3 we add an extra input for sin/cos of the angle.
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        # If the last dimension is 3, assume the third element is an angle
        # and replace it with its sin and cos.
        if x.shape[-1] == 3:
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x

# ---------------------------
# Neural MPC Class
# ---------------------------
class NeuralMPC:
    def __init__(self):
        # Global Constants and Parameters
        self.hidden_size = 64
        self.hidden_layers = 3
        self.nn_input_dim = 3

        self.N = 5
        self.dt = 0.5
        self.T = self.dt * self.N
        self.nx = 3  # Represents [x, y, theta]

        # Initialize the sensor bridge.
        self.bridge = BridgeClass(SENSORS)
        self.trees_pos = np.array(self.bridge.call_server({"trees_poses": None})["trees_poses"])
        lb, ub = self.get_domain(self.trees_pos)
        initial_state = self.generate_random_initial_state(lb, ub, margin=1.25)
        print(f"Initial State: {initial_state}")
        # Publish the initial random position.
        self.bridge.pub_data({ "cmd_pose": initial_state})
        rospy.sleep(2.5)
        self.latest_trees_scores  = self.bridge.update_robot_state() 

        # MPC horizon (number of steps)
        self.mpc_horizon = self.N
        # Initialize two lambda vectors: one for raw and one for ripe.
        self.lambda_k_raw = ca.DM.ones(self.trees_pos.shape[0], 1) * 0.5
        self.lambda_k_ripe = ca.DM.ones(self.trees_pos.shape[0], 1) * 0.5

    # ---------------------------
    # New: Generate a Random Initial State
    # ---------------------------
    def generate_random_initial_state(self, lb, ub, margin=1.25):
        """
        Generate a random initial state [x, y, theta] such that:
         - x is in [lb[0], ub[0]] and y in [lb[1], ub[1]]
         - The position is at least `margin` meters away from every tree.
         - theta is chosen uniformly from [-pi, pi].
        """
        while True:
            x = np.random.uniform(lb[0], ub[0])
            y = np.random.uniform(lb[1], ub[1])
            valid = True
            for tree in self.trees_pos:
                if np.linalg.norm(np.array([x, y]) - tree) < margin:
                    valid = False
                    break
            if valid:
                theta = np.random.uniform(-np.pi, np.pi)
                return np.array([x, y, theta]).reshape(-1,1)

    # ---------------------------
    # Sensor Callback Thread
    # ---------------------------
    def measurement_callback(self, measurement_interval=0.5):
        while not rospy.is_shutdown():
            data = self.bridge.get_data()
            if data["tree_scores"] is not None:
                # Assume tree_scores is an array with values in [0,1]
                scores = np.array(data["tree_scores"])
                raw_scores = scores.copy()
                ripe_scores = scores.copy()
                # For each tree score, if the value is less than 0.5,
                # flip it so that it lies between 0.5 and 1 for the raw branch,
                # and set the ripe branch to a neutral value (here 0.5).
                # Otherwise, update the ripe branch with the original value and keep raw neutral.
                for i, val in enumerate(scores):
                    if val < 0.5:
                        raw_scores[i] = 0.5 + (0.5 - val)  # flip: e.g., 0.2 becomes 0.8
                        ripe_scores[i] = 0.5
                    else:
                        raw_scores[i] = 0.5
                        ripe_scores[i] = val
                # Update both lambda arrays using the bayes update.
                self.lambda_k_raw = self.bayes(self.lambda_k_raw, ca.DM(raw_scores))
                self.lambda_k_ripe = self.bayes(self.lambda_k_ripe, ca.DM(ripe_scores))
                self.lambda_k_raw = np.ceil(self.lambda_k_raw * 1000) / 1000
                self.lambda_k_ripe = np.ceil(self.lambda_k_ripe * 1000) / 1000
                self.latest_trees_scores = scores
            time.sleep(measurement_interval)

    def start_measurement_thread(self, measurement_interval=0.5):
        meas_thread = threading.Thread(
            target=self.measurement_callback, args=(measurement_interval,)
        )
        meas_thread.daemon = True
        meas_thread.start()

    # ---------------------------
    # Utility Functions
    # ---------------------------
    def get_latest_best_model(self):
        # Get the directory where THIS script is stored.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Models are stored in a subdirectory "models/" relative to this script.
        model_dir = os.path.join(script_dir, "models")
        # Find the latest best model file matching the pattern.
        model_files = [
            f for f in os.listdir(model_dir)
            if re.match(r"best_model_epoch_(\d+)\.pth", f)
        ]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        latest_model = max(
            model_files,
            key=lambda x: int(re.match(r"best_model_epoch_(\d+)\.pth", x).group(1))
        )
        return os.path.join(model_dir, latest_model)

    @staticmethod
    def get_domain(tree_positions):
        """Return the domain (bounding box) of the tree positions."""
        x_min = np.min(tree_positions[:, 0])
        x_max = np.max(tree_positions[:, 0])
        y_min = np.min(tree_positions[:, 1])
        y_max = np.max(tree_positions[:, 1])
        return [x_min, y_min], [x_max, y_max]

    @staticmethod
    def kin_model(nx, dt):
        """
        Kinematic model: state X = [x, y, theta, vx, vy, omega] and control U = [ax, ay, angular_acc].
        Uses simple Euler integration.
        """
        X = ca.MX.sym('X', nx * 2)  # 6 states
        U = ca.MX.sym('U', nx)      # 3 controls
        rhs = ca.vertcat(X[nx:], U)
        f = ca.Function('f', [X, U], [rhs])
        intg_opts = {"number_of_finite_elements":1, "simplify":1}
        intg = ca.integrator('intg', 'rk', {'x': X, 'p': U, 'ode': f(X, U)}, 0, dt, intg_opts)
        xf = intg(x0=X, p=U)['xf']
        return ca.Function('F', [X, U], [xf])

    @staticmethod
    def bayes(lambda_prev, z):
        """
        Bayesian update for belief:
           lambda_next = (lambda_prev * z) / (lambda_prev * z + (1 - lambda_prev) * (1 - z))
        """
        prod = lambda_prev * z
        denom = prod + (1 - lambda_prev) * (1 - z) + 1e-9
        return prod / denom

    @staticmethod
    def entropy(p):
        """
        Compute the binary entropy of a probability p.
        Values are clipped to avoid log(0).
        """
        p = ca.fmax(ca.fmin(p, 1 - 1e-6), 1e-9)
        return (-p * ca.log10(p) - (1 - p) * ca.log10(1 - p)) / ca.log10(2)

    # ---------------------------
    # MPC Optimization Function
    # ---------------------------
    def mpc_opt(self, g_nn, trees, lb, ub, x0, lambda_vals, steps=10):
        nx_local = 3                   # For clarity in this function
        n_state = nx_local * 2         # 6-dimensional state: [x, y, theta, vx, vy, omega]
        n_control = nx_local           # 3-dimensional control: [ax, ay, angular_acc]
        opti = ca.Opti()
        F_ = self.kin_model(self.nx, self.dt)  # kinematic model function

        # Decision variables:
        X = opti.variable(n_state, steps + 1)
        U = opti.variable(n_control, steps)

        # Parameter vector: initial state and tree beliefs.
        num_trees = trees.shape[0]
        P0 = opti.parameter(n_state + num_trees*2)
        X0 = P0[: n_state]
        L0_raw = P0[n_state : n_state + num_trees]
        L0_ripe = P0[n_state + num_trees : n_state + 2 * num_trees]
        # Initialize belief evolution for both branches.
        lambda_evol_raw = [L0_raw]
        lambda_evol_ripe = [L0_ripe]

        # Convert tree positions to a CasADi DM.
        trees_dm = ca.DM(trees)  # Expected shape: (num_trees, 2)

        # Weights and safety parameters.
        w_control = 1e-2         # Control effort weight
        w_ang = 1e-4             # Angular control weight
        w_entropy = 1e1          # Weight for final entropy
        w_attract = 1e-2         # Weight for low-entropy attraction
        safe_distance = 1.0      # Safety margin (meters)

        # Initialize the objective.
        obj = 0

        # Initial condition constraint.
        opti.subject_to(X[:, 0] == X0)

        # Loop over the prediction horizon.
        for i in range(steps+1):
            # State and input bounds.
            opti.subject_to(opti.bounded(lb[0] - 2.5, X[0, i], ub[0] + 2.5))
            opti.subject_to(opti.bounded(lb[1] - 2.5, X[1, i], ub[1] + 2.5))
            opti.subject_to(opti.bounded(-6*np.pi, X[2, i], +6*np.pi))

            opti.subject_to(opti.bounded(0.0, ca.sumsqr(X[3:5, i]),4.00))
            opti.subject_to(opti.bounded(-3.14/4, X[5, i], 3.14 / 4))

            if i < steps:
                opti.subject_to(opti.bounded(0.0, ca.sumsqr(U[0:2, i]),8.0))
                opti.subject_to(opti.bounded(-3.14/2, U[-1, i], 3.14/2))
                obj += w_control * ca.sumsqr(U[0:2, i]) + w_ang * ca.sumsqr(U[2, i])

            # Collision avoidance: ensure safety margin from trees.
            delta = X[:2, i] - trees_dm.T
            sq_dists = ca.diag(ca.mtimes(delta.T, delta))
            opti.subject_to(ca.mmin(sq_dists) >= safe_distance**2)

            if i < steps:
                opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

        nn_batch = []
        for i in range(trees_dm.shape[0]):
            # For each tree, build NN input using the state at steps 1:steps+1.
            delta = X[:2, 1:] - trees_dm[i, :].T
            heading = X[2, 1:]
            nn_batch.append(ca.horzcat(delta.T, heading.T))

        g_out = g_nn(ca.vcat([*nn_batch]))

        # Use the NN output to update both branches over the horizon.
        z_k = ca.fmax(g_out, 0.5)
        for i in range(steps):
            lambda_next_raw = self.bayes(lambda_evol_raw[-1], z_k[i::steps])
            lambda_evol_raw.append(lambda_next_raw)
            lambda_next_ripe = self.bayes(lambda_evol_ripe[-1], z_k[i::steps])
            lambda_evol_ripe.append(lambda_next_ripe)

        # Compute the entropy evolution for each branch.
        entropy_future_raw = self.entropy(ca.vcat([*lambda_evol_raw[1:]]))
        entropy_future_ripe = self.entropy(ca.vcat([*lambda_evol_ripe[1:]]))
        # Combine the two entropy futures with softmax weighting.
        entropy_future_combined = ca.fmin(entropy_future_raw, entropy_future_ripe)
        entropy_term = ca.sum1(ca.vcat([ca.exp(-2*i)*ca.DM.ones(num_trees) for i in range(steps)]) *
                               (entropy_future_combined)) * w_entropy
        # Add terms to the objective.
        obj += entropy_term
        opti.minimize(obj)

        # Solver options.
        options = {
            "ipopt": {
                "tol": 1e-2,
                "warm_start_init_point": "yes",
                "hessian_approximation": "limited-memory",
                "print_level": 0,
                "sb": "no",
                "mu_strategy": "monotone",
                "max_iter": 3000
            }
        }
        opti.solver("ipopt", options)
        # Set the parameter values.
        opti.set_value(P0, ca.vertcat(x0, lambda_vals))
        sol = opti.solve()

        # Create the MPC step function for warm starting.
        inputs = [P0, opti.x, opti.lam_g]
        outputs = [U[:, 0], X, opti.x, opti.lam_g]
        mpc_step = opti.to_function("mpc_step", inputs, outputs)

        return (mpc_step,
                ca.DM(sol.value(U[:, 0])),
                ca.DM(sol.value(X)),
                ca.DM(sol.value(opti.x)),
                ca.DM(sol.value(opti.lam_g)))

    # ---------------------------
    # Simulation Function
    # ---------------------------
    def run_simulation(self, run_folder=None):
        F_ = self.kin_model(self.nx, self.dt)
        # Get tree positions from the server.
        lb, ub = self.get_domain(self.trees_pos)
        self.mpc_horizon = self.N
        # ---------------------------
        # Load the Learned Neural Network Model
        # ---------------------------
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model.load_state_dict(torch.load(self.get_latest_best_model(), weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, batched=True, device='cuda')

        # Initialize robot state: use the provided initial state.
        vx_k = ca.DM.zeros(self.nx)  # velocity component: [vx, vy, omega]
        x_k = ca.vertcat(ca.DM(self.bridge.update_robot_state() ), vx_k)

        # Containers for simulation output.
        all_trajectories = []
        lambda_history = []
        lambda_ripe_history = []
        lambda_raw_history = []
        entropy_history = []
        durations = []

        # ---------------------------
        # Velocity Command Logging
        # ---------------------------
        velocity_command_log = []
        pose_history = []      # [x, y, theta] for each step
        time_history = []      # simulation time at each step

        # ---------------------------
        # Metrics Tracking
        # ---------------------------
        sim_start_time = time.time()
        total_distance = 0.0
        total_commands = 0
        sum_vx = 0.0
        sum_vy = 0.0
        sum_yaw = 0.0
        sum_trans_speed = 0.0
        prev_x = float(x_k[0])
        prev_y = float(x_k[1])

        sim_time = 1800  # Total simulation time in seconds
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True

        # Start sensor measurement thread.
        self.start_measurement_thread(0.5)

        # Main MPC loop.
        while mpciter < sim_time and not rospy.is_shutdown():
            print('Step:', mpciter)
            print('Observe and update state')

            # Update state from sensor.
            current_state = self.bridge.update_robot_state()  # returns [x, y, theta]
            current_sim_time = time.time() - sim_start_time
            pose_history.append(current_state)
            time_history.append(current_sim_time)

            x_k = ca.vertcat(ca.DM(current_state), vx_k)

            while self.latest_trees_scores is None:
                time.sleep(0.05)

            print("Current state x_k:", x_k)
            print("Lambda raw:", self.lambda_k_raw)
            print("Lambda ripe:", self.lambda_k_ripe)
            lambda_score = (self.lambda_k_ripe > self.lambda_k_raw) * self.lambda_k_ripe + (self.lambda_k_ripe <= self.lambda_k_raw) * (1 - self.lambda_k_raw)
            self.bridge.pub_data({
                "tree_markers": {"trees_pos": self.trees_pos, "lambda": lambda_score.full().flatten()}
            })

            step_start_time = time.time()
            if warm_start:
                # Concatenate the two lambda arrays for the MPC parameter.
                lambdas_concat = ca.vertcat(self.lambda_k_raw, self.lambda_k_ripe)
                mpc_step, u, x_traj, x_dec, lam = self.mpc_opt(g_nn, self.trees_pos, lb, ub, x_k, lambdas_concat, self.mpc_horizon)
                warm_start = False
            else:
                lambdas_concat = ca.vertcat(self.lambda_k_raw, self.lambda_k_ripe)
                u, x_traj, x_dec, lam = mpc_step(ca.vertcat(x_k, lambdas_concat), x_dec, lam)
            durations.append(time.time() - step_start_time)
            # Log the MPC velocity command.
            u_np = np.array(u.full()).flatten()

            # Publish data and apply control.
            cmd_pose = F_(x_k, u[:, 0])
            self.bridge.pub_data({ "predicted_path": x_traj[:self.nx, 1:], "cmd_pose": cmd_pose})

            vx_k = cmd_pose[self.nx:]
            velocity_command_log.append([current_sim_time, "MPC", vx_k[0], vx_k[1], vx_k[2]])

            # Update metrics.
            vx_val = float(vx_k[0])
            vy_val = float(vx_k[1])
            yaw_val = float(vx_k[2])
            sum_vx += vx_val
            sum_vy += vy_val
            sum_yaw += yaw_val
            trans_speed = math.sqrt(vx_val**2 + vy_val**2)
            sum_trans_speed += trans_speed
            total_commands += 1

            curr_x = float(x_traj[0, 1])
            curr_y = float(x_traj[1, 1])
            distance_step = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_distance += distance_step
            prev_x, prev_y = curr_x, curr_y

            entropy_k = ca.sum1(ca.fmin(self.entropy(self.lambda_k_raw), self.entropy(self.lambda_k_ripe))).full().flatten()[0]
            lambda_history.append(lambda_score.full().flatten().tolist())
            lambda_ripe_history.append(self.lambda_k_ripe)
            lambda_raw_history.append(self.lambda_k_raw)
            entropy_history.append(entropy_k)
            all_trajectories.append(x_traj[:self.nx, :].full())

            mpciter += 1
            print("Entropy:", entropy_k)
            print(ca.fmax(self.lambda_k_ripe, self.lambda_k_raw).full().flatten())
            if all( v >=0.99 for v in ca.fmax(self.lambda_k_ripe, self.lambda_k_raw).full().flatten()) :
                break
            rate.sleep()

        # ---------------------------
        # Final Metrics Calculation and CSV Output
        # ---------------------------
        total_execution_time = time.time() - sim_start_time
        avg_wp_time = total_execution_time / total_commands if total_commands > 0 else 0.0
        avg_vx = sum_vx / total_commands if total_commands > 0 else 0.0
        avg_vy = sum_vy / total_commands if total_commands > 0 else 0.0
        avg_yaw = sum_yaw / total_commands if total_commands > 0 else 0.0
        avg_trans_speed = sum_trans_speed / total_commands if total_commands > 0 else 0.0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Use provided run_folder or create one inside a "baselines" directory.
        if run_folder is None:
            baselines_dir = os.path.join(script_dir, "../../baselines")
            os.makedirs(baselines_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_folder = os.path.join(baselines_dir, f"run_{timestamp}")
            os.makedirs(run_folder, exist_ok=True)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        perf_csv = os.path.join(run_folder, f"mpc_{timestamp}_performance_metrics.csv")
        with open(perf_csv, mode='w', newline='') as perf_file:
            headers = [
                "Total Execution Time (s)",
                "Total Distance (m)",
                "Average Waypoint-to-Waypoint Time (s)",
                "Final Entropy",
                "Total Commands"
            ]

            values = [
                total_execution_time,
                total_distance,
                avg_wp_time,
                entropy_history[-1] if entropy_history else "N/A",
                total_commands
            ]

            writer = csv.writer(perf_file)
            # Write data as columns: one label per row with its value
            for label, value in zip(headers, values):
                writer.writerow([label, value])

        vel_csv = os.path.join(run_folder, f"mpc_{timestamp}_velocity_commands.csv")
        with open(vel_csv, mode='w', newline='') as vel_file:
            writer = csv.writer(vel_file)
            writer.writerow(["Time (s)", "Tag", "x_velocity", "y_velocity", "yaw_velocity"])
            writer.writerows(velocity_command_log)

        print(f"Performance metrics saved to {perf_csv}")
        print(f"Velocity command log saved to {vel_csv}")

        plot_csv = os.path.join(run_folder, f"mpc_{timestamp}_plot_data.csv")
        with open(plot_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            tree_positions_flat = self.trees_pos.flatten().tolist()
            writer.writerow(["tree_positions"] + tree_positions_flat)
            header = ["time", "x", "y", "theta", "entropy"]
            if lambda_history and len(lambda_history[0]) > 0:
                num_trees = len(lambda_history[0])
                header += [f"lambda_{i}" for i in range(num_trees)]
            writer.writerow(header)
            for i in range(len(time_history)):
                time_val = time_history[i]
                x, y, theta = np.array(pose_history[i]).flatten()
                entropy_val = entropy_history[i]
                lambda_vals = lambda_history[i]
                row = [time_val, x, y, theta, entropy_val] + lambda_vals
                writer.writerow(row)
        print(f"Plot data saved to {plot_csv}")

        # ---------------------------
        # Save Lambda History Separately
        # ---------------------------
        lambda_csv = os.path.join(run_folder, f"mpc_{timestamp}_lambda_history.csv")
        with open(lambda_csv, mode='w', newline='') as lambda_file:
            writer = csv.writer(lambda_file)
            writer.writerow(["time"] + [f"lambda_{i}" for i in range(len(lambda_history[0]))])
            for i in range(len(time_history)):
                writer.writerow([time_history[i]] + lambda_history[i])
        print(f"Lambda history saved to {lambda_csv}")

        # Store time and entropy history for separate plotting.
        self.time_history = time_history
        self.entropy_history = entropy_history
        # Also store the lambda histories as class attributes for later plotting.
        self.lambda_ripe_history = [lr.full().flatten().tolist() for lr in lambda_ripe_history]
        self.lambda_raw_history = [lr.full().flatten().tolist() for lr in lambda_raw_history]

        return all_trajectories, entropy_history, lambda_history, durations, g_nn, self.trees_pos, lb, ub

    # ---------------------------
    # Plotting Function (Animated Trajectory etc.)
    # ---------------------------
    def plot_animated_trajectory_and_entropy_2d(self, all_trajectories, entropy_history, lambda_history, trees, lb, ub, computation_durations):
        # (The original plotting function remains unchanged.)
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model.load_state_dict(torch.load(self.get_latest_best_model(), weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, name='plotting_f', batched=True, device='cuda')

        x_trajectory = np.array([traj[0] for traj in all_trajectories])
        y_trajectory = np.array([traj[1] for traj in all_trajectories])
        theta_trajectory = np.array([traj[2] for traj in all_trajectories])
        all_trajectories = np.array(all_trajectories)
        lambda_history = np.array(lambda_history)

        # Compute predicted entropy reduction.
        entropy_mpc_pred = []
        for k in range(all_trajectories.shape[0]):
            lambda_k = lambda_history[k]
            entropy_mpc_pred_k = [entropy_history[k]]
            for i in range(all_trajectories.shape[2]-1):
                relative_position_robot_trees = np.tile(all_trajectories[k, :2, i+1], (trees.shape[0], 1)) - trees
                distance_robot_trees = np.sqrt(np.sum(relative_position_robot_trees**2, axis=1))
                theta = np.tile(all_trajectories[k, 2, i+1], (trees.shape[0], 1))
                input_nn = ca.horzcat(relative_position_robot_trees, theta)
                z_k =  ca.fmax(g_nn(input_nn), 0.5)
                lambda_k = self.bayes(lambda_k, z_k)
                reduction = ca.sum1(self.entropy(lambda_k)).full().flatten()[0]
                entropy_mpc_pred_k.append(reduction)
            entropy_mpc_pred.append(entropy_mpc_pred_k)

        entropy_mpc_pred = np.array(entropy_mpc_pred)
        sum_entropy_history = entropy_history
        cumulative_durations = np.cumsum(computation_durations)

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.6, 0.4],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )

        # MPC predicted trajectory.
        fig.add_trace(
            go.Scatter(
                x=x_trajectory[0],
                y=y_trajectory[0],
                mode="lines+markers",
                name="MPC Future Trajectory",
                line=dict(width=4),
                marker=dict(size=5)
            ),
            row=1, col=1
        )
        # Drone trajectory (empty initially).
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Drone Trajectory",
                line=dict(width=4),
                marker=dict(size=5)
            ),
            row=1, col=1
        )

        # Add tree markers.
        for i in range(trees.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=[trees[i, 0]],
                    y=[trees[i, 1]],
                    mode="markers+text",
                    marker=dict(size=10, colorscale=[[0, "#FF0000"], [1, "#00FF00"]]),
                    name=f"Tree {i}: {lambda_history[0][i]:.2f}",
                    text=[str(i)],
                    textposition="top center"
                ),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Sum of Entropies (Past)",
                line=dict(width=2),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Sum of Entropies (Future)",
                line=dict(width=2, dash="dot"),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Computation Durations",
                line=dict(width=2),
                marker=dict(size=5)
            ),
            row=2, col=2
        )

        # Create animation frames.
        frames = []
        for k in range(len(entropy_mpc_pred)):
            tree_data = []
            for i in range(trees.shape[0]):
                tree_data.append(
                    go.Scatter(
                        x=[trees[i, 0]],
                        y=[trees[i, 1]],
                        mode="markers+text",
                        marker=dict(size=10, color=[2*(lambda_history[k][i]-0.5)],
                                    colorscale=[[0, "#FF0000"], [1, "#00FF00"]]),
                        name=f"Tree {i}: {lambda_history[k][i]:.2f}",
                        text=[str(i)],
                        textposition="top center"
                    )
                )
            sum_entropy_past = sum_entropy_history[:k+1]
            sum_entropy_future = entropy_mpc_pred[k]
            computation_durations_past = computation_durations[:k+1]

            x_start = x_trajectory[k]
            y_start = y_trajectory[k]
            theta = theta_trajectory[k]
            x_end = x_start + 0.5 * np.cos(theta)
            y_end = y_start + 0.5 * np.sin(theta)
            list_of_actual_orientations = []
            for x0, y0, x1, y1 in zip(x_start, y_start, x_end, y_end):
                arrow = go.layout.Annotation(
                    dict(
                        x=x1, y=y1,
                        xref="x", yref="y",
                        showarrow=True,
                        ax=x0, ay=y0,
                        arrowhead=3, arrowwidth=1.5,
                        arrowcolor="red",
                    )
                )
                list_of_actual_orientations.append(arrow)
            for x0, y0, x1, y1 in zip(x_trajectory[:k+1, 0],
                                        y_trajectory[:k+1, 0],
                                        x_trajectory[:k+1, 0] + 0.5 * np.cos(theta_trajectory[:k+1, 0]),
                                        y_trajectory[:k+1, 0] + 0.5 * np.sin(theta_trajectory[:k+1, 0])):
                arrow = go.layout.Annotation(
                    dict(
                        x=x1, y=y1,
                        xref="x", yref="y",
                        showarrow=True,
                        ax=x0, ay=y0,
                        arrowhead=3, arrowwidth=1.5,
                        arrowcolor="orange",
                    )
                )
                list_of_actual_orientations.append(arrow)

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=x_trajectory[k],
                        y=y_trajectory[k],
                        mode="lines+markers",
                        line=dict(width=4),
                        marker=dict(size=5)
                    ),
                    go.Scatter(
                        x=x_trajectory[:k+1, 0],
                        y=y_trajectory[:k+1, 0],
                        mode="lines+markers",
                        line=dict(width=4),
                        marker=dict(size=5)
                    ),
                    *tree_data,
                    go.Scatter(
                        x=np.arange(len(sum_entropy_past)),
                        y=sum_entropy_past,
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=5)
                    ),
                    go.Scatter(
                        x=np.arange(k, k+len(sum_entropy_future)),
                        y=sum_entropy_future,
                        mode="lines+markers",
                        line=dict(width=2, dash="dot"),
                        marker=dict(size=5)
                    ),
                    go.Scatter(
                        x=np.arange(len(computation_durations_past)),
                        y=computation_durations_past,
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=5)
                    )
                ],
                name=f"Frame {k}",
                layout=dict(annotations=list_of_actual_orientations)
            )
            frames.append(frame)

        fig.frames = frames
        fig.update_layout(
            title="Drone Trajectory, Sum of Entropies, and Computation Durations",
            xaxis=dict(title="X Position", range=[lb[0] - 3, ub[0] + 3]),
            yaxis=dict(title="Y Position", range=[lb[1] - 3, ub[1] + 3]),
            xaxis2=dict(title="Time Step"),
            yaxis2=dict(title="Sum of Entropies"),
            xaxis3=dict(title="Time Step"),
            yaxis3=dict(title="Computation Duration (s)"),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                        )
                    ],
                    showactive=True,
                    x=0.1,
                    y=0
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 50, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(frames)
                ],
            }]
        )
        fig.show()
        fig.write_html(os.path.join(os.path.dirname(run_folder), 'neural_mpc_results.html'))
        return entropy_mpc_pred

    # ---------------------------
    # New Function: Plot Lambda Entropy Trends for Each Tree
    # ---------------------------
    def plot_tree_lambda_trends(self):
        """
        Plots for each tree the trend of the binary entropy computed from the raw and ripe lambda values over time.
        It uses the stored self.time_history, self.lambda_raw_history, and self.lambda_ripe_history.
        """
        # Ensure we have stored time and lambda history
        if (not hasattr(self, "time_history") or 
            not hasattr(self, "lambda_raw_history") or 
            not hasattr(self, "lambda_ripe_history")):
            print("Time or lambda history is not available. Run the simulation first.")
            return

        # Local helper to compute binary entropy.
        def binary_entropy(p):
            p = np.clip(p, 1e-9, 1-1e-9)
            return -p * np.log2(p) - (1-p) * np.log2(1-p)

        time_history = self.time_history
        num_steps = len(time_history)
        # Determine the number of trees from the first entry of the lambda history.
        num_trees = len(self.lambda_raw_history[0])

        # Create subplots: one row per tree.
        fig = make_subplots(rows=num_trees, cols=1, shared_xaxes=True,
                            subplot_titles=[f"Tree {i} Lambda Entropy Trend" for i in range(num_trees)])

        # For each tree, compute the entropy over time for both raw and ripe lambda values.
        for i in range(num_trees):
            raw_vals = [entry[i] for entry in self.lambda_raw_history]
            ripe_vals = [entry[i] for entry in self.lambda_ripe_history]
            raw_entropy = [binary_entropy(val) for val in raw_vals]
            ripe_entropy = [binary_entropy(val) for val in ripe_vals]

            # Add traces for raw and ripe entropy.
            fig.add_trace(go.Scatter(x=time_history, y=raw_entropy,
                                     mode='lines+markers', name=f'Tree {i} Raw Entropy',
                                     line=dict(width=2)),
                          row=i+1, col=1)
            fig.add_trace(go.Scatter(x=time_history, y=ripe_entropy,
                                     mode='lines+markers', name=f'Tree {i} Ripe Entropy',
                                     line=dict(width=2, dash='dot')),
                          row=i+1, col=1)

        fig.update_layout(title="Lambda Entropy Trends for Each Tree",
                          xaxis_title="Time (s)",
                          yaxis_title="Binary Entropy",
                          height=300 * num_trees)
        fig.show()
        fig.write_html('neural_mpc_tree_lambda_entropy.html')

    # ---------------------------
    # New Function: Plot Entropy Separately
    # ---------------------------
    def plot_entropy_separately(self):
        # Ensure we have stored time and entropy history
        if not hasattr(self, "time_history") or not hasattr(self, "entropy_history"):
            print("Time or entropy history is not available. Run the simulation first.")
            return
        # Create a separate Plotly figure for the entropy plot.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time_history, y=self.entropy_history,
                                 mode='lines+markers', name='Entropy'))
        fig.update_layout(title='Entropy over Time',
                          xaxis_title='Time (s)',
                          yaxis_title='Entropy')
        fig.show()
        fig.write_html('neural_mpc_entropy.html')
        print('ooooooo')

# ---------------------------
# Main Execution: Run 100 Tests
# ---------------------------
if __name__ == "__main__":
    mpc = NeuralMPC()
    # Get the field domain from the tree positions.
    lb, ub = mpc.get_domain(mpc.trees_pos)
    # Base folder to store all test run outputs.
    base_test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_runs")
    os.makedirs(base_test_folder, exist_ok=True)
    
    # Run 100 tests consecutively.
    for test_num in range(1, 101):
        print(f"================== Starting Test Run {test_num} ==================")
        # Generate a valid random initial state.
        initial_state = mpc.generate_random_initial_state(lb, ub, margin=1.25)
        print(f"Test {test_num} Initial State: {initial_state}")
        # Publish the initial random position.
        print(initial_state)
        mpc.bridge.pub_data({ "cmd_pose": initial_state})
        # Create a dedicated folder for this run.
        run_folder = os.path.join(base_test_folder, f"run_{test_num}")
        os.makedirs(run_folder, exist_ok=True)
        # Run the simulation with the specified initial state and output folder.
        sim_results = mpc.run_simulation(initial_state=initial_state, run_folder=run_folder)
        # Optionally, plot the entropy and tree lambda trends for this run.
        #mpc.plot_entropy_separately()
        #mpc.plot_tree_lambda_trends()
