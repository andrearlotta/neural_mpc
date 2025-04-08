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
    # Class-level type annotations for TorchScript
    input_layer: torch.nn.Linear
    hidden_layer: torch.nn.ModuleList
    out_layer: torch.nn.Linear

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

        self.N = 5  # Prediction horizon steps
        self.dt = 0.5
        self.T = self.dt * self.N
        self.nx = 3  # Represents [x, y, theta]
        self.n_state = self.nx * 2 # [x, y, theta, vx, vy, omega]
        self.n_control = self.nx   # [ax, ay, omega_dot]

        # --- NEW: Define number of trees for each purpose ---
        self.NUM_TARGET_TREES = 1    # Number of trees for information gain
        self.NUM_OBSTACLE_TREES = 100   # Number of nearest trees for collision avoidance
        # ---

        # For storing the latest tree scores from the sensor callback.
        self.latest_trees_scores = None

        # Initialize the sensor bridge.
        self.bridge = BridgeClass(SENSORS)

        # Get initial tree positions
        self.trees_pos = np.array(self.bridge.call_server({"trees_poses": None})["trees_poses"])
        self.num_total_trees = self.trees_pos.shape[0]

        # Initialize two lambda vectors: one for raw and one for ripe.
        self.lambda_k_raw = ca.DM.ones(self.num_total_trees, 1) * 0.5
        self.lambda_k_ripe = ca.DM.ones(self.num_total_trees, 1) * 0.5

    # ---------------------------
    # Sensor Callback Thread
    # ---------------------------
    def measurement_callback(self, measurement_interval=0.5):
        while not rospy.is_shutdown():
            data = self.bridge.get_data()
            if data["tree_scores"] is not None:
                # Assume tree_scores is an array with values in [0,1]
                scores = np.array(data["tree_scores"])
                # Ensure scores array has the correct size
                if len(scores) != self.num_total_trees:
                    rospy.logwarn(f"Received tree_scores array of length {len(scores)}, expected {self.num_total_trees}. Skipping update.")
                    time.sleep(measurement_interval)
                    continue

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

    def start_measurement_thread(self, measurement_interval=0.2):
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

    @staticmethod
    def numpy_entropy(p):
        p = np.clip(p, 1e-6, 1 - 1e-9)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    # ---------------------------
    # Function to select TARGET tree indices (for information gain)
    # RENAMED from get_selected_tree_indices
    # ---------------------------
    def get_target_tree_indices(self, robot_position, num_target=None, entropy_threshold=0.025):
        """
        Returns the indexes of the 'num_target' trees (from self.trees_pos) that are closest to
        the robot_position and have an effective entropy higher than 'entropy_threshold'.
        Effective entropy is min(H(lambda_raw), H(lambda_ripe)).
        If fewer than 'num_target' trees meet the criterion, the nearest ones meeting it are duplicated.
        If no trees meet the criterion, the 'num_target' overall nearest trees are returned.
        """
        if num_target is None:
            num_target = self.NUM_TARGET_TREES

        # Compute Euclidean distances from the robot to each tree.
        # Ensure robot_position is 1D array [x, y]
        robot_pos_1d = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos[:, :2] - robot_pos_1d, axis=1)

        # Compute binary entropies for raw and ripe lambdas.
        H_raw = self.numpy_entropy(np.array(self.lambda_k_raw.full()).flatten())
        H_ripe = self.numpy_entropy(np.array(self.lambda_k_ripe.full()).flatten())

        # Compute effective entropy (minimum of the two).
        H_min = np.minimum(H_raw, H_ripe)
        
        # Get indices where effective entropy is above the threshold.
        candidate_indices = np.where(H_min > entropy_threshold)[0]
        if candidate_indices.size == 0:
            # If no tree meets the entropy threshold, just take the overall nearest ones
            print("Warning: No trees above entropy threshold. Selecting nearest trees.")
            sorted_indices_all = np.argsort(distances)
            return sorted_indices_all[:num_target]

        # Sort candidate indices by distance.
        sorted_candidates = candidate_indices[np.argsort(distances[candidate_indices])]

        # If the number of selected candidates is less than num_target,
        # re-add (duplicate) the nearest ones among those above threshold.
        if sorted_candidates.size < num_target:
            repeats = int(np.ceil(num_target / sorted_candidates.size))
            # Duplicate the candidate array and slice the first num_target elements
            sorted_candidates = np.tile(sorted_candidates, repeats)[:num_target]
        # Return the first num_target indices.
        return sorted_candidates[:num_target]

    # ---------------------------
    # NEW Function to select NEAREST tree indices (for obstacle avoidance)
    # ---------------------------
    def get_nearest_tree_indices(self, robot_position, num_obstacle=None):
        """
        Returns the indices of the 'num_obstacle' trees (from self.trees_pos)
        that are closest to the robot_position.
        """
        if num_obstacle is None:
            num_obstacle = self.NUM_OBSTACLE_TREES
        if num_obstacle > self.num_total_trees:
            num_obstacle = self.num_total_trees # Cannot select more trees than exist

        # Ensure robot_position is 1D array [x, y]
        robot_pos_1d = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos - robot_pos_1d, axis=1)

        # Get indices sorted by distance
        sorted_indices = np.argsort(distances)

        # Return the first 'num_obstacle' indices
        return sorted_indices[:num_obstacle]


    # ---------------------------
    # MPC Optimization Function (MODIFIED)
    # ---------------------------
    def mpc_opt(self, g_nn, target_trees, target_lambdas, obstacle_trees, lb, ub, x0, steps=10):
        """
        MPC Optimization.
        Uses 'target_trees' and 'target_lambdas' for information gain objective.
        Uses 'obstacle_trees' for collision avoidance constraint.
        """
        opti = ca.Opti()
        F_ = self.kin_model(self.nx, self.dt)  # kinematic model function

        # Decision variables:
        X = opti.variable(self.n_state, steps + 1) # State trajectory [x,y,th,vx,vy,om]
        U = opti.variable(self.n_control, steps)   # Control trajectory [ax,ay,om_dot]

        # --- Parameter vector: initial state, target trees/lambdas, obstacle trees ---
        num_target_trees = target_trees.shape[0]
        num_obstacle_trees = obstacle_trees.shape[0]

        # P0 layout: [x0 (n_state), target_trees_flat (2*num_target), target_L0_raw (num_target), target_L0_ripe (num_target), obstacle_trees_flat (2*num_obstacle)]
        param_size = self.n_state + num_target_trees * 2 + num_target_trees * 2 + num_obstacle_trees * 2
        P0 = opti.parameter(param_size)

        # Unpack parameters
        p_idx = 0
        X0 = P0[p_idx : p_idx + self.n_state]; p_idx += self.n_state
        TARGET_TREES_param = P0[p_idx : p_idx + num_target_trees*2].reshape((num_target_trees, 2)); p_idx += num_target_trees*2
        L0_raw = P0[p_idx : p_idx + num_target_trees]; p_idx += num_target_trees
        L0_ripe = P0[p_idx : p_idx + num_target_trees]; p_idx += num_target_trees
        OBSTACLE_TREES_param = P0[p_idx : p_idx + num_obstacle_trees*2].reshape((num_obstacle_trees, 2)); p_idx += num_obstacle_trees*2
        # --- End Parameter Definition ---

        # Initialize belief evolution for both branches (using TARGET trees)
        lambda_evol_raw = [L0_raw]
        lambda_evol_ripe = [L0_ripe]

        # Weights and safety parameters.
        w_control = 1e-2         # Control effort weight
        w_ang = 1e-4             # Angular control weight
        w_entropy = 1e1          # Weight for final entropy (applied to target trees)
        # w_attract = 1e-2       # (Removed as focus is entropy reduction)
        safe_distance = 1.5      # Safety margin (meters) for obstacle trees

        # Initialize the objective.
        obj = 0

        # Initial condition constraint.
        opti.subject_to(X[:, 0] == X0)

        # Loop over the prediction horizon.
        for i in range(steps+1):
            # State and input bounds.
            opti.subject_to(opti.bounded(lb[0] - 3., X[0, i], ub[0] + 3.))
            opti.subject_to(opti.bounded(lb[1] - 3., X[1, i], ub[1] + 3.))
            opti.subject_to(opti.bounded(-ca.pi-0.1+X0[2], X[2, i], ca.pi+0.1+X0[2]))

            opti.subject_to(opti.bounded(-2.0, X[3, i], 2.0)) # vx bound
            opti.subject_to(opti.bounded(-2.0, X[4, i], 2.0)) # vy bound
            opti.subject_to(opti.bounded(-ca.pi/4, X[5, i], ca.pi / 4)) # omega bound

            if i < steps:
                opti.subject_to(opti.bounded(-4.0, U[0:2, i],4.0))
                opti.subject_to(opti.bounded(-ca.pi/2, U[2, i], ca.pi/2))
                obj += w_control * ca.sumsqr(U[0:2, i]) + w_ang * ca.sumsqr(U[2, i])

            # Collision avoidance: ensure safety margin from trees.
            delta = X[:2, i] - OBSTACLE_TREES_param.T
            sq_dists = ca.diag(ca.mtimes(delta.T, delta))
            opti.subject_to(ca.mmin(sq_dists) > safe_distance**2)

            # System dynamics constraint
            if i < steps:
                opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

        # --- Information Gain Objective: Use TARGET trees ---
        nn_batch = []
        for i in range(num_target_trees):
            # For each TARGET tree, build NN input using the state at steps 1:steps+1.
            delta_target = X[:2, 1:] - TARGET_TREES_param[i,:].T
            heading_target = X[2, 1:]
            # NN input: [delta_x, delta_y, heading]
            nn_batch.append(ca.horzcat(delta_target.T, heading_target.T))
        # Get NN output for all target trees over the horizon
        # Input shape: (num_target_trees * steps, 3)
        # Output shape: (num_target_trees * steps, 1)
        g_out = g_nn(ca.vcat([*nn_batch]))

        # Use the NN output to update both branches over the horizon.
        z_k = ca.fmax(g_out, 0.5)
        for i in range(steps):
            lambda_next_raw = self.bayes(lambda_evol_raw[-1], z_k[i::steps])
            lambda_evol_raw.append(lambda_next_raw)
            lambda_next_ripe = self.bayes(lambda_evol_ripe[-1], z_k[i::steps])
            lambda_evol_ripe.append(lambda_next_ripe)

        # Compute the entropy evolution for each branch.
        entropy_term = 0
        entropy_0 = ca.sum1(ca.fmin(self.entropy(lambda_evol_raw[0]), self.entropy(lambda_evol_ripe[0])))
        for i in range(1, steps + 1): # Steps k=1...N
            lambda_raw_k = lambda_evol_raw[i]
            lambda_ripe_k = lambda_evol_ripe[i]
            # Effective entropy at step k is sum over trees of min(H(raw), H(ripe))
            entropy_k = ca.sum1(ca.fmin(self.entropy(lambda_raw_k), self.entropy(lambda_ripe_k)))
            # Discount future entropy reduction (penalize high future entropy)
            entropy_term += ca.exp(-0.5 * i) * entropy_k
        # Add entropy term to the objective.
        obj += w_entropy * entropy_term
        # --- End Information Gain Objective ---

        opti.minimize(obj)

        # Solver options.
        options = {
            "ipopt": {
                "tol": 1e-3,
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-2,
                "warm_start_mult_bound_push": 1e-2,
                "mu_init": 1e-2,
                "bound_relax_factor": 1e-6,
                "hsllib": '/usr/local/lib/libcoinhsl.so', # Specify HSL library path if used
                "linear_solver": 'ma27',
                "hessian_approximation": "limited-memory",
                "print_level": 0,
                "sb": "no",
                "mu_strategy": "monotone",
                "max_iter": 3000
                }
        }
        opti.solver("ipopt", options)

        # Create the MPC step function for warm starting.
        # Input P0 needs: x0, target_trees_flat, target_L0_raw, target_L0_ripe, obstacle_trees_flat
        # We also need previous solution (x_dec) and multipliers (lam_g) for warm start
        inputs = [P0, opti.x]
        outputs = [U[:, 0], X,  opti.x]
        mpc_step_func = opti.to_function("mpc_step", inputs, outputs, ["p", "x_init"], ["u_opt", "x_pred", "x_opt"])

        # --- Solve for the first time ---
        # Concatenate initial parameter values correctly
        p0_val = ca.vertcat(
            x0,
            target_trees.flatten(),
            target_lambdas[:num_target_trees], # Raw lambdas
            target_lambdas[num_target_trees:], # Ripe lambdas
            obstacle_trees.flatten()
        )
        opti.set_value(P0, p0_val)
        try:
            sol = opti.solve()
            u_sol = sol.value(U[:, 0])
            x_traj_sol = sol.value(X)
            x_dec_sol = sol.value(opti.x)
            lam_g_sol = sol.value(opti.lam_g)
        except Exception as e:
            print(f"Solver failed: {e}")
            # Fallback: return zero control and current state if solver fails
            u_sol = ca.DM.zeros(self.n_control)
            x_traj_sol = ca.repmat(x0, 1, steps + 1)
            x_dec_sol = opti.initial(opti.x) # Use initial guess
            lam_g_sol = opti.initial(opti.lam_g) # Use initial guess

        return (mpc_step_func,
                ca.DM(u_sol),
                ca.DM(x_traj_sol),
                ca.DM(x_dec_sol),
                ca.DM(lam_g_sol))

    # ---------------------------
    # Simulation Function
    # ---------------------------
    def run_simulation(self):
        F_ = self.kin_model(self.nx, self.dt)
        # Get tree positions from the server.
        lb, ub = self.get_domain(self.trees_pos)

        # Start sensor measurement thread.
        self.start_measurement_thread(0.2)

        # ---------------------------
        # Load the Learned Neural Network Model
        # ---------------------------
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model_path = self.get_latest_best_model()
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, batched=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using L4CasADi on device: {g_nn.device}")

        # Initialize robot state
        initial_state = self.bridge.update_robot_state()  # [x, y, theta]
        vx_k = ca.DM.zeros(self.nx)  # Initial velocity [vx, vy, omega]
        x_k = ca.vertcat(ca.DM(initial_state), vx_k) # Full state [x, y, theta, vx, vy, omega]

        # Containers for simulation output.
        all_trajectories = []
        lambda_history = [] # Combined score: max(ripe, 1-raw)
        lambda_ripe_history = [] # Full ripe lambda history
        lambda_raw_history = []  # Full raw lambda history
        entropy_history = [] # Total effective entropy
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

        sim_time = 2800  # Total simulation time steps
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True
        x_dec_prev = None
        lam_g_prev = None
        mpc_step = None # Initialize mpc_step function

        # Main MPC loop.
        while mpciter < sim_time and not rospy.is_shutdown():
            loop_iter_start = time.time()
            print(f'\n--- MPC Step: {mpciter} ---')

            # Update state from sensor.
            current_pose = self.bridge.update_robot_state()  # [x, y, theta]
            current_sim_time = time.time() - sim_start_time
            pose_history.append(current_pose)
            time_history.append(current_sim_time)

            # Update full state (keep previous velocity command as estimate for current velocity)
            x_k = ca.vertcat(ca.DM(current_pose), vx_k)
            print(f'Current State (x,y,th,vx,vy,w): {x_k.full().flatten()}')

            # Wait for first measurement
            while self.latest_trees_scores is None and not rospy.is_shutdown():
                print("Waiting for initial tree scores...")
                time.sleep(0.1)
            if rospy.is_shutdown(): break

            # --- Select Trees ---
            robot_position_xy = np.array(current_pose[:2])
            target_indices = self.get_target_tree_indices(robot_position_xy, num_target=self.NUM_TARGET_TREES)
            obstacle_indices = self.get_nearest_tree_indices(robot_position_xy, num_obstacle=self.NUM_OBSTACLE_TREES)

            target_trees_subset = self.trees_pos[target_indices, :2] # Only need x,y for target trees as well
            obstacle_trees_subset = self.trees_pos[obstacle_indices, :2] # Only x,y needed

            # Get corresponding lambda values for TARGET trees
            target_lambdas_raw = self.lambda_k_raw[target_indices]
            target_lambdas_ripe = self.lambda_k_ripe[target_indices]
            target_lambdas_subset = ca.vertcat(target_lambdas_raw, target_lambdas_ripe)

            print(f'Selected {len(target_indices)} target trees (Indices: {target_indices})')
            print(f'Selected {len(obstacle_indices)} obstacle trees (Indices: {obstacle_indices})')
            # ---

            # Publish markers using the full lambda scores
            # Combined score: P(ripe) if ripe > raw, else P(ripe) = 1 - P(raw)
            lambda_score = (self.lambda_k_ripe > self.lambda_k_raw) * self.lambda_k_ripe + \
                           (self.lambda_k_ripe <= self.lambda_k_raw) * (1 - self.lambda_k_raw)
            self.bridge.pub_data({
                "tree_markers": {"trees_pos": self.trees_pos, "lambda": lambda_score.full().flatten()}
            })

            # --- Run MPC Optimization ---
            step_start_time = time.time()
            try:
                if warm_start or mpc_step is None:
                    print("Running MPC opt (first step / cold start)...")
                    mpc_step, u, x_traj, x_dec_prev, lam_g_prev = self.mpc_opt(
                        g_nn, target_trees_subset, target_lambdas_subset, obstacle_trees_subset, lb, ub, x_k, steps=self.N
                    )
                    warm_start = False
                    print("MPC opt finished.")
                else:
                    print("Running MPC step (warm start)...")
                    # Construct parameter vector for mpc_step
                    P0_val = ca.vertcat(
                        x_k,
                        target_trees_subset.flatten(),
                        target_lambdas_raw,
                        target_lambdas_ripe,
                        obstacle_trees_subset.flatten()
                    )
                    # Call the compiled MPC step function
                    u, x_traj, x_dec_prev = mpc_step(P0_val, x_dec_prev)
                    print("MPC step finished.")

                step_duration = time.time() - step_start_time
                durations.append(step_duration)
                print(f"MPC step duration: {step_duration:.4f} s")

            except Exception as e:
                rospy.logerr(f"Error during MPC optimization at step {mpciter}: {e}")
                # Implement fallback behavior (e.g., stop the robot, hover, use previous command)
                u = ca.DM.zeros(self.n_control) # Command zero acceleration
                x_traj = ca.repmat(x_k, 1, self.N + 1) # Predict staying still
                warm_start = True # Force re-optimization next time
                print("!!! MPC Solver Failed - Commanding Zero Acceleration !!!")


            # --- Apply Control and Log ---
            # Get the optimal control for the first step
            u_k = u[:, 0]
            # Simulate one step forward using the model to get next state (including velocity)
            next_state_pred = F_(x_k, u_k)
            # Extract the velocity part for the next iteration and logging
            vx_k = next_state_pred[self.nx:] # Update estimated velocity [vx, vy, omega]

            # Publish command pose (predicted state after one step)
            self.bridge.pub_data({"cmd_pose": next_state_pred[:self.nx]}) # Publish [x, y, theta] part
            # Optionally publish predicted path
            # self.bridge.pub_data({"predicted_path": x_traj[:self.nx, 1:]})

            # Log velocity command (which is vx_k, the velocity resulting from applied acceleration u_k)
            velocity_command_log.append([current_sim_time, "MPC", float(vx_k[0]), float(vx_k[1]), float(vx_k[2])])

            # --- Update Metrics ---
            vx_val = float(vx_k[0])
            vy_val = float(vx_k[1])
            yaw_val = float(vx_k[2])
            sum_vx += vx_val
            sum_vy += vy_val
            sum_yaw += yaw_val
            trans_speed = math.sqrt(vx_val**2 + vy_val**2)
            sum_trans_speed += trans_speed
            total_commands += 1

            # Use actual measured position for distance calculation if available and reliable,
            # otherwise use the first step of the predicted trajectory.
            # Using prediction here as it corresponds to the command sent.
            curr_x = float(x_traj[0, 1])
            curr_y = float(x_traj[1, 1])
            distance_step = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_distance += distance_step
            prev_x, prev_y = curr_x, curr_y

            # --- Store History ---
            # Calculate total effective entropy (using min over raw/ripe for *all* trees)
            entropy_k = ca.fmin(self.entropy(self.lambda_k_raw), self.entropy(self.lambda_k_ripe))
            lambda_history.append(lambda_score.full().flatten().tolist()) # Combined score history
            lambda_ripe_history.append(self.lambda_k_ripe.full().flatten().tolist()) # Store as list
            lambda_raw_history.append(self.lambda_k_raw.full().flatten().tolist()) # Store as list
            entropy_history.append(ca.sum1(entropy_k).full().flatten()[0])
            all_trajectories.append(x_traj[:self.nx, :].full()) # Store pose part of prediction

            mpciter += 1
            print(f"Total Effective Entropy: {ca.sum1(entropy_k).full().flatten()[0]:.4f}")
            if all( v <= 0.025 for v in entropy_k.full().flatten()):
                print("Entropy target reached.")
                break

            # Ensure loop runs at desired rate (approx self.dt)
            loop_elapsed = time.time() - loop_iter_start
            sleep_time = self.dt - loop_elapsed
            if sleep_time > 0:
                rate.sleep() # Use ROS rate limiter
            else:
                 print(f"Warning: Loop iteration {mpciter} took {loop_elapsed:.4f}s, longer than dt={self.dt}s")


        # --- Final Metrics Calculation and CSV Output ---
        total_execution_time = time.time() - sim_start_time
        # ... (rest of the saving and plotting code remains largely the same) ...
        # ... Ensure file paths are correct ...
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming 'baselines' directory is two levels up from the script's directory
        # Adjust relative path if needed: os.path.join(script_dir, "..", "..", "baselines")
        # Or use an absolute path if preferred
        baselines_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "baselines")) # Example adjustment
        os.makedirs(baselines_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"neural_mpc_{timestamp}"

        # Performance Metrics CSV
        perf_csv = os.path.join(baselines_dir, f"{prefix}_performance_metrics.csv")
        avg_wp_time = total_execution_time / total_commands if total_commands > 0 else 0.0
        avg_duration = np.mean(durations) if durations else 0.0
        with open(perf_csv, mode='w', newline='') as perf_file:
             headers = [
                 "Metric", "Value"
             ]
             values = [
                ("Total Execution Time (s)", total_execution_time),
                ("Total Simulation Steps", mpciter),
                ("Total Distance (m)", total_distance),
                ("Average Step Duration (s)", avg_duration),
                ("Final Total Entropy", entropy_history[-1] if entropy_history else "N/A"),
                ("Total Commands Sent", total_commands)
             ]
             writer = csv.writer(perf_file)
             writer.writerow(headers)
             writer.writerows(values)
        print(f"Performance metrics saved to {perf_csv}")

        # Velocity Commands CSV
        vel_csv = os.path.join(baselines_dir, f"{prefix}_velocity_commands.csv")
        with open(vel_csv, mode='w', newline='') as vel_file:
             writer = csv.writer(vel_file)
             writer.writerow(["Time (s)", "Tag", "x_velocity", "y_velocity", "yaw_velocity"])
             writer.writerows(velocity_command_log)
        print(f"Velocity command log saved to {vel_csv}")

        # Plot Data CSV (Pose, Entropy, Lambda Score)
        plot_csv = os.path.join(baselines_dir, f"{prefix}_plot_data.csv")
        with open(plot_csv, mode='w', newline='') as csvfile:
             writer = csv.writer(csvfile)
             # Tree positions header + data
             tree_positions_flat = self.trees_pos[:, :2].flatten().tolist() # Just x, y
             writer.writerow(["tree_positions"] + tree_positions_flat)
             # Main data header
             header = ["time", "x", "y", "theta", "entropy"]
             if lambda_history and len(lambda_history[0]) > 0:
                 num_lambda_scores = len(lambda_history[0])
                 header += [f"lambda_score_{i}" for i in range(num_lambda_scores)]
             writer.writerow(header)
             # Write data rows
             for i in range(len(time_history)):
                 time_val = time_history[i]
                 x_pose, y_pose, theta_pose = np.array(pose_history[i]).flatten()
                 entropy_val = entropy_history[i]
                 lambda_vals = lambda_history[i] # Combined score
                 row = [time_val, x_pose, y_pose, theta_pose, entropy_val] + lambda_vals
                 writer.writerow(row)
        print(f"Plot data saved to {plot_csv}")

        # Raw and Ripe Lambda History CSV
        lambda_detail_csv = os.path.join(baselines_dir, f"{prefix}_lambda_raw_ripe_history.csv")
        with open(lambda_detail_csv, mode='w', newline='') as lambda_file:
            writer = csv.writer(lambda_file)
            num_trees = self.num_total_trees
            header = ["time"] + [f"lambda_raw_{i}" for i in range(num_trees)] + [f"lambda_ripe_{i}" for i in range(num_trees)]
            writer.writerow(header)
            for i in range(len(time_history)):
                 writer.writerow([time_history[i]] + lambda_raw_history[i] + lambda_ripe_history[i])
        print(f"Raw/Ripe lambda history saved to {lambda_detail_csv}")


        # Store history for plotting methods
        self.time_history = time_history
        self.entropy_history = entropy_history
        self.lambda_ripe_history = lambda_ripe_history # Already stored as lists
        self.lambda_raw_history = lambda_raw_history   # Already stored as lists

        return all_trajectories, entropy_history, lambda_history, durations, g_nn, self.trees_pos, lb, ub


    # ---------------------------
    # Plotting Functions (Keep as is, they use the stored history)
    # ---------------------------
    def plot_animated_trajectory_and_entropy_2d(self, all_trajectories, entropy_history, lambda_history, trees, lb, ub, computation_durations):
        # ... (plotting code remains the same - it uses the data generated by run_simulation)
        # Note: The lambda_history used here is the combined score.
        # Note: The entropy prediction part inside this plot function might need adjustment
        #       if the NN was only trained/used on target trees in mpc_opt. It currently
        #       assumes it can predict for *all* trees passed to it.
        #       For simplicity, we can leave it as is, visualizing the *potential* reduction
        #       if the NN *were* applied to all trees shown.
        print("Generating animated plot...")
        # ... (rest of the plotting code) ...
        # --- Potential modification needed inside plot_animated_trajectory_and_entropy_2d ---
        # Inside the loop calculating `entropy_mpc_pred_k`:
        # The input `input_nn = ca.horzcat(relative_position_robot_trees, theta)`
        # uses *all* `trees`. If g_nn expects only inputs related to target trees,
        # this prediction might be inaccurate or cause dimension errors if g_nn isn't robust.
        # A safer approach might be to remove the entropy prediction plot if it causes issues,
        # or modify it to only predict for a subset if feasible.
        # --- End potential modification note ---

        # (The existing plotting code follows)
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model.load_state_dict(torch.load(self.get_latest_best_model(), weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, name='plotting_f', batched=True, device='cuda' if torch.cuda.is_available() else 'cpu')

        x_trajectory = np.array([traj[0] for traj in all_trajectories])
        y_trajectory = np.array([traj[1] for traj in all_trajectories])
        theta_trajectory = np.array([traj[2] for traj in all_trajectories])
        all_trajectories = np.array(all_trajectories) # Shape: (n_steps, nx, N+1)
        lambda_history = np.array(lambda_history) # Shape: (n_steps, n_total_trees)

        # Compute predicted entropy reduction (using the plotting g_nn instance)
        entropy_mpc_pred = []
        num_trees_total = trees.shape[0] # Should be self.num_total_trees
        print(f"Plotting: Predicting entropy for {num_trees_total} trees.")

        # Ensure lambda_history has the correct shape before prediction loop
        if lambda_history.shape[1] != num_trees_total:
             print(f"Warning: lambda_history shape {lambda_history.shape} doesn't match total trees {num_trees_total}. Entropy prediction might be incorrect.")
             # Fallback or adjust based on intended behavior
             # For now, proceed, but be aware of potential issues

        for k in range(all_trajectories.shape[0]): # Iterate through MPC steps taken
            # Use the *full* lambda state at step k for prediction start
            # Need both raw and ripe to calculate min entropy
            if k >= len(self.lambda_raw_history) or k >= len(self.lambda_ripe_history):
                 print(f"Warning: History lengths mismatch at step {k}. Skipping entropy prediction for this frame.")
                 entropy_mpc_pred.append([entropy_history[k]] * (self.N + 1)) # Fill with current entropy
                 continue

            lambda_k_raw = ca.DM(self.lambda_raw_history[k]).reshape((-1, 1))
            lambda_k_ripe = ca.DM(self.lambda_ripe_history[k]).reshape((-1, 1))

            if lambda_k_raw.shape[0] != num_trees_total or lambda_k_ripe.shape[0] != num_trees_total:
                  print(f"Warning: Stored lambda shapes at step {k} don't match total trees. Resizing/Padding needed.")
                  # Handle shape mismatch if necessary (e.g., pad with 0.5) - this indicates a data logging issue usually
                  lambda_k_raw = ca.DM.ones(num_trees_total, 1) * 0.5 # Fallback
                  lambda_k_ripe = ca.DM.ones(num_trees_total, 1) * 0.5 # Fallback


            current_total_entropy = ca.sum1(ca.fmin(self.entropy(lambda_k_raw), self.entropy(lambda_k_ripe))).full().flatten()[0]
            entropy_mpc_pred_k = [current_total_entropy] # Start with current measured entropy

            for i in range(self.N): # Predict N steps into the future (indices 1 to N of x_traj)
                # State at the *end* of prediction step i (time k+i+1)
                state_pred = all_trajectories[k, :, i+1]
                robot_pos_pred = state_pred[:2]
                robot_theta_pred = state_pred[2]

                # Prepare NN input for *all* trees relative to this predicted state
                relative_position_robot_trees = np.tile(robot_pos_pred, (num_trees_total, 1)) - trees[:, :2]
                # distance_robot_trees = np.sqrt(np.sum(relative_position_robot_trees**2, axis=1)) # Not needed for NN input
                theta_input = np.full((num_trees_total, 1), robot_theta_pred) # Tile theta for all trees

                # NN input: [delta_x, delta_y, heading] for all trees
                input_nn = ca.horzcat(relative_position_robot_trees, theta_input)

                try:
                    # Predict measurement likelihood z_k for all trees
                    z_k_pred = ca.fmax(g_nn(input_nn), 0.5) # Shape (num_trees_total, 1)

                    # Update lambdas using Bayes rule
                    lambda_k_raw = self.bayes(lambda_k_raw, z_k_pred) # Raw uses 1-z effectively via callback flip logic
                    lambda_k_ripe = self.bayes(lambda_k_ripe, z_k_pred)

                    # Calculate total effective entropy for the predicted state
                    predicted_entropy = ca.sum1(ca.fmin(self.entropy(lambda_k_raw), self.entropy(lambda_k_ripe))).full().flatten()[0]
                    entropy_mpc_pred_k.append(predicted_entropy)

                except Exception as e:
                    print(f"Error during entropy prediction in plotting at step {k}, prediction step {i}: {e}")
                    # Append previous value or NaN if error occurs
                    entropy_mpc_pred_k.append(entropy_mpc_pred_k[-1] if entropy_mpc_pred_k else np.nan)


            entropy_mpc_pred.append(entropy_mpc_pred_k)

        # Continue with the rest of the plotting code...
        entropy_mpc_pred = np.array(entropy_mpc_pred)
        sum_entropy_history = entropy_history # This is the actual measured total entropy history
        # computation_durations provided as input

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.6, 0.4],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            subplot_titles=("Robot Trajectory & Trees", "Entropy Trend", "Trajectory Details", "Computation Time")
        )

        # Initial Traces
        # 1. MPC Predicted Future Trajectory (from first step)
        fig.add_trace(
            go.Scatter(
                x=all_trajectories[0, 0, :], # x-coords from first MPC prediction
                y=all_trajectories[0, 1, :], # y-coords from first MPC prediction
                mode="lines+markers",
                name="MPC Future Trajectory (Initial)",
                line=dict(width=3, color='orange', dash='dash'),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        # 2. Drone Actual Path History (empty initially)
        fig.add_trace(
            go.Scatter(
                x=[], # Will be filled frame by frame
                y=[], # Will be filled frame by frame
                mode="lines+markers",
                name="Drone Path History",
                line=dict(width=4, color='blue'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        # 3. Tree Markers (initial state)
        initial_lambda_scores = np.array(lambda_history[0]) # Use combined score for initial color
        tree_colors = 2 * (initial_lambda_scores - 0.5) # Scale to [-1, 1] for colorscale
        tree_hover_texts = [f"Tree {i}: Score={initial_lambda_scores[i]:.2f}" for i in range(num_trees_total)]
        fig.add_trace(
            go.Scatter(
                x=trees[:, 0],
                y=trees[:, 1],
                mode="markers+text",
                marker=dict(size=12, color=tree_colors, colorscale='RdYlGn', cmin=-1, cmax=1, symbol='circle'),
                name="Trees",
                text=[str(i) for i in range(num_trees_total)],
                hovertext=tree_hover_texts,
                hoverinfo='text',
                textposition="top center"
            ),
            row=1, col=1
        )

        # 4. Entropy History (Past)
        fig.add_trace(
            go.Scatter(
                x=[], # Time steps (indices)
                y=[], # Actual entropy values
                mode="lines+markers",
                name="Total Entropy (Actual)",
                line=dict(width=2, color='green'),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        # 5. Entropy Prediction (Future)
        fig.add_trace(
            go.Scatter(
                x=[], # Time steps (indices offset)
                y=[], # Predicted entropy values
                mode="lines+markers",
                name="Total Entropy (Predicted)",
                line=dict(width=2, dash="dot", color='purple'),
                marker=dict(size=5)
            ),
            row=1, col=2
        )
        # 6. Computation Durations
        fig.add_trace(
            go.Scatter(
                x=[], # Time steps (indices)
                y=[], # Durations
                mode="lines+markers",
                name="MPC Step Duration",
                line=dict(width=2, color='red'),
                marker=dict(size=5)
            ),
            row=2, col=2 # Place in bottom right subplot
        )

        # Create animation frames.
        frames = []
        num_frames = len(all_trajectories) # Should match len(entropy_history), etc.

        for k in range(num_frames):
            # --- Data for frame k ---
            # 1. MPC Future Trajectory (prediction made at step k)
            frame_mpc_x = all_trajectories[k, 0, :]
            frame_mpc_y = all_trajectories[k, 1, :]

            # 2. Drone Path History (up to step k)
            frame_hist_x = x_trajectory[:k+1, 0] # Actual positions executed up to k
            frame_hist_y = y_trajectory[:k+1, 0]

            # 3. Tree Marker Colors (based on lambda score at step k)
            frame_lambda_scores = np.array(lambda_history[k])
            frame_tree_colors = 2 * (frame_lambda_scores - 0.5)
            frame_tree_hover_texts = [f"Tree {i}: Score={frame_lambda_scores[i]:.2f}" for i in range(num_trees_total)]

            # 4. Entropy History (Past - up to step k)
            frame_entropy_past_x = np.arange(k + 1)
            frame_entropy_past_y = sum_entropy_history[:k+1]

            # 5. Entropy Prediction (Future - made at step k)
            # Ensure prediction exists and has correct length
            if k < len(entropy_mpc_pred) and len(entropy_mpc_pred[k]) == (self.N + 1):
                frame_entropy_future_x = np.arange(k, k + self.N + 1) # Time indices k to k+N
                frame_entropy_future_y = entropy_mpc_pred[k]
            else: # Handle cases where prediction failed or history is short
                frame_entropy_future_x = [k]
                frame_entropy_future_y = [sum_entropy_history[k]] # Just show current point


            # 6. Computation Durations (up to step k)
            frame_durations_x = np.arange(k + 1)
            frame_durations_y = computation_durations[:k+1]


            # --- Orientation Arrows ---
            list_of_annotations = []
            # Current orientation arrow (from MPC prediction) - RED
            x_start_curr = all_trajectories[k, 0, 0]
            y_start_curr = all_trajectories[k, 1, 0]
            theta_curr = all_trajectories[k, 2, 0]
            arrow_len = 0.8 # Length of arrow
            x_end_curr = x_start_curr + arrow_len * np.cos(theta_curr)
            y_end_curr = y_start_curr + arrow_len * np.sin(theta_curr)
            current_arrow = go.layout.Annotation(
                    dict(
                        x=x_end_curr, y=y_end_curr, xref="x1", yref="y1", # Use axes from subplot 1,1
                        showarrow=True, ax=x_start_curr, ay=y_start_curr, axref="x1", ayref="y1",
                        arrowhead=3, arrowwidth=2, arrowcolor="red",
                    )
                )
            list_of_annotations.append(current_arrow)

            # History orientation arrows (from path history) - ORANGE (optional, can make plot busy)
            # for step_idx in range(k + 1):
            #     x_start_hist = x_trajectory[step_idx, 0]
            #     y_start_hist = y_trajectory[step_idx, 0]
            #     theta_hist = theta_trajectory[step_idx, 0]
            #     x_end_hist = x_start_hist + arrow_len * np.cos(theta_hist)
            #     y_end_hist = y_start_hist + arrow_len * np.sin(theta_hist)
            #     history_arrow = go.layout.Annotation(
            #             dict(
            #                 x=x_end_hist, y=y_end_hist, xref="x1", yref="y1",
            #                 showarrow=True, ax=x_start_hist, ay=y_start_hist, axref="x1", ayref="y1",
            #                 arrowhead=2, arrowwidth=1, arrowcolor="orange", opacity=0.6
            #             )
            #         )
            #     list_of_annotations.append(history_arrow)


            # --- Create Frame ---
            frame = go.Frame(
                data=[
                    # Update trace 0: MPC Future Traj
                    go.Scatter(x=frame_mpc_x, y=frame_mpc_y),
                    # Update trace 1: Drone Path History
                    go.Scatter(x=frame_hist_x, y=frame_hist_y),
                    # Update trace 2: Tree Markers (only update marker color and hovertext)
                    go.Scatter(marker=dict(color=frame_tree_colors), hovertext=frame_tree_hover_texts),
                    # Update trace 3: Entropy History (Past)
                    go.Scatter(x=frame_entropy_past_x, y=frame_entropy_past_y),
                    # Update trace 4: Entropy Prediction (Future)
                    go.Scatter(x=frame_entropy_future_x, y=frame_entropy_future_y),
                    # Update trace 5: Computation Durations
                    go.Scatter(x=frame_durations_x, y=frame_durations_y),
                ],
                name=f"Step {k}",
                # IMPORTANT: Specify which traces are being updated by index
                traces=[0, 1, 2, 3, 4, 5],
                layout=go.Layout(annotations=list_of_annotations) # Update annotations for arrows
            )
            frames.append(frame)

        # Assign frames to the figure
        fig.frames = frames

        # Configure animation settings
        play_button = dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": int(self.dt*1000), "redraw": True}, # Duration based on dt
                                         "fromcurrent": True, "transition": {"duration": 0}}] # Immediate transition
                        )
        pause_button = dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}]
                        )

        fig.update_layout(
            title="Neural MPC Simulation: Trajectory, Entropy, and Performance",
            xaxis=dict(title="X Position (m)", range=[lb[0] - 3, ub[0] + 3], scaleanchor="y", scaleratio=1), # Ensure aspect ratio
            yaxis=dict(title="Y Position (m)", range=[lb[1] - 3, ub[1] + 3]),
            xaxis2=dict(title="Time Step"),
            yaxis2=dict(title="Total Effective Entropy"),
            #xaxis3=dict(title="X Position (m)"), # Subplot 2,1 is now unused or can show details
            #yaxis3=dict(title="Y Position (m)"),
            xaxis4=dict(title="Time Step"), # Subplot 2,2 is Computation Time
            yaxis4=dict(title="Computation Duration (s)"),
            updatemenus=[dict(type="buttons", buttons=[play_button, pause_button], showactive=True, x=0.1, y=-0.1)],
            sliders=[{
                "active": 0,
                "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                           "label": str(k), "method": "animate"} for k, f in enumerate(frames)],
                "currentvalue": {"font": {"size": 16}, "prefix": "Step: ", "visible": True},
                "transition": {"duration": 0}, # Immediate transition on slider change
                "x": 0.2, "y": -0.15, "len": 0.8
            }],
            hovermode='closest', # Show hover info for nearest point
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legend on top
        )

        # Adjust subplot titles if needed
        fig.layout.annotations[0].update(text="Robot Trajectory & Trees") # subplot (1,1)
        fig.layout.annotations[1].update(text="Entropy Trend")            # subplot (1,2)
        # fig.layout.annotations[2].update(text="Details")                # subplot (2,1) - remove or repurpose
        fig.layout.annotations[2].update(text="MPC Step Duration")         # subplot (2,2)


        print("Displaying plot...")
        fig.show()
        html_path = os.path.join(os.path.dirname(self.get_latest_best_model()), '..', 'neural_mpc_results.html')
        fig.write_html(html_path)
        print(f"Plot saved to {html_path}")

        return entropy_mpc_pred # Return the predicted entropy calculated during plotting

    def plot_tree_lambda_trends(self):
        # ... (plotting code remains the same - uses self.time_history, self.lambda_raw_history, self.lambda_ripe_history)
        # Make sure these histories store the *full* lambda vectors for all trees.
        print("Generating tree lambda entropy trends plot...")
        # ... (rest of the plotting code) ...
        if (not hasattr(self, "time_history") or
            not hasattr(self, "lambda_raw_history") or
            not hasattr(self, "lambda_ripe_history")):
            print("Time or lambda history is not available. Run the simulation first.")
            return
        if not self.time_history or not self.lambda_raw_history or not self.lambda_ripe_history:
             print("Time or lambda history is empty. Cannot plot trends.")
             return

        # Local helper to compute binary entropy.
        def binary_entropy(p):
            p = np.clip(p, 1e-6, 1-1e-6) # Slightly wider clip for stability
            return -p * np.log2(p) - (1-p) * np.log2(1-p)

        time_history = self.time_history
        num_steps = len(time_history)
        # Determine the number of trees from the first entry of the lambda history.
        if not self.lambda_raw_history[0]:
            print("Lambda history entry is empty.")
            return
        num_trees = len(self.lambda_raw_history[0]) # Should be self.num_total_trees

        # Create subplots: one row per tree. Limit number of rows for readability if too many trees
        max_rows_plot = 20
        rows_to_plot = min(num_trees, max_rows_plot)
        fig = make_subplots(rows=rows_to_plot, cols=1, shared_xaxes=True,
                            subplot_titles=[f"Tree {i} Lambda Entropy Trend" for i in range(rows_to_plot)])

        # For each tree (up to max_rows_plot), compute the entropy over time
        for i in range(rows_to_plot):
            try:
                raw_vals = [entry[i] for entry in self.lambda_raw_history]
                ripe_vals = [entry[i] for entry in self.lambda_ripe_history]
                raw_entropy = [binary_entropy(val) for val in raw_vals]
                ripe_entropy = [binary_entropy(val) for val in ripe_vals]

                # Add traces for raw and ripe entropy.
                fig.add_trace(go.Scatter(x=time_history, y=raw_entropy,
                                         mode='lines', name=f'Tree {i} Raw H', legendgroup=f'tree{i}', showlegend=(i==0), # Show legend only once per type
                                         line=dict(width=2, color='blue')),
                              row=i+1, col=1)
                fig.add_trace(go.Scatter(x=time_history, y=ripe_entropy,
                                         mode='lines', name=f'Tree {i} Ripe H', legendgroup=f'tree{i}', showlegend=(i==0),
                                         line=dict(width=2, color='red', dash='dot')),
                              row=i+1, col=1)
            except IndexError:
                 print(f"Warning: Could not plot trends for Tree {i}, likely due to data inconsistency.")
                 continue # Skip this tree if data is bad

        fig.update_layout(title=f"Lambda Entropy Trends for First {rows_to_plot} Trees",
                          xaxis_title="Time (s)",
                          yaxis_title="Binary Entropy",
                          height=200 * rows_to_plot, # Adjust height based on rows
                          legend_title_text='Entropy Type')
        fig.show()
        html_path = os.path.join(os.path.dirname(self.get_latest_best_model()), '..', 'neural_mpc_tree_lambda_entropy.html')
        fig.write_html(html_path)
        print(f"Tree lambda entropy plot saved to {html_path}")


    def plot_entropy_separately(self):
        # ... (plotting code remains the same - uses self.time_history, self.entropy_history)
        print("Generating separate entropy plot...")
        # ... (rest of the plotting code) ...
        if not hasattr(self, "time_history") or not hasattr(self, "entropy_history"):
            print("Time or entropy history is not available. Run the simulation first.")
            return
        if not self.time_history or not self.entropy_history:
             print("Time or entropy history is empty. Cannot plot entropy.")
             return

        # Create a separate Plotly figure for the entropy plot.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time_history, y=self.entropy_history,
                                 mode='lines+markers', name='Total Effective Entropy'))
        fig.update_layout(title='Total Effective Entropy over Time',
                          xaxis_title='Time (s)',
                          yaxis_title='Entropy Sum (min(H_raw, H_ripe))')
        fig.show()
        html_path = os.path.join(os.path.dirname(self.get_latest_best_model()), '..', 'neural_mpc_entropy.html')
        fig.write_html(html_path)
        print(f"Entropy plot saved to {html_path}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    try:
        rospy.init_node('neural_mpc_node', anonymous=True)
        mpc = NeuralMPC()
        sim_results = mpc.run_simulation()

        # Check if simulation produced results before plotting
        if sim_results and len(sim_results) >= 8:
            # Unpack results for plotting
            all_trajectories, entropy_history, lambda_history, durations, _, trees_pos, lb, ub = sim_results

            # Call plotting methods
            # Animated plot (can be slow/resource intensive)
            mpc.plot_animated_trajectory_and_entropy_2d(all_trajectories, entropy_history, lambda_history, trees_pos, lb, ub, durations)

            # Separate static plots
            mpc.plot_entropy_separately()
            mpc.plot_tree_lambda_trends()
        else:
            print("Simulation did not produce valid results for plotting.")

    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optional: Add any cleanup code here
        print("Neural MPC script finished.")