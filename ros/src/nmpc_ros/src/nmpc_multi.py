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
import tf2_ros
from scipy.stats import norm

import l4casadi as l4c
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ROS message imports
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
import tf
from std_msgs.msg import Float32MultiArray

# Service import for tree poses (as in sensors.py)
from nmpc_ros.srv import GetTreesPoses

# Import helper functions from sensors.py
from nmpc_ros_package.ros_com_lib.sensors import create_path_from_mpc_prediction, create_tree_markers

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
# Neural MPC Class (Modified Version)
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

        # For storing the latest tree scores from the sensor callback.
        self.latest_trees_scores = None
        # For storing the latest robot state from the gps callback.
        self.current_state = None

        rospy.init_node("nmpc_node", anonymous=True, log_level=rospy.DEBUG)

        # agent number
        self.n_agent = rospy.get_param('~n_agent', 1) # default 1
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Variable to hold the current robot state [x, y, yaw]
        self.current_state = None

        # Start the robot state update thread at 30 Hz.
        self.robot_state_thread = threading.Thread(target=self.robot_state_update_thread)
        self.robot_state_thread.daemon = True
        self.robot_state_thread.start()
        # Subscribers
        rospy.Subscriber("tree_scores", Float32MultiArray, self.tree_scores_callback)
        # Publishers
        self.cmd_pose_pub = rospy.Publisher("cmd/pose", Pose, queue_size=10)
        self.pred_path_pub = rospy.Publisher("predicted_path", Path, queue_size=10)
        self.tree_markers_pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=10)
        # send lambda values
        self.lambda_pub = rospy.Publisher('lambda', Float32MultiArray, queue_size=10)

        # Get tree positions from service using the sensors.py serializer logic.
        self.trees_pos = self.get_trees_poses()
        self.lambda_k = ca.DM.ones(self.trees_pos.shape[0], 1) * 0.5

        # Consensus protocol
        # Network connection (fully connected)
        n_robots = 3
        neighbors = list(range(1, n_robots+1))
        neighbors.remove(int(self.n_agent))
        self.neighbors_id = neighbors
        # subscibers
        subscribers_net = []
        for neighbor in neighbors:
            topic = f"/agent_{neighbor}/lambda"
            sub = rospy.Subscriber(topic, Float32MultiArray, self.consensus_lambda)
            subscribers_net.append(sub)
        # neighbors' positions
        self.neighbors_pos = ca.DM.zeros(2*(n_robots+1), 1) # id vicino e id+1 = posizione x e y in lista
        # Lambda consensus variables
        self.lambda_cons = ca.DM.ones(self.trees_pos.shape[0], 1) * 0.5

        # Start the neighbors state update thread at 30 Hz.
        self.neighbors_state_thread = threading.Thread(target=self.neighbors_state_update_thread)
        self.neighbors_state_thread.daemon = True
        self.neighbors_state_thread.start()

        # MPC horizon (number of steps)
        self.mpc_horizon = self.N

    # ---------------------------
    # Callback Functions
    # ---------------------------
    def robot_state_update_thread(self):
        """Continuously update the robot's state using TF at 30 Hz."""
        rate = rospy.Rate(30)  # 30 Hz update rate
        while not rospy.is_shutdown():
            try:
                # Look up the transform from 'map' to 'base_link_n'
                link_str = 'base_link_' + str(self.n_agent)
                trans = self.tf_buffer.lookup_transform('map', link_str, rospy.Time(0))
                # Extract the yaw angle from the quaternion
                (_, _, yaw) = tf.transformations.euler_from_quaternion([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ])
                # Update current_state with [x, y, yaw]
                self.current_state = [trans.transform.translation.x,
                                      trans.transform.translation.y,
                                      yaw]
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("Failed to get transform: %s", e)
            rate.sleep()

    def neighbors_state_update_thread(self):
        """Continuously update the neighbors' state using TF at 30 Hz."""
        rate = rospy.Rate(30)  # 30 Hz update rate
        poss = [0.0 for _ in range(self.neighbors_pos.shape[0])]
        while not rospy.is_shutdown():
            for n in self.neighbors_id:
                try:
                    # Look up the transform from 'map' to 'base_link_n'
                    link_str = 'base_link_' + str(n)
                    trans = self.tf_buffer.lookup_transform('map', link_str, rospy.Time(0))
                    poss[2*n] = trans.transform.translation.x
                    poss[2*n+1] = trans.transform.translation.y
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn("Failed to get transform: %s", e)
            self.neighbors_pos = ca.DM(poss)
            rate.sleep()

    def tree_scores_callback(self, msg):
        """
        Callback for tree scores.
        """
        self.latest_trees_scores = np.array(msg.data).reshape(-1, 1)

    def consensus_lambda(self, msg):
        """
        Callback for consensus lambda.
        """
        # Take data
        neighbors = msg.data
        lambda_curr = self.lambda_cons.full().flatten()
        # Compute maximum/minimum
        result_max = np.maximum(neighbors, lambda_curr)
        # result_min = np.minimum(neighbors, lambda_curr)
        # for i in range(len(result_max)):
        #     val_max = result_max[i] - 0.5
        #     val_min = 0.5 - result_min[i]
        #     if val_max > val_min:
        #         result_min[i] = result_max[i]
        # self.lambda_cons = ca.DM(result_min)
        self.lambda_cons = ca.DM(result_max)

    # ---------------------------
    # Service Call to Get Trees Poses
    # ---------------------------
    def get_trees_poses(self):
        """
        Calls the GetTreesPoses service and returns tree positions as an (N,2) numpy array.
        The serializer logic is taken from sensors.py.
        """
        rospy.wait_for_service("/obj_pose_srv")
        try:
            trees_srv = rospy.ServiceProxy("/obj_pose_srv", GetTreesPoses)
            response = trees_srv()  # Adjust parameters if needed
            # Using the serializer from sensors.py:
            trees_pos = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
            return trees_pos
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return np.array([])

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
        intg_opts = {"number_of_finite_elements": 1, "simplify": 1}
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
    def penalty_2d(x, y, x_c, y_c, p=10, s=1, a=1):
        """Compute penalty term"""
        # return np.exp(-(x**p + y**p)/((2.0*s)**p))
        return a*ca.exp(-((x-x_c)**p + (y-y_c)**p)/((2.0*s)**p))
    
    def aggregation_2d(self, x, y, evol, idx, a=1, d=256):
        """Compute aggregation term"""
        x_c = self.trees_pos[idx][0]
        y_c = self.trees_pos[idx][1]
        lambda_c = evol[idx]
        return (a**2) * (1-lambda_c) * ((x - x_c)**2 + (y - y_c)**2) / (d**2)

    # ---------------------------
    # MPC Optimization Function 
    # ---------------------------
    def mpc_opt(self, g_nn, trees, lb, ub, x0, lambda_vals, neighbors_positions, steps=10):
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
        num_neighbors = neighbors_positions.shape[0]
        P0 = opti.parameter(n_state + num_trees + num_neighbors)
        X0 = P0[: n_state]
        L0 = P0[n_state:n_state+num_trees]
        N0 = P0[n_state+num_trees:]
        # Initialize belief evolution.
        lambda_evol = [L0]

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

        # Penalty term for cells
        penalty_cells = 0 # unassigned
        aggregation = 0   # assigned

        # Get unassigned cells. 
        # List of assigned trees (ID)
        if self.n_agent == 1:
            # assigned = [0, 1, 5, 6, 10, 11, 15, 16]
            assigned = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21]
        if self.n_agent == 2:
            # assigned = [2, 7, 12, 17, 20, 21, 22, 23, 24]
            assigned = [2, 7, 12, 17, 22]
        if self.n_agent == 3:
            # assigned = [3, 4, 8, 9, 13, 14, 18, 19]
            assigned = [3, 4, 8, 9, 13, 14, 18, 19, 23, 24]
        # Not assigned trees (ID)
        not_assigned = [num for num in list(range(num_trees)) if num not in assigned]

        # Loop over the prediction horizon.
        for i in range(steps+1):
            # State and input bounds.
            opti.subject_to(opti.bounded(lb[0] - 4., X[0, i], ub[0] + 4.))
            opti.subject_to(opti.bounded(lb[1] - 2., X[1, i], ub[1] + 2.))
            opti.subject_to(opti.bounded(-6*np.pi, X[2, i], +6*np.pi))

            opti.subject_to(opti.bounded(0.0, ca.sumsqr(X[3:5, i]),4.00))
            opti.subject_to(opti.bounded(-3.14/4, X[5, i], 3.14 / 4))

            # Robot-Robot Collision avoidance
            pi = X0[0:2] 
            for n in self.neighbors_id:
                pj = N0[2*n:2*n+2]
                pij = pj - pi
                dirs = ca.dot(X[0:2, i] - (pi + pj) / 2, pij)
                dist = 0.3 * ca.norm_2(pij)
                opti.subject_to(opti.bounded(-ca.inf, dirs+dist , 0))

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
        # Threshold the NN output
        z_k = ca.fmax(g_out, 0.5)

        for i in range(steps):
            lambda_next = self.bayes(lambda_evol[-1], z_k[i::steps])
            lambda_evol.append(lambda_next)

        # Limited area (Cells) and attraction
        for i in range(steps+1):
            for n_a in not_assigned:
                # penalty for unassigned cells
                penalty_cells += self.penalty_2d(X[0, i], X[1, i], self.trees_pos[n_a][0], self.trees_pos[n_a][1], p=10, s=1, a=5)
            for a_a in assigned:
                # aggregation term for assigned cells
                ### METTERE UN TERMINE CHE FA SCALARE PER IL NUMERO DI ALBERI !!!!!!!!!
                aggregation += self.aggregation_2d(X[0, i], X[1, i], lambda_evol[i], idx=a_a, a=10)

        # Compute entropy terms for the objective.
        entropy_future = self.entropy(ca.vcat([*lambda_evol[1:]]))
        entropy_term = ca.sum1( ca.vcat([ca.exp(-2*i)*ca.DM.ones(num_trees) for i in range(steps)]) * entropy_future) * w_entropy
        #--------------------------- Annulla la funzione degli alberi che non mi interessano
        # mask = ca.DM.zeros(num_trees * steps, 1)
        # for step in range(steps):
        #     for i in assigned:
        #         idx = i + step * num_trees  # indices step successivi
        #         mask[idx] = 1
        # exp_weights = ca.vcat([ca.exp(-2*i) * ca.DM.ones(num_trees, 1) for i in range(steps)])
        # entropy_term = ca.sum1((mask * exp_weights) * entropy_future) * w_entropy
        #---------------------------        
        # Add terms to the objective.
        obj += entropy_term
        obj += penalty_cells
        obj += aggregation
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
        opti.set_value(P0, ca.vertcat(x0, lambda_vals, neighbors_positions))
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
    def run_simulation(self):
        F_ = self.kin_model(self.nx, self.dt)
        lb, ub = self.get_domain(self.trees_pos)
        self.mpc_horizon = self.N

        # Wait until a GPS message has been received.
        rospy.loginfo("Waiting for GPS data...")
        while self.current_state is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        rospy.loginfo("GPS data received.")

        # ---------------------------
        # Load the Learned Neural Network Models
        # ---------------------------
        
        model = MultiLayerPerceptron(input_dim=self.nn_input_dim,
                                     hidden_size=self.hidden_size,
                                     hidden_layers=self.hidden_layers)
        model.load_state_dict(torch.load(self.get_latest_best_model(), weights_only=True))
        model.eval()
        g_nn = l4c.L4CasADi(model, batched=True, device='cuda')

        # Initialize robot state from the latest GPS callback.
        initial_state = self.current_state  # [x, y, theta]
        vx_k = ca.DM.zeros(self.nx)  # velocity component: [vx, vy, omega]
        x_k = ca.vertcat(ca.DM(initial_state), vx_k)

        # Containers for simulation output.
        all_trajectories = []
        lambda_history = []
        entropy_history = []
        durations = []

        velocity_command_log = []
        pose_history = []      # [x, y, theta] for each step
        time_history = []      # simulation time at each step

        sim_start_time = time.time()
        total_distance = 0.0
        total_commands = 0
        sum_vx = 0.0
        sum_vy = 0.0
        sum_yaw = 0.0
        sum_trans_speed = 0.0
        prev_x = float(x_k[0])
        prev_y = float(x_k[1])

        sim_time = 2800  # Total simulation time in seconds
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True

        # Main MPC loop.
        while mpciter < sim_time and not rospy.is_shutdown():
            rospy.loginfo('Step: %d', mpciter)
            # Update state from the latest GPS callback.
            while self.current_state is None and not rospy.is_shutdown():
                rospy.sleep(0.05)
            current_state = self.current_state
            current_sim_time = time.time() - sim_start_time
            self.lambda_k = self.lambda_cons
            pose_history.append(current_state)
            time_history.append(current_sim_time)
            x_k = ca.vertcat(ca.DM(current_state), vx_k)

            # Wait until tree scores have been received.
            while self.latest_trees_scores is None and not rospy.is_shutdown():
                rospy.sleep(0.05)
            latest_trees_scores = self.latest_trees_scores.copy()
            self.lambda_k = self.bayes(self.lambda_k, latest_trees_scores)
            self.lambda_k = np.ceil(self.lambda_k*1000)/1000
            rospy.loginfo("Current state x_k: %s", x_k)
            rospy.loginfo("Lambda: %s", self.lambda_k)
            rospy.loginfo("Current tree scores: %s", latest_trees_scores.flatten())
            rospy.loginfo("VICINI: %s", self.neighbors_pos)

            # Publish tree markers using the helper function from sensors.py.
            tree_markers_msg = create_tree_markers(self.trees_pos, self.lambda_k.full().flatten())
            self.tree_markers_pub.publish(tree_markers_msg)

            step_start_time = time.time()
            if warm_start:
                mpc_step, u, x_traj, x_dec, lam = self.mpc_opt(g_nn, self.trees_pos, lb, ub, x_k, self.lambda_k, self.neighbors_pos, self.mpc_horizon)
                warm_start = False
            else:
                u, x_traj, x_dec, lam = mpc_step(ca.vertcat(x_k, self.lambda_k, ca.DM(self.neighbors_pos)), x_dec, lam)
            durations.append(time.time() - step_start_time)
            # Log the MPC velocity command.
            u_np = np.array(u.full()).flatten()

            # Compute the command pose.
            cmd_pose = F_(x_k, u[:, 0])

            # Publish predicted path.
            predicted_path_msg = create_path_from_mpc_prediction(x_traj[:self.nx, 1:])
            self.pred_path_pub.publish(predicted_path_msg)

            # Build and publish the cmd_pose message.
            quaternion = tf.transformations.quaternion_from_euler(0, 0, float(cmd_pose[2]))
            cmd_pose_msg = Pose()
            cmd_pose_msg.position = Point(x=float(cmd_pose[0]), y=float(cmd_pose[1]), z=0.0)
            cmd_pose_msg.orientation = Quaternion(x=quaternion[0],
                                                  y=quaternion[1],
                                                  z=quaternion[2],
                                                  w=quaternion[3])
            self.cmd_pose_pub.publish(cmd_pose_msg)

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

            entropy_k = ca.sum1(self.entropy(self.lambda_k)).full().flatten()[0]
            lambda_history.append(self.lambda_k.full().flatten().tolist())
            entropy_history.append(entropy_k)
            all_trajectories.append(x_traj[:self.nx, :].full())

            # send lambda values
            lambda_msg = Float32MultiArray()
            lambda_msg.data = self.lambda_k.full().flatten()
            self.lambda_pub.publish(lambda_msg)

            mpciter += 1
            rospy.loginfo("Entropy: %s", entropy_k)
            if all( v >=0.99 for v in  self.lambda_k.full().flatten()):
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
        baselines_dir = os.path.join(script_dir, "baselines")
        os.makedirs(baselines_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_csv = os.path.join(baselines_dir, f"mpc_n{self.n_agent}_{timestamp}_performance_metrics.csv")
        with open(perf_csv, mode='w', newline='') as perf_file:
            writer = csv.writer(perf_file)
            writer.writerow([
                "Total Execution Time (s)",
                "Total Distance (m)",
                "Average Waypoint-to-Waypoint Time (s)",
                "Final Entropy",
                "Total Commands"
            ])
            writer.writerow([
                total_execution_time,
                total_distance,
                avg_wp_time,
                entropy_history[-1] if entropy_history else "N/A",
                total_commands
            ])

        vel_csv = os.path.join(baselines_dir, f"mpc_n{self.n_agent}_{timestamp}_velocity_commands.csv")
        with open(vel_csv, mode='w', newline='') as vel_file:
            writer = csv.writer(vel_file)
            writer.writerow(["Time (s)", "Tag", "x_velocity", "y_velocity", "yaw_velocity"])
            writer.writerows(velocity_command_log)

        rospy.loginfo("Performance metrics saved to %s", perf_csv)
        rospy.loginfo("Velocity command log saved to %s", vel_csv)

        plot_csv = os.path.join(baselines_dir, f"mpc_n{self.n_agent}_{timestamp}_plot_data.csv")
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
        rospy.loginfo("Plot data saved to %s", plot_csv)

        return all_trajectories, entropy_history, lambda_history, durations, g_nn, self.trees_pos, lb, ub

    # ---------------------------
    # Plotting Function
    # ---------------------------
def plot_animated_trajectory_and_entropy_2d(self, all_trajectories, entropy_history, lambda_history, trees, lb, ub, computation_durations):
        # Reload the model for plotting.
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
        fig.write_html('neural_mpc_results.html')
        return entropy_mpc_pred

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    rospy.init_node("neural_mpc_node", anonymous=True)
    mpc = NeuralMPC()
    sim_results = mpc.run_simulation()
    # Optionally, call the plotting method:
    # mpc.plot_animated_trajectory_and_entropy_2d(*sim_results[:-4], computation_durations=sim_results[3])
