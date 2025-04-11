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
# Removed torch import
import tf2_ros
# Removed scipy.stats import (norm)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import l4casadi as l4c # Keep for potential future use, though not in current core logic
# Removed plotly imports

# ROS message imports
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker # Import Marker
from nav_msgs.msg import Path
import tf

# Service import for tree poses
from nmpc_ros.srv import GetTreesPoses

# Import helper functions (adjusted)
# We'll define create_path_from_mpc_prediction and create_tree_markers locally
# or ensure they are correctly accessible from the package structure.
# Assuming they are defined later in this file or imported correctly.

# ---------------------------
# Helper Functions (from sensors.py or similar, potentially modified)
# ---------------------------
def create_path_from_mpc_prediction(mpc_prediction_states):
    """
    Creates a Path message from MPC predicted states (x, y, theta).
    Assumes mpc_prediction_states is a NumPy array [3, N].
    """
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"  # Or your relevant frame

    for i in range(mpc_prediction_states.shape[1]):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = path_msg.header.stamp + rospy.Duration(i * 0.1) # Example timing
        pose_stamped.header.frame_id = path_msg.header.frame_id
        pose_stamped.pose.position.x = float(mpc_prediction_states[0, i])
        pose_stamped.pose.position.y = float(mpc_prediction_states[1, i])
        pose_stamped.pose.position.z = 0.0 # Assuming 2D movement

        quaternion = tf.transformations.quaternion_from_euler(0, 0, float(mpc_prediction_states[2, i]))
        pose_stamped.pose.orientation = Quaternion(*quaternion)

        path_msg.poses.append(pose_stamped)

    return path_msg

def create_tree_markers(tree_positions, tree_scores):
    """
    Creates visualization markers for trees based on their positions and scores.
    Scores are assumed to be between 0 and 1, influencing color (e.g., Green to Red).
    """
    marker_array = MarkerArray()
    num_trees = tree_positions.shape[0]
    scores_np = np.array(tree_scores).flatten()

    if len(scores_np) != num_trees:
        rospy.logwarn(f"Mismatch between number of trees ({num_trees}) and scores ({len(scores_np)})")
        # Handle mismatch: maybe use default score or skip markers
        scores_np = np.ones(num_trees) * 0.5 # Default to neutral

    for i in range(num_trees):
        marker = Marker()
        marker.header.frame_id = "map" # Or your relevant frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trees"
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(tree_positions[i, 0])
        marker.pose.position.y = float(tree_positions[i, 1])
        marker.pose.position.z = 0.5 # Place marker slightly above ground
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.5 # Diameter
        marker.scale.y = 0.5
        marker.scale.z = 1.0 # Height

        # Color based on score: Green (high score) to Red (low score)
        score = max(0.0, min(1.0, scores_np[i])) # Clamp score
        marker.color.r = float(1.0 - score)
        marker.color.g = float(score)
        marker.color.b = 0.0
        marker.color.a = 0.8 # Alpha

        marker.lifetime = rospy.Duration(2.0) # How long the marker lasts

        marker_array.markers.append(marker)

    return marker_array

# ---------------------------
# CasADi FOV/GP Helper Functions (from test_fov_mpc.py)
# ---------------------------
def drone_objects_distances_casadi(drone_pos_2d, objects_pos):
    if not isinstance(objects_pos, (ca.MX, ca.SX)) and objects_pos.shape[1] != 2:
         raise ValueError("objects_pos should have shape (N, 2)")
    n_objects = objects_pos.shape[0]
    if hasattr(drone_pos_2d, 'shape') and drone_pos_2d.shape == (1, 2):
         drone_pos_2d = drone_pos_2d.T
    elif not (hasattr(drone_pos_2d, 'shape') and drone_pos_2d.shape == (2, 1)):
         if hasattr(drone_pos_2d, 'numel') and drone_pos_2d.numel() == 2:
             drone_pos_2d = drone_pos_2d.reshape((2,1))
         elif isinstance(drone_pos_2d, (np.ndarray, list)) and len(drone_pos_2d) == 2:
             drone_pos_2d = np.array(drone_pos_2d).reshape(2,1)
         else:
            if not isinstance(drone_pos_2d, (ca.SX, ca.MX)) or not drone_pos_2d.is_symbolic():
                 raise ValueError(f"drone_pos_2d should have shape (2, 1) or (1, 2), got {type(drone_pos_2d)}")
    diff = objects_pos - ca.repmat(drone_pos_2d.T, n_objects, 1)
    distances_sq = ca.sum2(diff**2)
    distances = ca.sqrt(distances_sq)
    return distances

def norm_sigmoid_ca(x, thresh=0.5, delta=0.1, alpha=1.0):
    k = alpha / delta
    c = thresh
    return 1 / (1 + ca.exp(-k * (x - c)))

def gaussian_ca(x, mu=0.0, sig=1.0):
    return ca.exp(-((x - mu)**2) / (2 * sig**2))

def fov_weight_fun_casadi_var_light(example_objects_pos, thresh_distance=5):
    sig = 1.5
    thresh = 0.7
    delta = 0.1
    alpha = 1.0
    epsilon = 1e-9
    drone_pos_sym = ca.MX.sym("drone_state", 3) # [x, y, theta]
    objects_pos_sym = ca.MX.sym("objects_pos", example_objects_pos.shape[0], example_objects_pos.shape[1])
    light_angle_sym = ca.MX.sym("light_angle")
    drone_xy_sym = drone_pos_sym[0:2]
    theta = drone_pos_sym[2]
    distances = drone_objects_distances_casadi(drone_xy_sym, objects_pos_sym)
    drone_view_dir_sym = ca.vertcat(ca.cos(theta), ca.sin(theta))
    object_directions = objects_pos_sym - ca.repmat(drone_xy_sym.T, objects_pos_sym.shape[0], 1)
    object_dir_norms = ca.sqrt(ca.sum2(object_directions**2))
    safe_object_dir_norms = object_dir_norms + epsilon * (object_dir_norms < epsilon)
    norm_object_directions_matrix = object_directions / ca.repmat(safe_object_dir_norms, 1, 2)
    vect_alignment = norm_object_directions_matrix @ drone_view_dir_sym
    alignment_input = ((vect_alignment + 1) / 2) ** 2
    alignment_score = norm_sigmoid_ca(alignment_input, thresh=thresh, delta=delta, alpha=alpha)
    distance_score = gaussian_ca(distances, mu=thresh_distance, sig=sig)
    light_dir_sym = ca.vertcat(ca.cos(light_angle_sym), ca.sin(light_angle_sym))
    vect_light_alignment_sym = ca.dot(light_dir_sym, drone_view_dir_sym)
    light_score_input = ((vect_light_alignment_sym + 1) / 2) ** 2
    light_score_sym = norm_sigmoid_ca(light_score_input, thresh=thresh, delta=delta, alpha=alpha)
    result = alignment_score * distance_score * light_score_sym
    return ca.Function('fov_function_var_light',
                       [drone_pos_sym, objects_pos_sym, light_angle_sym],
                       [result],
                       ['drone_state', 'object_positions', 'light_angle'],
                       ['weights'])

def rbf_kernel_casadi_2d(x1_2d, x2_matrix_2d, length_scale, constant_value):
    n_data = x2_matrix_2d.shape[0]
    n_dim = x2_matrix_2d.shape[1]
    if n_dim != 2: raise ValueError("rbf_kernel_casadi_2d expects 2D inputs (shape N x 2)")
    if hasattr(x1_2d, 'shape') and x1_2d.shape == (2, 1): x1_2d = x1_2d.T
    elif not (hasattr(x1_2d, 'shape') and x1_2d.shape == (1, 2)):
         if hasattr(x1_2d, 'numel') and x1_2d.numel() == n_dim: x1_2d = x1_2d.reshape((1,n_dim))
         else:
              if not isinstance(x1_2d, (ca.SX, ca.MX)) or not x1_2d.is_symbolic(): raise ValueError(f"x1_2d shape error")
    diff = x2_matrix_2d - ca.repmat(x1_2d, n_data, 1)
    if isinstance(length_scale, (int, float, ca.DM)) and (not hasattr(length_scale, 'numel') or length_scale.numel() == 1) :
        sq_dist = ca.sum2(diff**2)
        k_values = constant_value * ca.exp(-0.5 * sq_dist / length_scale**2)
    elif isinstance(length_scale, (list, np.ndarray, ca.DM)) and ca.DM(length_scale).numel() == n_dim:
         ls_ca = ca.DM(length_scale).T
         if ls_ca.shape[1] != n_dim: ls_ca = ls_ca.T
         scaled_diff_sq = (diff / ca.repmat(ls_ca, n_data, 1))**2
         sq_dist_sum = ca.sum1(scaled_diff_sq)
         k_values = constant_value * ca.exp(-0.5 * sq_dist_sum)
    else: raise TypeError(f"Unsupported length_scale type/shape")
    return k_values

# ---------------------------
# FOV MPC Class (Replaces NeuralMPC)
# ---------------------------
class FovMPC:
    def __init__(self):
        # Global Constants and Parameters
        self.N = 10      # Prediction horizon steps (from test_fov_mpc)
        self.dt = 0.25   # Time step (from test_fov_mpc)
        self.T = self.dt * self.N
        self.nx = 3      # Base state dim [x, y, theta]
        self.n_states = self.nx * 2 # Full state dim [x,y,th,vx,vy,om]
        self.n_controls = self.nx   # Control dim [ax,ay,alpha]

        # MPC Parameters (from test_fov_mpc)
        self.MEASUREMENT_NOISE_STD = 0.05 # Estimated noise in score measurement
        self.MAX_ACCEL = 4.0
        self.MAX_ANG_ACCEL = np.pi / 2
        self.MAX_VEL = 2.0
        self.MAX_ANG_VEL = np.pi / 4
        self.W_FOV = 100.0      # Weight for maximizing FOV score
        self.W_GP_ERROR = 100.0 # Weight for minimizing predicted GP error
        self.R_ACCEL = 0.1      # Control effort penalty (linear)
        self.R_ANG_ACCEL = 0.05 # Control effort penalty (angular)
        self.safe_distance = 1.0 # Safety margin for collision avoidance

        # GP-related state
        self.X_train_angles_list = [] # Store optimized 1D light angles
        self.Y_train_error_list = []  # Store calculated errors (Measured Score - Predicted Score)
        self.gp = None
        self.gp_params_available = False
        self.alpha_ = None # GP internal parameter for CasADi prediction
        self.opt_length_scale_ca = ca.DM(0.5) # Default GP kernel param
        self.opt_constant_value_ca = ca.DM(0.1) # Default GP kernel param
        self.kernel_2d = ConstantKernel(0.1, (1e-4, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 5e1)) \
                         + WhiteKernel(noise_level=self.MEASUREMENT_NOISE_STD**2,
                                       noise_level_bounds=(1e-7, 1e-1))

        # Internal state
        self.latest_trees_scores_sum = None # Store SUM of scores from callback
        self.current_state = None # [x, y, yaw] from TF
        self.current_state_full = None # [x, y, yaw, vx, vy, omega]
        self.light_theta_guess = 0.0 # Initial guess for light angle
        self.opti_solver_success = True # Track solver status for warm start
        self.U_guess = np.zeros((self.n_controls, self.N))
        self.X_guess = None # Initialized later when first state is available
        self.trees_pos = None # Loaded from service

        rospy.init_node("fov_mpc_node", anonymous=True, log_level=rospy.INFO) # Changed level

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Start the robot state update thread at 30 Hz.
        self.robot_state_thread = threading.Thread(target=self.robot_state_update_thread)
        self.robot_state_thread.daemon = True
        self.robot_state_thread.start()

        # Subscribers
        # Expects Float32MultiArray where data is [score_tree1, score_tree2, ...]
        rospy.Subscriber("tree_scores", Float32MultiArray, self.tree_scores_callback)

        # Publishers
        self.cmd_pose_pub = rospy.Publisher("agent_0/cmd/pose", Pose, queue_size=10)
        self.pred_path_pub = rospy.Publisher("agent_0/predicted_path", Path, queue_size=10)
        self.tree_markers_pub = rospy.Publisher("agent_0/tree_markers", MarkerArray, queue_size=10)

        # Get tree positions from service
        self.trees_pos = self.get_trees_poses()
        if self.trees_pos is None or self.trees_pos.shape[0] == 0:
             rospy.logfatal("Failed to get tree positions. Shutting down.")
             rospy.signal_shutdown("Failed to get tree positions")
             return
        rospy.loginfo(f"Received {self.trees_pos.shape[0]} tree positions.")

        # Initialize FOV function instance now that we have tree positions
        self.fov_func_instance = fov_weight_fun_casadi_var_light(self.trees_pos)

        # Kinematic model function
        self.F_dyn = self.kin_model(self.nx, self.dt)

    # ---------------------------
    # Callback Functions
    # ---------------------------
    def robot_state_update_thread(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                trans = self.tf_buffer.lookup_transform('map', 'drone_base_link', rospy.Time(0), rospy.Duration(0.1))
                (_, _, yaw) = tf.transformations.euler_from_quaternion([
                    trans.transform.rotation.x, trans.transform.rotation.y,
                    trans.transform.rotation.z, trans.transform.rotation.w
                ])
                self.current_state = [trans.transform.translation.x,
                                      trans.transform.translation.y,
                                      yaw]
                # Initialize full state and X_guess on first successful TF lookup
                if self.current_state_full is None:
                    self.current_state_full = np.array(self.current_state + [0.0, 0.0, 0.0])
                    self.X_guess = np.tile(self.current_state_full, (self.N + 1, 1)).T
                    rospy.loginfo(f"Initial robot state received: {self.current_state_full}")

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                if self.current_state is None: # Only warn if we haven't received state yet
                    rospy.logwarn_throttle(5.0, "Waiting for transform map -> drone_base_link: %s", e)
            rate.sleep()

    def tree_scores_callback(self, msg):
        """
        Callback for tree scores. Assumes msg.data is a list of scores,
        one per tree, influenced by the true (unknown) light angle.
        We store the SUM of these scores.
        """
        scores = np.array(msg.data)
        if len(scores) == self.trees_pos.shape[0]:
            self.latest_trees_scores_sum = float(np.sum(scores))
            # Optional: Store individual scores if needed for markers later
            self.latest_trees_scores_individual = scores
            # rospy.logdebug(f"Received tree scores. Sum: {self.latest_trees_scores_sum:.3f}")
        else:
             rospy.logwarn_throttle(5.0, f"Received scores length ({len(scores)}) does not match tree count ({self.trees_pos.shape[0]})")
             self.latest_trees_scores_sum = None # Invalidate score if mismatch


    # ---------------------------
    # Service Call to Get Trees Poses
    # ---------------------------
    def get_trees_poses(self):
        rospy.loginfo("Waiting for /obj_pose_srv service...")
        try:
            rospy.wait_for_service("/obj_pose_srv", timeout=15.0)
        except rospy.ROSException:
            rospy.logerr("Service /obj_pose_srv not available after 15 seconds.")
            return None
        try:
            trees_srv = rospy.ServiceProxy("/obj_pose_srv", GetTreesPoses)
            response = trees_srv()
            trees_pos = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
            if trees_pos.size == 0:
                 rospy.logwarn("Received empty tree poses from service.")
                 return np.array([])
            return trees_pos
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return None
        except Exception as e:
            rospy.logerr(f"Error processing tree poses: {e}")
            return None


    # ---------------------------
    # Utility Functions (Domain, Kinematics)
    # ---------------------------
    @staticmethod
    def get_domain(tree_positions):
        if tree_positions is None or tree_positions.shape[0] == 0:
            return [-10, -10], [10, 10] # Default domain if no trees
        x_min = np.min(tree_positions[:, 0]); x_max = np.max(tree_positions[:, 0])
        y_min = np.min(tree_positions[:, 1]); y_max = np.max(tree_positions[:, 1])
        return [x_min, y_min], [x_max, y_max]

    @staticmethod
    def kin_model(nx_base, dt):
        """ 2nd order Kinematic model using Euler integration """
        n_state = nx_base * 2 # [x, y, th, vx, vy, om]
        n_control = nx_base   # [ax, ay, alpha]
        X = ca.MX.sym('X', n_state)
        U = ca.MX.sym('U', n_control)
        # State: x, y, theta, vx, vy, omega
        # Control: ax, ay, alpha
        x, y, theta, vx, vy, omega = X[0], X[1], X[2], X[3], X[4], X[5]
        ax, ay, alpha = U[0], U[1], U[2]
        rhs = ca.vertcat(
            x + vx * dt,
            y + vy * dt,
            theta + omega * dt,
            vx + ax * dt,
            vy + ay * dt,
            omega + alpha * dt
        )
        return ca.Function('F_kin', [X, U], [rhs])

    # ---------------------------
    # MPC Optimization Function
    # ---------------------------
    def mpc_opt(self, x0_full, trees_pos_np, current_light_guess):
        """
        Sets up and solves the FOV MPC problem.
        x0_full: Current full state [x,y,th,vx,vy,om]
        trees_pos_np: Numpy array of tree positions (N_trees, 2)
        current_light_guess: Initial guess for the light angle optimization
        """
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.n_states, self.N + 1) # State trajectory
        U = opti.variable(self.n_controls, self.N)   # Control sequence
        light_theta = opti.variable()                # Estimated light angle

        # Parameters
        X0_param = opti.parameter(self.n_states)
        objects_pos_param = opti.parameter(trees_pos_np.shape[0], trees_pos_np.shape[1])

        # Objective function
        objective = 0
        lb, ub = self.get_domain(trees_pos_np) # Get bounds for state constraints

        # GP Error Term (Symbolic)
        gp_error_pred_sym = ca.DM(0.0) # Default to zero if GP not ready
        if self.gp_params_available and self.alpha_ is not None and len(self.X_train_angles_list) > 0:
            try:
                # Convert training data and GP params to CasADi DM
                X_train_angles_np = np.array(self.X_train_angles_list)
                X_train_gp_2d_np = np.stack([np.cos(X_train_angles_np), np.sin(X_train_angles_np)], axis=-1)
                X_train_ca_2d = ca.DM(X_train_gp_2d_np)
                alpha_ca = ca.DM(self.alpha_)

                # Ensure kernel params are CasADi DMs
                opt_length_scale_ca = ca.DM(self.opt_length_scale_ca) # Use stored optimized params
                opt_constant_value_ca = ca.DM(self.opt_constant_value_ca)

                # Symbolic representation of the angle to predict for
                light_theta_2d_sym = ca.horzcat(ca.cos(light_theta), ca.sin(light_theta))

                # Calculate kernel vector k(theta_est, X_train)
                k_star_sym = rbf_kernel_casadi_2d(light_theta_2d_sym, X_train_ca_2d,
                                                  opt_length_scale_ca, opt_constant_value_ca)

                # Symbolic GP prediction: E[error] = k* @ alpha
                gp_error_pred_sym = ca.dot(k_star_sym, alpha_ca)
                objective += self.W_GP_ERROR * (gp_error_pred_sym**2) # Penalize squared predicted error

            except Exception as e:
                rospy.logwarn_throttle(10.0, f"Error setting up symbolic GP term: {e}. Disabling GP objective term.")
                gp_error_pred_sym = ca.DM(0.0) # Fallback if symbolic setup fails
        else:
            gp_error_pred_sym = ca.DM(0.0) # GP not ready

        # Loop over the prediction horizon (stage costs)
        for k in range(self.N):
            state_k = X[:, k]
            control_k = U[:, k]
            state_k_pos_orient = state_k[0:3] # Extract [x, y, theta] for FOV

            # --- FOV Score Term ---
            # Calculate FOV weights using the *estimated* light_theta
            fov_results_sym = self.fov_func_instance(drone_state=state_k_pos_orient,
                                                     object_positions=objects_pos_param,
                                                     light_angle=light_theta)
            # Sum scores over all trees for this step k
            fov_score_k_sym = ca.sum1(fov_results_sym['weights'])
            # Objective: Maximize score -> Minimize (1 - score)^2
            objective += self.dt * self.W_FOV * (1 - fov_score_k_sym)**2

            # --- Control Effort Penalty ---
            objective += self.dt * (self.R_ACCEL * ca.sumsqr(control_k[0:2]) + \
                                    self.R_ANG_ACCEL * ca.sumsqr(control_k[2]))

            # --- Dynamics Constraint ---
            opti.subject_to(X[:, k+1] == self.F_dyn(state_k, control_k))

            # --- Collision Avoidance Constraint ---
            delta = state_k[:2] - objects_pos_param.T
            sq_dists = ca.sum1(delta**2) # squared distance to each tree
            opti.subject_to(sq_dists >= self.safe_distance**2) # Constraint for each tree

        # --- Constraints ---
        # Initial state
        opti.subject_to(X[:, 0] == X0_param)

        # State bounds (applied over the horizon)
        opti.subject_to(opti.bounded(lb[0] - 2.0, X[0, :], ub[0] + 2.0)) # X pos bounds
        opti.subject_to(opti.bounded(lb[1] - 2.0, X[1, :], ub[1] + 2.0)) # Y pos bounds
        # No explicit theta bound needed if using atan2 logic later
        opti.subject_to(opti.bounded(-self.MAX_VEL, X[3, :], self.MAX_VEL)) # vx
        opti.subject_to(opti.bounded(-self.MAX_VEL, X[4, :], self.MAX_VEL)) # vy
        opti.subject_to(opti.bounded(-self.MAX_ANG_VEL, X[5, :], self.MAX_ANG_VEL)) # omega

        # Control bounds
        opti.subject_to(opti.bounded(-self.MAX_ACCEL, U[0:2, :], self.MAX_ACCEL)) # ax, ay
        opti.subject_to(opti.bounded(-self.MAX_ANG_ACCEL, U[2, :], self.MAX_ANG_ACCEL)) # alpha

        # Light angle constraint (keep within [-pi, pi] range, relative to guess for solver stability)
        # Allow wrapping around the circle
        opti.subject_to(opti.bounded(-np.pi + current_light_guess, light_theta, np.pi + current_light_guess))


        # --- Set Parameter Values and Initial Guesses ---
        opti.set_value(X0_param, x0_full)
        opti.set_value(objects_pos_param, trees_pos_np)

        # Warm starting
        if self.opti_solver_success: # Use previous solution if successful
            opti.set_initial(X, self.X_guess)
            opti.set_initial(U, self.U_guess)
            opti.set_initial(light_theta, self.light_theta_guess) # Use wrapped guess
        else: # Reset guess if previous solve failed
             rospy.logwarn("Previous MPC solve failed, resetting initial guess.")
             # Create a basic feasible guess (e.g., stay still)
             X_reset_guess = np.tile(x0_full, (self.N + 1, 1)).T
             U_reset_guess = np.zeros((self.n_controls, self.N))
             opti.set_initial(X, X_reset_guess)
             opti.set_initial(U, U_reset_guess)
             opti.set_initial(light_theta, current_light_guess) # Use current guess


        # --- Solver Setup and Solve ---
        options = {
            "ipopt": {
                "tol": 1e-3, # Slightly looser tolerance maybe
                "constr_viol_tol": 1e-3,
                "acceptable_tol": 1e-2,
                "warm_start_init_point": "yes",
                "hessian_approximation": "limited-memory",
                "print_level": 0, # Suppress verbose IPOPT output
                "sb": "yes", # Suppress IPOPT banner
                "max_iter": 150, # Limit iterations
            },
            "print_time": False # Suppress CasADi timing info
        }
        opti.solver("ipopt", options)

        try:
            sol = opti.solve()
            u_opt_first = ca.DM(sol.value(U[:, 0]))
            x_traj_opt = ca.DM(sol.value(X))
            light_theta_opt = float(sol.value(light_theta))
            # Wrap the optimized angle to [-pi, pi] for consistent storage/use
            light_theta_opt_wrapped = math.atan2(math.sin(light_theta_opt), math.cos(light_theta_opt))

            # Store warm start info for next iteration
            self.X_guess = sol.value(X)
            self.U_guess = sol.value(U)
            # Store the *wrapped* angle as the guess for the next iteration
            self.light_theta_guess = light_theta_opt_wrapped
            self.opti_solver_success = True

            # Retrieve predicted GP error value if GP was used
            final_gp_error_pred_opt = float(sol.value(gp_error_pred_sym)) if self.gp_params_available else 0.0

            return u_opt_first, x_traj_opt, light_theta_opt_wrapped, final_gp_error_pred_opt, True # Success

        except Exception as e:
            rospy.logerr(f"MPC Solver failed: {e}")
            # Use debug values or a safe fallback (e.g., zero control)
            u_fallback = ca.DM.zeros(self.n_controls)
            x_fallback = ca.DM(opti.debug.value(X)) # Last feasible state trajectory if available
            light_theta_fallback = float(opti.debug.value(light_theta)) # Last feasible angle
            light_theta_fallback_wrapped = math.atan2(math.sin(light_theta_fallback), math.cos(light_theta_fallback))

            # Don't update guess if solver failed
            self.opti_solver_success = False
            # Keep the previous light_theta_guess

            return u_fallback, x_fallback, light_theta_fallback_wrapped, 0.0, False # Failure


    # ---------------------------
    # Main Simulation Loop
    # ---------------------------
    def run_simulation(self):
        rospy.loginfo("Waiting for initial robot state from TF...")
        while self.current_state_full is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if rospy.is_shutdown(): return
        rospy.loginfo("Initial robot state received. Starting MPC loop.")

        # Simulation parameters
        sim_time_steps = 200 # Number of MPC steps
        mpciter = 0
        rate = rospy.Rate(1.0 / self.dt) # Match MPC step time

        # Data Logging Setup
        log_data = []
        log_headers = [
            "Iteration", "Timestamp",
            "State_x", "State_y", "State_theta", "State_vx", "State_vy", "State_omega",
            "Control_ax", "Control_ay", "Control_alpha",
            "Est_Light_Angle", "Measured_FOV_Score", "Predicted_FOV_Score_Current",
            "GP_Error_Pred_Optimized", "Calculated_Error_GP", "GP_Points",
            "Solve_Time", "Solver_Success"
        ]
        sim_start_time = rospy.get_time()
        last_print_time = sim_start_time

        # --- Main MPC Loop ---
        while mpciter < sim_time_steps and not rospy.is_shutdown():
            iter_start_time = time.time()

            # --- 1. Get Current State (already updated by thread) ---
            if self.current_state_full is None:
                rospy.logwarn_throttle(2.0,"Waiting for current state...")
                rospy.sleep(0.1)
                continue
            current_state_full_np = self.current_state_full.copy()

            # --- 2. Update/Train GP (Error Model) ---
            if len(self.X_train_angles_list) >= 2: # Need at least a few points
                try:
                    X_train_angles_np = np.array(self.X_train_angles_list)
                    X_train_gp_2d_np = np.stack([np.cos(X_train_angles_np), np.sin(X_train_angles_np)], axis=-1)
                    Y_train_gp_error_np = np.array(self.Y_train_error_list)

                    self.gp = GaussianProcessRegressor(kernel=self.kernel_2d,
                                                       n_restarts_optimizer=5 if mpciter > 5 else 1,
                                                       alpha=self.MEASUREMENT_NOISE_STD**2,
                                                       normalize_y=False,
                                                       random_state=mpciter) # Seed changes
                    self.gp.fit(X_train_gp_2d_np, Y_train_gp_error_np)

                    self.alpha_ = self.gp.alpha_ # Store for CasADi
                    params = self.gp.kernel_.get_params()
                    self.opt_constant_value_ca = ca.DM(params['k1__k1__constant_value'])
                    self.opt_length_scale_ca = ca.DM(params['k1__k2__length_scale'])
                    # fitted_noise_level = params.get('k2__noise_level', 'N/A')
                    self.gp_params_available = True
                    # rospy.logdebug(f"GP trained LML: {self.gp.log_marginal_likelihood_value_:.3f}, "
                    #                f"Scale={self.opt_constant_value_ca}, Len={self.opt_length_scale_ca}, "
                    #                f"Noise={fitted_noise_level:.4g}")

                except Exception as e:
                    rospy.logwarn(f"GP training failed at iter {mpciter}: {e}. Disabling GP term.")
                    self.gp = None
                    self.gp_params_available = False
                    self.alpha_ = None
            else:
                # Not enough data yet
                self.gp_params_available = False
                self.alpha_ = None


            # --- 3. Solve MPC Optimization ---
            u_opt, x_traj_opt, light_theta_opt, gp_error_pred_opt, success = self.mpc_opt(
                current_state_full_np,
                self.trees_pos,
                self.light_theta_guess # Provide the current wrapped guess
            )
            solve_time = time.time() - iter_start_time

            # --- 4. Apply First Control & Update State ---
            # Calculate the *next* state based on the *first* optimal control
            next_state_sim = self.F_dyn(current_state_full_np, u_opt).full().flatten()
            # Ensure theta wraps correctly
            next_state_sim[2] = math.atan2(math.sin(next_state_sim[2]), math.cos(next_state_sim[2]))

            # In a real system, we wouldn't set self.current_state_full directly.
            # Instead, we'd send the command, and the TF thread would update it.
            # For simulation/testing purposes, we update it here.
            # In ROS, the command publishing happens next. The state update relies on the robot moving
            # and TF reflecting that change before the *next* iteration.
            self.current_state_full = next_state_sim # Update internal state for next MPC step

            # --- 5. Publish Commands and Visualizations ---
            # Calculate target pose for the *end* of the dt interval
            target_pose_state = next_state_sim # Use the simulated next state
            cmd_pose_msg = Pose()
            cmd_pose_msg.position = Point(x=float(target_pose_state[0]), y=float(target_pose_state[1]), z=0.0)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, float(target_pose_state[2]))
            cmd_pose_msg.orientation = Quaternion(*quaternion)
            self.cmd_pose_pub.publish(cmd_pose_msg)

            # Publish predicted path (from optimization result)
            predicted_path_msg = create_path_from_mpc_prediction(x_traj_opt[:self.nx, :]) # Only x,y,theta
            self.pred_path_pub.publish(predicted_path_msg)

            # Publish tree markers based on *latest measured* individual scores
            if hasattr(self, 'latest_trees_scores_individual') and self.latest_trees_scores_individual is not None:
                 tree_markers_msg = create_tree_markers(self.trees_pos, self.latest_trees_scores_individual)
                 self.tree_markers_pub.publish(tree_markers_msg)
            elif self.trees_pos is not None: # Publish neutral markers if no scores yet
                 tree_markers_msg = create_tree_markers(self.trees_pos, np.ones(self.trees_pos.shape[0]) * 0.5)
                 self.tree_markers_pub.publish(tree_markers_msg)


            # --- 6. Calculate Actual Error for GP Training ---
            measured_true_fov_score = self.latest_trees_scores_sum # Get sum score from callback
            calculated_error_gp = None
            predicted_fov_score_current = None

            if measured_true_fov_score is not None and success: # Only update GP if MPC succeeded and we have a score
                # Calculate the *predicted* score at the *current* state using the *estimated* light angle
                fov_pred_result = self.fov_func_instance(
                    drone_state=current_state_full_np[0:3], # State *before* moving
                    object_positions=self.trees_pos,
                    light_angle=light_theta_opt # Use angle estimated by MPC
                )
                predicted_fov_score_current = float(ca.sum1(fov_pred_result['weights']).full().item())

                # Error = Measured Score - Predicted Score (at current state)
                # Note: test_fov_mpc used True-Pred. Here using Measured-Pred seems more direct.
                # Let's stick to Measured - Predicted for the GP target
                calculated_error_gp = measured_true_fov_score - predicted_fov_score_current

                # Add to GP training data only if score is reasonably high (indicates informative view)
                # And if the error is reasonably bounded (avoid outliers)
                if measured_true_fov_score > 0.1 and abs(calculated_error_gp) < 1.0: # Heuristic thresholds
                    self.X_train_angles_list.append(light_theta_opt) # Store optimized angle
                    self.Y_train_error_list.append(calculated_error_gp) # Store associated error
                    # rospy.logdebug(f"Added GP point: Angle={light_theta_opt:.3f}, Error={calculated_error_gp:.4f}")
                # else:
                    # rospy.logdebug(f"Skipped GP point: Score={measured_true_fov_score:.3f}, Error={calculated_error_gp:.4f}")

            else:
                 # Cannot calculate error if no measurement or MPC failed
                 calculated_error_gp = None
                 predicted_fov_score_current = None
                 if measured_true_fov_score is None:
                     rospy.logwarn_throttle(5.0,"Waiting for tree scores...")


            # --- 7. Logging ---
            current_rostime = rospy.get_time()
            log_entry = {
                "Iteration": mpciter, "Timestamp": current_rostime - sim_start_time,
                "State_x": current_state_full_np[0], "State_y": current_state_full_np[1], "State_theta": current_state_full_np[2],
                "State_vx": current_state_full_np[3], "State_vy": current_state_full_np[4], "State_omega": current_state_full_np[5],
                "Control_ax": float(u_opt[0]), "Control_ay": float(u_opt[1]), "Control_alpha": float(u_opt[2]),
                "Est_Light_Angle": light_theta_opt if success else None,
                "Measured_FOV_Score": measured_true_fov_score,
                "Predicted_FOV_Score_Current": predicted_fov_score_current,
                "GP_Error_Pred_Optimized": gp_error_pred_opt if success else None,
                "Calculated_Error_GP": calculated_error_gp,
                "GP_Points": len(self.X_train_angles_list),
                "Solve_Time": solve_time,
                "Solver_Success": success
            }
            log_data.append(log_entry)

            if current_rostime - last_print_time > 2.0: # Print every 2 seconds
                 rospy.loginfo(f"Iter {mpciter}/{sim_time_steps}, State:({current_state_full_np[0]:.1f},{current_state_full_np[1]:.1f},{np.rad2deg(current_state_full_np[2]):.0f}deg), "
                               f"Est Light: {np.rad2deg(light_theta_opt):.0f}deg, Meas Score: {measured_true_fov_score:.2f}, "
                               f"Pred Score: {predicted_fov_score_current:.2f}, GP Pts: {len(self.X_train_angles_list)}, "
                               f"T: {solve_time:.3f}s, Success: {success}")
                 last_print_time = current_rostime


            mpciter += 1
            rate.sleep() # Maintain loop frequency

        # --- End of Loop ---
        rospy.loginfo("Simulation finished.")

        # --- Save Log Data ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_csv_path = os.path.join(log_dir, f"fov_mpc_{timestamp}_log.csv")

        try:
            with open(log_csv_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=log_headers)
                writer.writeheader()
                writer.writerows(log_data)
            rospy.loginfo(f"Log data saved to {log_csv_path}")
        except IOError as e:
            rospy.logerr(f"Failed to write log file: {e}")

        # Add final summary statistics if desired (e.g., average solve time, final angle error if TRUE_LIGHT known)


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    try:
        mpc_node = FovMPC()
        # Check if initialization failed (e.g., couldn't get tree poses)
        if mpc_node.trees_pos is not None:
             mpc_node.run_simulation()
        else:
             rospy.logerr("FOV MPC Node initialization failed. Shutting down.")
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in FovMPC: {e}", exc_info=True)
    finally:
        rospy.loginfo("FOV MPC node shutting down.")