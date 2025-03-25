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
import tf2_ros
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
# import l4casadi as l4c # Keep commented out unless needed

# ROS message imports
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path
import tf
import matplotlib.pyplot as plt

# Service import for tree poses
from nmpc_ros.srv import GetTreesPoses

# ---------------------------
# Helper Functions (Visualization) - Defined locally
# ---------------------------
def create_path_from_mpc_prediction(mpc_prediction_states, frame_id="map", dt=0.1):
    """
    Creates a Path message from MPC predicted states (x, y, theta).
    Assumes mpc_prediction_states is a NumPy array [>=3, N].
    """
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = frame_id

    for i in range(mpc_prediction_states.shape[1]):
        pose_stamped = PoseStamped()
        # Stagger timestamps slightly into the future
        pose_stamped.header.stamp = path_msg.header.stamp + rospy.Duration(i * dt)
        pose_stamped.header.frame_id = path_msg.header.frame_id
        pose_stamped.pose.position.x = float(mpc_prediction_states[0, i])
        pose_stamped.pose.position.y = float(mpc_prediction_states[1, i])
        pose_stamped.pose.position.z = 0.0 # Assuming 2D movement

        # Ensure theta is valid before converting
        theta = float(mpc_prediction_states[2, i])
        if not np.isfinite(theta):
            rospy.logwarn_throttle(5.0, f"Invalid theta ({theta}) in MPC prediction at step {i}. Using 0.")
            theta = 0.0

        try:
            quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
            pose_stamped.pose.orientation = Quaternion(*quaternion)
        except ValueError as e:
            rospy.logwarn_throttle(5.0, f"Error creating quaternion from theta={theta}: {e}. Using identity.")
            pose_stamped.pose.orientation.w = 1.0


        path_msg.poses.append(pose_stamped)

    return path_msg

def create_tree_markers(tree_positions, tree_scores, frame_id="map"):
    """
    Creates visualization markers for trees based on their positions and scores.
    Scores are assumed to be between 0 and 1, influencing color (Green=high, Red=low).
    """
    marker_array = MarkerArray()
    num_trees = tree_positions.shape[0]

    # Ensure scores are a flat numpy array
    if isinstance(tree_scores, (list, tuple)):
        scores_np = np.array(tree_scores).flatten()
    elif isinstance(tree_scores, np.ndarray):
        scores_np = tree_scores.flatten()
    else:
        rospy.logwarn_throttle(5.0, f"Unexpected type for tree_scores: {type(tree_scores)}. Using default.")
        scores_np = np.ones(num_trees) * 0.5 # Default to neutral

    if len(scores_np) != num_trees:
        rospy.logwarn_throttle(5.0, f"Mismatch between number of trees ({num_trees}) and scores ({len(scores_np)}). Using default scores.")
        scores_np = np.ones(num_trees) * 0.5 # Default to neutral

    for i in range(num_trees):
        marker = Marker()
        marker.header.frame_id = frame_id
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
    """
    Calculates the Euclidean distances between a drone's 2D position and multiple object positions using CasADi.
    """
    # Ensure objects_pos is treated as a matrix (N_obj x 2)
    if not isinstance(objects_pos, (ca.MX, ca.SX)) and objects_pos.shape[1] != 2:
         raise ValueError("objects_pos should have shape (N, 2)")
    n_objects = objects_pos.shape[0]

    # Ensure drone_pos_2d is a column vector (2x1) for broadcasting
    if hasattr(drone_pos_2d, 'shape') and drone_pos_2d.shape == (1, 2):
         drone_pos_2d = drone_pos_2d.T # Transpose to 2x1 if it's 1x2
    elif not (hasattr(drone_pos_2d, 'shape') and drone_pos_2d.shape == (2, 1)):
         # Attempt to reshape if it's a flat vector (e.g., from vertcat) or numpy array
         if hasattr(drone_pos_2d, 'numel') and drone_pos_2d.numel() == 2:
             drone_pos_2d = drone_pos_2d.reshape((2,1))
         elif isinstance(drone_pos_2d, (np.ndarray, list)) and len(drone_pos_2d) == 2:
             drone_pos_2d = np.array(drone_pos_2d).reshape(2,1)
         else:
            # If it's already a CasADi SX/MX scalar operation might handle it, otherwise raise error
            if not isinstance(drone_pos_2d, (ca.SX, ca.MX)) or not drone_pos_2d.is_symbolic():
                 raise ValueError(f"drone_pos_2d should have shape (2, 1) or (1, 2), got {type(drone_pos_2d)}")

    # Calculate differences: objects_pos (N_obj x 2) - drone_pos_2d.T (1 x 2)
    diff = objects_pos - ca.repmat(drone_pos_2d.T, n_objects, 1) # Result is (N_obj x 2)
    distances_sq = ca.sum2(diff**2)
    distances = ca.sqrt(distances_sq) # Result is (N_obj x 1)
    return distances

def norm_sigmoid_ca(x, thresh=0.5, delta=0.1, alpha=1.0):
    """Normalized sigmoid function using CasADi."""
    k = alpha / delta
    c = thresh
    return 1 / (1 + ca.exp(-k * (x - c)))

def gaussian_ca(x, mu=0.0, sig=1.0):
    """Gaussian function using CasADi."""
    return ca.exp(-((x - mu)**2) / (2 * sig**2))

def fov_weight_fun_casadi_var_light(example_objects_pos, thresh_distance=5):
    """
    Creates a CasADi function to calculate FOV weights based on drone state [x,y,theta],
    object positions, and light angle. Accepts symbolic inputs.
    """
    sig = 1.5
    thresh = 0.7
    delta = 0.1
    alpha = 1.0
    epsilon = 1e-9

    # Define symbolic inputs matching expected types
    drone_pos_sym = ca.MX.sym("drone_state", 3) # Expecting [x, y, theta]
    objects_pos_sym = ca.MX.sym("objects_pos", example_objects_pos.shape[0], example_objects_pos.shape[1])
    light_angle_sym = ca.MX.sym("light_angle") # Scalar

    drone_xy_sym = drone_pos_sym[0:2]
    theta = drone_pos_sym[2]

    distances = drone_objects_distances_casadi(drone_xy_sym, objects_pos_sym)
    drone_view_dir_sym = ca.vertcat(ca.cos(theta), ca.sin(theta)) # Shape (2,1)

    object_directions = objects_pos_sym - ca.repmat(drone_xy_sym.T, objects_pos_sym.shape[0], 1) # Shape (N_obj, 2)
    object_dir_norms = ca.sqrt(ca.sum2(object_directions**2)) # Shape (N_obj, 1)

    # Ensure norms are not zero before division
    safe_object_dir_norms = object_dir_norms + epsilon * (object_dir_norms < epsilon)
    norm_object_directions_matrix = object_directions / ca.repmat(safe_object_dir_norms, 1, 2) # Shape (N_obj, 2)

    # Alignment score calculation
    vect_alignment = norm_object_directions_matrix @ drone_view_dir_sym # (N_obj, 2) @ (2, 1) -> (N_obj, 1)
    alignment_input = ((vect_alignment + 1) / 2) ** 2
    alignment_score = norm_sigmoid_ca(alignment_input, thresh=thresh, delta=delta, alpha=alpha)

    # Distance score calculation
    distance_score = gaussian_ca(distances, mu=thresh_distance, sig=sig) # Shape (N_obj, 1)

    # Light score calculation
    light_dir_sym = ca.vertcat(ca.cos(light_angle_sym), ca.sin(light_angle_sym)) # Shape (2,1)
    # Dot product between light direction and drone view direction
    vect_light_alignment_sym = ca.dot(light_dir_sym, drone_view_dir_sym) # (1,2) @ (2,1) -> scalar
    light_score_input = ((vect_light_alignment_sym + 1) / 2) ** 2
    light_score_sym = norm_sigmoid_ca(light_score_input, thresh=thresh, delta=delta, alpha=alpha) # Scalar

    # Combine scores: Element-wise product + broadcasted light score
    result = alignment_score * distance_score * light_score_sym # (N_obj, 1) * (N_obj, 1) * scalar -> (N_obj, 1)

    # Return a CasADi function
    return ca.Function('fov_function_var_light',
                       [drone_pos_sym, objects_pos_sym, light_angle_sym],
                       [result],
                       ['drone_state', 'object_positions', 'light_angle'],
                       ['weights'])

def rbf_kernel_casadi_2d(x1_2d, x2_matrix_2d, length_scale, constant_value):
    """CasADi implementation of scaled RBF kernel vector k(x1, X_train) for 2D inputs.
       Handles isotropic (scalar length_scale) and anisotropic (vector length_scale).
    """
    n_data = x2_matrix_2d.shape[0]
    n_dim = x2_matrix_2d.shape[1] # Should be 2 for this method

    if n_dim != 2:
        raise ValueError("rbf_kernel_casadi_2d expects 2D inputs (shape N x 2)")

    # Ensure x1_2d is a row vector (1x2) for broadcasting/repmat
    if hasattr(x1_2d, 'shape') and x1_2d.shape == (2, 1):
         x1_2d = x1_2d.T # Transpose to 1x2 if it's 2x1
    elif not (hasattr(x1_2d, 'shape') and x1_2d.shape == (1, 2)):
         # Attempt reshape if flat (e.g., from vertcat)
         if hasattr(x1_2d, 'numel') and x1_2d.numel() == n_dim:
             x1_2d = x1_2d.reshape((1,n_dim))
         else:
              # Check if symbolic allows broadcast
              if not isinstance(x1_2d, (ca.SX, ca.MX)) or not x1_2d.is_symbolic():
                   raise ValueError(f"x1_2d should have shape (1, {n_dim}) or ({n_dim}, 1), got {type(x1_2d)} shape {getattr(x1_2d,'shape','N/A')}")
              # Assume symbolic is ok if shape check fails (Casadi might handle it)


    # Calculate differences: x2_matrix_2d (N x 2) - x1_2d (1 x 2)
    diff = x2_matrix_2d - ca.repmat(x1_2d, n_data, 1) # Result is (N x 2)

    # Calculate squared Euclidean distance based on length_scale type
    if isinstance(length_scale, (int, float, ca.DM)) and (not hasattr(length_scale, 'numel') or length_scale.numel() == 1) : # Isotropic
        sq_dist = ca.sum2(diff**2) # Sum squares across columns -> (N x 1)
        k_values = constant_value * ca.exp(-0.5 * sq_dist / length_scale**2)
    elif isinstance(length_scale, (list, np.ndarray, ca.DM)) and ca.DM(length_scale).numel() == n_dim: # Anisotropic
         # Ensure length_scale is casadi DM type for calculation
         ls_ca = ca.DM(length_scale).T # Shape (1, n_dim)
         if ls_ca.shape[1] != n_dim: # Check after potential transpose
              ls_ca = ls_ca.T
         scaled_diff_sq = (diff / ca.repmat(ls_ca, n_data, 1))**2 # Element-wise scaled diff sq (N x n_dim)
         sq_dist_sum = ca.sum1(scaled_diff_sq) # Sum across dims -> (N x 1)
         k_values = constant_value * ca.exp(-0.5 * sq_dist_sum)
    else:
         raise TypeError(f"Unsupported length_scale type/shape: {type(length_scale)} value: {length_scale} for {n_dim} dims")

    return k_values # Returns a column vector (N x 1)

# ---------------------------
# FOV MPC Class (Integrated)
# ---------------------------
class FovMPC:
    def __init__(self):
        # --- ROS Initialization ---
        rospy.init_node("fov_mpc_node", anonymous=True, log_level=rospy.INFO)

        # --- Parameters ---
        # MPC Timing & Horizon
        self.N = rospy.get_param("~prediction_horizon", 10)      # Prediction horizon steps
        self.dt = rospy.get_param("~time_step", 0.25)            # Time step [s]
        self.T = self.dt * self.N                                # Total prediction time

        # State/Control Dimensions
        self.nx_base = 3 # Base state dim [x, y, theta]
        self.n_states = self.nx_base * 2 # Full state dim [x,y,th,vx,vy,om]
        self.n_controls = self.nx_base   # Control dim [ax,ay,alpha]

        # Constraints
        self.MAX_ACCEL = rospy.get_param("~max_linear_accel", 4.0) # m/s^2
        self.MAX_ANG_ACCEL = rospy.get_param("~max_angular_accel", np.pi / 2) # rad/s^2
        self.MAX_VEL = rospy.get_param("~max_linear_vel", 2.0) # m/s
        self.MAX_ANG_VEL = rospy.get_param("~max_angular_vel", np.pi / 4) # rad/s
        self.safe_distance = rospy.get_param("~collision_safe_distance", 1.0) # Safety margin (Currently unused in cost)

        # Cost Function Weights
        self.W_FOV = rospy.get_param("~weight_fov", 100.0)           # Weight for maximizing FOV score (minimize 1-score)
        self.W_GP_ERROR = rospy.get_param("~weight_gp_error", 100.0) # Weight for minimizing predicted GP error (squared)
        self.R_ACCEL = rospy.get_param("~weight_control_linear", 0.1) # Control effort penalty (linear accel)
        self.R_ANG_ACCEL = rospy.get_param("~weight_control_angular", 0.05) # Control effort penalty (angular accel)

        # GP Parameters
        self.MEASUREMENT_NOISE_STD = rospy.get_param("~score_measurement_noise_std", 0.05) # Estimated noise in score measurement
        self.GP_TRAIN_MIN_SCORE_THRESH = rospy.get_param("~gp_train_min_score_thresh", 0.1) # Min measured score to add GP point
        self.GP_TRAIN_MAX_ERROR_THRESH = rospy.get_param("~gp_train_max_error_thresh", 1.0) # Max abs error to add GP point
        self.GP_N_RESTARTS = rospy.get_param("~gp_n_restarts", 5) # Restarts for GP hyperparameter optimization

        # Simulation/Run Parameters
        self.sim_time_steps = rospy.get_param("~max_iterations", 200) # Number of MPC steps to run
        self.log_data_flag = rospy.get_param("~log_data", True)
        self.log_dir = rospy.get_param("~log_directory", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

        # ROS Frame IDs
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "drone_base_link") # Frame for TF lookup

        # --- GP-related state ---
        self.X_train_angles_list = [] # Store optimized 1D light angles
        self.Y_train_error_list = []  # Store calculated errors (Measured Score - Predicted Score)
        self.gp = None
        self.gp_params_available = False
        self.alpha_ = None # GP internal parameter for CasADi prediction
        # Default GP kernel params (will be updated by GP fit)
        self.opt_length_scale_ca = ca.DM(0.5)
        self.opt_constant_value_ca = ca.DM(0.1)
        # Define the kernel structure (hyperparameters optimized during fit)
        self.kernel_2d = ConstantKernel(0.1, (1e-4, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 5e1)) \
                         + WhiteKernel(noise_level=self.MEASUREMENT_NOISE_STD**2,
                                       noise_level_bounds=(1e-7, 1e-1))

        # --- Internal state ---
        self.latest_trees_scores_sum = None # Store SUM of scores from callback
        self.latest_trees_scores_individual = None # Store individual scores for markers
        self.current_state_3d = None # [x, y, yaw] from TF
        self.current_state_full = None # [x, y, yaw, vx, vy, omega] - 6D state
        self.current_velocity = ca.DM.zeros(self.nx_base) # [vx, vy, omega] - Estimated velocity
        self.light_theta_guess = 0.0 # Initial guess for light angle (1D)
        self.opti_solver_success = True # Track solver status for warm start
        self.U_guess = np.zeros((self.n_controls, self.N)) # Control guess
        self.X_guess = None # State guess (initialized later)
        self.trees_pos = None # Loaded from service
        self.num_total_trees = 0

        # --- ROS Communication Setup ---
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0)) # Increased buffer duration
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        rospy.Subscriber("agent_0/tree_scores", Float32MultiArray, self.tree_scores_callback, queue_size=1) # Smaller queue

        # Publishers
        self.cmd_pose_pub = rospy.Publisher("agent_0/cmd/pose", Pose, queue_size=5) # Publish PoseStamped
        self.pred_path_pub = rospy.Publisher("agent_0/predicted_path", Path, queue_size=5)
        self.tree_markers_pub = rospy.Publisher("agent_0/tree_markers", MarkerArray, queue_size=5)
        self.est_light_pub = rospy.Publisher("agent_0/estimated_light_angle", Float32MultiArray, queue_size=5) # Publish estimated angle

        # --- Initialization ---
        # Get Tree Positions
        self.trees_pos = self.get_trees_poses()
        if self.trees_pos is None or self.trees_pos.shape[0] == 0:
             rospy.logfatal("Failed to get tree positions or received empty list. Shutting down.")
             rospy.signal_shutdown("Failed to get tree positions")
             # Exit __init__ early if critical info is missing
             # Set a flag or check for self.trees_pos later before running
             self.initialization_failed = True
             return
        else:
            self.initialization_failed = False

        self.num_total_trees = self.trees_pos.shape[0]
        rospy.loginfo(f"Received {self.num_total_trees} tree positions.")

        # Initialize FOV function instance now that we have tree positions
        self.fov_func_instance = fov_weight_fun_casadi_var_light(self.trees_pos)

        # Kinematic model function (using 2nd order model)
        self.F_dyn = self.kin_model_dt(self.dt)

        # Wait for the first state update
        rospy.loginfo("Waiting for initial robot state from TF...")
        rate = rospy.Rate(10) # Check for TF at 10 Hz initially
        while self.current_state_3d is None and not rospy.is_shutdown():
            self.update_robot_state_from_tf()
            rate.sleep()

        if rospy.is_shutdown():
            self.initialization_failed = True
            return # Exit if ROS shuts down during init

        if self.current_state_3d is None:
            rospy.logfatal("Failed to get initial robot state from TF. Shutting down.")
            rospy.signal_shutdown("Failed to get initial robot state")
            self.initialization_failed = True
            return
        else:
             # Initialize the full 6D state and the state guess for MPC
             self.current_state_full = ca.vertcat(ca.DM(self.current_state_3d), self.current_velocity)
             self.X_guess = np.tile(self.current_state_full.full().flatten(), (self.N + 1, 1)).T
             rospy.loginfo(f"Initial robot state received: {self.current_state_full.full().flatten()}")


    # ---------------------------
    # Callback Functions
    # ---------------------------
    def update_robot_state_from_tf(self):
        """Continuously update the robot's 3D state [x, y, yaw] using TF."""
        try:
            # Use rospy.Time(0) to get the latest available transform
            trans = self.tf_buffer.lookup_transform(self.map_frame, self.robot_base_frame, rospy.Time(0), rospy.Duration(0.1)) # Short timeout
            # Extract the yaw angle from the quaternion
            (_, _, yaw) = tf.transformations.euler_from_quaternion([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            # Update current_state with [x, y, yaw]
            self.current_state_3d = [trans.transform.translation.x,
                                     trans.transform.translation.y,
                                     yaw]
            # rospy.logdebug(f"TF Update: x={self.current_state_3d[0]:.2f}, y={self.current_state_3d[1]:.2f}, yaw={self.current_state_3d[2]:.2f}")
            return True # Indicate success
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"Failed to get transform from {self.map_frame} to {self.robot_base_frame}: {e}")
            self.current_state_3d = None # Invalidate state if TF fails
            return False # Indicate failure

    def tree_scores_callback(self, msg):
        """
        Callback for tree scores. Stores sum and individual scores.
        """
        scores = np.array(msg.data)
        if len(scores) == self.num_total_trees:
            self.latest_trees_scores_sum = float(np.sum(scores))
            self.latest_trees_scores_individual = scores
            # rospy.logdebug(f"Received {len(scores)} tree scores. Sum: {self.latest_trees_scores_sum:.3f}")
        else:
             rospy.logwarn_throttle(5.0, f"Received scores length ({len(scores)}) does not match tree count ({self.num_total_trees})")
             self.latest_trees_scores_sum = None # Invalidate score if mismatch
             self.latest_trees_scores_individual = None


    # ---------------------------
    # Service Call to Get Trees Poses
    # ---------------------------
    def get_trees_poses(self):
        service_name = "/obj_pose_srv"
        rospy.loginfo(f"Waiting for {service_name} service...")
        try:
            rospy.wait_for_service(service_name, timeout=15.0)
        except rospy.ROSException:
            rospy.logerr(f"Service {service_name} not available after 15 seconds.")
            return None
        try:
            trees_srv = rospy.ServiceProxy(service_name, GetTreesPoses)
            response = trees_srv()
            # Ensure poses exist before accessing
            if not response.trees_poses or not response.trees_poses.poses:
                 rospy.logwarn("Received empty tree poses from service.")
                 return np.array([]) # Return empty array, not None

            trees_pos = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
            if trees_pos.size == 0:
                 rospy.logwarn("Processed tree poses resulted in an empty array.")
                 return np.array([])
            return trees_pos
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
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
        padding = 2.0 # Add some padding
        return [x_min - padding, y_min - padding], [x_max + padding, y_max + padding]

    @staticmethod
    def kin_model_dt(dt):
        """ 2nd order Kinematic model using Euler integration (CasADi Function) """
        n_state = 6 # [x, y, th, vx, vy, om]
        n_control = 3   # [ax, ay, alpha]
        X = ca.MX.sym('X', n_state)
        U = ca.MX.sym('U', n_control)
        # State: x, y, theta, vx, vy, omega
        # Control: ax, ay, alpha
        x, y, theta, vx, vy, omega = X[0], X[1], X[2], X[3], X[4], X[5]
        ax, ay, alpha = U[0], U[1], U[2]
        rhs = ca.vertcat(
            x + vx * dt,
            y + vy * dt,
            theta + omega * dt, # Simple Euler integration for angle
            vx + ax * dt,
            vy + ay * dt,
            omega + alpha * dt
        )
        # Wrap theta: atan2(sin(theta_new), cos(theta_new)) - More robust than modulo
        # Note: Applying this within the dynamics can sometimes complicate optimization.
        # It's often better to apply wrapping *after* the dynamics step if needed,
        # or rely on the cost function/constraints if angle matters directly.
        # Let's keep it simple Euler for now and wrap outside if necessary.
        # rhs[2] = ca.atan2(ca.sin(rhs[2]), ca.cos(rhs[2])) # Optional: wrap inside dynamics

        return ca.Function('F_kin_dt', [X, U], [rhs], ['state', 'control'], ['next_state'])

    # ---------------------------
    # GP Update Function
    # ---------------------------
    def _update_gp_model(self):
        """Trains the GP model using the collected data."""
        self.gp = None
        self.gp_params_available = False
        self.alpha_ = None
        fitted_params = {}

        if len(self.X_train_angles_list) < 1: # Need at least one point
            # rospy.logdebug("Not enough data to train GP yet.")
            return fitted_params # Return empty dict

        # Prepare training data (convert angles to 2D for GP input)
        X_train_angles = np.array(self.X_train_angles_list)
        # Ensure angles are wrapped to [-pi, pi] before converting to 2D
        X_train_angles_wrapped = np.arctan2(np.sin(X_train_angles), np.cos(X_train_angles))
        X_train_gp_2d = np.stack([np.cos(X_train_angles_wrapped), np.sin(X_train_angles_wrapped)], axis=-1)
        Y_train_gp_error = np.array(self.Y_train_error_list)

        # Create and fit the GP model
        self.gp = GaussianProcessRegressor(kernel=self.kernel_2d,
                                           n_restarts_optimizer=self.GP_N_RESTARTS if len(self.X_train_angles_list) > 5 else 1, # Fewer restarts initially
                                           alpha=self.MEASUREMENT_NOISE_STD**2, # Prior noise variance
                                           normalize_y=False, # Usually False for error modeling unless errors vary wildly
                                           random_state=int(time.time())) # Add some randomness

        try:
            self.gp.fit(X_train_gp_2d, Y_train_gp_error)
            rospy.logdebug(f"GP (Error Model, 2D Input) trained with {len(self.X_train_angles_list)} points. LML: {self.gp.log_marginal_likelihood_value_:.3f}")

            # Store internal parameters needed for CasADi prediction
            self.alpha_ = self.gp.alpha_ # Weights for prediction

            # Extract optimized hyperparameters for CasADi kernel function
            try:
                params = self.gp.kernel_.get_params()
                self.opt_constant_value_ca = ca.DM(params['k1__k1__constant_value'])
                self.opt_length_scale_ca = ca.DM(params['k1__k2__length_scale']) # Can be scalar or vector
                fitted_noise = params.get('k2__noise_level', 'N/A')
                self.gp_params_available = True
                rospy.logdebug(f"  GP Params: Scale={self.opt_constant_value_ca}, Len={self.opt_length_scale_ca}, Noise={fitted_noise:.4g}")
                fitted_params = {
                    'constant_value': float(self.opt_constant_value_ca.full().item()),
                    'length_scale': self.opt_length_scale_ca.full().flatten().tolist(), # Store as list (can be scalar or vector)
                    'noise_level': fitted_noise
                }
            except Exception as e:
                 rospy.logwarn(f"GP Parameter extraction failed: {e}. Using defaults for CasADi prediction.")
                 # Keep default self.opt_... values, but mark params as unavailable for CasADi use
                 self.gp_params_available = False
                 self.alpha_ = None

        except Exception as e:
            rospy.logerr(f"GP training failed: {e}")
            self.gp = None
            self.gp_params_available = False
            self.alpha_ = None

        return fitted_params # Return extracted params or empty dict


    # ---------------------------
    # Main Execution Loop
    # ---------------------------
    def run_simulation(self):
        """ Main ROS execution loop """

        # Randomize true light angle in [-pi, pi]
        TRUE_LIGHT_ANGLE = np.random.uniform(-np.pi, np.pi)
        # === MPC Parameters ===
        DT = 0.25  # Time step [s]
        N_HORIZON = 10  # Prediction horizon
        MAX_ITER = 50 # Number of MPC iterations (simulation steps)
        MEASUREMENT_NOISE_STD = 0.005
        PLOT_UPDATE_INTERVAL = 0.1 # Faster plotting

        # Control constraints
        MAX_ACCEL = 4.0 # m/s^2
        MAX_ANG_ACCEL = np.pi / 2 # rad/s^2
        # State constraints (example)
        MAX_VEL = 2.0 # m/s
        MAX_ANG_VEL = np.pi/4 # rad/s

        # === Cost Function Weights ===
        W_FOV = 100.0      # Weight for maximizing FOV score (negative cost)
        W_GP_ERROR = 100.0  # Weight for minimizing predicted GP error (squared)
        R_ACCEL = 0.1     # Weight for penalizing linear acceleration
        R_ANG_ACCEL = 0.05 # Weight for penalizing angular acceleration

        # Create dynamics and FOV function instances
        dynamics_func = self.F_dyn
        fov_func_instance = fov_weight_fun_casadi_var_light(self.trees_pos)

        # --- GP Setup (Method 1: 2D Input) ---
        kernel_2d = ConstantKernel(0.1, (1e-4, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 5e1)) \
                + WhiteKernel(noise_level=MEASUREMENT_NOISE_STD**2,
                            noise_level_bounds=(1e-7, 1e-1))

        # --- Initial Training Data (for GP Error Model) ---
        X_train_angles_list = [] # Store estimated 1D light angles from optimization
        Y_train_error_list = [] # Store calculated errors (True Score - Predicted Score)

        print(f"Using Method 1: Mapping angles to 2D [cos(theta), sin(theta)] for GP input.")
        print(f"True Light Angle: {TRUE_LIGHT_ANGLE:.3f} rad ({np.rad2deg(TRUE_LIGHT_ANGLE):.1f} deg)")

        # --- History Storage ---
        self.update_robot_state_from_tf()
        initial_drone_state_full = np.array(
            self.current_state_3d+[ 
            0.0, # vx
            0.0, # vy
            0.0  # omega
        ])
        history_drone_states = [self.current_state_3d + [0.0,0.0,0.0]] # Store full 6D state
        history_controls = [] # Store applied controls [ax, ay, alpha]
        history_predicted_trajectories = [] # Store the X prediction from opti
        history_light_angles_est = [] # Store optimized 1D light angles
        history_true_fov_scores = []
        history_predicted_fov_scores_at_current = [] # Predicted score at the current state using est. angle
        history_gp_error_preds = [] # Store GP's prediction of error at the optimized angle
        history_objective_vals = []
        history_calculated_errors = [] # Actual error measured at each step

        # --- Setup Interactive Plotting ---
        plt.ion()
        fig_live, axes_live = plt.subplots(1, 2, figsize=(18, 8)) # Wider figure
        ax_traj = axes_live[0]
        ax_gp = axes_live[1]
        fig_live.suptitle("Live MPC Progress (GP learns Error, 2D Input)", fontsize=16)

        # Pre-calculate angles for GP plotting (still plot against 1D angle)
        plot_angles_gp_1d = np.linspace(-np.pi - 0.5, np.pi + 0.5, 200).reshape(-1, 1)
        # Convert plot angles to 2D for GP prediction
        plot_angles_gp_2d = np.hstack([np.cos(plot_angles_gp_1d), np.sin(plot_angles_gp_1d)])

        # --- Iterative MPC Loop ---
        current_drone_state = initial_drone_state_full.copy()
        current_light_angle_guess = 0.0 # Initial guess for 1D light angle
        gp = None
        gp_params_available = False
        alpha_ = None
        opti_solver_success = True # Track solver status

        # Store initial guesses for warm starting
        U_guess = np.zeros((self.n_controls, N_HORIZON))
        X_guess = np.tile(current_drone_state, (N_HORIZON + 1, 1)).T
        light_theta_guess = current_light_angle_guess

        for iter_num in range(MAX_ITER):
            start_time = time.time()
            print(f"\n--- MPC Iteration {iter_num + 1}/{MAX_ITER} ---")
            print(f"Current State: x={current_drone_state[0]:.2f}, y={current_drone_state[1]:.2f}, th={current_drone_state[2]:.2f}, "
                f"vx={current_drone_state[3]:.2f}, vy={current_drone_state[4]:.2f}, w={current_drone_state[5]:.2f}")

            # --- 1. Update/Train GP (on Error Data using 2D Input) ---
            alpha_ = None # Reset alpha_
            constant_value = 0.1 # Default
            length_scale = 0.5 # Default (might become array if anisotropic)
            fitted_noise_level = 'Not Optimized' # Default

            if len(X_train_angles_list) >= 1:
                X_train_angles = np.array(X_train_angles_list)
                X_train_gp_2d = np.stack([np.cos(X_train_angles), np.sin(X_train_angles)], axis=-1)
                Y_train_gp_error = np.array(Y_train_error_list)

                gp = GaussianProcessRegressor(kernel=kernel_2d,
                                            n_restarts_optimizer=5 if iter_num > 2 else 1, # Fewer restarts initially
                                            alpha=MEASUREMENT_NOISE_STD**2,
                                            normalize_y=False)
                try:
                    gp.fit(X_train_gp_2d, Y_train_gp_error)
                    print(f"GP (Error Model, 2D Input) trained with {len(X_train_angles_list)} points. LML: {gp.log_marginal_likelihood_value_:.3f}")
                    alpha_ = gp.alpha_
                    try:
                        params = gp.kernel_.get_params()
                        constant_value = params['k1__k1__constant_value']
                        length_scale = params['k1__k2__length_scale']
                        fitted_noise_level = params.get('k2__noise_level', 'Not Optimized')
                        gp_params_available = True
                        print(f"  GP Params (2D): Scale={constant_value:.4f}, Len={length_scale}, Noise={fitted_noise_level:.4g}")
                    except Exception as e:
                        print(f"Warning: GP Parameter extraction failed: {e}. Using defaults.")
                        gp_params_available = False
                        alpha_ = None
                except Exception as e:
                    print(f"GP training failed: {e}. Skipping GP error term in objective.")
                    gp = None
                    gp_params_available = False
                    alpha_ = None
            else:
                print("Not enough data to train GP yet.")
                gp = None
                gp_params_available = False
                alpha_ = None

            # --- 2. Setup CasADi Optimization (MPC) ---
            opti = ca.Opti()

            # Decision variables
            X = opti.variable(self.n_states, N_HORIZON + 1) # State trajectory (includes initial state)
            U = opti.variable(self.n_controls, N_HORIZON)   # Control sequence
            light_theta = opti.variable()             # Estimated light angle (single value for the horizon)

            # Parameters
            X0_param = opti.parameter(self.n_states)          # Initial state
            objects_pos_param = opti.parameter(self.trees_pos.shape[0], self.trees_pos.shape[1]) # Object positions

            # Objective function
            objective = 0
            # Term 2: Predicted GP *error* (constant over horizon, depends only on light_theta)
            gp_error_pred_sym = ca.DM(0.0)
            if gp_params_available and alpha_ is not None and len(X_train_angles_list) > 0:
                X_train_ca_2d = ca.DM(X_train_gp_2d)
                alpha_ca = ca.DM(alpha_)
                opt_length_scale_ca = ca.DM(length_scale)
                opt_constant_value_ca = ca.DM(constant_value)
                light_theta_2d_sym = ca.horzcat(ca.cos(light_theta), ca.sin(light_theta))
                k_star_sym = rbf_kernel_casadi_2d(light_theta_2d_sym, X_train_ca_2d,
                                                opt_length_scale_ca, opt_constant_value_ca)
                gp_error_pred_sym = ca.dot(k_star_sym, alpha_ca)
                objective += W_GP_ERROR * (gp_error_pred_sym**2) # Penalize squared predicted error
            else:
                gp_error_pred_sym = ca.DM(0.0) # Set to zero if GP not available

            # Loop over the prediction horizon (stage costs)
            for k in range(N_HORIZON):
                state_k = X[:, k]
                control_k = U[:, k]
                state_k_pos_orient = state_k[0:3] # Extract [x, y, theta] for FOV

                # Term 1: FOV score (want to maximize -> minimize negative score or (1-score)^2)
                # Using (1-score)^2 encourages reaching high scores (close to 1)
                fov_results_sym = fov_func_instance(drone_state=state_k_pos_orient,
                                                    object_positions=objects_pos_param,
                                                    light_angle=light_theta) # Use the estimated angle
                fov_score_k_sym = ca.sum1(fov_results_sym['weights'])
                objective += DT * W_FOV * (1 - fov_score_k_sym)**2 # Minimize deviation from max score

                # Term 3: Control effort penalty
                objective += DT * (R_ACCEL * ca.sumsqr(control_k[0:2]) + R_ANG_ACCEL * ca.sumsqr(control_k[2]))

            opti.minimize(objective)

            # --- Constraints ---
            # Initial state constraint
            opti.subject_to(X[:, 0] == X0_param)

            # Dynamics constraints (multiple shooting)
            for k in range(N_HORIZON):
                opti.subject_to(X[:, k+1] == dynamics_func(X[:, k], U[:, k]))

            # State constraints (example: velocity limits)
            opti.subject_to(opti.bounded(-MAX_VEL, X[3, :], MAX_VEL)) # vx bounds
            opti.subject_to(opti.bounded(-MAX_VEL, X[4, :], MAX_VEL)) # vy bounds
            opti.subject_to(opti.bounded(-MAX_ANG_VEL, X[5, :], MAX_ANG_VEL)) # omega bounds
            # Add position bounds if needed: opti.subject_to(opti.bounded(-20, X[0:2, :], 20))

            # Control constraints
            opti.subject_to(opti.bounded(-MAX_ACCEL, U[0:2, :], MAX_ACCEL)) # ax, ay bounds
            opti.subject_to(opti.bounded(-MAX_ANG_ACCEL, U[2, :], MAX_ANG_ACCEL)) # alpha bounds

            # Light angle constraint (keep within [-pi, pi]) - important!
            opti.subject_to(opti.bounded(-np.pi +light_theta_guess, light_theta, np.pi+light_theta_guess))


            # --- Set Parameter Values and Initial Guesses ---
            opti.set_value(X0_param, current_drone_state)
            opti.set_value(objects_pos_param, self.trees_pos)

            # Warm starting: Use previous solution shifted
            if opti_solver_success: # Only use if previous solve was successful
                X_guess[:, :-1] = X_guess[:, 1:] # Shift state guess
                X_guess[:, -1] = X_guess[:, -2] # Repeat last state
                U_guess[:, :-1] = U_guess[:, 1:] # Shift control guess
                U_guess[:, -1] = U_guess[:, -2] # Repeat last control (or set to zero)
            else: # If previous failed, reset guess
                X_guess = np.tile(current_drone_state, (N_HORIZON + 1, 1)).T
                U_guess = np.zeros((self.n_controls, N_HORIZON))
                # Keep light_theta_guess from previous iteration or reset

            opti.set_initial(X, X_guess)
            opti.set_initial(U, U_guess)
            opti.set_initial(light_theta, light_theta_guess) # Use previous estimate


            # --- Solve the OCP ---
            ipopt_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 150}
            opti.solver('ipopt', ipopt_opts)

            optimization_successful_this_iter = False
            try:
                sol = opti.solve()
                # Extract results
                X_opt = sol.value(X)
                U_opt = sol.value(U)
                opt_light_theta = sol.value(light_theta)
                final_objective = sol.value(objective)
                final_gp_error_pred_opt = sol.value(gp_error_pred_sym) # GP error prediction at opt angle

                # Get the first control input to apply
                u0 = U_opt[:, 0]

                # Compute the command pose.
                cmd_pose = self.F_dyn(current_drone_state, u0)

                # Publish predicted path.
                predicted_path_msg = create_path_from_mpc_prediction(X_opt[:self.nx_base, 1:])
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
                print(f"MPC Optimization successful. Est. Light Angle: {opt_light_theta:.3f} rad ({np.rad2deg(opt_light_theta):.1f} deg)")
                print(f"  Applied Control: ax={u0[0]:.2f}, ay={u0[1]:.2f}, alpha={u0[2]:.2f}")
                print(f"  Predicted GP Error (at est. angle): {final_gp_error_pred_opt:.4f}")

                # Update guesses for next iteration (warm start)
                X_guess = X_opt
                U_guess = U_opt
                light_theta_guess = ca.atan2(ca.sin(opt_light_theta), ca.cos(opt_light_theta)) # Update 1D guess
                optimization_successful_this_iter = True
                opti_solver_success = True

            except RuntimeError as e:
                print(f"MPC Optimization failed: {e}")
                # Handle failure: Apply zero control or previous control? Let's apply zero.
                u0 = np.zeros(self.n_controls)
                X_opt = np.tile(current_drone_state, (N_HORIZON + 1, 1)).T # Placeholder prediction
                opt_light_theta = light_theta_guess # Keep previous guess
                final_objective = np.nan
                final_gp_error_pred_opt = np.nan
                opti_solver_success = False
                # Do not update guesses X_guess, U_guess if failed

            # --- 3. Simulate System (Apply First Control) ---
            next_state_true = dynamics_func(current_drone_state, u0).full().flatten()
            # Wrap angle theta to [-pi, pi]
            next_state_true[2] = np.arctan2(np.sin(next_state_true[2]), np.cos(next_state_true[2]))
            # Update current state for the next iteration
            current_drone_state = next_state_true


            # --- 4. Calculate Actual Error (using True Light Angle) for NEXT GP training ---
            # Get "true" score + noise at the NEW state, true angle
            true_fov_result = fov_func_instance(
                drone_state=current_drone_state[0:3], # Use [x,y,theta] from new state
                object_positions=self.trees_pos,
                light_angle=TRUE_LIGHT_ANGLE
            )
            true_fov_score_raw = ca.sum1(true_fov_result['weights']).full().item()
            measured_true_fov_score = max(0, true_fov_score_raw + np.random.normal(0, MEASUREMENT_NOISE_STD))

            # Get predicted score at NEW state, estimated angle from this iteration's optimization
            predicted_fov_result = fov_func_instance(
                drone_state=current_drone_state[0:3], # Use [x,y,theta] from new state
                object_positions=self.trees_pos,
                light_angle=opt_light_theta # Use the estimated 1D angle from this step
            )
            predicted_fov_score = ca.sum1(predicted_fov_result['weights']).full().item()

            # Calculate error: Measured True - Predicted (Using score at the *new* state)
            current_error = - ca.fabs(predicted_fov_score - measured_true_fov_score) # CHANGED to True - Pred for GP target
            print(f"Measured Score (True Light): {measured_true_fov_score:.4f}")
            print(f"Predicted Score (Est. Light): {predicted_fov_score:.4f}")
            print(f"Calculated Error for GP Training (True - Pred): {current_error:.4f}")


            # --- 5. Update GP Training Data ---
            if optimization_successful_this_iter and  predicted_fov_score >= 0.9:
                # Store the 1D estimated angle from this step and the corresponding error
                X_train_angles_list.append(opt_light_theta)
                Y_train_error_list.append(current_error) # Store True - Pred
                print(f"Added GP training point: Angle={opt_light_theta:.3f}, Error={current_error:.4f}")
            else:
                print("Skipping GP data update due to optimization failure.")


            # --- 6. Store History ---
            history_drone_states.append(current_drone_state.copy())
            history_controls.append(u0.copy())
            history_predicted_trajectories.append(X_opt) # Store the full predicted state trajectory
            history_light_angles_est.append(opt_light_theta)
            history_true_fov_scores.append(measured_true_fov_score)
            history_predicted_fov_scores_at_current.append(predicted_fov_score)
            history_gp_error_preds.append(final_gp_error_pred_opt)
            history_objective_vals.append(final_objective)
            history_calculated_errors.append(current_error) # Store True - Pred error


            # --- 7. Update Live Plots ---
            # Trajectory Plot
            ax_traj.cla()
            current_history_states_plot = np.array(history_drone_states) # shape (iter+2, 6)
            ax_traj.plot(current_history_states_plot[:, 0], current_history_states_plot[:, 1], 'o-', label='Drone Path', markersize=4, linewidth=1, color='tab:blue', zorder=2)
            ax_traj.scatter(self.trees_pos[:, 0], self.trees_pos[:, 1], color='red', marker='s', s=80, label='Objects', zorder=1)
            ax_traj.plot(current_history_states_plot[0, 0], current_history_states_plot[0, 1], 'go', markersize=8, label='Start', zorder=3)
            ax_traj.plot(current_history_states_plot[-1, 0], current_history_states_plot[-1, 1], 'mo', markersize=8, label='Current', zorder=3)

            # Plot predicted trajectory from this step
            if optimization_successful_this_iter:
                ax_traj.plot(X_opt[0, :], X_opt[1, :], 'x--', color='purple', alpha=0.6, linewidth=1.5, markersize=4, label=f'Predicted Traj (k={iter_num+1})', zorder=2)

            arrow_len = 1.0 # Adjusted arrow length
            ax_traj.quiver(current_drone_state[0], current_drone_state[1],
                        arrow_len * np.cos(current_drone_state[2]), arrow_len * np.sin(current_drone_state[2]),
                        color='magenta', scale_units='xy', scale=1, width=0.008, headwidth=4, label='Current Orientation', zorder=4)
            ax_traj.quiver(current_drone_state[0], current_drone_state[1],
                        arrow_len * 1.2 * np.cos(opt_light_theta), arrow_len * 1.2 * np.sin(opt_light_theta),
                        color='orange', scale_units='xy', scale=1, width=0.006, headwidth=4, label=f'Est. Light ({np.rad2deg(opt_light_theta):.1f}°)', zorder=4)
            ax_traj.quiver(0, 0,
                        arrow_len * 1.2 * np.cos(TRUE_LIGHT_ANGLE), arrow_len * 1.2 * np.sin(TRUE_LIGHT_ANGLE),
                        color='green', scale_units='xy', scale=1, width=0.007, headwidth=4, label=f'True Light ({np.rad2deg(TRUE_LIGHT_ANGLE):.1f}°)', zorder=4, alpha=0.7)
            ax_traj.set_xlabel("X Coordinate"); ax_traj.set_ylabel("Y Coordinate")
            ax_traj.set_title(f"Trajectory & Prediction (MPC Iter {iter_num + 1})")
            ax_traj.legend(fontsize='small', loc='upper right'); ax_traj.grid(True); ax_traj.set_aspect('equal', adjustable='box')
            all_x_traj = np.concatenate((current_history_states_plot[:, 0], self.trees_pos[:, 0], [0], X_opt[0,:]))
            all_y_traj = np.concatenate((current_history_states_plot[:, 1], self.trees_pos[:, 1], [0], X_opt[1,:]))
            margin_traj = max(2.0, 0.15 * (max(all_x_traj.max()-all_x_traj.min(), all_y_traj.max()-all_y_traj.min()))) if len(all_x_traj)>1 else 2.0
            ax_traj.set_xlim(all_x_traj.min() - margin_traj, all_x_traj.max() + margin_traj)
            ax_traj.set_ylim(all_y_traj.min() - margin_traj, all_y_traj.max() + margin_traj)


            # GP Plot (Shows Error vs. 1D Angle)
            ax_gp.cla()
            if gp is not None and gp_params_available:
                try:
                    gp_mean_pred_error, gp_std_pred_error = gp.predict(plot_angles_gp_2d, return_std=True)
                    ax_gp.plot(plot_angles_gp_1d.ravel(), gp_mean_pred_error, color='blue', linestyle='-', linewidth=2, label='GP Mean (Error)', zorder=2)
                    ax_gp.fill_between(plot_angles_gp_1d.ravel(),
                                    gp_mean_pred_error - 1.96 * gp_std_pred_error,
                                    gp_mean_pred_error + 1.96 * gp_std_pred_error,
                                    color='blue', alpha=0.2, label='GP 95% CI (Error)', zorder=1)
                except Exception as plot_e:
                    print(f"Could not plot GP prediction: {plot_e}")

            if len(X_train_angles_list) > 0:
                X_train_gp_1d_plot = np.array(X_train_angles_list)
                Y_train_gp_error_plot = np.array(Y_train_error_list)
                ax_gp.scatter(X_train_gp_1d_plot.ravel(), Y_train_gp_error_plot.ravel(), color='red', marker='x', s=50, label='Training Data (Error)', zorder=3)

            ax_gp.axvline(TRUE_LIGHT_ANGLE, color='green', linestyle='--', label=f'True Light ({TRUE_LIGHT_ANGLE:.2f})', zorder=4)
            ax_gp.axhline(0, color='black', linestyle=':', linewidth=0.8, label='Zero Error', zorder=1)
            ax_gp.set_xlabel("Estimated Light Angle (rad)")
            ax_gp.set_ylabel("Calculated Error (True - Pred Score)")
            ax_gp.set_title(f"GP Fit (Error Model - Iter {iter_num + 1}, {len(X_train_angles_list)} pts)")
            ax_gp.legend(fontsize='small', loc='best')
            ax_gp.grid(True)
            ax_gp.set_xticks(np.linspace(-np.pi, np.pi, 5))
            ax_gp.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
            all_errors_plot = np.array(Y_train_error_list)
            if len(all_errors_plot) > 0:
                y_min_err, y_max_err = all_errors_plot.min(), all_errors_plot.max()
                y_range_err = y_max_err - y_min_err if y_max_err > y_min_err else 0.1
                plot_margin_err = 0.2 * y_range_err + 0.05 # Adjusted margin
                y_abs_max_err = max(abs(y_min_err), abs(y_max_err), 0.01) # Ensure range > 0
                ax_gp.set_ylim(min(y_min_err, -y_abs_max_err) - plot_margin_err,
                            max(y_max_err, y_abs_max_err) + plot_margin_err)
            else:
                ax_gp.set_ylim(-0.1, 0.1) # Default range

            # Refresh plot window
            plt.draw()
            plt.pause(PLOT_UPDATE_INTERVAL)

            end_time = time.time()
            print(f"MPC Iteration {iter_num + 1} time: {end_time - start_time:.2f} seconds")


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    try:
        mpc_node = FovMPC()
        # Check if initialization failed (e.g., couldn't get tree poses or initial state)
        if not mpc_node.initialization_failed:
             mpc_node.run_simulation()
        else:
             rospy.logerr("FOV MPC Node initialization failed. Shutting down.")
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in FovMPC main: {e}", exc_info=True) # Log traceback
    finally:
        rospy.loginfo("FOV MPC node shutting down.")