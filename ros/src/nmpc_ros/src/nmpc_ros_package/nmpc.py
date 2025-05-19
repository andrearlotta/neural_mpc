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

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
import tf


from nmpc_ros.srv import GetTreesPoses


from nmpc_ros_package.ros_com_lib.sensors import create_path_from_mpc_prediction, create_tree_markers




class MultiLayerPerceptron(torch.nn.Module):
    input_layer: torch.nn.Linear
    hidden_layer: torch.nn.ModuleList
    out_layer: torch.nn.Linear

    def __init__(self, input_dim, hidden_size=64, hidden_layers=3):
        super().__init__()
        in_features = input_dim if input_dim != 3 else input_dim + 1
        self.input_layer = torch.nn.Linear(in_features, hidden_size)
        self.hidden_layer = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):

        if x.shape[-1] == 3:
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            x = torch.cat([x[..., :-1], sin_cos], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x




class NeuralMPC:
    def __init__(self, run_dir=None, initial_randomic=False):
        
        self.hidden_size = 64
        self.hidden_layers = 3
        self.nn_input_dim = 3

        self.N = 5
        self.dt = 0.2
        self.T = self.dt * self.N
        self.nx = 3  
        self.n_control = self.nx
        self.n_state = self.nx * 2
        self.NUM_TARGET_TREES = 10
        self.NUM_OBSTACLE_TREES = 2

        self.entropy_target = self.entropy_f(self.NUM_TARGET_TREES)
        self.latest_trees_scores = None


        rospy.init_node("nmpc_node", anonymous=True, log_level=rospy.DEBUG)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        
        rospy.Subscriber("tree_scores", Float32MultiArray, self.tree_scores_callback)
        
        self.cmd_pose_pub = rospy.Publisher("cmd/pose", Pose, queue_size=10)
        self.pred_path_pub = rospy.Publisher("predicted_path", Path, queue_size=10)
        self.tree_markers_pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=10)

        
        self.trees_pos, self.trees_gt_id = self.get_trees_poses_and_types()
        self.num_total_trees = self.trees_pos.shape[0]
        self.entropy_entire_field = self.entropy_f(self.num_total_trees)

        
        self.lambda_k = ca.DM.ones(self.num_total_trees, 1) * 0.5

        
        self.mpc_horizon = self.N
        self.baselines_dir = os.path.join( os.path.dirname(os.path.abspath(__file__)), "../../baselines") if run_dir is None else run_dir
        self.initial_randomic = initial_randomic

    
    
    

    def robot_state_update(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'drone_base_link', rospy.Time())
            (_, _, yaw) = tf.transformations.euler_from_quaternion([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            return [trans.transform.translation.x,
                                  trans.transform.translation.y,
                                  yaw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to get transform: %s", e)
            return None

    def tree_scores_callback(self, msg):
        """
        Callback for tree scores.
        """
        scores = np.array(msg.data).reshape(-1, 1)
        self.latest_trees_scores = scores

    
    
    
    def get_trees_poses_and_types(self):
        """
        Calls the GetTreesPoses service and returns:
        - tree_positions: An (N,2) numpy array of [x, y] positions.
        - tree_types: An (N,) numpy array of types (0 for raw, 1 for ripe).
        """
        service_name = "/obj_pose_srv"
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            trees_srv = rospy.ServiceProxy(service_name, GetTreesPoses)
            response = trees_srv()

            trees_pos = np.array([[pose.position.x, pose.position.y]
                                for pose in response.trees_poses.poses])
            
            if isinstance(response.tree_types, bytes):
                tree_types_list = list(response.tree_types) # Converts b'\x00\x01\x02' to [0, 1, 2]
            else:
                rospy.logerr(f"Unexpected type for raw tree types data: {type(response.tree_types)}")
                tree_types_list = []
            tree_types = np.array(tree_types_list, dtype=np.uint8)

            if len(trees_pos) != len(tree_types):
                rospy.logwarn(f"Mismatch in number of poses ({len(trees_pos)}) and types ({len(tree_types)}).")
                return np.array([]), np.array([])
                
            return trees_pos, tree_types
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to {service_name} failed: {e}")
            return np.array([]), np.array([])
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to connect to service {service_name}: {e}")
            return np.array([]), np.array([])
        except AttributeError as e:
            rospy.logerr(f"Failed to access 'tree_types' in service response: {e}. "
                        "Ensure .srv file and messages are updated.")
            return np.array([]), np.array([])

    # ---------------------------
    # Utility Functions
    # ---------------------------
    def generate_random_initial_state(self, lb, ub, margin=1.5):
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

    def get_latest_best_model(self, cls=''):
    
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
        model_dir = os.path.join(script_dir, "models", cls)
    
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
        print( os.path.join(model_dir, latest_model))
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
        nx = 6
    
        nu = 3
        
    
    
        x_sym = ca.SX.sym('x', nx)
        u_sym = ca.SX.sym('u', nu)
        
    
        px, py, pw, vx, vy, vw = [x_sym[i] for i in range(nx)]
        ax, ay, aw = [u_sym[i] for i in range(nu)]
        
    
        x_dot = ca.vertcat(vx, vy, vw, ax, ay, aw)
        
    
        f_continuous = ca.Function('f_cont', [x_sym, u_sym], [x_dot], ['x', 'u'], ['x_dot'])
        k1 = f_continuous(x_sym, u_sym)
        k2 = f_continuous(x_sym + dt / 2 * k1, u_sym)
        k3 = f_continuous(x_sym + dt / 2 * k2, u_sym)
        k4 = f_continuous(x_sym + dt * k3, u_sym)
        x_next = x_sym + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
    
        F = ca.Function('F', [x_sym, u_sym], [x_next], ['x_k', 'u_k'], ['x_k1'])
        return F

    @staticmethod
    def bayes(lambda_prev, z):
        """
        Bayesian update for belief:
           lambda_next = (lambda_prev * z) / (lambda_prev * z + (1 - lambda_prev) * (1 - z))
        """
        prod = lambda_prev * z
        denom = prod + (1 - lambda_prev) * (1 - z)
        return prod / denom

    @staticmethod
    def entropy_f(num_targets):
        """
        Compute the binary entropy of a probability p.
        Values are clipped to avoid log(0).
        """
        p = ca.MX.sym(f'input_entropy_f{num_targets}_dim', num_targets)
        output = -4*(p-0.5)**2 + 1 #(-p * ca.log10(p) - (1 - p) * ca.log10(1 - p)) / ca.log10(2)
        return ca.Function(f'entropy_f_{num_targets}_dim', [p], [output])

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

        robot_pos_1d = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos[:, :2] - robot_pos_1d, axis=1)

        H = self.entropy_entire_field(self.lambda_k)

        H_min = H.full()
        
        candidate_indices = np.where(H_min > entropy_threshold)[0]
        if candidate_indices.size == 0:
            print("Warning: No trees above entropy threshold. Selecting nearest trees.")
            sorted_indices_all = np.argsort(distances)
            return sorted_indices_all[:num_target]


        sorted_candidates = candidate_indices[np.argsort(distances[candidate_indices])]

        if sorted_candidates.size < num_target:
            repeats = int(np.ceil(num_target / sorted_candidates.size))
            sorted_candidates = np.tile(sorted_candidates, repeats)[:num_target]
        return sorted_candidates[:num_target]
    def get_nearest_tree_indices(self, robot_position, num_obstacle=None):
        """
        Returns the indices of the 'num_obstacle' trees (from self.trees_pos)
        that are closest to the robot_position.
        """
        if num_obstacle is None:
            num_obstacle = self.NUM_OBSTACLE_TREES
        if num_obstacle > self.num_total_trees:
            num_obstacle = self.num_total_trees
        robot_pos_1d = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos - robot_pos_1d, axis=1)
        sorted_indices = np.argsort(distances)
        return sorted_indices[:num_obstacle]

    # ---------------------------
    # MPC Optimization Function
    # ---------------------------
    def mpc_opt(self, g_nn, target_trees, target_lambdas, obstacle_trees, lb, ub, x0, steps=10):
        opti = ca.Opti()
        F_ = self.kin_model(self.nx, self.dt)

        # Decision variables
        X = opti.variable(self.n_state, steps + 1)
        U = opti.variable(self.n_control, steps)

        # Parameters
        num_target_trees = target_trees.shape[0]
        num_obstacle_trees = obstacle_trees.shape[0]
        param_size = self.n_state + num_target_trees * 3 + num_obstacle_trees * 2
        P0 = opti.parameter(param_size)

        p_idx = 0
        X0 = P0[p_idx : p_idx + self.n_state]; p_idx += self.n_state
        TARGET_TREES_param = P0[p_idx : p_idx + num_target_trees*2].reshape((num_target_trees,2)).T; p_idx += num_target_trees*2
        L0 = P0[p_idx : p_idx + num_target_trees]; p_idx += num_target_trees
        OBSTACLE_TREES_param = P0[p_idx : p_idx + num_obstacle_trees*2].reshape((num_obstacle_trees,2)).T
        lambda_evol = [L0]

        # Weights
        Q_dist = 1e-3
        R_xy = 1e-2
        R_theta = 1e-2
        attraction = 0
        safe_distance = 1.0
        obj = 0

        # Initial condition
        opti.subject_to(X[:, 0] == X0)
        ca_batch = []

        # Dynamics + bounds
        for i in range(steps):
            opti.subject_to(opti.bounded(lb[0] - 3.0, X[0, i], ub[0] + 3.0))
            opti.subject_to(opti.bounded(lb[1] - 3.0, X[1, i], ub[1] + 3.0))
            opti.subject_to(opti.bounded(-3*np.pi, X[2, i], +3*np.pi))
            opti.subject_to(opti.bounded(-1.75, X[3:5, i], 1.75))
            opti.subject_to(opti.bounded(-np.pi/4, X[5, i], np.pi/4))

            opti.subject_to(opti.bounded(-5.0, U[0:2, i], 5.0))
            opti.subject_to(opti.bounded(-np.pi, U[2, i], np.pi))
            opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

            # Collision avoidance
            for j in range(num_obstacle_trees):
                obs_j_pos = OBSTACLE_TREES_param[:, j]
                dist_sq_obs = ca.sumsqr(X[:2, i+1] - obs_j_pos)
                opti.subject_to(dist_sq_obs >= safe_distance**2)

            distances_sq = []
            nn_batch = []
            for j in range(num_target_trees):
                obj_j_pos = TARGET_TREES_param[:, j]
                diff = X[:2, i + 1] - obj_j_pos
                distances_sq.append(ca.sumsqr(diff) +1e-6)
                heading_target = X[2, i+1]
                nn_batch.append(ca.horzcat(diff.T, heading_target))
            ca_batch.append(ca.vcat([*nn_batch]))
            min_dist_sq = distances_sq[0]
            for j in range(1, num_target_trees):
                min_dist_sq = ca.fmin(min_dist_sq, distances_sq[j])
            attraction = attraction + min_dist_sq * Q_dist
            obj = obj + R_xy * ca.sumsqr(U[:2, i]) + R_theta * ca.sumsqr(U[2, i])

        g_out = g_nn(ca.vcat(ca_batch)) 
        
        z_k = ca.fmax(g_out, 0.5)
        L0_ext = ca.vcat([L0 for _ in range(steps)])
        z_k_bin = (L0_ext>=0.5)*z_k + (L0_ext<0.5)*(1-z_k)
        for i in range(steps):
            lambda_next = self.bayes(lambda_evol[-1], z_k_bin[i*num_target_trees:(i+1)*num_target_trees])
            lambda_evol.append(lambda_next)
        entropy_obj = 0

        for i in range(1, steps+1):
            entropy_future = self.entropy_target(lambda_evol[i])
            entropy_obj += ca.exp(-2*i)*ca.logsumexp(-40*entropy_future)


        sq_dist_to_targets = ca.sum1((X0[:2] - TARGET_TREES_param)**2)
        min_sq_dist = ca.mmin(sq_dist_to_targets)

        
        threshold_sq_dist = 81.0
        sigmoid_steepness = 10.0
        sigmoid_factor = 1.0 / (1.0 + ca.exp(-sigmoid_steepness * (min_sq_dist - threshold_sq_dist)))

        
        modulated_attraction_term = attraction * sigmoid_factor

        opti.minimize(obj - 10*entropy_obj + modulated_attraction_term)                                 
        options = {
            "ipopt": {
                "tol":1e-6,
                "warm_start_init_point": "yes",
                "print_level": 0,
                "sb": "no",
                "warm_start_bound_push": 1e-8,
                "warm_start_mult_bound_push": 1e-8,
                "mu_init": 1e-5,
                "bound_relax_factor": 1e-9,
                #"hsllib": '/usr/local/lib/libcoinhsl.so', # Specify HSL library path if used
                #"linear_solver": 'ma27',
                "hessian_approximation":'limited-memory',
                "mu_strategy": "monotone",
                "max_iter": 1000,
            }
        }
        opti.solver("ipopt", options)
        inputs = [P0, opti.x, opti.lam_g]
        outputs = [U[:, 0], X,  opti.x, opti.lam_g]

        # --- Solve for the first time ---
        p0_val = ca.vertcat(
            x0,
            ca.reshape(target_trees, 2* self.NUM_TARGET_TREES, 1),
            target_lambdas[:num_target_trees], # Raw lambdas
            target_lambdas[num_target_trees:], # Ripe lambdas
            ca.reshape(obstacle_trees, 2* self.NUM_OBSTACLE_TREES, 1)
        )
        opti.set_value(P0, p0_val)

        sol = opti.solve()
        u_sol = sol.value(U[:, 0])
        x_traj_sol = sol.value(X)
        x_dec_sol = sol.value(opti.x)
        lam_g_sol = sol.value(opti.lam_g)
        mpc_step_func = opti.to_function("mpc_step", inputs, outputs, ["p", "x_init", "x_lam"], ["u_opt", "x_pred", "x_opt", "lam_opt"])


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
        lb, ub = self.get_domain(self.trees_pos)
        self.mpc_horizon = self.N
        current_state = None

        if self.initial_randomic:
            current_state = self.generate_random_initial_state(lb, ub, margin=1.5)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, float(current_state[2]))
            cmd_pose_msg = Pose()
            cmd_pose_msg.position = Point(x=float(current_state[0]), y=float(current_state[1]), z=0.0)
            cmd_pose_msg.orientation = Quaternion(x=quaternion[0],
                                                  y=quaternion[1],
                                                  z=quaternion[2],
                                                  w=quaternion[3])
            self.cmd_pose_pub.publish(cmd_pose_msg)    
            rospy.sleep(1.5)

        # Wait until a position message has been received.
        rospy.loginfo("Waiting for GPS data...")
        while current_state is None and not rospy.is_shutdown():
            current_state =  self.robot_state_update()
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
        g_nn = l4c.L4CasADi(model, batched=True, device='cuda', generate_jac_jac=True, generate_adj1=False, generate_jac_adj1=False)


        initial_state = current_state  
        vx_k = ca.DM.zeros(self.nx)  
        x_k = ca.vertcat(ca.DM(initial_state), vx_k)

        all_trajectories = []
        lambda_history = []
        entropy_history = []
        durations = []

        velocity_command_log = []
        pose_history = []
        time_history = []

        sim_start_time = time.time()
        total_distance = 0.0
        total_commands = 0
        sum_vx = 0.0
        sum_vy = 0.0
        sum_yaw = 0.0
        sum_trans_speed = 0.0
        prev_x = float(x_k[0])
        prev_y = float(x_k[1])

        sim_time = 1200
        mpciter = 0
        rate = rospy.Rate(int(1/self.dt))
        warm_start = True
        x_dec_prev = None
        lam_g_prev = None
        mpc_step = None
        current_state = None
        # Main MPC loop.
        while mpciter < sim_time and not rospy.is_shutdown():
            loop_iter_start = time.time()
            rospy.loginfo('Step: %d', mpciter)
            current_state =  self.robot_state_update()
            while current_state is None and not rospy.is_shutdown():
                current_state = self.robot_state_update()
                rospy.sleep(0.05)
            current_sim_time = time.time() - sim_start_time
            pose_history.append(current_state)
            time_history.append(current_sim_time)
            x_k = ca.vertcat(ca.DM(current_state), vx_k)

            while self.latest_trees_scores is None:
                rospy.sleep(0.05)
            scores = self.latest_trees_scores.copy()
            if (mpciter % 2 == 0): self.lambda_k = self.bayes(self.lambda_k, ca.DM(scores))

            tree_markers_msg = create_tree_markers(self.trees_pos, self.lambda_k.full().flatten())
            self.tree_markers_pub.publish(tree_markers_msg)
            
            robot_position_xy = np.array(current_state[:2])
            target_indices = self.get_target_tree_indices(robot_position_xy, num_target=self.NUM_TARGET_TREES)
            obstacle_indices = self.get_nearest_tree_indices(robot_position_xy, num_obstacle=self.NUM_OBSTACLE_TREES)

            target_trees_subset = self.trees_pos[target_indices]
            obstacle_trees_subset = self.trees_pos[obstacle_indices]

            target_lambdas = self.lambda_k[target_indices]

            step_start_time = time.time()
            try:
                if warm_start or mpc_step is None:
                    print("Running MPC opt (first step / cold start)...")
                    mpc_step, u, x_traj, x_dec_prev, lam_g_prev = self.mpc_opt(
                        g_nn, target_trees_subset, target_lambdas, obstacle_trees_subset, lb, ub, x_k, steps=self.N
                    )
                    warm_start = False
                    print("MPC opt finished.")
                else:
                    print("Running MPC step (warm start)...")
                    P0_val = ca.vertcat(
                        x_k,
                        ca.reshape(target_trees_subset, 2* self.NUM_TARGET_TREES, 1),
                        target_lambdas,
                        ca.reshape(obstacle_trees_subset, 2* self.NUM_OBSTACLE_TREES, 1)
                    )
                    u, x_traj, x_dec_prev, lam_g_prev = mpc_step(P0_val, x_dec_prev, lam_g_prev)
                    print("MPC step finished.")

                step_duration = time.time() - step_start_time
                durations.append(step_duration)
                print(f"MPC step duration: {step_duration:.4f} s")

            except Exception as e:
                rospy.logerr(f"Error during MPC optimization at step {mpciter}: {e}")
                u = ca.DM.zeros(self.n_control)
                x_traj = ca.repmat(x_k, 1, self.N + 1)
                warm_start = True
                print("!!! MPC Solver Failed - Commanding Zero Acceleration !!!")
                return

            durations.append(time.time() - step_start_time)

            # Log the MPC velocity command.
            u_np = np.array(u.full()).flatten()

            # Compute the command pose.
            cmd_pose = F_(x_k, u[:, 0])

            # Publish predicted path.
            predicted_path_msg = create_path_from_mpc_prediction(x_traj[:self.nx, 1:])
            self.pred_path_pub.publish(predicted_path_msg)

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

            entropy_k = self.entropy_entire_field(self.lambda_k)
            lambda_history.append(self.lambda_k.full().flatten().tolist())
            entropy_history.append(ca.sum1(entropy_k).full().flatten()[0])
            all_trajectories.append(x_traj[:self.nx, :].full())

            mpciter += 1
            rospy.loginfo("Entropy: %s", entropy_history[-1])
            if all( v <= 0.025 for v in entropy_k.full().flatten()):
                rospy.loginfo("Entropy target reached.")
                break

        
            loop_elapsed = time.time() - loop_iter_start
            sleep_time = self.dt - loop_elapsed
            current_state = None
            if sleep_time > 0:
                rate.sleep()
            else:
                 print(f"Warning: Loop iteration {mpciter} took {loop_elapsed:.4f}s, longer than dt={self.dt}s")

            rospy.sleep(0.1) 
        
        # ---------------------------
        # Final Metrics Calculation and CSV Output
        # ---------------------------
        total_execution_time = time.time() - sim_start_time
        avg_wp_time = total_execution_time / total_commands if total_commands > 0 else 0.0
        avg_vx = sum_vx / total_commands if total_commands > 0 else 0.0
        avg_vy = sum_vy / total_commands if total_commands > 0 else 0.0
        avg_yaw = sum_yaw / total_commands if total_commands > 0 else 0.0
        avg_trans_speed = sum_trans_speed / total_commands if total_commands > 0 else 0.0

        os.makedirs(self.baselines_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_csv = os.path.join(self.baselines_dir, f"mpc_{timestamp}_performance_metrics.csv")
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
            for label, value in zip(headers, values):
                writer.writerow([label, value])

        vel_csv = os.path.join(self.baselines_dir, f"mpc_{timestamp}_velocity_commands.csv")
        with open(vel_csv, mode='w', newline='') as vel_file:
            writer = csv.writer(vel_file)
            writer.writerow(["Time (s)", "Tag", "x_velocity", "y_velocity", "yaw_velocity"])
            writer.writerows(velocity_command_log)

        rospy.loginfo("Performance metrics saved to %s", perf_csv)
        rospy.loginfo("Velocity command log saved to %s", vel_csv)

        plot_csv = os.path.join(self.baselines_dir, f"mpc_{timestamp}_plot_data.csv")
        with open(plot_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            
            tree_positions_flat = self.trees_pos.flatten().tolist()
            writer.writerow(["tree_positions"] + tree_positions_flat)

            writer.writerow(["trees_gt_id"] + self.trees_gt_id.tolist())

            header = ["time", "x", "y", "theta", "entropy"]
            if lambda_history and lambda_history[0] is not None and len(lambda_history[0]) > 0:
                num_lambdas = len(lambda_history[0]) 
                header += [f"lambda_{i}" for i in range(num_lambdas)]
            writer.writerow(header)
            
            for i in range(len(time_history)):
                time_val = time_history[i]
                
                current_pose = np.array(pose_history[i]).flatten()
                if len(current_pose) == 3:
                    x, y, theta = current_pose
                else: 
                    x, y, theta = 0.0, 0.0, 0.0 
                    rospy.logwarn(f"Unexpected pose format at index {i}: {pose_history[i]}")

                entropy_val = entropy_history[i]
                lambda_vals = lambda_history[i]
                row = [time_val, x, y, theta, entropy_val] + lambda_vals
                writer.writerow(row)
                
        rospy.loginfo("Plot data saved to %s", plot_csv)

        return all_trajectories, entropy_history, lambda_history, durations, g_nn, self.trees_pos, lb, ub




if __name__ == "__main__":
    mpc = NeuralMPC()
    sim_results = mpc.run_simulation()
