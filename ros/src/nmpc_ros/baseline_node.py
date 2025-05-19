#!/usr/bin/env python
import rospy
import math
import numpy as np
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, Twist
import tf
import csv
import time
import os
import threading

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
import tf
from nmpc_ros_package.ros_com_lib.sensors import create_tree_markers

from nmpc_ros.srv import GetTreesPoses

class PIDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0

    def update(self, error):
        output = self.kp * error + self.kd * (error - self.prev_error)
        self.prev_error = error
        return output


class Logger:
    def __init__(self, trajectory_type, run_folder="baselines", filename='performance_metrics.csv'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
        # Create a filename that includes trajectory type and timestamp
        self.filename = os.path.join(os.path.join(script_dir, run_folder),
                                     f"{trajectory_type}_{self.timestamp}_{filename}")

        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header with additional columns for Position and Entropy Reduction.
        self.csv_writer.writerow(['Waypoint', 'Position', 'Execution Time (s)',
                                  'Cumulative Distance (m)', 'Waypoint-to-Waypoint Time (s)', 'Entropy Reduction'])
        self.performance_data = []
        self.start_time = None
        self.total_distance = 0

        # Open an additional CSV file to log injected velocity commands.
        self.velocity_filename = os.path.join(os.path.join(script_dir,run_folder),
                                              f"{trajectory_type}_{self.timestamp}_velocity_commands.csv")
        self.velocity_csv_file = open(self.velocity_filename, 'w', newline='')
        self.velocity_csv_writer = csv.writer(self.velocity_csv_file)
        self.velocity_csv_writer.writerow(['Time (s)', 'Tag', 'x_velocity', 'y_velocity', 'yaw_velocity'])

    def record_performance(self, wp_index, execution_time, distance, wp_to_wp_time, entropy_reduction=None, position=None):
        log_entry = {
            'Waypoint': wp_index,
            'Position': position,
            'Execution Time': execution_time,
            'Distance': distance,
            'Waypoint-to-Waypoint Time': wp_to_wp_time,
            'Entropy Reduction': entropy_reduction
        }
        self.performance_data.append(log_entry)
        self.csv_writer.writerow([wp_index, position, execution_time, distance, wp_to_wp_time, entropy_reduction])
        self.csv_file.flush()

    def log_velocity(self, x_vel, y_vel, yaw_vel, tag):
        current_time = time.time() - self.start_time if self.start_time is not None else 0
        self.velocity_csv_writer.writerow([current_time, tag, x_vel, y_vel, yaw_vel])
        self.velocity_csv_file.flush()

    def start_logging(self):
        self.start_time = time.time()

    def add_distance(self, distance):
        self.total_distance += distance

    def finalize_performance_metrics(self, final_entropy=None, final_bayes=None):
        total_time = time.time() - self.start_time if self.start_time is not None else 0
        # Compute average waypoint-to-waypoint time if available.
        wp_times = [data['Waypoint-to-Waypoint Time'] for data in self.performance_data]
        avg_wp_time = np.mean(wp_times) if wp_times else 0

        self.csv_writer.writerow([])
        self.csv_writer.writerow(['Total Execution Time (s)', total_time])
        self.csv_writer.writerow(['Total Distance (m)', self.total_distance])
        self.csv_writer.writerow(['Average Waypoint-to-Waypoint Time (s)', avg_wp_time])
        if final_entropy is not None:
            self.csv_writer.writerow(['Final Entropy', final_entropy])
        if final_bayes is not None:
            self.csv_writer.writerow(['Final Bayes Values', final_bayes])
        self.csv_file.close()
        self.velocity_csv_file.close()

        print(f"Total Execution Time: {total_time:.2f} s")
        print(f"Total Distance: {self.total_distance:.2f} m")
        print(f"Average Waypoint-to-Waypoint Time: {avg_wp_time:.2f} s")


class TrajectoryGenerator:
    def __init__(self, trajectory_type, run_folder="baselines", random_initial_state=True):
        self.run_folder = run_folder
        # Get trajectory mode and initialize Logger only once.
        self.trajectory_type = trajectory_type
        self.logger = Logger(trajectory_type, run_folder)

        rospy.init_node("baseline_node", anonymous=True, log_level=rospy.DEBUG)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_pose_pub = rospy.Publisher("cmd/pose", Pose, queue_size=10)
        self.tree_markers_pub = rospy.Publisher("tree_markers", MarkerArray, queue_size=10)
        rospy.Subscriber("tree_scores", Float32MultiArray, self.tree_scores_callback)

        # New history lists for storing poses and bayesian (lambda) values
        self.pose_history = []      # Each entry: [x, y, theta]
        self.lambda_history = []    # Each entry: copy of lambda_values at that step
        self.entropy_history = []
        self.time_history = []
        self.latest_trees_scores = None
        self.is_mower =( self.trajectory_type == 'mower')
        # Initialize PID controllers.
        self.pid_controller_x = PIDController(kp=3.0 if self.is_mower else 7.0, kd=1)
        self.pid_controller_y = PIDController(kp=3.0 if self.is_mower else 7.0, kd=1)
        self.pid_controller_yaw = PIDController(kp=3.0 if self.is_mower else .25, kd=0.1)

        self.idx = None

        self.circle_tree_event = threading.Event()
 
        # Get tree positions from service using the sensors.py serializer logic.
        self.tree_positions, self.trees_gt_id = self.get_trees_poses_and_types()
        self.num_total_trees = self.tree_positions.shape[0]
        lb, ub = self.get_domain(self.tree_positions)
        initial_state = self.generate_random_initial_state(lb, ub, margin=1.75)
        print(f"Initial State: {initial_state}")
        # Publish the initial random position.
        if random_initial_state:
            current_state = self.generate_random_initial_state(lb, ub, margin=1.5)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, float(current_state[2]))
            cmd_pose_msg = Pose()
            cmd_pose_msg.position = Point(x=float(current_state[0]), y=float(current_state[1]), z=0.0)
            cmd_pose_msg.orientation = Quaternion(x=quaternion[0],
                                                  y=quaternion[1],
                                                  z=quaternion[2],
                                                  w=quaternion[3])
            self.cmd_pose_pub.publish(cmd_pose_msg)            
        rospy.sleep(2.5)
        self.x, self.y, self.theta = np.array(self.robot_state_update_thread()).flatten()
        self.lambda_values = np.full(len(self.tree_positions), 0.5)  # Initialize lambda


        tree_markers_msg = create_tree_markers(self.tree_positions, self.lambda_values.flatten())
        self.tree_markers_pub.publish(tree_markers_msg)

        self.max_velocity = 0.45 if self.is_mower else 3.0         # Maximum velocity of the robot
        self.max_yaw_velocity = np.pi/4      # Maximum yaw velocity (rad/s)

        self.dt = 0.1
        self.rate = rospy.Rate(int(1/self.dt))  # Control loop rate

        # Maximum time allowed for observation around a tree (in seconds)
        self.max_observe_time = 30.0
        self.rate = rospy.Rate(int(1 / self.dt))
        self.measurement_timer = rospy.Timer(rospy.Duration(0.4), self.measurement_callback)

    # ---------------------------
    # Service Call to Get Trees Poses
    # ---------------------------
    def get_trees_poses_and_types(self):
        """
        Calls the GetTreesPoses service and returns:
        - tree_positions: An (N,2) numpy array of [x, y] positions.
        - tree_types: An (N,) numpy array of types (0 for raw, 1 for ripe).
        """
        service_name = "/obj_pose_srv" # Assicurati che corrisponda al server C#
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            trees_srv = rospy.ServiceProxy(service_name, GetTreesPoses)
            response = trees_srv()

            trees_pos = np.array([[pose.position.x, pose.position.y]
                                for pose in response.trees_poses.poses])
            
            # Assuming each byte in raw_byte_string is a tree type
            if isinstance(response.tree_types, bytes):
                tree_types_list = list(response.tree_types) # Converts b'\x00\x01\x02' to [0, 1, 2]
            else:
                # Handle other cases or raise an error if the type is unexpected
                rospy.logerr(f"Unexpected type for raw tree types data: {type(response.tree_types)}")
                tree_types_list = [] # Or some default
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
    # Callback Functions
    # ---------------------------
    def tree_scores_callback(self, msg):
        """
        Callback for tree scores.
        """
        scores = np.array(msg.data).reshape(-1, 1)
        self.latest_trees_scores = scores

    def robot_state_update_thread(self):
        """Continuously update the robot's state using TF at 30 Hz."""
        try:
            # Look up the transform from 'map' to 'drone_base_link'
            trans = self.tf_buffer.lookup_transform('map', 'drone_base_link', rospy.Time())
            # Extract the yaw angle from the quaternion
            (_, _, yaw) = tf.transformations.euler_from_quaternion([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            # Update current_state with [x, y, yaw]
            return [trans.transform.translation.x,
                                  trans.transform.translation.y,
                                  yaw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to get transform: %s", e)
            return None


    def generate_random_initial_state(self, lb, ub, margin=1.25):
        """
        Generate a random initial state [x, y, theta].
        - If mode is 'linear', start outside the field bounds.
        - If mode is 'mower', start from one of the four field corners.
        """
        threshold = 1.5  # Extra margin
        direction_map = {'N': np.pi/2, 'E': 0, 'S': -np.pi/2, 'W': np.pi}
        orientations = ['N', 'S', 'E', 'W']

        while True:
            if self.trajectory_type == "linear":
                # Generate x outside the x domain
                if np.random.rand() < 0.5:
                    x = np.random.uniform(lb[0] - threshold, lb[0])
                else:
                    x = np.random.uniform(ub[0], ub[0] + threshold)

                # Generate y outside the y domain
                if np.random.rand() < 0.5:
                    y = np.random.uniform(lb[1] - threshold, lb[1])
                else:
                    y = np.random.uniform(ub[1], ub[1] + threshold)

                theta = np.random.uniform(-np.pi, np.pi)

            elif self.trajectory_type == "mower":
                # Randomly pick one of the 4 vertices
                vertices = [
                    (lb[0]-1.5, lb[1]-1.5),  # bottom-left
                    (lb[0]-1.5, ub[1]+1.5),  # top-left
                    (ub[0]-1.5, lb[1]-1.5),  # bottom-right
                    (ub[0]-1.5, ub[1]+1.5)   # top-right
                ]
                x, y = vertices[np.random.randint(0, 4)]

                # Random heading from cardinal directions
                orientation_choice = np.random.choice(orientations)
                theta = direction_map[orientation_choice]

                print(f"Selected start at vertex: ({x:.2f}, {y:.2f}) with orientation {orientation_choice}")

            else:
                # Default fallback: inside field, away from trees
                x = np.random.uniform(lb[0], ub[0])
                y = np.random.uniform(lb[1], ub[1])
                theta = np.random.uniform(-np.pi, np.pi)

            # Ensure initial position is not too close to trees
            valid = True
            for tree in self.tree_positions:
                if np.linalg.norm(np.array([x, y]) - tree) < margin:
                    valid = False
                    break

            if valid:
                return np.array([x, y, theta]).reshape(-1, 1)

    @staticmethod
    def get_domain(tree_positions):
        """Return the domain (bounding box) of the tree positions."""
        x_min = np.min(tree_positions[:, 0])
        x_max = np.max(tree_positions[:, 0])
        y_min = np.min(tree_positions[:, 1])
        y_max = np.max(tree_positions[:, 1])
        return [x_min, y_min], [x_max, y_max]

    def measurement_callback(self, event):
        if not (self.is_mower or self.circle_tree_event.is_set()):
            return
        if  self.latest_trees_scores is None : return
        z_k = self.latest_trees_scores
        
        if self.is_mower:
            for i in range(len(z_k)):
                self.lambda_values[i] = self.bayes(self.lambda_values[i], z_k[i])
                self.lambda_values = self.bayes(self.lambda_values, z_k)
        elif self.idx is not None:
            self.lambda_values[self.idx] = self.bayes(self.lambda_values[self.idx], z_k[self.idx])
            # When the lambda value reaches a threshold, clear the event to stop observation.
            if self.calculate_entropy(self.lambda_values[self.idx]) <= 0.025:
                self.circle_tree_event.clear()

        current_entropy = self.calculate_entropy(self.lambda_values)
        tree_markers_msg = create_tree_markers(self.tree_positions, self.lambda_values.flatten())
        self.tree_markers_pub.publish(tree_markers_msg)

    def calculate_entropy(self, lambda_values):
        """Simple entropy calculation based on lambda values."""
        lambda_values = np.fmin(lambda_values, 1 - 1e-6)
        return -np.sum(lambda_values * np.log2(lambda_values) + (1 - lambda_values) * np.log2(1 - lambda_values))

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calculate_motion_vector(self, current_position, goal_position):
        """
        Compute a motion vector for the drone that steers it toward the goal but 
        deviates tangentially when near obstacles (trees).
        """
        att_vector = goal_position - current_position
        att_magnitude = np.linalg.norm(att_vector)
        if att_magnitude < 1:
            att_unit = goal_position - current_position
        else:
            att_unit = att_vector / att_magnitude
        if self.is_mower:
            return att_vector
        deviation = np.zeros(2)
        unsafe = False
        ray_activation = 3.5
        ray_avoidance = 3.25

        for tree in self.tree_positions:
            if  np.linalg.norm(tree -goal_position)< 2.0:
                continue
            diff = current_position - tree
            d = np.linalg.norm(diff)
            if d < ray_activation:
                angle = np.arccos(np.clip(np.dot(att_unit, -diff / d), -1.0, 1.0))
                if angle < np.pi / 3:
                    unsafe = True
                    u = diff / d
                    tangent1 = np.array([-u[1], u[0]])
                    tangent2 = np.array([u[1], -u[0]])
                    chosen_tangent = tangent1 if np.dot(tangent1, att_unit) > np.dot(tangent2, att_unit) else tangent2
                    deviation += chosen_tangent * ray_avoidance - 1.75 * u

        if not unsafe:
            return att_unit * att_magnitude
        else:
            combined = att_unit + 3.5 * deviation
            combined_norm = np.linalg.norm(combined)
            if combined_norm < 1e-6:
                return att_unit * att_magnitude
            return combined / combined_norm
    
    # ----------------------- Modified move_to_waypoint -----------------------
    def move_to_waypoint(self, target_x, target_y, tolerance=.2, observe=False, desired_heading=None):
        """
        Move the drone to the specified waypoint.
        An optional desired_heading (in radians) can be provided to override the default
        heading computed from the goal position.
        """
        goal_position = np.array([target_x, target_y])
        start_time = time.time()

        while not rospy.is_shutdown():
            current_state = np.array(self.robot_state_update_thread()).flatten()
            self.x, self.y, self.theta = current_state[:3]
            self.pose_history.append(current_state[:3])
            self.time_history.append(time.time() - self.logger.start_time)
            self.lambda_history.append(self.lambda_values.copy())
            self.entropy_history.append(self.calculate_entropy(self.lambda_values))

            if np.linalg.norm(current_state[:2] - goal_position) < tolerance:
                break

            motion_vector = self.calculate_motion_vector(current_state[:2], goal_position)
            # Use the provided desired_heading if given; otherwise, compute from the goal.
            if desired_heading is None:
                desired_theta = math.atan2(goal_position[1] - current_state[1],
                                           goal_position[0] - current_state[0])
            else:
                desired_theta = desired_heading

            yaw_error = self.normalize_angle(desired_theta - self.theta)

            # Scale the motion vector to the maximum velocity.
            velocity = (motion_vector / np.linalg.norm(motion_vector))

            x_output = self.pid_controller_x.update(velocity[0])
            y_output = self.pid_controller_y.update(velocity[1])
            yaw_output = self.pid_controller_yaw.update(yaw_error)

            # Clip the PID outputs to their maximum limits.
            x_output = np.clip(x_output, -self.max_velocity, self.max_velocity)
            y_output = np.clip(y_output, -self.max_velocity, self.max_velocity)
            yaw_output = np.clip(yaw_output, -self.max_yaw_velocity, self.max_yaw_velocity)

            # Log the injected velocities.
            self.logger.log_velocity(x_output, y_output, yaw_output, tag="move_to_waypoint")
    
            cmd_pose = np.array([self.x + x_output * self.dt, self.y + y_output * self.dt, self.theta + yaw_output * self.dt]).reshape(-1, 1)
            # Build and publish the cmd_pose message.
            quaternion = tf.transformations.quaternion_from_euler(0, 0, float(cmd_pose[2]))
            cmd_pose_msg = Pose()
            cmd_pose_msg.position = Point(x=float(cmd_pose[0]), y=float(cmd_pose[1]), z=0.0)
            cmd_pose_msg.orientation = Quaternion(x=quaternion[0],
                                                  y=quaternion[1],
                                                  z=quaternion[2],
                                                  w=quaternion[3])
            self.cmd_pose_pub.publish(cmd_pose_msg)

            # Calculate the distance traveled in this step.
            step_distance = np.linalg.norm([x_output * self.dt,  y_output * self.dt])
            self.logger.add_distance(step_distance)
            self.rate.sleep()

        execution_time = time.time() - start_time
        entropy_reduction = self.calculate_entropy(self.lambda_values) if observe else None
        return execution_time, entropy_reduction
    # ------------------------------------------------------------------------

    def observe_tree(self):
        # Set the event to start observation.
        self.circle_tree_event.set()
        center_x, center_y = self.tree_positions[self.idx]
        start_time = time.time()

        self.x, self.y, self.theta = np.array(self.robot_state_update_thread()).flatten()
        phi = math.atan2(self.y - center_y, self.x - center_x)
        radius = np.sqrt((self.x - center_x) ** 2 + (self.y - center_y) ** 2)
        delta_phi = self.dt * 15 * math.pi / 180  # Angular increment (in radians)

        while self.circle_tree_event.is_set():
            desired_x = center_x + radius * math.cos(phi)
            desired_y = center_y + radius * math.sin(phi)
            desired_theta = phi + math.pi  # Face the tree center

            self.x, self.y, self.theta = np.array(self.robot_state_update_thread()).flatten()
            self.pose_history.append([self.x, self.y, self.theta])
            self.lambda_history.append(self.lambda_values.copy())
            self.entropy_history.append(self.calculate_entropy(self.lambda_values))
            self.time_history.append(time.time() - self.logger.start_time)

            x_error = desired_x - self.x
            y_error = desired_y - self.y
            yaw_error = self.normalize_angle(desired_theta - self.theta)

            x_output = self.pid_controller_x.update(x_error)
            y_output = self.pid_controller_y.update(y_error)
            yaw_output = self.pid_controller_yaw.update(yaw_error)

            # Clip the outputs
            x_output = np.clip(x_output, -self.max_velocity, self.max_velocity)
            y_output = np.clip(y_output, -self.max_velocity, self.max_velocity)
            yaw_output = np.clip(yaw_output, -self.max_yaw_velocity, self.max_yaw_velocity)

            self.logger.log_velocity(x_output, y_output, yaw_output, tag=f"observe_tree_{self.idx}")

            cmd_pose = np.array([self.x + x_output * self.dt, self.y + y_output * self.dt, self.theta + yaw_output]).reshape(-1, 1)
            # Build and publish the cmd_pose message.
            quaternion = tf.transformations.quaternion_from_euler(0, 0, float(cmd_pose[2]))
            cmd_pose_msg = Pose()
            cmd_pose_msg.position = Point(x=float(cmd_pose[0]), y=float(cmd_pose[1]), z=0.0)
            cmd_pose_msg.orientation = Quaternion(x=quaternion[0],
                                                  y=quaternion[1],
                                                  z=quaternion[2],
                                                  w=quaternion[3])
            self.cmd_pose_pub.publish(cmd_pose_msg)
            phi += delta_phi
            step_distance = np.linalg.norm([x_output * self.dt, y_output * self.dt])
            self.logger.add_distance(step_distance)
            rospy.sleep(self.dt)

        return time.time() - start_time

    def bayes(self, lambda_prev, z):
        """Bayesian update for confidence (lambda)."""
        numerator = lambda_prev * z
        denominator = numerator + (1 - lambda_prev) * (1 - z)
        return numerator / denominator

    def generate_mower_path_mower(self, offset=2.0, spacing=4.0, heading_direction='O', axis='x'):
        """
        Generate a mower path that includes external rows outside the tree grid.

        The path now includes external rows to the left and right of the tree grid,
        along with the existing rows between the trees.        """
        if len(self.tree_positions) == 0:
            return []

        # Determine field boundaries based on tree positions
        x_min = np.min(self.tree_positions[:, 0]) - offset 
        x_max = np.max(self.tree_positions[:, 0]) + offset
        y_min = np.min(self.tree_positions[:, 1]) - offset
        y_max = np.max(self.tree_positions[:, 1]) + offset

        # Use initial position and orientation
        drone_x,drone_y,_ = self.robot_state_update_thread()


        def find_nearest_vertex(drone_x, drone_y, x_min_inner, x_max_inner, y_min_inner, y_max_inner):
            # Define the four vertices of the inner field
            vertices = [
                (x_min_inner, y_min_inner),  # Bottom-left
                (x_min_inner, y_max_inner),  # Top-left
                (x_max_inner, y_min_inner),  # Bottom-right
                (x_max_inner, y_max_inner)   # Top-right
            ]
            
            min_distance_sq = float('inf')
            nearest_vertex = None
            
            for vertex in vertices:
                dx = drone_x - vertex[0]
                dy = drone_y - vertex[1]
                distance_sq = dx ** 2 + dy ** 2
                # Update if this vertex is closer
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    nearest_vertex = vertex
            
            return nearest_vertex
        # Find the nearest vertex to the drone's position
        nearest_vertex = find_nearest_vertex(drone_x, drone_y, x_min, x_max, y_min, y_max)
        x_start, y_start = nearest_vertex

        # Determine direction for x and y based on the nearest vertex
        x_step = spacing if x_start == x_min else -spacing
        y_step = spacing if y_start == y_min else -spacing

        # Generate x coordinates in the correct order
        if x_step > 0:
            x_coords = np.arange(x_min , x_max + x_step , x_step)
        else:
            x_coords = np.arange(x_max , x_min + x_step , x_step)

        # Generate y coordinates in the correct order
        if y_step > 0:
            y_coords = np.arange(y_min , y_max + y_step, y_step)
        else:
            y_coords = np.arange(y_max , y_min + y_step, y_step)


        # Create grid waypoints

        waypoints = []
        direction_map = {'N': np.pi/2, 'E': 0, 'S': -np.pi/2, 'W': np.pi}
        fixed_heading = np.random.choice(list(direction_map.values()))
        axis = np.random.choice(['x', 'y'])

        if axis == 'x':
            for y in y_coords:
                for x in x_coords:
                    waypoints.append((x, y, fixed_heading))
                x_coords = x_coords[::-1]       
        else:
            for x in x_coords:
                for y in y_coords:
                    waypoints.append((x, y, fixed_heading))
                y_coords = y_coords[::-1]       
        print( fixed_heading , axis)
        return waypoints


    def run_mower_trajectory(self):
        """
        Execute a simple mower trajectory using the generated mower path.

        The robot will follow the sequence of waypoints generated by generate_simple_mower_path,
        always maintaining the fixed heading (e.g. facing east). Performance metrics (e.g. time
        and distance) are recorded along the way.
        """
        self.logger.start_logging()

        # Generate the simple mower path.
        # You can adjust offset, spacing, and heading_direction as needed.
        path = self.generate_mower_path_mower(offset=2.0, spacing=4.0, heading_direction='E', axis='y')
        
        #self.plot_path(path, self.tree_positions)

        total_time = 0
        for i in range(1, len(path)):
            current_point = np.array(path[i][:2])
            heading = path[i][2]
            print(f"Moving to waypoint {i}: ({current_point[0]:.2f}, {current_point[1]:.2f}) with heading: {heading:.2f} rad")

            wp_time, _ = self.move_to_waypoint(
                current_point[0], current_point[1],
                tolerance=.20, observe=True, desired_heading=heading
            )
            total_time += wp_time
        
        final_entropy = self.calculate_entropy(self.lambda_values)
        final_bayes = self.lambda_values.tolist()
        self.logger.finalize_performance_metrics(final_entropy, final_bayes)
        print("Simple mower trajectory complete")

    def generate_linear_path(self):
        if len(self.tree_positions) == 0:
            return []

        tree_indices = list(range(len(self.tree_positions)))
        current_pos = np.array([self.x, self.y])
        path = []
        same_row_tol = 1e-2

        while tree_indices:
            same_row_indices = [
                idx for idx in tree_indices
                if abs(self.tree_positions[idx][1] - current_pos[1]) < same_row_tol
            ]
            if same_row_indices:
                distances = [
                    (idx, abs(self.tree_positions[idx][0] - current_pos[0]))
                    for idx in same_row_indices
                ]
                next_idx, _ = min(distances, key=lambda x: x[1])
            else:
                distances = [
                    (idx, np.linalg.norm(current_pos - self.tree_positions[idx]))
                    for idx in tree_indices
                ]
                next_idx, _ = min(distances, key=lambda x: x[1])
            path.append(next_idx)
            current_pos = self.tree_positions[next_idx]
            tree_indices.remove(next_idx)

        return path

    def get_bayes_value(self, position):
        if len(self.tree_positions) == 0:
            return 0.5
        distances = np.linalg.norm(self.tree_positions - np.array(position), axis=1)
        nearest_index = np.argmin(distances)
        return self.lambda_values[nearest_index]

    def plot_path(self, path, tree_positions):
        tree_x = [tree[0] for tree in tree_positions]
        tree_y = [tree[1] for tree in tree_positions]
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=tree_x, y=tree_y,
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Trees'
        ))

        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            name='Trajectory'
        ))

        fig.add_trace(go.Scatter(
            x=[path_x[0]], y=[path_y[0]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Start'
        ))
        fig.add_trace(go.Scatter(
            x=[path_x[-1]], y=[path_y[-1]],
            mode='markers',
            marker=dict(size=15, color='yellow', symbol='star'),
            name='End'
        ))

        for i, tree in enumerate(tree_positions):
            fig.add_annotation(
                x=tree[0], y=tree[1],
                text=f'T{i}',
                showarrow=False,
                yshift=10
            )

        fig.update_layout(
            title='Drone Trajectory with Tree Observations',
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            showlegend=True
        )

        fig.show()

    def save_plot_data_csv(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baselines_dir = os.path.join(script_dir, self.run_folder)
        if not os.path.exists(baselines_dir):
            os.makedirs(baselines_dir)
        filename = os.path.join(baselines_dir, f"{self.trajectory_type}_{self.logger.timestamp}_plot_data.csv")
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write tree positions (existing)
            tree_positions_flat = self.tree_positions.flatten().tolist()
            writer.writerow(["tree_positions"] + tree_positions_flat)

            writer.writerow(["trees_gt_id"] + self.trees_gt_id.tolist())

            header = ["time", "x", "y", "theta", "entropy"]
            if self.lambda_history and self.lambda_history[0] is not None and len(self.lambda_history[0]) > 0:
                num_lambdas = len(self.lambda_history[0]) # Number of trees based on lambda
                header += [f"lambda_{i}" for i in range(num_lambdas)]
            writer.writerow(header)
            
            for i in range(len(self.time_history)):
                time_val = self.time_history[i]
                # Ensure pose_history[i] is a flat list/array of 3 elements
                current_pose = np.array(self.pose_history[i]).flatten()
                if len(current_pose) == 3:
                    x, y, theta = current_pose
                else: # Fallback or error
                    x, y, theta = 0.0, 0.0, 0.0 
                    rospy.logwarn(f"Unexpected pose format at index {i}: {self.pose_history[i]}")

                entropy_val = self.entropy_history[i]
                lambda_vals = self.lambda_history[i] if self.lambda_history[i] is not None else []
                row = [time_val, x, y, theta, entropy_val] + lambda_vals.tolist()
                writer.writerow(row)
                
        rospy.loginfo("Plot data saved to %s", filename)

    def run_linear_trajectory(self):
            self.logger.start_logging()
            tree_order = self.generate_linear_path()
            tree_path = [self.tree_positions[idx].tolist() for idx in tree_order]
            #self.plot_path(tree_path, self.tree_positions)

            total_time = 0
            total_distance = 0
            current_state = np.array(self.robot_state_update_thread()).flatten()[:2]
            prev_point = current_state

            for idx in tree_order:
                self.idx = idx
                angle = np.pi*(np.random.random()*2 -1)
                tree_pos = self.tree_positions[idx]
                print(f"Moving to tree {idx} at position: ({tree_pos[0]:.2f}, {tree_pos[1]:.2f})")
                wp_time, _ = self.move_to_waypoint(tree_pos[0], tree_pos[1], desired_heading=None, tolerance=2.0)
                total_time += wp_time
                print(f"Observing tree {idx}...")
                # Run observe_tree in a separate thread.
                obs_thread = threading.Thread(target=self.observe_tree)
                obs_thread.start()
                obs_thread.join()  # Optionally, wait for observation to finish before moving on.

                # Alternatively, if you want to continue without waiting:
                # obs_thread.start()
                # ... continue with other tasks
            final_entropy = self.calculate_entropy(self.lambda_values)
            final_bayes = self.lambda_values.tolist()
            self.logger.finalize_performance_metrics(final_entropy, final_bayes)
            print("Tree-to-tree trajectory complete")

    def run_greedy_trajectory(self):
        """
        Greedy approach: iteratively choose the nearest candidate (based on a bayes value)
        from a combined list of mower waypoints and tree positions.
        """
        self.logger.start_logging()

        candidates = []
        for i, tree in enumerate(self.tree_positions):
            bayes = self.get_bayes_value(tree)
            candidates.append({
                'type': 'tree',
                'index': i,
                'position': tree,
                'bayes': bayes
            })
        total_time = 0
        total_distance = 0
        observation_distance_threshold = 2.0

        while candidates:
            current_pos = np.array(self.robot_state_update_thread()).flatten()[:2]
            candidates.sort(key=lambda c: (c['bayes'], np.linalg.norm(np.array(c['position']) - current_pos)))
            next_candidate = candidates[0]
            angle = np.pi*(np.random.random()*2 -1)
            target = np.array(next_candidate['position'])
            target_type = next_candidate['type']
            target_index = next_candidate['index']
            self.idx = next_candidate['index']

            print(f"Moving to {target_type} {target_index} at position: ({target[0]:.2f}, {target[1]:.2f}), bayes: {next_candidate['bayes']:.2f}")
            self.move_to_waypoint(target[0], target[1],
                                    desired_heading=None,
                                    tolerance=observation_distance_threshold,
                                    observe=(target_type == 'tree'))

            if target_type == 'tree':
                print(f"Observing tree {target_index}...")
                # Run observe_tree in a separate thread.
                obs_thread = threading.Thread(target=self.observe_tree)
                obs_thread.start()
                obs_thread.join()  # Optionally, wait for observation to finish before moving on.
            candidates.pop(0)

        final_entropy = self.calculate_entropy(self.lambda_values)
        final_bayes = self.lambda_values.tolist()
        self.logger.finalize_performance_metrics(final_entropy, final_bayes)
        print("Drone trajectory complete (greedy approach)")

    def run(self):
        """
        Main method that runs the trajectory generation based on the specified mode.
        After completion, the trajectory and entropy r
            eduction are plotted.
        """
        if self.trajectory_type == "mower":
            self.is_mower = True
            self.run_mower_trajectory()
        elif self.trajectory_type == "linear":
            self.run_linear_trajectory()
        elif  self.trajectory_type == "greedy":
            self.run_greedy_trajectory()
        
        self.shutdown()
        # Corrected the parameter passed for plotting from an undefined 'mode' to self.trajectory_type.
        #self.plot_animated_trajectory_and_entropy_2d(self.trajectory_type)

    def shutdown(self):
        if hasattr(self, 'measurement_timer'):
            self.measurement_timer.shutdown()

if __name__ == '__main__':
    try:
        # Initialize and run the trajectory generator
        modes = ['greedy', 'linear', 'mower']
        for mode in modes:
            for test_num in range(0, 2):
                import re
                # Define base folder
                base_test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"batch_test_100trees_{mode}_gt")
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
                trajectory_generator = TrajectoryGenerator(mode, run_folder=run_folder)
                trajectory_generator.run()
                trajectory_generator.save_plot_data_csv()
    except rospy.ROSInterruptException:
        pass
