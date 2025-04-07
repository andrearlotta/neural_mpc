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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nmpc_ros_package.ros_com_lib.bridge_class import BridgeClass
from nmpc_ros_package.ros_com_lib.sensors import SENSORS


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
    def __init__(self, trajectory_type, filename='performance_metrics.csv', save_dir='baselines'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
        # Create a filename that includes trajectory type and timestamp
        self.filename = os.path.join(os.path.join(script_dir, save_dir),
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
        self.velocity_filename = os.path.join(os.path.join(script_dir, save_dir),
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
    def __init__(self, trajectory_type, save_dir='baselines'):
        # Get trajectory mode and initialize Logger only once.
        self.trajectory_type = trajectory_type
        self.logger = Logger(trajectory_type)

        # New history lists for storing poses and bayesian (lambda) values
        self.pose_history = []      # Each entry: [x, y, theta]
        self.lambda_history = []    # Each entry: copy of lambda_values at that step
        self.entropy_history = []
        self.time_history = []

        # Initialize PID controllers.
        self.pid_controller_x = PIDController(kp=3.0, kd=1)
        self.pid_controller_y = PIDController(kp=3.0, kd=1)
        self.pid_controller_yaw = PIDController(kp=.25, kd=0.1)

        self.idx = None
        self.is_mower = False
        self.circle_tree_event = threading.Event()
        # Bridge for robot state and sensor data.
        self.bridge = BridgeClass(SENSORS)
        # Get tree positions from the server.
        self.tree_positions = np.array(self.bridge.call_server({"trees_poses": None})["trees_poses"])
        self.x, self.y, self.theta = np.array(self.bridge.update_robot_state()).flatten()
        self.lambda_values = np.full(len(self.tree_positions), 0.5)  # Initialize lambda

        self.bridge.pub_data({
            "tree_markers": {"trees_pos": self.tree_positions, "lambda": self.lambda_values}
        })

        self.max_velocity = 0.50 if self.is_mower else 1.5          # Maximum velocity of the robot
        self.max_yaw_velocity = np.pi/4      # Maximum yaw velocity (rad/s)

        self.dt = 0.1
        self.rate = rospy.Rate(int(1/self.dt))  # Control loop rate

        # Maximum time allowed for observation around a tree (in seconds)
        self.max_observe_time = 30.0
        self.rate = rospy.Rate(int(1 / self.dt))
        self.measurement_timer = rospy.Timer(rospy.Duration(0.5), self.measurement_callback)
        while self.bridge.get_data()["tree_scores"] is None:
            pass

    def measurement_callback(self, event):
        if not (self.is_mower or self.circle_tree_event.is_set()):
            return
        new_data = self.bridge.get_data()
        if new_data is None or new_data.get("tree_scores") is None:
            return
        z_k = new_data["tree_scores"].flatten()

        if self.is_mower:
            self.lambda_values = self.bayes(self.lambda_values, z_k)
        elif self.idx is not None:
            self.lambda_values[self.idx] = self.bayes(self.lambda_values[self.idx], z_k[self.idx])
            # When the lambda value reaches a threshold, clear the event to stop observation.
            if self.lambda_values[self.idx] * 1e3 >= 999:
                self.circle_tree_event.clear()

        current_entropy = self.calculate_entropy(self.lambda_values)
        self.bridge.pub_data({
            "tree_markers": {"trees_pos": self.tree_positions, "lambda": self.lambda_values}
        })

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
            if (tree == goal_position).all():
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
    def move_to_waypoint(self, target_x, target_y, tolerance=2.0, observe=False, desired_heading=None):
        """
        Move the drone to the specified waypoint.
        An optional desired_heading (in radians) can be provided to override the default
        heading computed from the goal position.
        """
        goal_position = np.array([target_x, target_y])
        start_time = time.time()

        while not rospy.is_shutdown():
            current_state = np.array(self.bridge.update_robot_state()).flatten()
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
            self.bridge.pub_data({"cmd_pose": cmd_pose})

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

        self.x, self.y, self.theta = np.array(self.bridge.update_robot_state()).flatten()
        phi = math.atan2(self.y - center_y, self.x - center_x)
        radius = np.sqrt((self.x - center_x) ** 2 + (self.y - center_y) ** 2)
        delta_phi = self.dt * 5.0 * math.pi / 180  # Angular increment (in radians)

        while self.circle_tree_event.is_set():
            desired_x = center_x + radius * math.cos(phi)
            desired_y = center_y + radius * math.sin(phi)
            desired_theta = phi + math.pi  # Face the tree center

            self.x, self.y, self.theta = np.array(self.bridge.update_robot_state()).flatten()
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
            self.bridge.pub_data({"cmd_pose": cmd_pose})
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
    
    def generate_mower_path_between_rows(self, offset=2.0, spacing=4.0, heading_direction='O', axis='x'):
        """
        Generate a mower path that includes external rows outside the tree grid.

        The path now includes external rows to the left and right of the tree grid,
        along with the existing rows between the trees.
        """
        if len(self.tree_positions) == 0:
            return []

        # Determine field boundaries based on tree positions
        x_min = np.min(self.tree_positions[:, 0]) - offset 
        x_max = np.max(self.tree_positions[:, 0]) + offset
        y_min = np.min(self.tree_positions[:, 1]) - offset
        y_max = np.max(self.tree_positions[:, 1]) + offset

        [drone_x],[drone_y],[_] = self.bridge.update_robot_state()

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
            x_coords = np.arange(x_min, x_max + 1e-9, x_step)
        else:
            x_coords = np.arange(x_max, x_min - 1e-9, x_step)

        # Generate y coordinates in the correct order
        if y_step > 0:
            y_coords = np.arange(y_min, y_max + 1e-9, y_step)
        else:
            y_coords = np.arange(y_max, y_min - 1e-9, y_step)

        # Create grid waypoints: iterate over y first (rows), then x (columns)
        waypoints = []
        direction_map = {'N': np.pi/2, 'E': 0, 'S': -np.pi/2, 'W': np.pi}
        fixed_heading = direction_map.get(heading_direction.upper(), 0)
        

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

        return waypoints

    def run_between_rows_trajectory(self):
        """
        Execute a simple mower trajectory using the generated mower path.

        The robot will follow the sequence of waypoints generated by generate_simple_mower_path,
        always maintaining the fixed heading (e.g. facing east). Performance metrics (e.g. time
        and distance) are recorded along the way.
        """
        self.logger.start_logging()

        # Generate the simple mower path.
        # You can adjust offset, spacing, and heading_direction as needed.
        path = self.generate_mower_path_between_rows(offset=2.0, spacing=4.0, heading_direction='E', axis='y')
        self.plot_path(path, self.tree_positions)

        total_time = 0
        for i in range(1, len(path)):
            current_point = np.array(path[i][:2])
            heading = path[i][2]
            print(f"Moving to waypoint {i}: ({current_point[0]:.2f}, {current_point[1]:.2f}) with heading: {heading:.2f} rad")

            wp_time, _ = self.move_to_waypoint(
                current_point[0], current_point[1],
                tolerance=0.25, observe=True, desired_heading=heading
            )
            total_time += wp_time
        
        final_entropy = self.calculate_entropy(self.lambda_values)
        final_bayes = self.lambda_values.tolist()
        self.logger.finalize_performance_metrics(final_entropy, final_bayes)
        print("Simple mower trajectory complete")

    def generate_tree_to_tree_path(self):
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

    def save_plot_data_csv(self, trajectory_type, save_dir='baselines'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baselines_dir = os.path.join(script_dir, save_dir)
        if not os.path.exists(baselines_dir):
            os.makedirs(baselines_dir)
        filename = os.path.join(baselines_dir, f"{trajectory_type}_{self.logger.timestamp}_plot_data.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            tree_positions_flat = self.tree_positions.flatten().tolist()
            writer.writerow(["tree_positions"] + tree_positions_flat)

            header = ["time", "x", "y", "theta", "entropy"]
            if len(self.lambda_history) > 0:
                num_trees = len(self.lambda_history[0])
                header += [f"lambda_{i}" for i in range(num_trees)]
            writer.writerow(header)

            for i in range(len(self.time_history)):
                time_val = self.time_history[i]
                x, y, theta = self.pose_history[i]
                entropy = self.entropy_history[i]
                lambda_vals = list(self.lambda_history[i])
                row = [time_val, x, y, theta, entropy] + lambda_vals
                writer.writerow(row)
        print(f"Plot data saved to {filename}")

    def plot_animated_trajectory_and_entropy_2d(self, trajectory_type):
        self.save_plot_data_csv(trajectory_type)

        x_trajectory = np.array([traj[0] for traj in self.pose_history])
        y_trajectory = np.array([traj[1] for traj in self.pose_history])
        theta_trajectory = np.array([traj[2] for traj in self.pose_history])
        self.pose_history = np.array(self.pose_history)
        lambda_history = np.array(self.lambda_history)
        time_history = np.array(self.time_history)

        x_min = np.min(self.tree_positions[:, 0])
        x_max = np.max(self.tree_positions[:, 0])
        y_min = np.min(self.tree_positions[:, 1])
        y_max = np.max(self.tree_positions[:, 1])
        lb, ub = [x_min, y_min], [x_max, y_max]

        fig_static = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )

        fig_static.add_trace(
            go.Scatter(
                x=x_trajectory,
                y=y_trajectory,
                mode="lines+markers",
                name="Drone Trajectory",
                line=dict(color="orange", width=4),
                marker=dict(size=5, color="orange")
            ),
            row=1, col=1
        )

        for i in range(self.tree_positions.shape[0]):
            fig_static.add_trace(
                go.Scatter(
                    x=[self.tree_positions[i, 0]],
                    y=[self.tree_positions[i, 1]],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=[2 * (lambda_history[-1][i] - 0.5)],
                        colorscale=[[0, "#FF0000"], [1, "#00FF00"]],
                        cmin=0,
                        cmax=1,
                        showscale=False
                    ),
                    name=f"Tree {i}: {lambda_history[-1][i]:.2f}",
                    text=[str(i)],
                    textposition="top center"
                ),
                row=1, col=1
            )

        fig_static.add_trace(
            go.Scatter(
                x=time_history,
                y=self.entropy_history,
                mode="lines+markers",
                name="Sum of Entropies",
                line=dict(color="blue", width=2),
                marker=dict(size=5, color="blue")
            ),
            row=1, col=2
        )

        fig_static.update_layout(
            title=f"Complete Drone Trajectory and Sum of Entropies: {trajectory_type} baseline ",
            xaxis=dict(title="X Position", range=[lb[0] - 3, ub[0] + 3]),
            yaxis=dict(title="Y Position", range=[lb[1] - 3, ub[1] + 3]),
            xaxis2=dict(title="Time (s)"),
            yaxis2=dict(title="Sum of Entropies"),
        )

        arrow_length = 0.5
        arrow_annotations = []
        for x0, y0, theta in zip(x_trajectory, y_trajectory, theta_trajectory):
            x1 = x0 + arrow_length * np.cos(theta)
            y1 = y0 + arrow_length * np.sin(theta)
            arrow_annotations.append(
                go.layout.Annotation(
                    x=x1,
                    y=y1,
                    xref="x", yref="y",
                    ax=x0,
                    ay=y0,
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowwidth=1.5,
                    arrowcolor="orange"
                )
            )
        fig_static.update_layout(annotations=arrow_annotations)
        fig_static.show()

        sample_rate = 10
        sampled_indices = range(0, len(self.entropy_history), sample_rate)
        sampled_time_history = time_history[sampled_indices]
        sampled_entropy_history = np.array(self.entropy_history)[sampled_indices]
        sampled_x_trajectory = x_trajectory[sampled_indices]
        sampled_y_trajectory = y_trajectory[sampled_indices]
        sampled_theta_trajectory = theta_trajectory[sampled_indices]
        sampled_lambda_history = lambda_history[sampled_indices]

        fig_animated = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )

        fig_animated.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Drone Trajectory",
                line=dict(color="orange", width=4),
                marker=dict(size=5, color="orange")
            ),
            row=1, col=1
        )

        for i in range(self.tree_positions.shape[0]):
            fig_animated.add_trace(
                go.Scatter(
                    x=[self.tree_positions[i, 0]],
                    y=[self.tree_positions[i, 1]],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color="#FF0000",
                        colorscale=[[0, "#FF0000"], [1, "#00FF00"]],
                        cmin=0,
                        cmax=1,
                        showscale=False
                    ),
                    name=f"Tree {i}: {sampled_lambda_history[0][i]:.2f}",
                    text=[str(i)],
                    textposition="top center"
                ),
                row=1, col=1
            )

        fig_animated.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name="Sum of Entropies",
                line=dict(color="blue", width=2),
                marker=dict(size=5, color="blue")
            ),
            row=1, col=2
        )

        frames = []
        for k in range(len(sampled_entropy_history)):
            tree_data = []
            for i in range(self.tree_positions.shape[0]):
                tree_data.append(
                    go.Scatter(
                        x=[self.tree_positions[i, 0]],
                        y=[self.tree_positions[i, 1]],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color=[2 * (sampled_lambda_history[k][i] - 0.5)],
                            colorscale=[[0, "#FF0000"], [1, "#00FF00"]],
                            cmin=0,
                            cmax=1,
                            showscale=False
                        ),
                        name=f"Tree {i}: {sampled_lambda_history[k][i]:.2f}",
                        text=[str(i)],
                        textposition="top center"
                    )
                )

            sum_entropy_past = sampled_entropy_history[:k + 1]
            x_start_traj = sampled_x_trajectory[:k + 1]
            y_start_traj = sampled_y_trajectory[:k + 1]
            theta_traj = sampled_theta_trajectory[:k + 1]
            x_end_traj = x_start_traj + 0.5 * np.cos(theta_traj)
            y_end_traj = y_start_traj + 0.5 * np.sin(theta_traj)
            list_of_actual_orientations = []
            for x0, y0, x1, y1 in zip(x_start_traj, y_start_traj, x_end_traj, y_end_traj):
                arrow = go.layout.Annotation(
                    dict(
                        x=x1,
                        y=y1,
                        xref="x", yref="y",
                        text="",
                        showarrow=True,
                        axref="x", ayref="y",
                        ax=x0,
                        ay=y0,
                        arrowhead=3,
                        arrowwidth=1.5,
                        arrowcolor="orange",
                    )
                )
                list_of_actual_orientations.append(arrow)

            time_label = go.layout.Annotation(
                dict(
                    x=0.95,
                    y=0.95,
                    xref="paper", yref="paper",
                    text=f"Time: {sampled_time_history[k]:.2f} s",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            )

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=sampled_x_trajectory[:k + 1],
                        y=sampled_y_trajectory[:k + 1],
                        mode="lines+markers",
                        line=dict(color="orange", width=4),
                        marker=dict(size=5, color="orange")
                    ),
                    *tree_data,
                    go.Scatter(
                        x=sampled_time_history[:k + 1],
                        y=sum_entropy_past,
                        mode="lines+markers",
                        line=dict(color="blue", width=2),
                        marker=dict(size=5, color="blue")
                    ),
                ],
                name=f"Frame {k}",
                layout=dict(annotations=[*list_of_actual_orientations, time_label])
            )
            frames.append(frame)

        fig_animated.frames = frames

        fig_animated.update_layout(
            title=f"Drone Trajectory and Sum of Entropies: {trajectory_type} baseline ",
            xaxis=dict(title="X Position", range=[lb[0] - 3, ub[0] + 3]),
            yaxis=dict(title="Y Position", range=[lb[1] - 3, ub[1] + 3]),
            xaxis2=dict(title="Time (s)"),
            yaxis2=dict(title="Sum of Entropies"),
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
                    "prefix": "Frame Time: ",
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
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "mode": "immediate",
                                "layout": {"annotations": f.layout.annotations}
                            }
                        ],
                        "label": f"{sampled_time_history[k]:.2f} s",
                        "method": "animate",
                    }
                    for k, f in enumerate(frames)
                ],
            }]
        )

        fig_animated.show()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(os.path.join(script_dir, "baselines"),
                                f"{trajectory_type}_{self.logger.timestamp}_baseline_results.html")
        fig_animated.write_html(filename)

    def run_tree_to_tree_trajectory(self):
            self.logger.start_logging()
            tree_order = self.generate_tree_to_tree_path()
            tree_path = [self.tree_positions[idx].tolist() for idx in tree_order]
            self.plot_path(tree_path, self.tree_positions)

            total_time = 0
            total_distance = 0
            current_state = np.array(self.bridge.update_robot_state()).flatten()[:2]
            prev_point = current_state

            for idx in tree_order:
                self.idx = idx
                tree_pos = self.tree_positions[idx] + np.array([np.cos(np.pi/2), np.sin(np.pi/2)]) * 2
                print(f"Moving to tree {idx} at position: ({tree_pos[0]:.2f}, {tree_pos[1]:.2f})")
                wp_time, _ = self.move_to_waypoint(tree_pos[0], tree_pos[1], desired_heading=-np.pi/2, tolerance=0.2)
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
        current_pos = np.array(self.bridge.update_robot_state()).flatten()[:2]
        observation_distance_threshold = .20

        while candidates:
            candidates.sort(key=lambda c: (c['bayes'], np.linalg.norm(np.array(c['position']) - current_pos)))
            next_candidate = candidates[0]
            target = np.array(next_candidate['position']) + np.array([np.cos(np.pi/2), np.sin(np.pi/2)]) * 2
            target_type = next_candidate['type']
            target_index = next_candidate['index']
            self.idx = next_candidate['index']

            print(f"Moving to {target_type} {target_index} at position: ({target[0]:.2f}, {target[1]:.2f}), bayes: {next_candidate['bayes']:.2f}")
            self.move_to_waypoint(target[0], target[1],
                                    desired_heading=-np.pi/2,
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
        After completion, the trajectory and entropy reduction are plotted.
        """
        if self.trajectory_type == "between_rows":
            self.is_mower = True
            self.run_between_rows_trajectory()
        elif self.trajectory_type == "tree_to_tree":
            self.run_tree_to_tree_trajectory()
        else:
            self.run_greedy_trajectory()

        # Corrected the parameter passed for plotting from an undefined 'mode' to self.trajectory_type.
        self.plot_animated_trajectory_and_entropy_2d(self.trajectory_type)


if __name__ == '__main__':
    try:
        mode = rospy.get_param('~trajectory_mode', 'greedy')
        trajectory_generator = TrajectoryGenerator(mode, save_dir='results/random_field')
        trajectory_generator.run()
    except rospy.ROSInterruptException:
        pass
