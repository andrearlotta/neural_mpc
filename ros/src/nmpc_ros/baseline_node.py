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
    def __init__(self, trajectory_type, filename='performance_metrics.csv'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
        # Create a filename that includes trajectory type and timestamp
        self.filename = os.path.join(os.path.join(script_dir, "baselines"),
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
        self.velocity_filename = os.path.join(os.path.join(script_dir, "baselines"),
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
    def __init__(self, trajectory_type):
        # Get trajectory mode and initialize Logger only once.
        self.trajectory_type = trajectory_type
        self.logger = Logger(trajectory_type)
        
        # New history lists for storing poses and bayesian (lambda) values
        self.pose_history = []      # Each entry: [x, y, theta]
        self.lambda_history = []    # Each entry: copy of lambda_values at that step
        self.entropy_history = []
        self.time_history = []

        # Initialize PID controllers.
        self.pid_controller_x = PIDController(kp=0.1, kd=0.1)
        self.pid_controller_y = PIDController(kp=0.1, kd=0.1)
        self.pid_controller_yaw = PIDController(kp=0.1, kd=0.1)

        self.cruise_velocity = 0.5  # m/s
        self.circle_radius = 2.3    # Not used in mower trajectory

        # Bridge for robot state and sensor data.
        self.bridge = BridgeClass(SENSORS)
        # Get tree positions from the server.
        self.tree_positions = np.array(self.bridge.call_server({"trees_poses": None})["trees_poses"])
        self.x, self.y, self.theta = np.array(self.bridge.update_robot_state()).flatten()
        self.lambda_values = np.full(len(self.tree_positions), 0.5)  # Initialize lambda

        self.bridge.pub_data({
            "tree_markers": {"trees_pos": self.tree_positions, "lambda": self.lambda_values}
        })

        self.safe_distance = 2.5         # Minimum safe distance from trees (in meters)
        self.repulsive_gain = 5.0        # Gain for the repulsive force (unused in new motion vector)
        self.attractive_gain = 1.0       # Gain for the attractive force (unused in new motion vector)
        self.max_repulsive_distance = 5.0  # Maximum distance for repulsive effect to take place
        self.max_velocity = 1.0          # Maximum velocity of the robot

        self.dt = 0.25
        self.rate = rospy.Rate(int(1/self.dt))  # Control loop rate

        # Maximum time allowed for observation around a tree (in seconds)
        self.max_observe_time = 30.0

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
        if att_magnitude < 1e-6:
            return np.zeros(2)
        att_unit = att_vector / att_magnitude

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
                if angle < np.pi / 4:
                    unsafe = True
                    u = diff / d
                    tangent1 = np.array([-u[1], u[0]])
                    tangent2 = np.array([u[1], -u[0]])
                    chosen_tangent = tangent1 if np.dot(tangent1, att_unit) > np.dot(tangent2, att_unit) else tangent2
                    deviation += chosen_tangent * ray_avoidance - 2 * u

        if not unsafe:
            return att_unit * att_magnitude
        else:
            combined = att_unit + 3 * deviation
            combined_norm = np.linalg.norm(combined)
            if combined_norm < 1e-6:
                return att_unit * att_magnitude
            return combined / combined_norm

    def calculate_attractive_force(self, current_position, goal_position):
        distance = np.linalg.norm(goal_position - current_position)
        return self.attractive_gain * (goal_position - current_position) / distance

    def calculate_repulsive_force(self, current_position, tree_positions):
        repulsive_force = np.zeros(2)
        for tree in tree_positions:
            tree = np.array(tree)
            distance = np.linalg.norm(current_position - tree)
            if distance < self.max_repulsive_distance:
                force_magnitude = self.repulsive_gain * (1 / distance - 1 / self.max_repulsive_distance) * (1 / distance**2)
                force_direction = (current_position - tree) / distance
                repulsive_force += force_magnitude * force_direction
        return repulsive_force

    def move_to_waypoint(self, target_x, target_y, tolerance=2.0, observe=False):
        """
        Move the robot toward the waypoint. Runs until the goal is reached (within tolerance)
        and then returns the execution time (and entropy reduction if observe=True).
        """
        goal_position = np.array([target_x, target_y])
        start_time = time.time()

        while not rospy.is_shutdown():
            state = np.array(self.bridge.update_robot_state()).flatten()
            self.x, self.y, self.theta = state
            self.pose_history.append([self.x, self.y, self.theta])
            self.lambda_history.append(self.lambda_values.copy())
            self.entropy_history.append(self.calculate_entropy(self.lambda_values))
            self.time_history.append(time.time() - self.logger.start_time)
            
            current_position = np.array([self.x, self.y])
            if np.linalg.norm(current_position - goal_position) < tolerance:
                break

            motion_vector = self.calculate_motion_vector(current_position, goal_position)
            desired_theta = math.atan2(goal_position[1] - current_position[1],
                                       goal_position[0] - current_position[0])
            yaw_error = self.normalize_angle(desired_theta - self.theta)

            # Scale the motion vector to the maximum velocity.
            velocity = (motion_vector / np.linalg.norm(motion_vector)) * self.max_velocity

            x_output = self.pid_controller_x.update(velocity[0])
            y_output = self.pid_controller_y.update(velocity[1])
            yaw_output = self.pid_controller_yaw.update(yaw_error)
            
            # Log the injected velocities.
            self.logger.log_velocity(x_output, y_output, yaw_output, tag="move_to_waypoint")

            cmd_pose = np.array([self.x + x_output, self.y + y_output, self.theta + yaw_output]).reshape(-1, 1)
            self.bridge.pub_data({"cmd_pose": cmd_pose})

            if observe:
                new_data = self.bridge.get_data()
                while new_data["tree_scores"] is None:
                    new_data = self.bridge.get_data()
                z_k = new_data["tree_scores"].flatten()
    
                self.lambda_values = self.bayes(self.lambda_values, z_k)
                self.bridge.pub_data({
                    "tree_markers": {"trees_pos": self.tree_positions, "lambda": self.lambda_values}
                })
            self.rate.sleep()

        execution_time = time.time() - start_time
        entropy_reduction = self.calculate_entropy(self.lambda_values) if observe else None
        return execution_time, entropy_reduction

    def observe_tree(self, tree_idx):
        """
        Circle the tree from the current position and update lambda until it 
        reaches saturation (λ>=1.0) or until max_observe_time is reached.
        """
        center_x, center_y = self.tree_positions[tree_idx]
        start_time = time.time()

        self.x, self.y, self.theta = np.array(self.bridge.update_robot_state()).flatten()
        phi = math.atan2(self.y - center_y, self.x - center_x)
        radius = np.sqrt((self.x - center_x)**2 + (self.y - center_y)**2)
        delta_phi = 3 * math.pi / 180  # Angular increment (in radians)

        while (time.time() - start_time) < self.max_observe_time and self.lambda_values[tree_idx] < 1.0:
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
            
            # Log the injected velocities for tree observation.
            self.logger.log_velocity(x_output, y_output, yaw_output, tag=f"observe_tree_{tree_idx}")

            cmd_pose = np.array([self.x + x_output, self.y + y_output, self.theta + yaw_output]).reshape(-1, 1)
            self.bridge.pub_data({"cmd_pose": cmd_pose})

            new_data = self.bridge.get_data()
            while new_data["tree_scores"] is None:
                new_data = self.bridge.get_data()
            z_k = new_data["tree_scores"]
            self.lambda_values[tree_idx] = self.bayes(self.lambda_values[tree_idx].flatten(), z_k[tree_idx].flatten())
            self.bridge.pub_data({
                "tree_markers": {"trees_pos": self.tree_positions, "lambda": self.lambda_values}
            })

            phi += delta_phi
            rospy.sleep(self.dt)

        return time.time() - start_time

    def bayes(self, lambda_prev, z):
        """Bayesian update for confidence (lambda)."""
        z = np.clip(z, 0.5, 1.0 - 1e-6)
        numerator = lambda_prev * z
        denominator = numerator + (1 - lambda_prev) * (1 - z)
        return numerator / denominator

    def generate_mower_trajectory(self, spacing=1.0, margin=1.0):
        if len(self.tree_positions) == 0:
            return []

        x_min = np.min(self.tree_positions[:, 0]) - margin
        x_max = np.max(self.tree_positions[:, 0]) + margin
        y_min = np.min(self.tree_positions[:, 1]) - margin
        y_max = np.max(self.tree_positions[:, 1]) + margin

        waypoints = []
        y_vals = np.arange(y_min, y_max, spacing)
        for i, y in enumerate(y_vals):
            if i % 2 == 0:
                waypoints.append([x_min, y])
                waypoints.append([x_max, y])
            else:
                waypoints.append([x_max, y])
                waypoints.append([x_min, y])
        return waypoints

    def generate_mower_path_between_rows(self, margin=1.0):
        if len(self.tree_positions) == 0:
            return []
        current_pos = np.array(self.bridge.update_robot_state()).flatten()[:2]
        sorted_trees = self.tree_positions[self.tree_positions[:, 1].argsort()]
        rows = []
        current_row = [sorted_trees[0]]
        threshold = 0.5
        for tree in sorted_trees[1:]:
            if abs(tree[1] - current_row[-1][1]) < threshold:
                current_row.append(tree)
            else:
                rows.append(np.array(current_row))
                current_row = [tree]
        if current_row:
            rows.append(np.array(current_row))

        if len(rows) < 2:
            return self.generate_mower_trajectory(margin=margin)

        gaps = []
        for i in range(len(rows) - 1):
            y_avg_1 = np.mean(rows[i][:, 1])
            y_avg_2 = np.mean(rows[i + 1][:, 1])
            gap_y = (y_avg_1 + y_avg_2) / 2
            gaps.append(gap_y)

        x_min = np.min(self.tree_positions[:, 0]) - margin
        x_max = np.max(self.tree_positions[:, 0]) + margin

        waypoints = []
        waypoints.append(current_pos)
        robot_x = current_pos[0]
        robot_y = current_pos[1]
        if abs(robot_x - x_min) <= abs(robot_x - x_max):
            start_side = x_min
        else:
            start_side = x_max

        waypoints.append([start_side, robot_y])
        current_side = start_side
        for gap in gaps:
            waypoints.append([current_side, gap])
            other_side = x_max if current_side == x_min else x_min
            waypoints.append([other_side, gap])
            current_side = other_side

        return waypoints

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

    def save_plot_data_csv(self, trajectory_type):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baselines_dir = os.path.join(script_dir, "baselines")
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
                        color=[2 * (sampled_lambda_history[-1][i] - 0.5)],
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
    
    def run_greedy_trajectory(self):
        """
        Greedy approach: iteratively choose the nearest candidate (based on a bayes value)
        from a combined list of mower waypoints and tree positions.
        """
        self.logger.start_logging()
        mower_path = self.generate_mower_trajectory(spacing=1.0, margin=1.0)

        candidates = []
        for i, waypoint in enumerate(mower_path):
            bayes = self.get_bayes_value(waypoint)
            candidates.append({
                'type': 'mower',
                'index': i,
                'position': waypoint,
                'bayes': bayes
            })
        for i, tree in enumerate(self.tree_positions):
            bayes = self.get_bayes_value(tree)
            candidates.append({
                'type': 'tree',
                'index': i,
                'position': tree,
                'bayes': bayes
            })

        self.plot_path(mower_path, self.tree_positions)

        total_time = 0
        total_distance = 0
        current_pos = np.array(self.bridge.update_robot_state()).flatten()[:2]
        observation_distance_threshold = 3.0

        while candidates:
            candidates.sort(key=lambda c: (c['bayes'], np.linalg.norm(np.array(c['position']) - current_pos)))
            next_candidate = candidates[0]
            target = np.array(next_candidate['position'])
            target_type = next_candidate['type']
            target_index = next_candidate['index']

            print(f"Moving to {target_type} {target_index} at position: ({target[0]:.2f}, {target[1]:.2f}), bayes: {next_candidate['bayes']:.2f}")
            wp_time, entropy_reduction = self.move_to_waypoint(target[0], target[1],
                                                               tolerance=observation_distance_threshold,
                                                               observe=(target_type == 'tree'))
            total_time += wp_time
            step_distance = np.linalg.norm(target - current_pos)
            total_distance += step_distance
            self.logger.add_distance(step_distance)
            current_pos = target
            self.logger.record_performance(f"{target_type}{target_index}", wp_time, total_distance, wp_time,
                                           entropy_reduction, position=target)

            if target_type == 'tree':
                print(f"Observing tree {target_index}...")
                observe_time = self.observe_tree(target_index)
                total_time += observe_time
                self.logger.record_performance(f"T_obs_{target_index}", observe_time, total_distance, observe_time,
                                               position=self.tree_positions[target_index])

            candidates.pop(0)
        
        final_entropy = self.calculate_entropy(self.lambda_values)
        final_bayes = self.lambda_values.tolist()
        self.logger.finalize_performance_metrics(final_entropy, final_bayes)
        print("Drone trajectory complete (greedy approach)")

    def run_between_rows_trajectory(self):
        """
        Run mode: follow a path that takes the drone between the rows of trees.
        """
        self.logger.start_logging()
        mower_path = self.generate_mower_path_between_rows(margin=1.0)
        self.plot_path(mower_path, self.tree_positions)

        total_time = 0
        total_distance = 0
        current_state = np.array(self.bridge.update_robot_state()).flatten()[:2]
        prev_point = current_state

        for i, waypoint in enumerate(mower_path):
            target_x, target_y = waypoint
            print(f"Moving to between-row waypoint {i}: ({target_x:.2f}, {target_y:.2f})")
            wp_time, entropy_reduction = self.move_to_waypoint(target_x, target_y, tolerance=0.1, observe=True)
            total_time += wp_time
            step_distance = np.linalg.norm(np.array(waypoint) - prev_point)
            total_distance += step_distance
            self.logger.add_distance(step_distance)
            prev_point = np.array(waypoint)
            self.logger.record_performance(i, wp_time, total_distance, wp_time, entropy_reduction, position=waypoint)
            
        final_entropy = self.calculate_entropy(self.lambda_values)
        final_bayes = self.lambda_values.tolist()
        self.logger.finalize_performance_metrics(final_entropy, final_bayes)
        print("Between-rows trajectory complete")

    def run_tree_to_tree_trajectory(self):
        """
        Run mode: move from tree to tree (nearest-neighbor ordering). Upon reaching a tree,
        perform the observation procedure before continuing.
        """
        self.logger.start_logging()
        tree_order = self.generate_tree_to_tree_path()
        tree_path = [self.tree_positions[idx].tolist() for idx in tree_order]
        self.plot_path(tree_path, self.tree_positions)

        total_time = 0
        total_distance = 0
        current_state = np.array(self.bridge.update_robot_state()).flatten()[:2]
        prev_point = current_state

        for idx in tree_order:
            tree_pos = self.tree_positions[idx]
            print(f"Moving to tree {idx} at position: ({tree_pos[0]:.2f}, {tree_pos[1]:.2f})")
            wp_time, _ = self.move_to_waypoint(tree_pos[0], tree_pos[1], tolerance=3)
            total_time += wp_time
            step_distance = np.linalg.norm(tree_pos - prev_point)
            total_distance += step_distance
            self.logger.add_distance(step_distance)
            prev_point = tree_pos
            self.logger.record_performance(f"T{idx}", wp_time, total_distance, wp_time, position=tree_pos)
            
            print(f"Observing tree {idx}...")
            observe_time = self.observe_tree(idx)
            total_time += observe_time
            self.logger.record_performance(f"T_obs_{idx}", observe_time, total_distance, observe_time, position=tree_pos)
            
        final_entropy = self.calculate_entropy(self.lambda_values)
        final_bayes = self.lambda_values.tolist()
        self.logger.finalize_performance_metrics(final_entropy, final_bayes)
        print("Tree-to-tree trajectory complete")

    def run(self):
        """
        Main method that runs the trajectory generation based on the specified mode.
        After completion, the trajectory and entropy reduction are plotted.
        """

        if self.trajectory_type == "between_rows":
            self.run_between_rows_trajectory()
        elif self.trajectory_type == "tree_to_tree":
            self.run_tree_to_tree_trajectory()
        else:
            self.run_greedy_trajectory()

        self.plot_animated_trajectory_and_entropy_2d(mode)


if __name__ == '__main__':
    try:
        mode = rospy.get_param('~trajectory_mode', 'between_rows')
        trajectory_generator = TrajectoryGenerator(mode)
        trajectory_generator.run()
    except rospy.ROSInterruptException:
        pass
