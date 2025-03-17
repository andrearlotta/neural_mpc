#!/usr/bin/env python
import os
import re
import time
import csv
import math
import rospy

import casadi as ca
import numpy as np
import torch
from scipy.stats import norm 

from nmpc_ros_package.ros_com_lib.bridge_class import BridgeClass
from nmpc_ros_package.ros_com_lib.sensors import SENSORS

import l4casadi as l4c

# ---------------------------
# Global Constants
# ---------------------------
hidden_size = 64
hidden_layers = 3
nn_input_dim = 3

N = 5
dt = 0.5
T = dt * N
nx = 3  # Represents [x, y, theta]; state X is 6-D ([pos; vel])

# ---------------------------
# Simple Neural Network
# ---------------------------
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim):
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

def get_latest_best_model():
    # Get the directory where THIS script is stored
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

# ---------------------------
# Domain and Dynamics Functions
# ---------------------------
def get_domain(tree_positions):
    """Return the domain (bounding box) of the tree positions."""
    x_min = np.min(tree_positions[:, 0])
    x_max = np.max(tree_positions[:, 0])
    y_min = np.min(tree_positions[:, 1])
    y_max = np.max(tree_positions[:, 1])
    return [x_min, y_min], [x_max, y_max]

def kin_model():
    """
    Kinematic model: state X = [x, y, theta, vx, vy, omega] and control U = [ax, ay, angular_acc].
    Uses simple Euler integration.
    """
    X = ca.MX.sym('X', nx * 2)  # 6 states
    U = ca.MX.sym('U', nx)      # 3 controls
    rhs = ca.vertcat(X[nx:], U)
    f = ca.Function('f', [X,U], [rhs])
    intg_opts = {"number_of_finite_elements":1, "simplify":1}
    intg = ca.integrator('intg', 'rk', {'x':X, 'p':U, 'ode':f(X,U)}, 0, dt, intg_opts)
    xf = intg(x0=X, p=U)['xf']
    return ca.Function('F', [X, U], [xf])

def bayes(lambda_prev, z):
    """
    Bayesian update for belief:
       lambda_next = (lambda_prev * z) / (lambda_prev * z + (1 - lambda_prev) * (1 - z) + epsilon)
    """
    prod = lambda_prev*z
    denom = prod + (1 - lambda_prev)* (1 - z)
    return prod / denom

def entropy(p):
    """
    Compute the binary entropy of a probability p.
    Values are clipped to avoid log(0).
    """
    p = ca.fmin(p, 1 - 1e-6)
    return (-p*ca.log10(p) - (1 - p)*ca.log10(1 - p)) / ca.log10(2)

# =============================================================================
# Modified MPC Optimization Function
# =============================================================================
def mpc_opt(g_nn, trees, lb, ub, x0, lambda_vals, steps=10):
    """
    Set up and solve the MPC optimization problem with an added objective to attract
    the agent towards low-entropy trees when entropy reduction is insufficient.
    """
    nx_local = 3                   # For clarity in this function
    n_state = nx_local * 2         # 6-dimensional state: [x, y, theta, vx, vy, omega]
    n_control = nx_local           # 3-dimensional control: [ax, ay, angular_acc]
    opti = ca.Opti()
    F_ = kin_model()  # kinematic model function

    # Decision variables:
    X = opti.variable(n_state, steps + 1)
    U = opti.variable(n_control, steps)
    
    # Parameter vector: initial state and tree beliefs.
    num_trees = trees.shape[0]
    P0 = opti.parameter(n_state + num_trees)
    X0 = P0[: n_state]
    L0 = P0[n_state:]
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

    # Loop over the prediction horizon.
    for i in range(steps+1):
        # State and input bounds.
        opti.subject_to(opti.bounded(lb[0] - 3.,   X[0, i], ub[0] + 3.))
        opti.subject_to(opti.bounded(lb[1] - 3.,   X[1, i], ub[1] + 3.))
        opti.subject_to(opti.bounded(-6*np.pi,    X[2, i], +6*np.pi))

        opti.subject_to(opti.bounded(0.0, ca.sumsqr(X[3:5, i]), 4.0))
        opti.subject_to(opti.bounded(-3.14/4,   X[5, i], 3.14 / 4))

        if i < steps: opti.subject_to(opti.bounded(0.0, ca.sumsqr(U[0:2, i]), 8.0))
        if i < steps: opti.subject_to(opti.bounded(-3.14/2, U[-1, i], 3.14/2))
        if i < steps: obj += w_control * ca.sumsqr(U[0:2, i]) + w_ang * ca.sumsqr(U[2, i])

        # Collision avoidance: ensure safety margin from trees.
        delta = X[:2, i] - trees_dm.T
        sq_dists = ca.diag(ca.mtimes(delta.T, delta))
        opti.subject_to(ca.mmin(sq_dists) >= safe_distance**2)

        if i < steps: opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))
        
    nn_batch = []
    for i in range(trees_dm.shape[0]):
        # Ensure the robot remains at least safe_distance away from every tree.
        delta = X[:2,1:] - trees_dm[i,:].T
        # --- Neural Network Prediction & Belief Update ---
        heading = X[2,1:]
        nn_batch.append(ca.horzcat(delta.T, heading.T))

    g_out = g_nn(ca.vcat([*nn_batch]))
    alpha = 20.0  # Steepness parameter.
    z_k = ca.fmax(g_out, 0.5)  # threshold the NN output

    for i in range(steps):
        lambda_next = bayes(lambda_evol[-1], z_k[i::steps])
        lambda_evol.append(lambda_next)
    
    epsilon_dist = 1e-3
    #dist_vector = ca.hcat([(ca.sum1((trees_dm.T - X[:2, i+1])**2)+ epsilon_dist) for i in range(0,steps,2)]).T
    # Compute entropy terms for the objective
    entropy_future = entropy(ca.vcat([*lambda_evol[1:]])) #/ (5*(-ca.tanh(0.25*dist_vector)**2+ 1)+1)

    #sum_per_step = ca.sum1(entropy_future)
    # Compute softmax weights
    #exp_sum = ca.exp(entropy_future)
    #softmax_weights = exp_sum / sum_per_step

    # Entropy term with softmax
    entropy_term = ca.sum1( ca.vcat([ca.exp(-2*i)*ca.DM.ones(num_trees) for i in range(steps)])*( entropy_future - ca.vcat([lambda_evol[0] for i in range(steps)])) )* w_entropy

    # Attraction term

    #dist_vector = ca.hcat([(ca.sum1((trees_dm.T - X[:2, i+1])**2) + epsilon_dist) for i in range(0,steps,2)]).T
    #attraction_term = w_attract * ca.sum1(ca.tanh((1.0 - 2.0*(ca.vcat([lambda_evol[0] for i in range(steps)]) - 0.5)**2 / dist_vector)))

    # Add terms to the objective
    obj += entropy_term #- attraction_term
    opti.minimize(obj)
    
    # Solver options.
    options = {
        "ipopt": {
            "tol": 1e-2,
            "bound_relax_factor" :1e-4,
            "bound_push" : 1e-6,
            "warm_start_init_point": "no",
            "hessian_approximation": "limited-memory",
            "print_level": 5,
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

# =============================================================================
# Simulation Function with Metrics Collection and CSV Generation
# =============================================================================
def run_simulation():
    F_ = kin_model()
    bridge = BridgeClass(SENSORS)
    trees_pos = np.array(bridge.call_server({"trees_poses": None})["trees_poses"])
    lb, ub = get_domain(trees_pos)

    # Initialize belief as a column vector.
    lambda_k = ca.reshape(ca.DM.ones(len(trees_pos)) * 0.5, (len(trees_pos), 1))
    mpc_horizon = N

    # ---------------------------
    # Load the Learned Neural Network Model
    # ---------------------------
    model = MultiLayerPerceptron(input_dim=nn_input_dim)
    model.load_state_dict(torch.load(get_latest_best_model(), weights_only=True))
    model.eval()
    g_nn = l4c.L4CasADi(model, batched=True, device='cuda')

    # Initialize robot state: assume update_robot_state() returns [x, y, theta].
    initial_state = bridge.update_robot_state()  # e.g., [x, y, theta]
    vx_k = ca.DM.zeros(nx)  # velocity component: [vx, vy, omega]
    x_k = ca.vertcat(ca.DM(initial_state), vx_k)

    # Containers for simulation output.
    all_trajectories = []
    lambda_history = []
    entropy_history = []
    durations = []

    # ---------------------------
    # Velocity Command Logging List
    # ---------------------------
    # This list will store rows of the form [time (s), tag, x_vel, y_vel, yaw_vel]
    velocity_command_log = []

    # ---------------------------
    # New Lists to Record Plot Data
    # ---------------------------
    # These will be used to save the plot data (time, pose, entropy, lambda)
    pose_history = []      # Each element is [x, y, theta]
    time_history = []      # Each element is the simulation time at that step

    # ---------------------------
    # Metrics Tracking Initialization
    # ---------------------------
    sim_start_time = time.time()           # Start simulation time.
    total_distance = 0.0
    total_commands = 0
    sum_vx = 0.0
    sum_vy = 0.0
    sum_yaw = 0.0
    sum_trans_speed = 0.0
    # Use initial position (first two elements of state) as previous position.
    prev_x = float(x_k[0])
    prev_y = float(x_k[1])

    sim_time = 700  # Total simulation time (s)
    mpciter = 0
    rate = rospy.Rate(int(1/dt))
    warm_start = True

    # ---------------------------
    # Main MPC Loop
    # ---------------------------
    while mpciter  < sim_time and not rospy.is_shutdown():
        print('Step:', mpciter)
        print('Observe and update state')

        # Update state from sensor.
        current_state = bridge.update_robot_state()  # returns [x, y, theta]
        # Record the pose and current simulation time for the plot data.
        current_sim_time = time.time() - sim_start_time
        pose_history.append(current_state)
        time_history.append(current_sim_time)

        x_k = ca.vertcat(ca.DM(current_state), vx_k)

        new_data = bridge.get_data()
        while new_data["tree_scores"] is None:
            new_data = bridge.get_data()
        print("Tree scores:", new_data["tree_scores"].flatten())

        # Update belief using tree scores.
        z_k = ca.reshape(ca.DM(new_data["tree_scores"]), (len(new_data["tree_scores"]), 1))
        lambda_k = bayes(lambda_k, z_k)
        lambda_k = np.ceil(lambda_k*1000)/1000
        print("Current state x_k:", x_k)
        print("Lambda:", lambda_k)

        bridge.pub_data({
            "tree_markers": {"trees_pos": trees_pos, "lambda": lambda_k.full().flatten()}
        })

        step_start_time = time.time()
        if warm_start:
            # Initialize MPC.
            mpc_step, u, x_traj, x_dec, lam = mpc_opt(g_nn, trees_pos, lb, ub, x_k, lambda_k, mpc_horizon)
            warm_start = False
        else:
            # Warm-start MPC.
            u, x_traj, x_dec, lam = mpc_step(ca.vertcat(x_k, lambda_k), x_dec, lam)
        durations.append(time.time() - step_start_time)

        # ---------------------------
        # Log the MPC Velocity Command
        # ---------------------------
        # Convert u (CasADi DM) to a NumPy array and flatten to extract individual velocity components.
        u_np = np.array(u.full()).flatten()
        velocity_command_log.append([current_sim_time, "MPC", u_np[0], u_np[1], u_np[2]])

        # ---------------------------
        # Publish Data and Apply Control
        # ---------------------------
        bridge.pub_data({
            "predicted_path": x_traj[:nx, 1:],
        })
        cmd_pose = F_( x_k, u[:, 0])
        bridge.pub_data({"cmd_pose": cmd_pose})

        # Update velocity from predicted trajectory.
        vx_k = x_traj[nx:, 1]

        # ---------------------------
        # Metrics Update
        # ---------------------------
        # Record velocity commands from vx_k.
        vx_val = float(vx_k[0])
        vy_val = float(vx_k[1])
        yaw_val = float(vx_k[2])
        sum_vx += vx_val
        sum_vy += vy_val
        sum_yaw += yaw_val
        trans_speed = math.sqrt(vx_val**2 + vy_val**2)
        sum_trans_speed += trans_speed
        total_commands += 1

        # Compute distance traveled from previous position to current.
        curr_x = float(x_traj[0, 1])
        curr_y = float(x_traj[1, 1])
        distance_step = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        total_distance += distance_step
        prev_x, prev_y = curr_x, curr_y

        # Compute and log overall entropy.
        entropy_k = ca.sum1(entropy(lambda_k)).full().flatten()[0]
        lambda_history.append(lambda_k.full().flatten().tolist())
        entropy_history.append(entropy_k)
        all_trajectories.append(x_traj[:nx, :].full())

        mpciter += 1
        print("Entropy:", entropy_k)
        if entropy_k < 0.10:
            break
        rate.sleep()

    # ---------------------------
    # Final Metrics Calculation
    # ---------------------------
    total_execution_time = time.time() - sim_start_time
    avg_wp_time = total_execution_time / total_commands if total_commands > 0 else 0.0
    avg_vx = sum_vx / total_commands if total_commands > 0 else 0.0
    avg_vy = sum_vy / total_commands if total_commands > 0 else 0.0
    avg_yaw = sum_yaw / total_commands if total_commands > 0 else 0.0
    avg_trans_speed = sum_trans_speed / total_commands if total_commands > 0 else 0.0

    # ---------------------------
    # Save Performance Metrics to CSV Files
    # ---------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baselines_dir = os.path.join(script_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
    # Create a filename that includes trajectory type and timestamp
    perf_csv = os.path.join(baselines_dir, f"mpc_{timestamp}_performance_metrics.csv")
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

    # ---------------------------
    # Save the Logged MPC Velocity Commands to CSV
    # ---------------------------
    vel_csv = os.path.join(baselines_dir, f"mpc_{timestamp}_velocity_commands.csv")
    with open(vel_csv, mode='w', newline='') as vel_file:
        writer = csv.writer(vel_file)
        writer.writerow(["Time (s)", "Tag", "x_velocity (m/s)", "y_velocity (m/s)", "yaw_velocity (rad/s)"])
        writer.writerows(velocity_command_log)

    print(f"Performance metrics saved to {perf_csv}")
    print(f"Velocity command log saved to {vel_csv}")

    # ---------------------------
    # Save the Plot Data to CSV
    # ---------------------------
    plot_csv = os.path.join(baselines_dir, f"mpc_{timestamp}_plot_data.csv")
    with open(plot_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write tree positions (flattened)
        tree_positions_flat = trees_pos.flatten().tolist()
        writer.writerow(["tree_positions"] + tree_positions_flat)

        # Build header row for plot data: time, pose (x, y, theta), entropy, then lambda values.
        header = ["time", "x", "y", "theta", "entropy"]
        if lambda_history and len(lambda_history[0]) > 0:
            num_trees = len(lambda_history[0])
            header += [f"lambda_{i}" for i in range(num_trees)]
        writer.writerow(header)

        # Write each recorded data row.
        for i in range(len(time_history)):
            time_val = time_history[i]
            x, y, theta = pose_history[i]
            entropy_val = entropy_history[i]
            lambda_vals = lambda_history[i]
            row = [time_val, x, y, theta, entropy_val] + lambda_vals
            writer.writerow(row)
    print(f"Plot data saved to {plot_csv}")

    return all_trajectories, entropy_history, lambda_history, durations, g_nn, trees_pos, lb, ub


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_animated_trajectory_and_entropy_2d( all_trajectories, entropy_history, lambda_history, trees, lb, ub, computation_durations):
    
    model = MultiLayerPerceptron(input_dim=nn_input_dim) 
    # Note: Adjust 'weights_only' if needed by your model saving logic.
    model.load_state_dict(torch.load(get_latest_best_model(), weights_only=True))
    model.eval()
    g_nn = l4c.L4CasADi(model, name='plotting_f', batched=True, device='cuda')
    

    # Extract trajectory data
    x_trajectory = np.array([traj[0] for traj in all_trajectories])
    y_trajectory = np.array([traj[1] for traj in all_trajectories])
    theta_trajectory = np.array([traj[2] for traj in all_trajectories])  # Drone orientation (yaw)
    all_trajectories = np.array(all_trajectories)
    lambda_history = np.array(lambda_history)
    
    # Compute entropy reduction for each step
    entropy_mpc_pred = []
    for k in range(all_trajectories.shape[0]):
        lambda_k = lambda_history[k]
        entropy_mpc_pred_k = [entropy_history[k]]  # Start with the initial entropy value
        for i in range(all_trajectories.shape[2]-1):
            relative_position_robot_trees = np.tile(all_trajectories[k, :2, i+1], (trees.shape[0], 1)) - trees
            distance_robot_trees = np.sqrt(np.sum(relative_position_robot_trees**2, axis=1))
            theta = np.tile(all_trajectories[k, 2, i+1], (trees.shape[0], 1))  # Drone yaw
            # Build NN input using CasADi horzcat
            input_nn = ca.horzcat(relative_position_robot_trees, theta)
            # If the tree is far, set z_k to 0.5; otherwise use the NN prediction (thresholded)
            z_k = (distance_robot_trees > 10) * 0.5 + (distance_robot_trees <= 10) * ca.fmax(g_nn(input_nn), 0.5)
            lambda_k = bayes(lambda_k, z_k)
            reduction = ca.sum1(entropy(lambda_k)).full().flatten()[0]
            entropy_mpc_pred_k.append(reduction)
        entropy_mpc_pred.append(entropy_mpc_pred_k)
    
    entropy_mpc_pred = np.array(entropy_mpc_pred)
    sum_entropy_history = entropy_history  # use provided history

    # Compute cumulative computation durations (if needed later)
    cumulative_durations = np.cumsum(computation_durations)

    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.6, 0.4],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],  # First row: 2D map and entropy plot
            [{"type": "scatter"}, {"type": "scatter"}]   # Second row: (empty) and computation durations plot
        ]
    )

    # Add the initial MPC predicted trajectory to the first subplot.
    fig.add_trace(
        go.Scatter(
            x=x_trajectory[0],
            y=y_trajectory[0],
            mode="lines+markers",
            name="MPC Future Trajectory",
            line=dict(color="red", width=4),
            marker=dict(size=5, color="blue")
        ),
        row=1, col=1
    )

    # Add an (initially empty) trace for the drone trajectory.
    fig.add_trace(
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

    # Add tree markers with text showing the tree number.
    # The legend name also includes the initial lambda value.
    for i in range(trees.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=[trees[i, 0]],
                y=[trees[i, 1]],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="#FF0000",  # initial color (red)
                    colorscale=[[0, "#FF0000"], [1, "#00FF00"]],  # Red to green scale
                    cmin=0,
                    cmax=1,
                    showscale=False
                ),
                name=f"Tree {i}: {lambda_history[0][i]:.2f}",
                text=[str(i)],        # Plot the tree number as text
                textposition="top center"
            ),
            row=1, col=1
        )

    # Add the sum of entropies plot to the top-right subplot.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Sum of Entropies (Past)",
            line=dict(color="blue", width=2),
            marker=dict(size=5, color="blue")
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Sum of Entropies (Future)",
            line=dict(color="purple", width=2, dash="dot"),
            marker=dict(size=5, color="purple")
        ),
        row=1, col=2
    )

    # Add the computation durations plot to the bottom-right subplot.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Computation Durations",
            line=dict(color="green", width=2),
            marker=dict(size=5, color="green")
        ),
        row=2, col=2
    )

    # Create frames for animation
    frames = []
    for k in range(len(entropy_mpc_pred)):
        # For each frame, update tree markers with the current lambda values.
        tree_data = []
        for i in range(trees.shape[0]):
            tree_data.append(
                go.Scatter(
                    x=[trees[i, 0]],
                    y=[trees[i, 1]],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        # Use a simple linear mapping to set the color; adjust if needed.
                        color=[2*(lambda_history[k][i] - 0.5)],
                        colorscale=[[0, "#FF0000"], [1, "#00FF00"]],
                        cmin=0,
                        cmax=1,
                        showscale=False
                    ),
                    # Update the legend name with the current lambda value.
                    name=f"Tree {i}: {lambda_history[k][i]:.2f}",
                    text=[str(i)],  # Show tree number as text
                    textposition="top center"
                )
            )

        # Update the entropy curves
        sum_entropy_past = sum_entropy_history[:k+1]
        sum_entropy_future = entropy_mpc_pred[k]

        # Update computation durations (past values)
        computation_durations_past = computation_durations[:k+1]

        # Prepare drone orientation arrows for the current frame.
        x_start = x_trajectory[k]
        y_start = y_trajectory[k]
        theta = theta_trajectory[k]
        x_end = x_start + 0.5 * np.cos(theta)
        y_end = y_start + 0.5 * np.sin(theta)

        list_of_actual_orientations = []
        # For the current MPC trajectory (red arrow)
        for x0, y0, x1, y1 in zip(x_start, y_start, x_end, y_end):
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
                    arrowcolor="red",
                )
            )
            list_of_actual_orientations.append(arrow)

        # Also add past drone trajectory arrows (orange)
        x_start_traj = x_trajectory[:k+1, 0]
        y_start_traj = y_trajectory[:k+1, 0]
        theta_traj = theta_trajectory[:k+1, 0]
        x_end_traj = x_start_traj + 0.5 * np.cos(theta_traj)
        y_end_traj = y_start_traj + 0.5 * np.sin(theta_traj)
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

        frame = go.Frame(
            data=[
                # MPC future trajectory (red)
                go.Scatter(
                    x=x_trajectory[k],
                    y=y_trajectory[k],
                    mode="lines+markers",
                    line=dict(color="red", width=4),
                    marker=dict(size=5, color="blue")
                ),
                # Drone actual trajectory (orange)
                go.Scatter(
                    x=x_trajectory[:k+1, 0],
                    y=y_trajectory[:k+1, 0],
                    mode="lines+markers",
                    line=dict(color="orange", width=4),
                    marker=dict(size=5, color="orange")
                ),
                *tree_data,  # Tree markers with updated lambda and tree number
                # Past entropy
                go.Scatter(
                    x=np.arange(len(sum_entropy_past)),
                    y=sum_entropy_past,
                    mode="lines+markers",
                    line=dict(color="blue", width=2),
                    marker=dict(size=5, color="blue")
                ),
                # Future entropy prediction
                go.Scatter(
                    x=np.arange(k, k+len(sum_entropy_future)),
                    y=sum_entropy_future,
                    mode="lines+markers",
                    line=dict(color="purple", width=2, dash="dot"),
                    marker=dict(size=5, color="purple")
                ),
                # Computation durations
                go.Scatter(
                    x=np.arange(len(computation_durations_past)),
                    y=computation_durations_past,
                    mode="lines+markers",
                    line=dict(color="green", width=2),
                    marker=dict(size=5, color="green")
                )
            ],
            name=f"Frame {k}",
            layout=dict(annotations=list_of_actual_orientations)
        )
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Update layout for the subplots and animation controls.
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

    # Show and save the figure.
    fig.show()
    fig.write_html('neural_mpc_results.html')
    return entropy_mpc_pred

