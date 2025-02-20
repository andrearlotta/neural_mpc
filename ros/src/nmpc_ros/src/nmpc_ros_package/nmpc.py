import os
import re
import time

import casadi as ca
import numpy as np
import torch

from nmpc_ros_package.ros_com_lib.bridge_class import BridgeClass
from nmpc_ros_package.ros_com_lib.sensors import SENSORS

import l4casadi as l4c
from scipy.stats import norm 

# ---------------------------
# Global Constants (defined early for use in the NN class)
# ---------------------------
hidden_size = 64
hidden_layers = 3
nn_input_dim = 3
T = 1.0
N = 5
dt = T / N
nx = 3  # Represents [x, y, theta] (position/heading); state X is 6-D ([pos; vel])

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
    X_next = X + dt * ca.vertcat(X[nx:], U)
    return ca.Function('F', [X, U], [X_next])

def bayes(lambda_prev, z):
    """
    Bayesian update for belief:
       lambda_next = (lambda_prev * z) / (lambda_prev * z + (1 - lambda_prev) * (1 - z) + epsilon)
    """
    z = ca.fmin(ca.fmax(z,0.5), 1 - 1e-6)
    prod = lambda_prev*z
    denom = prod + (1 - lambda_prev)* (1 - z)
    return prod / denom

def entropy(p):
    """
    Compute the binary entropy of a probability p.
    Values are clipped to avoid log(0). Uses element-wise multiplication.
    """
    p = ca.fmin(p, 1 - 1e-6)
    return (-p*ca.log10(p) - (1 - p)* (ca.log10(1 - p))) / ca.log10(2)

# =============================================================================
# MPC Optimization Function
# =============================================================================
def mpc_opt(g_nn, trees, lb, ub, x0, lambda_vals, steps=10):
    """
    Set up and solve the MPC optimization problem with the following modifications:
      - The robot moves according to the kinematic model.
      - Only positions and translational velocities are constrained (orientation is free).
      - A collision-avoidance (inequality) constraint ensures the robot stays a safe
        distance from each tree.
      - The objective now includes an entropy term that rewards the reduction in belief 
        uncertainty between the initial and final steps.
    
    The objective minimizes:
      1. Attraction to the nearest tree that has a belief (Bayes value) lower than 0.9.
      2. The misalignment between the robot's heading and the direction toward that tree.
      3. Control effort.
      4. Final entropy (to maximize the reduction between initial and final entropy).
    """
    # Dimensions: state is 6-D; control is 3-D.
    n_state = nx * 2   # 6
    n_control = nx     # 3

    opti = ca.Opti()
    F_ = kin_model()  # Kinematic model function

    # Decision variables:
    X = opti.variable(n_state, steps + 1)
    U = opti.variable(n_control, steps)

    # Parameter vector: initial state and tree beliefs
    P0 = opti.parameter(n_state + trees.shape[0])
    opti.subject_to(X[:, 0] == P0[: n_state])
    # In this formulation, we use the current belief as a parameter (constant over the horizon)
    lambda_param = P0[n_state:]

    # Convert tree positions to a CasADi DM for convenience.
    trees_dm = ca.DM(trees)
    num_trees = trees_dm.size1()

    # ---------------------------
    # Weights and Safety Parameters (tuned)
    # ---------------------------
    w_orient  = 5*1e-3     # Weight for orientation misalignment
    w_control = 1e-6    # Weight for translational control effort
    w_ang     = 1e-8    # Weight for angular control effort
    w_attr    = 1e-2    # Weight for attraction cost
    w_entropy = 1.0    # Weight for final entropy reduction term

    safe_distance = 1.5  # Minimum allowed distance from any tree

    # Initialize the objective.
    obj = 0

    # Create a list to hold the evolution of the belief.
    # lambda_evol[0] is the initial belief.
    lambda_evol = [lambda_param]

    for i in range(steps):
        # --- Constrain the State and Input ---
        opti.subject_to(opti.bounded(lb[0] - 3.0, X[0, i + 1], ub[0] + 3.0))
        opti.subject_to(opti.bounded(lb[1] - 3.0, X[1, i + 1], ub[1] + 3.0))
        opti.subject_to(opti.bounded(-3, X[3, i + 1], 3))
        opti.subject_to(opti.bounded(-3, X[4, i + 1], 3))
        opti.subject_to(opti.bounded(-3.14/2, X[-1, i], 3.14/2))
        # No constraints on orientation (X[2]) or angular velocity (X[5])
        opti.subject_to(opti.bounded(-3, U[0:2, i], 3))
        opti.subject_to(opti.bounded(-3.14, U[-1, i], 3.14))
        
        # --- Dynamics Constraint ---
        opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

        # --- Obstacle Avoidance Constraint ---
        # Ensure the robot remains at least safe_distance away from every tree.
        delta = X[0:2, i+1] - trees_dm.T  # (2 x num_trees)
        # Squared distances for each tree
        sq_dists = ca.diag(ca.mtimes(delta.T, delta))
        # The closest tree must be at least safe_distance away.
        opti.subject_to(ca.mmin(sq_dists) >= safe_distance**2)
        
        # --- Neural Network Prediction & Bayesian Update ---
        relative_pos_bayes = ca.repmat(X[0:2, i+1].T, num_trees, 1) - trees_dm
        heading_bayes = ca.repmat(X[2, i+1], num_trees, 1)
        nn_input = ca.horzcat(relative_pos_bayes, heading_bayes)
        g_out = g_nn(nn_input)
        z_k = ca.fmax(g_out, 0.5)  # Ensure a minimum confidence level
        # Update the belief using the Bayesian update rule.
        lambda_next = bayes(lambda_evol[-1], z_k)
        lambda_evol.append(lambda_next)
        obj += w_entropy * ca.sum1(entropy(lambda_next ))
        
        # --- Attraction & Orientation Cost ---
        pos_i = X[0:2, i + 1]   # current (x, y)
        theta_i = X[2, i + 1]   # current orientation
        theta_i_wrapped = ca.atan2(ca.sin(theta_i), ca.cos(theta_i))  # Wrap to [-π, π)
        cost_list = []
        d_threshold = 5.0  # threshold distance (in meters) for attraction activation
        
        for j in range(num_trees):
            tree_j = trees_dm[j, :].T
            rel_ij = tree_j - pos_i          # vector from robot to tree j
            d_ij = ca.norm_2(rel_ij)           # Euclidean distance
            theta_des = ca.atan2(rel_ij[1], rel_ij[0])  # desired heading angle toward tree j
            orient_error = 1 - ca.cos(theta_i_wrapped - theta_des)  # misalignment cost
            
            # Penalty: high if the belief for this tree is high (i.e., tree is "certain").
            penalty = 1e6 * (0.5 + 0.5 * ca.tanh(100 * (lambda_evol[-1][j] - 0.95)))
            
            # Attraction multiplier: activates more when farther than d_threshold.
            attraction_multiplier = 0.5 * (1 + ca.tanh(10 * (d_ij - d_threshold)))
            
            cost_j =  w_attr* d_ij + w_orient * orient_error  + penalty
            cost_list.append(cost_j)
        cost_vec = ca.vertcat(*cost_list)
        # Use the minimum cost among trees.
        step_cost = ca.mmin(cost_vec)
        obj += step_cost

        # --- Control Effort Regularization ---
        obj += w_control * ca.sumsqr(U[0:2, i]) + w_ang * ca.sumsqr(U[2, i])

    # Set the overall objective.
    opti.minimize(obj)

    # ---------------------------
    # Solver Options and Problem Solve
    # ---------------------------
    options = {
        "ipopt": {
            "tol": 1e-3,
            "warm_start_init_point": "no",
            "hessian_approximation": "limited-memory",
            "print_level": 0,
            "sb": "no",
            "mu_strategy": "monotone",
            "max_iter": 1000
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
# Simulation Function
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
    # Note: Adjust 'weights_only' if needed by your model saving logic.
    model.load_state_dict(torch.load(get_latest_best_model(), weights_only=True))
    model.eval()
    g_nn = l4c.L4CasADi(model, batched=True, device='cuda')
    
    # Initialize robot state: assume update_robot_state() returns [x, y, theta].
    vx_k = ca.DM.ones(nx)  * 1e-4
    # Logging containers.
    all_trajectories = []
    lambda_history = []
    entropy_history = []
    durations = []

    sim_time = 200  # Total simulation time in seconds
    mpciter = 0

    # ---------------------------
    # Main MPC Loop
    # ---------------------------
    while mpciter * T < 500:
        print('Step:', mpciter)
        print('Observe and update state')
        new_data = bridge.get_data()
        x_k = ca.vertcat(ca.DM(bridge.update_robot_state()), vx_k)
        while new_data["tree_scores"] is None:
            new_data = bridge.get_data()
        print(new_data["tree_scores"])

        # Ensure tree_scores are a column vector.
        z_k = ca.reshape(ca.DM(new_data["tree_scores"]), (len(new_data["tree_scores"]), 1))
        lambda_k = bayes(lambda_k, z_k)
        print("Current state x_k:", x_k)
        print("Lambda: ", lambda_k)
        start_time = time.time()
        if mpciter == 0:
            # Initialize MPC.
            mpc_step, u, x_, x_dec, lam = mpc_opt(g_nn, trees_pos, lb, ub, x_k, lambda_k, mpc_horizon)
        else:
            # Warm-start MPC.
            u, x_, x_dec, lam = mpc_step(ca.vertcat(x_k, lambda_k), x_dec, lam)
        durations.append(time.time() - start_time)

        # ---------------------------
        # Publish Data and Apply Control
        # ---------------------------
        bridge.pub_data({
            "predicted_path": x_[:nx, 1:], 
            "tree_markers": {"trees_pos": trees_pos, "lambda": lambda_k.full().flatten()}
        })
        
        bridge.pub_data({"cmd_pose": x_[:nx, 1]})
        time.sleep(T)

        # Update velocity component from the predicted trajectory.
        vx_k = x_[nx:, 1]

        # Compute and log overall entropy.
        entropy_k = ca.sum1(entropy(ca.fmin(lambda_k, 1 - 1e-3))).full().flatten()[0]
        lambda_history.append(lambda_k.full().flatten().tolist())
        entropy_history.append(entropy_k)
        all_trajectories.append(x_[:nx, :].full())

        mpciter += 1
        print(entropy_k)
        if entropy_k < 1.5:
            break

    return all_trajectories, entropy_history, lambda_history, durations, g_nn, trees_pos, lb, ub


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_animated_trajectory_and_entropy_2d(g_nn, all_trajectories, entropy_history, lambda_history, trees, lb, ub, computation_durations):
    print(np.array(all_trajectories).shape)
    
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

