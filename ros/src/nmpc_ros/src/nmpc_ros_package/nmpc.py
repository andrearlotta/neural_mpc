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

def fake_confidence(drone_pos, tree_pos, include_yaw=True, fov=20):
    """
    Simulates the confidence output of the NN.
    """
    direction = tree_pos - drone_pos[:2]
    distance = np.linalg.norm(direction)
    result = np.full_like(distance, 0.5)
    
    if include_yaw:
        theta = drone_pos[2]
        drone_forward = np.array([np.cos(theta), np.sin(theta)])
        n_direction = direction / distance
        ang_fov_tree = n_direction.T @ drone_forward  # Dot product (cos(angolo))

        fov_threshold = np.cos(np.deg2rad(fov))

    if distance < 0.001 or distance > 10:
        result = 0.5
    elif include_yaw and ang_fov_tree < fov_threshold:  # Tree oustide the fov
        result = 0.5
    else:
        result = norm.pdf(distance, loc=2.5, scale=1) + 0.5
        
    return result

# Simple NN
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()


        self.input_layer = torch.nn.Linear(input_dim if not input_dim== 3 else input_dim+1, hidden_size)

        hidden_layers = []
        for i in range(3):
            hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):

        if x.shape[-1] == 3:  # Check if the input has 3 dimensions
            # Replace the angle with its sin and cos values
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
    
    # If your models are in a subdirectory (e.g., "models/") relative to the script:
    model_dir = os.path.join(script_dir, "models")  # Adjust subdirectory name as needed
    
    # Find the latest best model file
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

# Constants
hidden_size = 64
hidden_layers = 3
nn_input_dim = 3
T = 1.0
N = 10
nx = 3  # Represents position (x, y, theta) and velocity (vx, vy, omega)


def get_domain(tree_positions):
    """Return the domain (bounding box) of the tree positions."""
    x_min = np.min(tree_positions[:, 0])
    x_max = np.max(tree_positions[:, 0])
    y_min = np.min(tree_positions[:, 1])
    y_max = np.max(tree_positions[:, 1])
    return [x_min,y_min], [x_max, y_max]

# Define the entropy function
def entropy(lambda_val):
    return (-lambda_val * ca.log10(lambda_val) - (1 - lambda_val) * ca.log10(1 - lambda_val))/ca.log10(2)

def kin_model(T=1.0, N=20):
    # Correct kinematic model with state [x, y, theta, vx, vy, omega]
    x = ca.MX.sym('x', nx*2)
    u = ca.MX.sym('u', nx)  # Control: [acc_x, acc_y, angular_acc]
    xf = x + T/N * ca.vertcat(x[nx:], u)
    return ca.Function('F', [x, u], [xf])

# Bayesian update function
def bayes(lambda_prev, z):
    prod = lambda_prev * z
    return prod / (prod + (1 - lambda_prev) * (1 - z))

# MPC optimization using CasADi
def mpc_opt(g_nn, trees, lb, ub, x0, lambda_vals, steps=10):
    opti = ca.Opti()
    F_ = kin_model(T=1.0, N=steps)

    P0 = opti.parameter(nx*2 + len(trees))
    X = opti.variable(nx*2, steps + 1)
    U = opti.variable(nx, steps)

    opti.subject_to(X[:, 0] == P0[:nx*2])
    lambda_k = P0[nx*2:]

    for i in range(steps):
        # State constraints
        opti.subject_to(opti.bounded(lb[0]-3, X[0, i+1], ub[0]+3))
        opti.subject_to(opti.bounded(lb[1]-3, X[1, i+1], ub[1]+3))
        opti.subject_to(opti.bounded(-10, X[3, i+1], 10))  # Velocity constraints
        opti.subject_to(opti.bounded(-10, X[4, i+1], 10))  # Velocity constraints
        opti.subject_to(opti.bounded(-3.14, X[5, i+1], 3.14))  # Angular velocity
        opti.subject_to(opti.bounded(-10, U[:2, i], 10))  # Acceleration constraints
        opti.subject_to(opti.bounded(-3.14, U[2, i], 3.14))  # Angular acceleration
        opti.subject_to(X[:, i+1] == F_(X[:, i], U[:, i]))

        # Calculate relative positions and distances
        relative_pos = ca.repmat(X[:2, i+1].T, trees.shape[0], 1) - trees
        squared_dist = ca.sum2(relative_pos**2)  # Correct distance calculation
        within_range = squared_dist <= 100

        # Neural network input and prediction
        heading = ca.repmat(X[2, i+1], trees.shape[0], 1)
        nn_input = ca.horzcat(relative_pos, heading)
        g_out = g_nn(nn_input)
        already_seen = (lambda_k[:,-1] < 0.9)
        z_k = ca.fmax(g_out, 0.5) #* already_seen + 0.5 * (1- already_seen)

        # Bayesian update
        bayes_update = bayes(lambda_k[:,-1], z_k)
        lambda_k = ca.horzcat(lambda_k, bayes_update)

    # Objective function: Minimize the entropy
    obj = 0
    for i in range(steps):
        # Compute here the relative position of the drone wrt to the tree
        obj += ca.sum1(entropy(lambda_k[:, i+1]) - entropy(lambda_k[:, i]))

    obj += 1e-5 * ca.sumsqr(U[:2,:]) + 1e-8 * ca.sumsqr(U[2,:])
    opti.minimize(obj)
    options = {"ipopt": {"tol":1e-5,  "hessian_approximation": "limited-memory", "print_level":0, "sb": "no", "mu_strategy": "monotone", "max_iter":500}} #reduced print level
    opti.solver('ipopt', options)

    # Initial step    
    opti.set_value(P0, ca.vertcat(x0, lambda_vals))

    sol = opti.solve()

    inputs = [P0, opti.x,opti.lam_g]
    outputs = [U[:,0], X, opti.x, opti.lam_g]
    mpc_step = opti.to_function('mpc_step',inputs,outputs)

    return mpc_step,ca.DM(sol.value(U[:,0])), ca.DM(sol.value(X)),ca.DM(sol.value(opti.x)),ca.DM(sol.value(opti.lam_g))

def run_simulation():
    F_ = kin_model(T=T, N=N)
    bridge = BridgeClass(SENSORS)
    trees_pos = np.array(bridge.call_server({"trees_poses": None})["trees_poses"])
    lb, ub = get_domain(trees_pos)
    lambda_k =ca.DM.ones(len(trees_pos)) * 0.5
    z_k = np.ones(len(trees_pos)) * 0.5
    mpc_horizon = N

    # Step 1: Load learned g(.) which works for one tree.
    model = MultiLayerPerceptron(input_dim=nn_input_dim) 
    model.load_state_dict(torch.load(get_latest_best_model(), weights_only=True))
    model.eval()
    g_nn = l4c.L4CasADi(model, generate_jac_jac=True, batched=True, device='cuda')
    
    # Initialize robot state
    x_k = ca.vertcat(ca.DM(bridge.update_robot_state()), ca.DM.zeros(nx))
    vx_k = ca.DM.zeros(nx)
    # Log
    all_trajectories = []
    lambda_history = []
    entropy_history = []
    durations = []

# =============================================================================
# SIMULATION DEFINITION
# =============================================================================

    sim_time = 150  #s Total simulation time
    mpciter = 0

    """
    Run the simulation loop.
    """
    # Main MPC loop
    while mpciter * T < sim_time :
        print('Step:', mpciter)

        # Step 5: Get a real observation (simulated)
        print('Observe and update state')
        new_data = bridge.get_data()
        x_k = ca.vertcat(ca.DM(bridge.update_robot_state()), vx_k)
        z_k = ca.DM(new_data["tree_scores"]) if new_data["tree_scores"] is not None else ca.vertcat(*[fake_confidence(x_k.full().flatten(), tree, fov=25) for tree in trees_pos])
        lambda_k = bayes(lambda_k, z_k)
        print(x_k)
        start_time = time.time()
        if mpciter == 0 :
             # Step 2.5: Initialize mpc
            mpc_step, u, x_, x, lam = mpc_opt(g_nn, trees_pos, lb, ub, x_k, lambda_k, mpc_horizon)
        else:
            # Step 3: Run MPC
            u, x_, x, lam = mpc_step(ca.vertcat(x_k, lambda_k), x, lam)
        durations.append( time.time() - start_time)

        # Step 4: Apply command to the drone (update its pose)
        # Apply the first control input and update the state
        bridge.pub_data({"predicted_path": x_[:nx,1:], "tree_markers": {"trees_pos":trees_pos, "lambda":lambda_k.full().flatten()}})
        bridge.pub_data({"cmd_pose": (T/N) * F_(x_k, u[:, 0])[nx:].full()})

        print( (T/N) * F_(x_k, u[:, 0])[nx:])
        # Log
        vx_k = x_[nx:,0]
        entropy_k = ca.sum1(entropy(lambda_k)).full().flatten()[0]
        lambda_history.append(lambda_k.full().flatten().tolist())
        entropy_history.append(ca.sum1(entropy(lambda_k)).full().flatten().tolist()[0])
        all_trajectories.append( x_[:nx,:].full())

        # Sleep for the remaining time to maintain the 10 Hz frequency
        time.sleep(T)
        
        mpciter += 1
        if entropy_k < 0.5:
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
            relative_position_robot_trees = np.tile(all_trajectories[k,:2,i+1], (trees.shape[0], 1)) - trees
            distance_robot_trees = np.sqrt(np.sum(relative_position_robot_trees**2, axis=1))
            theta =  np.tile(all_trajectories[k,2,i+1], (trees.shape[0], 1))  # Drone yaw
            input_nn = ca.horzcat(relative_position_robot_trees, theta) # horzcat(np.tile(all_trajectories[k,:3,i+1], (trees.shape[0], 1)), trees)
            z_k = (distance_robot_trees > 10) * 0.5 + (distance_robot_trees <= 10) * ca.fmax(g_nn(input_nn), 0.5)
            lambda_k = bayes(lambda_k, z_k)
            reduction = ca.sum1(entropy(lambda_k)).full().flatten()[0]
            entropy_mpc_pred_k.append(reduction)
        entropy_mpc_pred.append(entropy_mpc_pred_k)
    
    entropy_mpc_pred = np.array(entropy_mpc_pred)

    # Compute the sum of entropies for all trees at each frame
    sum_entropy_history = entropy_history

    # Compute cumulative computation durations
    cumulative_durations = np.cumsum(computation_durations)
    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.6, 0.4],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],  # First row: 2D map and entropy plot
            [{"type": "scatter"}, {"type": "scatter"}]   # Second row: empty and computation durations plot
        ]
    )

    # Add the initial trajectory (2D scatter plot) to the first subplot
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

    # Add trees as circles with color based on lambda to the first subplot
    for i in range(trees.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=[trees[i, 0]],
                y=[trees[i, 1]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="#FF0000",  # Initial color (red)
                    colorscale=[[0, "#FF0000"], [1, "#00FF00"]],  # Red to green
                    cmin=0,
                    cmax=1,
                    showscale=False
                ),
                name=f"Tree {i}"
            ),
            row=1, col=1
        )

    # Add the sum of entropies plot to the second subplot (top-right)
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

    # Add the computation durations plot to the fourth subplot (bottom-right)
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
        # Update tree colors based on lambda values
        tree_data = []
        for i in range(trees.shape[0]):
            tree_data.append(
                go.Scatter(
                    x=[trees[i, 0]],
                    y=[trees[i, 1]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=[2*(lambda_history[k][i] - 0.5)],  # Color based on lambda value
                        colorscale=[[0, "#FF0000"], [1, "#00FF00"]],  # Red to green
                        cmin=0,
                        cmax=1,
                        showscale=False
                    ),
                    name=f"Tree {i}"
                )
            )

        # Update the sum of entropies plot
        sum_entropy_past = sum_entropy_history[:k+1]
        sum_entropy_future = entropy_mpc_pred[k]

        # Update the computation durations plot
        computation_durations_past = computation_durations[:k+1]

        # Add drone orientation as an arrow
        x_start = x_trajectory[k]  # Drone x position
        y_start = y_trajectory[k]  # Drone y position
        theta = theta_trajectory[k]  # Drone yaw angle
        x_end = x_start + 0.5 * np.cos(theta)  # Arrow end x
        y_end = y_start + 0.5 * np.sin(theta)  # Arrow end y

        list_of_actual_orientations = []
        for x0,y0,x1,y1 in zip(x_start, y_start, x_end, y_end):
            arrow = go.layout.Annotation(
                dict(
                    x=x1,  # Arrow end x
                    y=y1,  # Arrow end y
                    xref="x", yref="y",
                    text="",
                    showarrow=True,
                    axref="x", ayref="y",
                    ax=x0,  # Arrow start x
                    ay=y0,  # Arrow start y
                    arrowhead=3,  # Arrowhead size
                    arrowwidth=1.5,  # Arrow width
                    arrowcolor="red",  # Arrow color
                )
            )
            list_of_actual_orientations.append(arrow)


        # Add drone orientation as an arrow
        x_start = x_trajectory[:k+1,0]  # Drone x position
        y_start = y_trajectory[:k+1,0]  # Drone y position
        theta = theta_trajectory[:k+1,0]  # Drone yaw angle
        x_end = x_start + 0.5 * np.cos(theta)  # Arrow end x
        y_end = y_start + 0.5 * np.sin(theta)  # Arrow end y

        for x0,y0,x1,y1 in zip(x_start, y_start, x_end, y_end):
            arrow = go.layout.Annotation(
                dict(
                    x=x1,  # Arrow end x
                    y=y1,  # Arrow end y
                    xref="x", yref="y",
                    text="",
                    showarrow=True,
                    axref="x", ayref="y",
                    ax=x0,  # Arrow start x
                    ay=y0,  # Arrow start y
                    arrowhead=3,  # Arrowhead size
                    arrowwidth=1.5,  # Arrow width
                    arrowcolor="orange",  # Arrow color
                )
            )
            list_of_actual_orientations.append(arrow)
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x_trajectory[k],
                    y=y_trajectory[k],
                    mode="lines+markers",
                    line=dict(color="red", width=4),
                    marker=dict(size=5, color="blue")
                ),
                go.Scatter(
                    x=x_trajectory[:k+1,0],
                    y=y_trajectory[:k+1,0],
                    mode="lines+markers",
                    line=dict(color="orange", width=4),
                    marker=dict(size=5, color="orange")
                ),
                *tree_data,  # Add tree data for this frame
                go.Scatter(
                    x=np.arange(len(sum_entropy_past)),
                    y=sum_entropy_past,
                    mode="lines+markers",
                    line=dict(color="blue", width=2),
                    marker=dict(size=5, color="blue")
                ),
                go.Scatter(
                    x=np.arange(k, k+len(sum_entropy_future)),
                    y=sum_entropy_future,
                    mode="lines+markers",
                    line=dict(color="purple", width=2, dash="dot"),
                    marker=dict(size=5, color="purple")
                ),
                go.Scatter(
                    x=np.arange(len(computation_durations_past)),
                    y=computation_durations_past,
                    mode="lines+markers",
                    line=dict(color="green", width=2),
                    marker=dict(size=5, color="green")
                )
            ],
            name=f"Frame {k}",
            layout=dict(annotations=list_of_actual_orientations)  # Add the annotation and arrow to the frame
        )
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Update layout for the subplots
    fig.update_layout(
        title="Drone Trajectory, Sum of Entropies, and Computation Durations",
        xaxis=dict(title="X Position", range=[lb[0] - 3 , ub[0] + 3]), 
        yaxis=dict(title="Y Position", range=[lb[1] - 3 , ub[1] + 3]),
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
                for k, f in enumerate(fig.frames)
            ],
        }]
    )

    # Show the figure
    fig.show()
    fig.write_html('neural_mpc_results.html')
    return entropy_mpc_pred
