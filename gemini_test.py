import numpy as np
from casadi import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools import *
import imageio
T = .50
N = 5
dt = T / N  # Declare dt globally

def kin_model(T=1.0, N=20):
    nx = 4

    # Construct a CasADi function for the ODE right-hand side
    x = MX.sym('x', nx)  # states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
    u = MX.sym('u', 2)  # control force [N]
    rhs = vertcat(x[2:4], u)

    # Continuous system dynamics as a CasADi Function
    f = Function('f', [x, u], [rhs])
    xf = x + dt * f(x, u)

    F_ = Function('F', [x, u], [xf])

    return F_

def distance_function(trees_p):
    # Definire variabili simboliche
    x_sym = MX.sym('x_')
    y_sym = MX.sym('y_')
    # Definire l'espressione delle distanze simbolicamente
    distances_expr = []
    for k in range(len(trees_p)):
        dx = x_sym - trees_p[k, 0]
        dy = y_sym - trees_p[k, 1]
        dist = dx**2 + dy**2
        distances_expr.append(dist)

    distances_ca = vertcat(*distances_expr)

    # Create a CasADi function for evaluating distances
    return Function('f_Z', [x_sym, y_sym], [distances_ca])

def generate_max_value_function(gaussians_func, dim):
    """
    Wraps the Gaussian function to output the maximum value among all outputs.
    """
    drone_statex = MX.sym('_x')
    drone_statey = MX.sym('_y')
    trees_lambda = MX.sym('_centers', dim, 2)
    weights = MX.sym('weights', dim)

    gaussian_values = weights * gaussians_func(drone_statex, drone_statey, trees_lambda)
    max_value = logsumexp(gaussian_values)
    max_func = Function('max_func', [drone_statex, drone_statey, trees_lambda, weights], [max_value])
    return max_func

def maximize_with_opti_2d(gaussians_func, centers, weights, sigma, lb, ub, steps=10, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0):
    """Uses CasADi's Opti to minimize entropy."""
    opti = Opti()
    F_ = kin_model(T=1.0, N=steps)
    bayes_f = bayes_func(len(centers))
    entropy_f = entropy_func(len(centers))
    f_Z = distance_function(centers)

    X = opti.variable(4, steps + 1)  # State trajectory
    U = opti.variable(2, steps)  # Control inputs

    X[:, 0] = [x0, y0, vx0, vy0]  # Initialize initial state

    for i in range(steps):
        opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))  # State update using kinematic model
        opti.subject_to((lb[0] <= X[0, i + 1]) <= ub[0])
        opti.subject_to((lb[1] <= X[1, i + 1]) <= ub[1])
        opti.subject_to((-20 <= U[0, i]) <= 20)
        opti.subject_to((-20 <= U[1, i]) <= 20)

    # Objective function: Minimize the entropy
    obj = 0
    current_weights = weights #initialize the weights
    exploration_weight = 1 # Tune this parameter
    for i in range(steps):
        y_z = gaussians_func(X[0, i], X[1, i], centers)+ 0.5
        current_weights = bayes_f(current_weights, y_z)
        exploration_term = - mmax(current_weights)*mmax(-(1+f_Z(X[0, i], X[1, i]))*( 1 - weights)**-1)
        obj += exploration_weight * exploration_term

    opti.minimize(obj)  # Minimize the objective

    options = {"ipopt": {"hessian_approximation": "limited-memory", "print_level": 0, "sb": "no", "mu_strategy": "monotone"}} #reduced print level
    opti.solver('ipopt', options)

    sol = opti.solve()

    optimal_trajectory = sol.value(X)
    return optimal_trajectory, sol.value(U)

def main():
    # Parameters
    n_points = 100
    np.random.seed(2)
    sigma = 1.0
    lb = [-100, -100]
    ub = [100, 100]
    max_iterations = 100  # Maximum number of MPC iterations

    # Generate random 2D Gaussian centers
    centers = np.array([np.random.uniform(lb, ub) for _ in range(n_points)])
    weights = DM.ones(n_points) * 0.5

    gaussians_func = create_l4c_nn_f(len(centers))
    max_gaussians_func = generate_max_value_function(gaussians_func, len(centers))
    bayes_f = bayes_func(len(centers))
    steps = N
    F_ = kin_model(T=T, N=steps)
    bayes_f = bayes_func(len(centers))
    entropy_f = entropy_func(len(centers))
    f_Z = distance_function(centers)

    # Initialize robot state
    x0 = 0.0
    y0 = 0.0
    vx0 = 0.0
    vy0 = 0.0

    all_trajectories = []
    weights_history = []
    entropy_history = [] #log the entropy
    os.makedirs("frames", exist_ok=True)

    for iteration in range(450):
        weights_history.append(weights.full().flatten().tolist())
        entropy_history.append(entropy_f(weights).full().flatten().tolist()[0])

        optimal_trajectory, U = maximize_with_opti_2d(gaussians_func, centers, weights, sigma, lb, ub, steps, x0, y0, vx0, vy0)

        x_steps = optimal_trajectory[0, :]
        y_steps = optimal_trajectory[1, :]
        all_trajectories.append((x_steps, y_steps))

        x0 = x_steps[1]
        y0 = y_steps[1]
        vx0 = optimal_trajectory[2, 1]
        vy0 = optimal_trajectory[3, 1]

        y_z = gaussians_func(x0, y0, centers).full().flatten() + 0.5
        weights = bayes_f(weights, DM(y_z))

        print(f"Iteration {iteration}: x={x0:.2f}, y={y0:.2f}, Entropy = {entropy_f(weights).full().flatten()[0]}")

        if np.all(np.abs(weights.full() - 1) < 1e-2):
            print(f"Converged at iteration {iteration}")
            break
        print(f"Iteration {iteration}: x={x0:.2f}, y={y0:.2f}, Weights sum = {sum1(weights).full().flatten()}") #log the position and weight sum


    # Calculate Z values for the surface plot (using the Gaussian function)
    x_vals = np.linspace(lb[0] - 2.5, ub[0] + 2.5, 100)
    y_vals = np.linspace(lb[1] - 2.5, ub[1] + 2.5, 100)

    print(weights)
    z_vals = np.zeros((100, 100))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            y_z = gaussians_func(x,y,centers)+ 0.5
            z_vals[j, i] = mmax(y_z)*mmax(-(1+f_Z(x, y))*( 1 - weights)**-1) # np.sum((1-weights.full()) * gaussians_func(x, y, centers).full())

    # Create the figure
    fig = go.Figure()

    # Add the surface plot
    fig.add_trace(
        go.Surface(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            showscale=True,
            opacity=0.7 #make surface transparent
        )
    )

    # Add the trajectory as a 3D line
    z_traj_all = []
    for frame_num, (x, y) in enumerate(all_trajectories):
        z_traj_all.append(np.sum(weights.full() * gaussians_func(x[1], y[1], centers).full()))
    print(np.array(all_trajectories).shape)
    fig.add_trace(
        go.Scatter3d(
            x=np.array(all_trajectories)[:,0,1],
            y=np.array(all_trajectories)[:,1,1],
            z=z_traj_all,
            mode='markers+lines',
            marker=dict(color='red', size=5),
            line=dict(color='blue', width=2),
            name="Optimal Trajectory"
        )
    )

    fig.show()

    fig_entropy = go.Figure()
    fig_entropy.add_trace(go.Scatter(y=entropy_history, mode='lines+markers'))
    fig_entropy.update_layout(title="Entropy over Iterations", xaxis_title="Iteration", yaxis_title="Entropy")
    fig_entropy.show()

if __name__ == "__main__":
    main()


