import numpy as np
from casadi import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools import *
import time

T = 1.0
N = 5
dt = T / N  # Declare dt globally

def kin_model(T=1.0, N=20):
    nx = 2

    # Construct a CasADi function for the ODE right-hand side
    x = MX.sym('x', nx)  # states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
    u = MX.sym('u', 2)  # control force [N]
    rhs = vertcat(u)

    # Continuous system dynamics as a CasADi Function
    f = Function('f', [x, u], [rhs])
    xf = x + dt *  u

    F_ = Function('F', [x, u], [xf])

    return F_

def distance_function(trees_p, max_dist=2):
    # Definire variabili simboliche
    x_sym = MX.sym('x_')
    y_sym = MX.sym('y_')
    # Definire l'espressione delle distanze simbolicamente
    distances_expr = []
    for k in range(len(trees_p)):
        dx = x_sym - trees_p[k, 0] + 1e-5
        dy = y_sym - trees_p[k, 1] + 1e-5
        dist = dx**2 + dy**2 
        distances_expr.append(dist)

    distances_ca = vertcat(*distances_expr)

    # Create a CasADi function for evaluating distances
    return Function('f_Z', [x_sym, y_sym], [distances_ca])

def maximize_with_opti_2d(gaussians_func, centers, lb, ub, x0, y0, vx0, vy0, weights, steps=10):
    """Uses CasADi's Opti to minimize entropy."""
    opti = Opti()
    
    F_ = kin_model(T=1.0, N=steps)
    bayes_f = bayes_func(len(centers))
    entropy_f = entropy_func(len(centers))
    f_Z = distance_function(centers)

    X0 = opti.parameter(2 + len(centers))
    
    X = opti.variable(2, steps + 1)  # State trajectory
    U = opti.variable(2, steps)  # Control inputs


    opti.subject_to( X[:, 0] ==  X0[:2])

    for i in range(steps):
        opti.subject_to((lb[0]-5 <= X[0, i + 1]) <= ub[0]+5)
        opti.subject_to((lb[1]-5 <= X[1, i + 1]) <= ub[1]+5)
        opti.subject_to((-10 <= U[0, i]) <= 10)
        opti.subject_to((-10 <= U[1, i]) <= 10)
        opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))  # State update using kinematic model

    # Objective function: Minimize the entropy
    obj = 0
    current_weights = X0[2:] #initialize the weights
    w0 =  X0[2:]
    exploration_weight = 1 # Tune this parameter
    for i in range(steps):
        z_k = gaussians_func(X[0, i+1], X[1, i+1], centers)+ 0.5
        current_weights = bayes_f(current_weights, z_k)
        dist =  mmax(-(1+log10(0.001*f_Z(X[0, i+1], X[1, i+1])+1))*(1e-6+ 1- 2*(w0-0.5))**-2)
        transition = smooth_transition(-dist)
        obj += -(10*sum1(current_weights*(1- 2*(w0-0.5))) * (1-transition)  + transition * 0.1*dist)
    opti.minimize(obj)  # Minimize the objective

    options = {"ipopt": {"hessian_approximation": "limited-memory", "print_level":0, "sb": "no", "mu_strategy": "monotone", "tol":1e-3}} #reduced print level
    opti.solver('ipopt', options)
    
    opti.set_value(X0, vertcat([x0],[y0], weights))

    start_time = time.time()
    sol = opti.solve()

    duration = time.time() - start_time

    inputs = [X0, opti.x,opti.lam_g]
    outputs = [U[:,0], X, opti.x,opti.lam_g]


    mpc_step = opti.to_function('mpc_step',inputs,outputs)
    u = sol.value(U[:,0])
    x = sol.value(opti.x)
    X_ = sol.value(X)
    lam = sol.value(opti.lam_g)


    print(type(u))
    print(type(x))
    print(type(X_))
    return mpc_step, u, X_, x, lam
    
def main():
    # Parameters
    n_points = 100
    np.random.seed(2)
    lb = [-100, -100]
    ub = [100, 100]
    max_iterations = 100  # Maximum number of MPC iterations

    # Generate random 2D Gaussian centers
    centers = generate_tree_positions ([10,10],8) #np.array([np.random.uniform(lb, ub) for _ in range(n_points)])
    lb, ub = get_domain(centers)
    weights = DM.ones(len(centers)) * 0.5

    gaussians_func = create_l4c_nn_f(len(centers), dev='cuda')
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
    durations = [] #log iteration durations
    os.makedirs("frames", exist_ok=True)

    mpc_step, u, x_, x, lam = maximize_with_opti_2d(gaussians_func, centers,lb, ub, x0, y0, vx0, vy0, weights,steps)
    
    for iteration in range(10000):

        weights_history.append(weights.full().flatten().tolist())
        entropy_history.append(entropy_f(weights).full().flatten().tolist()[0])
        x_steps = x_[0,:]
        y_steps = x_[1,:]
        all_trajectories.append((x_steps, y_steps))

        x0 = x_steps[1]
        y0 = y_steps[1]
        vx0 = u[ 0 ]
        vy0 = u[ 1 ]

        z_k = np.round(gaussians_func(x0, y0, centers).full().flatten() + 0.5, 2)

        weights = bayes_f(weights, DM(z_k))
        
        print(f"Iteration {iteration}: x={x0:.2f}, y={y0:.2f}, Entropy = {entropy_f(weights).full().flatten()[0]}")

        if np.all(np.abs(weights.full() - 1) < 1e-2):
            print(f"Converged at iteration {iteration}")
            break
        print(f"Iteration {iteration}: x={x0:.2f}, y={y0:.2f}, Weights sum = {sum1(weights).full().flatten()}") #log the position and weight sum
        u, x_, x, lam = mpc_step(vertcat([x0], [y0], weights), x, lam)
        u   = u.full()
        x_  = x_.full()





    # Calculate Z values for the surface plot (using the Gaussian function)
    x_vals = np.linspace(lb[0] - 2.5, ub[0] + 2.5, 100)
    y_vals = np.linspace(lb[1] - 2.5, ub[1] + 2.5, 100)

    print(weights)
    z_vals = np.zeros((100, 100))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_k = gaussians_func(x,y,centers)+ 0.5
            bayes = bayes_f(z_k, weights)
            z_vals[j, i] = -mmax(-log10(10*(f_Z(x,y)+1))*( 1 + 20*(weights-0.5)))+  + sum1(z_k * (1- 2*(weights-0.5))) #mmax(gaussians_func(x, y, centers).full().flatten() + 0.5)

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
        z_k = gaussians_func(x[0],y[0],centers)+ 0.5
        bayes = bayes_f(z_k, weights)
        z_traj_all.append((-mmax(-log10(10*(f_Z(x[0],y[0])+1))*( 1 + 20*(weights-0.5)))+  + sum1(z_k * (1- 2*(weights-0.5)))).full())
    print(np.array(all_trajectories).shape)
    fig.add_trace(
        go.Scatter3d(
            x=np.array(all_trajectories)[:,0,1],
            y=np.array(all_trajectories)[:,1,1],
            z=np.array(z_traj_all).flatten(),
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

    # Plot iteration durations
    fig_durations = go.Figure()
    fig_durations.add_trace(go.Scatter(y=durations, mode='lines+markers'))
    fig_durations.update_layout(title="Optimizer Duration per Iteration", xaxis_title="Iteration", yaxis_title="Duration (seconds)")
    fig_durations.show()


    # Calculate Z values for the surface plot (using the Gaussian function)
    x_vals = np.linspace(lb[0] - 2.5, ub[0] + 2.5, 100)
    y_vals = np.linspace(lb[1] - 2.5, ub[1] + 2.5, 100)

    print(weights)
    z_vals = np.zeros((100, 100))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_k = gaussians_func(x,y,centers)+ 0.5
            bayes = bayes_f(z_k, weights)
            z_vals[j, i] = mmax(z_k ) #mmax(gaussians_func(x, y, centers).full().flatten() + 0.5)

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
        z_k = gaussians_func(x[0],y[0],centers)+ 0.5
        bayes = bayes_f(z_k, weights)
        z_traj_all.append((mmax(z_k)).full())
    print(np.array(all_trajectories).shape)
    fig.add_trace(
        go.Scatter3d(
            x=np.array(all_trajectories)[:,0,1],
            y=np.array(all_trajectories)[:,1,1],
            z=np.array(z_traj_all).flatten(),
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
