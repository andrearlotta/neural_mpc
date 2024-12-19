from tools import *
import plotly.graph_objects as go
import numpy as np
from casadi import MX, DM, Opti, rootfinder, logsumexp, log10
from casadi import *

for i in range(1):
    # Parameters
    n_trees = 20
    lbx = -20
    ubx = 20


    np.random.seed(i)
    lb = [-20, -20]  # Lower bounds for x and y
    ub = [20, 20]  # Upper bounds for x and y
    # Generate random 2D Gaussian centers
    trees_p = np.array([np.random.uniform(lb, ub) for _ in range(n_trees)])
    
    l4c_nn_f = create_l4c_nn_f(n_trees, dev='cpu', model_name='models/rbfnn_model_2d_synthetic.pth')
    gaussian_2d_f = gaussian_2d_casadi(0.0,2.0,0.45)
    # Analytical Function Rootfinding Example
    X = MX.sym('x', 1)  # Define symbolic variable
    Y = MX.sym('y', 1)  # Define symbolic variable
    options = {'print_iteration': True, 'error_on_fail': False}

    # Rootfinder for 1 - Gaussian
    rf = rootfinder('solver', 'newton', {'x': vec(vertcat(X,Y)), 'g':vertcat(1 - gaussian_2d_f(vertcat(X,Y)),1 - gaussian_2d_f(vertcat(X,Y))) }, options)
    print(rf)
    print("Root for 1 - Gaussian:")
    print(rf([0.,5], []))
    print(rf.stats())

    # Rootfinder for Neural Network-based LogSumExp
    T = MX.sym('trees', len(trees_p))  # Symbolic tree positions
    rf_nn = rootfinder(
        'solver', 'newton',
        {'x':  vec(vertcat(X,Y)), 'g':vertcat(1/logsumexp(0.5-l4c_nn_f(X,Y, trees_p), 0.01)**2,1/logsumexp(0.5-l4c_nn_f(X,Y, trees_p), 0.01)**2) },
        options
    )

    print("Rootfinder with Neural Network LogSumExp:")
    print("Root:", rf_nn(DM([1.5]), []))
    print(rf_nn.stats())

    # Optimization Problem with Opti
    opti = Opti()
    x = opti.variable()  # Define optimization variable
    opti.subject_to(x <= ubx + 10)  # Upper bound
    opti.subject_to(x >= lbx - 10)  # Lower bound
    y = opti.variable()  # Define optimization variable
    opti.subject_to(y <= ubx + 10)  # Upper bound
    opti.subject_to(y >= lbx - 10)  # Lower bound

    f_Z = dist_f(trees_p)
    entropy_f = entropy_func(len(trees_p))
    bayes_f = bayes_func(len(trees_p))

    b0 = np.array([np.random.uniform(0.5, 0.9) for _ in range(n_trees)]) 
    opti.minimize(- mmax((l4c_nn_f(x,y, trees_p) + 0.5)*(-(1+f_Z(x,y)))) )

    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory","mu_strategy":"adaptive"}}
    opti.solver('ipopt', options)

    # Solve optimization
    sol = opti.solve()
    print('optimal x,y: ',sol.value(x), sol.value(y))
    print('optimal z: ',sol.value(mmax((-(1+f_Z(x,y))))))

    x_vals = np.linspace(lb[0] - 2.5, ub[0] + 2.5, 100)
    y_vals = np.linspace(lb[1] - 2.5, ub[1] + 2.5, 100)
    z_vals = np.zeros((len(x_vals), len(y_vals)))

    b0 = np.array([np.random.uniform(0.5, 0.9) for _ in range(n_trees)])

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_k =  l4c_nn_f(x, y, trees_p) 
            z_vals[j, i] =  (mmax(l4c_nn_f(x,y, trees_p)+0.5)+(mmax(-(1+log10(0.001*f_Z(x,y)+1))*(1- 2*(b0-0.5))**-2))).full().flatten()

    fig_a = go.Figure()

    fig_a.add_trace(
        go.Surface(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            showscale=True
        )
    )

    fig_a.update_layout(
        title="Plot A: LogSumExp-based Measurements with Tree Positions",
        xaxis_title="res (varying between -10 and 10)",
        yaxis_title="Measurement Value",
        template="plotly",
        showlegend=True
    )
    fig_a.show()

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_k =  l4c_nn_f(x, y, trees_p) 
            z_vals[j, i] = entropy_f(z_k + 0.5).full().flatten()

    fig_a = go.Figure()

    fig_a.add_trace(
        go.Surface(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            showscale=True
        )
    )

    fig_a.update_layout(
        title="Plot A: LogSumExp-based Measurements with Tree Positions",
        xaxis_title="res (varying between -10 and 10)",
        yaxis_title="Measurement Value",
        template="plotly",
        showlegend=True
    )
    fig_a.show()