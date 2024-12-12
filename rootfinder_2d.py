from tools import *
import plotly.graph_objects as go
import numpy as np
from casadi import MX, DM, Opti, rootfinder, logsumexp, log10
from casadi import *
count = 0

# Funzione Entropia
def entropy(confidences):
    epsilon = 1e-6
    return -logsumexp(confidences * log10(confidences + epsilon)/ log10(2))

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
    
    l4c_nn_f = create_l4c_nn_f(n_trees,dev='cpu')

    # Analytical Function Rootfinding Example
    X = MX.sym('x', 1)  # Define symbolic variable
    Y = MX.sym('y', 1)  # Define symbolic variable
    options = {'print_iteration': True, 'error_on_fail': False}

    # Rootfinder for 1 - Gaussian
    rf = rootfinder('solver', 'newton', {'x': vec(vertcat(X,Y)), 'g':vertcat(1 - gaussian_2d(1,1,X,Y),1 - gaussian_2d(1,1,X,Y)) }, options)
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
    opti.set_initial(x, 0)
    opti.subject_to(x <= ubx + 10)  # Upper bound
    opti.subject_to(x >= lbx - 10)  # Lower bound
    y = opti.variable()  # Define optimization variable
    opti.set_initial(x, 0)
    opti.subject_to(y <= ubx + 10)  # Upper bound
    opti.subject_to(y >= lbx - 10)  # Lower bound

    # Definire variabili simboliche
    x_sym = MX.sym('x_')
    y_sym = MX.sym('y_')
    # Definire l'espressione delle distanze simbolicamente
    distances_expr = []
    for k in range(n_trees):
        dx = x_sym - trees_p[k, 0]
        dy = y_sym - trees_p[k, 1]
        dist = dx**2 + dy**2
        distances_expr.append(dist)

    distances_ca = vertcat(*distances_expr)

    # Definire l'espressione LogSumExp
    # Z = -logsumexp(-distances)

    Z_expr = distances_ca

    # Creare una funzione CasADi per valutare Z
    f_Z = Function('f_Z', [x_sym, y_sym], [Z_expr])
    z = opti.variable() 
    opti.subject_to(z > f_Z(x,y))
    opti.minimize(- mmax((l4c_nn_f(x,y, trees_p) + 0.5)*(-(1+f_Z(x,y)))) )
    #- mmax((l4c_nn_f(x,y, trees_p) + 0.5)*(-(1+f_Z(x,y)))))# 1/logsumexp(-0.5+l4c_nn_f(x, trees_p), 0.01)**.5)  # Objective function

    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory","mu_strategy":"adaptive"}}
    opti.solver('ipopt', options)

    # Solve optimization
    sol = opti.solve()

    sol_x = sol.value(x)
    sol_y = sol.value(y)
    sol_z = logsumexp(sol.value(l4c_nn_f(x,y, trees_p)), 0.01).full().flatten()
    print("Optimal solution (x):", sol.value(x))
    print("Optimal solution (y):", sol.value(y))
    print("Optimal NN output:",    mmax(sol.value(l4c_nn_f(x,y, trees_p))))
    print("Optimal NN root:", - sol.value(mmax((l4c_nn_f(x,y, trees_p) + 0.5)*(-f_Z(x,y)))))
    print(i)
    if sol_z > 0.2: count +=1

    print( count )
    if i > 1: continue
    # Plot A: LogSumExp-based Measurements with Tree Positions
    fig_a = go.Figure()

    x_vals = np.linspace(lb[0] - 2.5, ub[0] + 2.5, 100)
    y_vals = np.linspace(lb[1] - 2.5, ub[1] + 2.5, 100)

    z_vals = np.zeros((100, 100))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_vals[j, i] =   - mmax((l4c_nn_f(x,y, trees_p) + 0.5)*(-f_Z(x,y)))#mmax(gaussians_func(x, y, centers)).full().flatten()[0]

    entropy_f = entropy_func(len(trees_p))
    bayes_f = bayes_func(len(trees_p))
    entropy_grid = np.zeros((len(x_vals), len(y_vals)))
    b0 = np.array([np.random.uniform(0.5, 0.9) for _ in range(n_trees)])
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            entropy_grid[j, i] = entropy(bayes_f(l4c_nn_f(x, y, trees_p) + 0.5, b0 )).full().flatten()
            print(entropy_grid[j, i])
    fig_a.add_trace(
        go.Surface(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            showscale=True
        )
    )

    # Add markers for tree positions
    for tree_pos in trees_p:
        fig_a.add_trace(go.Scatter3d(
            x=[tree_pos[0]], 
            y=[tree_pos[1]],
            
            z=[0],

            mode='markers',
            marker=dict(color='red', size=5),
        ))

    fig_a.add_trace(go.Scatter3d(
        x=[sol_x], 
        y=[sol_y],
        
        z=sol_z,

        mode='markers',
        marker=dict(color='green', size=5),
        name="Optimal Points Path"
    ))


    fig_a.update_layout(
        title="Plot A: LogSumExp-based Measurements with Tree Positions",
        xaxis_title="res (varying between -10 and 10)",
        yaxis_title="Measurement Value",
        template="plotly",
        showlegend=True
    )
    fig_a.show()

    # Plot B: Entropy Surface
    fig_b = go.Figure()
    fig_b.add_trace(
        go.Surface(
            z=entropy_grid,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            showscale=True
        )
    )
    fig_b.update_layout(
        title="Entropy Surface",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly",
        showlegend=True
    )
    fig_b.show()
