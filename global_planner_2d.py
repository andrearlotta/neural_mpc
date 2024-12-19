import numpy as np
from casadi import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools import *

def maximize_with_opti_2d(gaussians_func, centers, weights, lb, ub, steps=10, x0= 0.0, y0=0.0):
    """
    Uses CasADi's Opti to minimize the entropy .

    Returns:
    - The optimal (x, y).
    """
    opti = Opti()

    bayes_f = bayes_func(len(centers))
    entropy_f = entropy_func(len(centers))
    distance_f = dist_f(centers)

    # Decision variables
    x = opti.variable()
    opti.set_initial(x,x0)
    y = opti.variable()
    opti.set_initial(y,y0)
    W = []

    X = []
    Y = []

    Obj = []
    for i in range(steps):
        # Constraints for (x, y) bounds
        opti.subject_to((x >= (lb[0] -2)) <= ub[0]+2)
        opti.subject_to((y >= (lb[1] -2)) <= ub[1]+2)

        z_k = gaussians_func(x, y, centers)
        
        W.append(bayes_f(z_k + 0.5 , (weights if i == 0 else W[-1])))
        Obj.append(entropy_f(W[-1]))
        X.append(x)
        Y.append(y)
        if i < steps -1 :
            x = opti.variable()
            y = opti.variable()

    opti.minimize(sum2(hcat(Obj)))
    
    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory", "print_level":5, "sb": "no", "mu_strategy":"monotone", "tol":1e-3}}
    opti.solver('ipopt', options)

    sol = opti.solve()

    return sol.value(X[1]),  sol.value(Y[1])

def main():
    # Parameters
    n_points = 100

    np.random.seed(2)
    lb = [-100, -100]  # Lower bounds for x and y
    ub = [100, 100]  # Upper bounds for x and y
    steps = 10

    # Generate random 2D Gaussian centers
    centers = np.array([np.random.uniform(lb, ub) for _ in range(n_points)])
    weights = DM.ones(n_points) * 0.5  # Uniform weights

    # Display the generated centers and weights
    print(f"Generated centers: {centers}")
    print(f"Generated weights: {weights}")

    gaussians_func = create_l4c_nn_f(len(centers), model_name='models/rbfnn_model_2d_synthetic.pth')
    bayes_f = bayes_func(len(centers))
    distance_f = dist_f(centers)
    entropy_f = entropy_func(len(centers))
    
    gaussians_func_min = create_l4c_nn_f_min(len(centers))

    x_steps = [0.01]
    y_steps = [0.01]
    z_steps = weights

    i = 0

    while (sum1(weights) <= len(centers) -1e-1 )and (i < 100) : 
        # Optimize to find the maximum
        optimal_point_x,optimal_point_y = maximize_with_opti_2d(gaussians_func, centers, weights, lb, ub, steps , x_steps[-1], y_steps[-1])
        x_steps = horzcat(x_steps, optimal_point_x)
        y_steps = horzcat(y_steps, optimal_point_y)
        weights = bayes_f(gaussians_func(optimal_point_x, optimal_point_y, centers) + 0.5, weights)
        z_steps = horzcat(z_steps, weights)
        print("x,y: ", optimal_point_x, optimal_point_y)
        print("sum: ", sum1(weights))
        i += 1
    
    x_steps = x_steps.full().flatten()
    y_steps = y_steps.full().flatten()
    z_steps = z_steps.T.full()

    # Visualization
    grid_dim = 100
    x_grid = np.linspace(lb[0] - 2.5, ub[0] + 2.5, grid_dim)
    y_grid = np.linspace(lb[1] - 2.5, ub[1] + 2.5, grid_dim)
    z_vals = np.zeros((grid_dim, grid_dim))
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            z_k = gaussians_func(x, y, centers)
            z_vals[j, i] =   entropy_f(z_k).full().flatten()

    Z = []
    xy_pairs = zip(x_steps[:-1], y_steps[:-1])  # Combine x_steps and y_steps into pairs
    for x, y in xy_pairs:
            z_k = gaussians_func(x, y, centers)
            Z = vertcat(Z,  entropy_f(z_k))
    Z = Z.full().flatten()

    # Create the figure
    fig = go.Figure()

    # Add the surface plot
    fig.add_trace(
        go.Surface(
            z=z_vals,
            x=x_grid.flatten(),
            y=y_grid.flatten(),
            colorscale='Viridis',
            showscale=True
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_steps.flatten(),
            y=y_steps.flatten(),
            z=Z.flatten(),
            mode='markers+lines',
            marker=dict(color='red', size=5),
            line=dict(color='blue', width=2),
            name="Optimal Points Path"
        )
    )

    # Update layout
    fig.update_layout(
        title="Optimization Path on Weighted 2D Gaussians",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)'
        ),
        template="plotly_white"
    )

    fig.show()


if __name__ == "__main__":
    main()
