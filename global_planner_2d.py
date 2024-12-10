import numpy as np
from casadi import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools import *

def generate_max_value_function(gaussians_func, dim):
    """
    Wraps the Gaussian function to output the maximum value among all outputs.
    Args:
    - gaussians_func: CasADi function that computes Gaussian values for a given x.

    Returns:
    - CasADi function that computes the maximum value among all Gaussian values.
    """
    drone_statex = MX.sym('_x')
    drone_statey = MX.sym('_y')
    trees_lambda = MX.sym('_centers', dim,2)
    weights = MX.sym('weights', dim)

    gaussian_values =  weights * gaussians_func(drone_statex, drone_statey, trees_lambda)
    max_value = mmax(gaussian_values)
    max_func = Function('max_func', [drone_statex, drone_statey, trees_lambda, weights], [max_value])
    return max_func


def maximize_with_opti_2d(max_gaussians_func, gaussians_func, centers, weights, sigma, lb, ub, steps=1, x0= 0.0, y0=0.0):
    """
    Uses CasADi's Opti to maximize the output of the 2D Gaussian function.
    Args:
    - gaussians_func: CasADi function that computes Gaussian values.
    - centers: The Gaussian centers.

    Returns:
    - The optimal (x, y) value and the corresponding maximum value.
    """
    opti = Opti()

    bayes_f = bayes_func(len(centers))
    entropy_f = entropy_func(len(centers))
    # Decision variables
    x = opti.variable()
    opti.set_initial(x,x0)
    y = opti.variable()
    opti.set_initial(y,y0)
    W = []

    X = []
    Y = []
    gradient_magnitude = []
    Obj = []
    for i in range(steps):
        # Constraints for (x, y) bounds
        w_next = opti.variable(len(centers))
        opti.subject_to(w_next == bayes_f(gaussians_func(x, y, centers) + 0.5 ,  W[-1] if i !=0 else weights))
        W.append(w_next)        
        Obj.append(entropy_f(W[-1]))

        opti.subject_to((x >= (lb[0] -2)) <= ub[0]+2)
        opti.subject_to((y >= (lb[1] -2)) <= ub[1]+2)
        if i < steps -1:
            X.append(x)
            x = opti.variable()
            Y.append(y)
            y = opti.variable()

    opti.minimize(sumsqr(1-vcat(W[1:])))

    # Objective: Maximize the output of the Gaussian function with guiding term
    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory", "print_level":5, "sb": "no", "mu_strategy":"adaptive", "tol":1e-3}}
    opti.solver('ipopt', options)
    

    hessian(opti.f,opti.x)[0].sparsity().spy()
    jacobian(opti.g,opti.x).sparsity().spy()
 
    sol = opti.solve()

    optimal_x = sol.value(X[1])
    optimal_y = sol.value(Y[1])
    max_value = sol.value(gaussians_func(optimal_x, optimal_y, centers) * weights) + 0.5
    next_weights = sol.value(W[1])
    return optimal_x, optimal_y, max_value, next_weights


def main():
    # Parameters
    n_points = 10 #int(input("Enter the number of Gaussian centers (e.g., 5): "))  # User input for the number of centers
    sigma = 1.0  # Standard deviation
    lb = [-5, -5]  # Lower bounds for x and y
    ub = [5, 5]  # Upper bounds for x and y
    steps = 10
    # Generate random 2D Gaussian centers
    centers = np.array([np.random.uniform(lb, ub) for _ in range(n_points)])
    weights = DM.ones(n_points) * 0.5  # Uniform weights

    # Display the generated centers and weights
    print(f"Generated centers: {centers}")
    print(f"Generated weights: {weights}")

    gaussians_func = create_l4c_nn_f(len(centers))


    max_gaussians_func = generate_max_value_function(gaussians_func, len(centers))
    x_steps = []
    y_steps = []
    z_steps = weights

    x = [DM([0.0])]
    y = [DM([0.0])]
    i = 0
    while (sum1(weights) <= len(centers) -1e-1 )and (i < 100) : 

        # Optimize to find the maximum
        optimal_point_x,optimal_point_y, max_value, next_weights = maximize_with_opti_2d(max_gaussians_func, gaussians_func, centers, weights, sigma, lb, ub, 10, x[-1], y[-1])
        x_steps = horzcat(x_steps, optimal_point_x)
        y_steps = horzcat(y_steps, optimal_point_y)
        z_steps = horzcat(z_steps, next_weights)
        x.append(optimal_point_x)
        y.append(optimal_point_y)
        weights = next_weights
        print(sum1(next_weights))
        i += 1
    
    x_steps = x_steps.full().flatten()
    y_steps = y_steps.full().flatten()
    z_steps = z_steps.T.full()


    print(max_value.T)
    # Visualization
    x_vals = np.linspace(lb[0] - 2.5, ub[0] + 2.5, 100)
    y_vals = np.linspace(lb[1] - 2.5, ub[1] + 2.5, 100)

    z_vals = np.zeros((100, 100))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_vals[j, i] = mmax(gaussians_func(x, y, centers)).full().flatten()[0]

    # Create the figure
    fig = go.Figure()

    # Add the surface plot
    fig.add_trace(
        go.Surface(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            showscale=True
        )
    )

    print(x_steps)
    print(y_steps)
    print(z_steps)
    Z = []

    xy_pairs = zip(x_steps, y_steps)  # Combine x_steps and y_steps into pairs
    for x, y in xy_pairs:
        Z = vertcat(Z, mmax(gaussians_func(x, y, centers)))

    Z = Z.full().flatten()

    fig.add_trace(
        go.Scatter3d(
            x=x_steps,
            y=y_steps,
            z=Z,
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
