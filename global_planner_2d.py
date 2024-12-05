import numpy as np
from casadi import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools import *


def smooth_radial_bounds_function_casadi(max_value, min_value, max_distance):
    """
    Smooth radial function implemented in CasADi.
    Decreases from max_value at the center to min_value at the borders.
    
    Returns:
        A CasADi function that computes the radial function.
    """
    # Define symbolic variables
    x = MX.sym("x")
    y = MX.sym("y")

    # Compute squared distance from the origin
    r_squared = (x)**2 + (y)**2
    r = r_squared**0.5

    # Normalize distance to the range [0, 1]
    r_normalized = r / max_distance
    r_normalized = MX.fmin(MX.fmax(r_normalized, 0), 1)  # Clip to [0, 1]

    # Smooth transition from max_value to min_value
    value = min_value + (max_value - min_value) * (1 - r_normalized**2)

    # Create the CasADi function
    return Function("smooth_radial_bounds_function", 
                    [x, y], 
                    [value])

def gaussian_2d(mu, sigma, x, y):
    """
    2D Gaussian function definition using CasADi, scaled to range between 0.5 and 1.
    Args:
    - mu: Mean of the Gaussian (2D vector).
    - sigma: Standard deviation of the Gaussian.
    - x, y: Symbolic variables for the inputs.

    Returns:
    - Scaled Gaussian expression.
    """
    raw_gaussian = (1 / (2 * pi * sigma**2)) * exp(
        -0.5 * ((x - mu[0])**2 + (y - mu[1])**2) / sigma**2
    )
    max_gaussian = 1 / (2 * pi * sigma**2)  # Maximum value of the Gaussian
    scaled_gaussian = 0.5 + 0.5 * (raw_gaussian / max_gaussian)  # Scale to range [0.5, 1]
    return scaled_gaussian


def generate_gaussian_function_2d(centers, sigma):
    """
    Creates a CasADi function to compute weighted 2D Gaussian values for given (x, y) and weights.
    Args:
    - centers: Array of 2D Gaussian centers.
    - sigma: Standard deviation of the Gaussians.

    Returns:
    - CasADi function that computes Gaussian values for a given (x, y).
    - The centers and weights as numpy arrays for reference.
    """
    x_input = MX.sym('x')
    y_input = MX.sym('y')
    weights = MX.sym('weights', len(centers))  # Symbolic weights

    gaussian_exprs = [
        gaussian_2d(mu, sigma, x_input, y_input) for mu in centers
    ]
    gaussian_exprs = sum(weights[i] * gaussian_exprs[i] for i in range(len(centers)))
    gaussians_func = Function('gaussians_func', [x_input, y_input, weights], [gaussian_exprs])
    return gaussians_func


def maximize_with_opti_2d(gaussians_func, centers, weights, sigma, lb, ub):
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
        
    # Decision variables
    x = opti.variable()
    y = opti.variable()

    # Compute Gaussian values at (x, y)
    gaussian_values = bayes_f(gaussians_func(x, y, centers) + 0.5 , weights)
    
    # Compute gradients of the Gaussian function with respect to x and y
    grad_x = jacobian(gaussian_values, x)
    grad_y = jacobian(gaussian_values, y)

    # Gradient magnitude (norm) to guide towards higher values
    gradient_magnitude = logsumexp(sqrt(grad_x**2 + grad_y**2))

    # Constraints for (x, y) bounds
    opti.subject_to((lb[0] - 5 <= x) <= ub[0] + 5)
    opti.subject_to((lb[1] - 5 <= y) <= ub[1] + 5)

    # Objective: Maximize the output of the Gaussian function with guiding term
    opti.minimize(-gradient_magnitude)

    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory", "print_level":0,"tol":1e-5, "sb": "no", "mu_strategy":"adaptive"}}
    opti.solver('ipopt', options)

    sol = opti.solve()

    optimal_x = sol.value(x)
    optimal_y = sol.value(y)
    max_value = sol.value(gaussians_func(optimal_x, optimal_y, centers) * weights)
    next_weights = sol.value(gaussian_values)
    return optimal_x, optimal_y, max_value, next_weights


def main():
    # Parameters
    n_points = int(input("Enter the number of Gaussian centers (e.g., 5): "))  # User input for the number of centers
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

    x_steps = []
    y_steps = []
    z_steps = weights

    while sum1(weights) <= len(centers) -1e-1:

        # Optimize to find the maximum
        optimal_point_x,optimal_point_y, max_value, next_weights = maximize_with_opti_2d(gaussians_func, centers, weights, sigma, lb, ub)
        x_steps = horzcat(x_steps, optimal_point_x)
        y_steps = horzcat(y_steps, optimal_point_y)
        z_steps = horzcat(z_steps, next_weights)
        weights = next_weights
        print(sum1(next_weights))
    
    x_steps = x_steps.full().flatten()
    y_steps = y_steps.full().flatten()
    z_steps = z_steps.T.full()


    print(max_value.T)
    # Visualization
    x_vals = np.linspace(lb[0], ub[0], 100)
    y_vals = np.linspace(lb[1], ub[1], 100)

    z_vals = np.zeros((100, 100))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_vals[j, i] = mmax(np.ones(n_points) * 0.5  * gaussians_func(x, y, centers)).full().flatten()[0]

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
        Z = vertcat(Z, mmax(np.ones(n_points) * 0.5 * gaussians_func(x, y, centers)))

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
