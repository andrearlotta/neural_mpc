import numpy as np
from casadi import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools import *

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

    # Decision variables
    x = opti.variable()
    y = opti.variable()

    # Constraints for (x, y) bounds
    opti.subject_to((lb[0] <= x) <= ub[0])
    opti.subject_to((lb[1] <= y) <= ub[1])

    # Objective: Maximize the output of the Gaussian function
    opti.minimize(-sum1(gaussians_func(x, y, centers)*weights))

    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory"}}
    opti.solver('ipopt', options)

    sol = opti.solve()

    optimal_x = sol.value(x)
    optimal_y = sol.value(y)
    max_value = sol.value(mmax(gaussians_func(optimal_x, optimal_y, centers) * weights))

    return (optimal_x, optimal_y), max_value


def main():
    # Parameters
    n_points = int(input("Enter the number of Gaussian centers (e.g., 5): "))  # User input for the number of centers
    sigma = 1.0  # Standard deviation
    lb = [-10, -10]  # Lower bounds for x and y
    ub = [10, 10]  # Upper bounds for x and y

    # Generate random 2D Gaussian centers
    centers = np.array([np.random.uniform(lb, ub) for _ in range(n_points)])
    weights = np.ones(n_points) * 0.5  # Uniform weights

    # Display the generated centers and weights
    print(f"Generated centers: {centers}")
    print(f"Generated weights: {weights}")

    gaussians_func = create_l4c_nn_f(len(centers))

    # Optimize to find the maximum
    optimal_point, max_value = maximize_with_opti_2d(gaussians_func, centers, weights, sigma, lb, ub)
    print(f"Optimal point: {optimal_point}")
    print(f"Maximum value: {max_value}")

    # Visualization
    x_vals = np.linspace(lb[0], ub[0], 100)
    y_vals = np.linspace(lb[1], ub[1], 100)
    z_vals = np.zeros((100, 100))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            z_vals[j, i] = (mmax(weights * gaussians_func(x, y, centers))).full().flatten()[0]

    # Create the 2D plot
    fig = go.Figure(
        data=[
            go.Surface(
                z=z_vals,
                x=x_vals,
                y=y_vals,
                colorscale='Viridis',
                showscale=True
            )
        ]
    )

    # Add the optimal point as a scatter
    fig.add_trace(
        go.Scatter3d(
            x=[optimal_point[0]],
            y=[optimal_point[1]],
            z=[max_value],
            mode='markers',
            marker=dict(color='red', size=5),
            name='Optimal Point'
        )
    )

    fig.update_layout(
        title="Maximum Value of Weighted 2D Gaussians",
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
