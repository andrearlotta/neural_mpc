# Load the saved state dictionary into the model
from trainRBF import *
import l4casadi as l4c
from casadi import *

# Define the Gaussian function
def gaussian__ca(x, mean=0, std=1):
    return 0.5 + 0.5 * exp(-((x - mean) ** 2) / (2 * std**2))

def create_l4c_nn_f(trees_number, loaded_model = RBFNN(input_dim=2, num_centers=20), model_name="rbfnn_model_2d.pth", device="cuda"):

    loaded_model.load_state_dict(torch.load(model_name))
    loaded_model.eval()  # Set the model to evaluation mode_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device="cuda")
    l4c_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device=device)

    for i in range(10):
        x_sym = MX.sym('x_', trees_number,2)
        y_sym = l4c_model(x_sym)
        f_ = Function('y_', [x_sym], [y_sym])
        df_ = Function('dy', [x_sym], [jacobian(y_sym, x_sym)])

        x = DM([[0., 2.] for _ in range(trees_number)])
        l4c_model(x)
        f_(x)
        df_(x)

    drone_statex = MX.sym('drone_pos_x')
    drone_statey = MX.sym('drone_pos_y')
    drone_state1 = horzcat(drone_statex, drone_statey)
    trees_lambda = MX.sym('trees_lambda', trees_number,2)
    diff_ = repmat(drone_state1, trees_lambda.shape[0], 1) - trees_lambda
    output = l4c_model(diff_)
    return Function('F_single', [drone_statex, drone_statey, trees_lambda], [output])

def bayes_func(shape_):
    """
    Performs a Bayesian update.
    Args:
        shape_: Shape of the lambda and y arrays.

    Returns:
        A CasADi function that performs a Bayesian update.
    """
    lambda_k = MX.sym('lambda_', shape_)
    y_z = MX.sym('y_', shape_)

    numerator = times(lambda_k, y_z)
    denominator = numerator + times((1 - lambda_k), (1 - y_z))
    output = numerator / denominator

    return Function('bayes_func', [lambda_k, y_z], [output])


def _entropy_func(dim):
    """
    Calculates the entropy of a probability distribution.

    Args:
        lambda_: Probability distribution (CasADi MX or DM object).

    Returns:
        Entropy value (CasADi MX or DM object).
    """
    # Adding a small epsilon to avoid log(0)
    lambda__ = MX.sym('lambda__', dim)  # To avoid log(0) issues
    output =  -sum1((lambda__ * log10(lambda__) / log10(2) + (1-lambda__) * log10(1-lambda__) / log10(2)))
    return Function('entropy_f', [lambda__], [output])



def entropy_func(dim):
    """
    Calculates the entropy of a probability distribution.

    Args:
        lambda_: Probability distribution (CasADi MX or DM object).

    Returns:
        Entropy value (CasADi MX or DM object).
    """
    # Adding a small epsilon to avoid log(0)
    lambda__ = MX.sym('lambda__', dim)  # To avoid log(0) issues
    epsilon = 1e-12
    
    lambda_min = horzcat(lambda__, MX.ones(lambda__.shape[0])*epsilon)
    lambda_min__ = []
    for i  in range(lambda__.shape[0]):
        lambda_min__ = vertcat(lambda_min__, logsumexp(lambda_min[i,:].T ,0.01))

    one_minus_lambda = horzcat(1 - lambda__, MX.ones(lambda__.shape[0])*epsilon)
    one_minus_lambda__ = []
    for i  in range(lambda__.shape[0]):
        one_minus_lambda__ = vertcat(one_minus_lambda__, logsumexp(one_minus_lambda[i,:].T ,0.01))

    output =  -sum1((lambda_min__ * log10(lambda_min__ + 1e-10) / log10(2) + one_minus_lambda__* log10(one_minus_lambda__) / log10(2)))
    return Function('entropy_f', [lambda__], [output])


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