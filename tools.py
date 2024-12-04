# Load the saved state dictionary into the model
from trainRBF import *
import l4casadi as l4c
from casadi import *

# Define the Gaussian function
def gaussian__ca(x, mean=0, std=1):
    return 0.5 + 0.5 * exp(-((x - mean) ** 2) / (2 * std**2))

def create_l4c_nn_f(trees_number, loaded_model = RBFNN(input_dim=1, num_centers=20), model_name="rbfnn_model.pth", device="gpu"):

    loaded_model.load_state_dict(torch.load(model_name))
    loaded_model.eval()  # Set the model to evaluation mode_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device="cuda")
    l4c_model = l4c.L4CasADi(loaded_model, generate_jac_jac=False, batched=True, device=device)

    for i in range(10):
        x_sym = MX.sym('x_', trees_number)
        y_sym = l4c_model(x_sym)
        f_ = Function('y_', [x_sym], [y_sym])
        df_ = Function('dy', [x_sym], [jacobian(y_sym, x_sym)])

        x = DM([[0., 2., np.pi/18] for _ in range(trees_number)])
        l4c_model(x)
        f_(x)
        df_(x)

    drone_state1 = MX.sym('drone_pos')
    trees_lambda = MX.sym('trees_lambda', trees_number)
    diff_ = repmat(drone_state1, trees_lambda.shape[0], 1) - trees_lambda
    output = l4c_model(diff_)
    return Function('F_single', [drone_state1, trees_lambda], [output])

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
    output =  -sum1((lambda__ * log10(lambda__) / log10(2) + (1-lambda__) * log10(1-lambda__) / log10(2)))
    return Function('entropy_f', [lambda__], [output])
