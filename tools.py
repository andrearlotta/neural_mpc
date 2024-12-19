# Load the saved state dictionary into the model
from trainRBF import *
import l4casadi as l4c
from casadi import *

def gaussian_np(x, mu, sig=1/np.sqrt(2*np.pi), norm=True):
    a = 1 if not norm else (sig * np.sqrt(2 * np.pi))
    return a * (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))


def sigmoid_np(x, alpha=10.0):
    return 1 / (1 + np.exp(-alpha * x))


def norm_sigmoid_np(x, thresh=6, delta=0.5, alpha=10.0):
    x_min = thresh - delta
    x_max = thresh + delta
    y_min = 0.0
    y_max = 1.0
    
    normalized_x = ((x - x_min) - (x_max - x_min) / 2) / (x_max - x_min)
    normalized_y = sigmoid_np(normalized_x, alpha)
    mapped_y = y_min + (normalized_y * (y_max - y_min))
    
    return mapped_y


def drone_objects_distances_numpy(drone_pos, trees_pos):
    return np.linalg.norm(trees_pos - drone_pos, axis=1)


def fov_weight_fun_numpy(drone_pos, trees_pos, thresh_distance=5):
    sig = 1.5
    thresh = 0.7
    delta = 0.1
    alpha = 1.0

    # Calculate distance between the drone and each tree
    distances = drone_objects_distances_numpy(drone_pos[:2], trees_pos)

    # Calculate direction from drone to each tree
    theta = drone_pos[2]
    drone_dir = np.array([np.cos(theta), np.sin(theta)])
    light_direction = np.array([1, 0])
    tree_directions = trees_pos - drone_pos[:2]

    # Normalize the tree direction vector
    norm_factor = np.linalg.norm(tree_directions, axis=1, keepdims=True)
    norm_tree_directions = tree_directions / norm_factor  # Normalize the tree directions

    # Compute vector alignment between drone direction and tree directions
    vect_alignment = np.dot(norm_tree_directions, drone_dir)
    vect_light_alignment = np.dot(light_direction, drone_dir)

    # Apply sigmoid and Gaussian functions
    alignment_score = norm_sigmoid_np(((vect_alignment + 1) / 2) ** 4, thresh=thresh, delta=delta, alpha=alpha)
    light_score = norm_sigmoid_np(((vect_light_alignment + 1) / 2) ** 2, thresh=thresh, delta=delta, alpha=alpha)
    distance_score = gaussian_np(distances, mu=thresh_distance, sig=sig)
    
    result = np.minimum(distance_score * light_score * 0.5, 1.0)
    return result


def generate_fake_dataset(num_samples, is_polar, n_input=2):
    synthetic_X = []
    synthetic_Y = []
    tree_pos = np.zeros((1, 2))
    
    # Define ranges for drone position (X, Y) and yaw
    x = np.linspace(-8, 8, num_samples)
    y = np.linspace(-8, 8, num_samples)

    X, Y = np.meshgrid(x, y)
    # Iterate over the grid to vary the drone's position and yaw
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            drone_pos = np.array([X[i, j], Y[i, j]])
            yaw = np.arctan2(Y[i, j], X[i, j])
            value = fov_weight_fun_numpy(np.hstack((drone_pos, yaw)), tree_pos, 3)
            synthetic_X.append([X[i, j], Y[i, j], yaw] if n_input == 3 else [X[i, j], Y[i, j]])
            synthetic_Y.append(value)
    
    return torch.Tensor(np.array(synthetic_X)), torch.Tensor(np.array(synthetic_Y))


def create_l4c_nn_f(trees_number, loaded_model = RBFNN(input_dim=2, num_centers=20), model_name="models/rbfnn_model_2d_syntehtic.pth", dev="cpu"):

    loaded_model.load_state_dict(torch.load(model_name))
    loaded_model.eval()  # Set the model to evaluation mode_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device="cuda")
    l4c_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device=dev)

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


def create_l4c_nn_f_min(trees_number, loaded_model = RBFNN(input_dim=2, num_centers=20), model_name="models/rbfnn_model_2d_syntehtic.pth", device="cpu"):

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
    casadi_quad_approx_sym_out = l4c_model(diff_)

    casadi_quad_approx_sym_out = horzcat(casadi_quad_approx_sym_out, 0.5* MX.ones(casadi_quad_approx_sym_out.shape[0]))
    output = []
    for i  in range(casadi_quad_approx_sym_out.shape[0]):
        output = vertcat(output, mmax(casadi_quad_approx_sym_out[i,:]))
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
