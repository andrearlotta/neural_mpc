# Load the saved state dictionary into the model
import l4casadi as l4c
from casadi import *
import torch
import torch.nn.functional as F
import numpy as np


def generate_tree_positions(grid_size, spacing):
    """Generate tree positions in a grid."""
    x_positions = np.arange(0, grid_size[0]*spacing, spacing)
    y_positions = np.arange(0, grid_size[1]*spacing, spacing)
    xv, yv = np.meshgrid(x_positions, y_positions)
    tree_positions = np.vstack([xv.ravel(), yv.ravel()]).T
    return tree_positions

def get_domain(tree_positions):
    """Return the domain (bounding box) of the tree positions."""
    x_min = np.min(tree_positions[:, 0])
    x_max = np.max(tree_positions[:, 0])
    y_min = np.min(tree_positions[:, 1])
    y_max = np.max(tree_positions[:, 1])
    return [x_min,y_min], [x_max, y_max]

def dist_f(trees_p):
    # Definire variabili simboliche
    x_sym = MX.sym('x_')
    y_sym = MX.sym('y_')
    # Definire l'espressione delle distanze simbolicamente
    distances_expr = []
    for k in range(len(trees_p)):
        dx = x_sym - trees_p[k, 0]
        dy = y_sym - trees_p[k, 1]
        dist = sqrt(dx**2 + dy**2)
        distances_expr.append(dist)

    distances_ca = vertcat(*distances_expr)

    # Definire l'espressione LogSumExp
    # Z = -logsumexp(-distances)
    
    Z_expr = distances_ca

    # Creare una funzione CasADi per valutare Z



    # Find the index of the nearest tree
    min_dist_expr = MX.sym('min_dist')
    min_index_expr = MX.sym('min_index')
    
    # Initialize with the first distance
    min_dist_expr = distances_expr[0]
    min_index_expr = 0
    
    # Loop through the distances to find the minimum
    for k in range(1, len(distances_expr)):
        min_dist_expr = if_else(distances_expr[k] < min_dist_expr, distances_expr[k], min_dist_expr)
        min_index_expr = if_else(distances_expr[k] < min_dist_expr, k, min_index_expr)

    # Creare una funzione CasADi per valutare Z e l'indice dell'albero più vicino
    return Function('f_Z', [x_sym, y_sym], [distances_ca, min_index_expr])

def gaussian_2d_casadi(mean, std, max_val=0.45):
    """
    Computes the 2D Gaussian function in CasADi.
    Parameters:
        x (MX): Input symbolic tensor of shape (N, 2).
        mean (list or MX): Mean tensor of shape (2,).
        std (float): Standard deviation (scalar).
    Returns:
        MX: Gaussian values scaled to a maximum of 0.4.
    """
    x = MX.sym('x_gauss_2d',2)
    # Compute the difference
    diff = x - mean

    # Compute the raw Gaussian function
    raw_gaussian = exp(-0.5 * sum1((diff / std) ** 2)) / (max_val* 2 * pi * std ** 2)

    return Function('gaussian_2d_f', [x],[raw_gaussian])

# Define the Gaussian function for ground truth in 2D
def gaussian_2d_torch(x, mean, std):
    """
    Computes the 2D Gaussian function.
    Parameters:
        x (torch.Tensor): Input tensor of shape (N, 2).
        mean (torch.Tensor): Mean tensor of shape (2,).
        std (float): Standard deviation (scalar).
    Returns:
        torch.Tensor: Gaussian values.
    """
    diff = x - mean
    raw_gaussian = torch.exp(-0.5 * torch.sum((diff / std) ** 2, dim=1)) / (2 * torch.pi * std ** 2)
    return 0.4 * raw_gaussian / raw_gaussian.max()


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
    
    result = np.minimum(distance_score * light_score * alignment_score * 0.5, 1.0)
    return result * 35


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
            yaw = np.arctan2(Y[i, j], X[i, j]) if n_input == 2 else (np.random.rand() -0.5)*4*np.pi
            value = fov_weight_fun_numpy(np.hstack((drone_pos, yaw)), tree_pos, 3)
            synthetic_X.append([X[i, j], Y[i, j], yaw] if n_input == 3 else [X[i, j], Y[i, j]])
            synthetic_Y.append(value)
    
    return torch.Tensor(np.array(synthetic_X)), torch.Tensor(np.array(synthetic_Y))


def create_l4c_nn_f(loaded_model ,trees_number,model_name, input_dim=2 , dev="cpu", synthetic=True):
    loaded_model.load_state_dict(torch.load(model_name))
    loaded_model.eval()  # Set the model to evaluation mode_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device="cuda")
    l4c_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device=dev)

    for i in range(10):
        x_sym = MX.sym('x_', trees_number,input_dim)
        y_sym = l4c_model(x_sym)
        f_ = Function('y_', [x_sym], [y_sym])
        df_ = Function('dy', [x_sym], [jacobian(y_sym, x_sym)])

        x = DM([[0., 2.] for _ in range(trees_number)])
        l4c_model(x)
        f_(x)
        df_(x)

    drone_statex = MX.sym('drone_pos_x')
    if input_dim>1: drone_statey = MX.sym('drone_pos_y')
    drone_state1 = horzcat(drone_statex, drone_statey) if input_dim>1 else drone_statex
    trees_lambda = MX.sym('trees_lambda', trees_number,input_dim)
    diff_ = repmat(drone_state1, trees_lambda.shape[0], 1) - trees_lambda
    output = l4c_model(diff_)
    output += 0.5  if synthetic else 0.0
    return Function('F_single', [drone_statex, drone_statey, trees_lambda] if input_dim>1 else [drone_statex, trees_lambda], [output])


def create_l4c_nn_f_min(trees_number, input_dim=2, model_name="models/rbfnn_model_2d_synthetic.pth", device="cpu"):
    loaded_model = RBFNN(input_dim=input_dim, num_centers=20)
    loaded_model.load_state_dict(torch.load(model_name))
    loaded_model.eval()  # Set the model to evaluation mode_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device="cuda")
    l4c_model = l4c.L4CasADi(loaded_model, generate_jac_jac=True, batched=True, device=device)

    for i in range(10):
        x_sym = MX.sym('x_', trees_number,input_dim)
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
    trees_lambda = MX.sym('trees_lambda', trees_number,input_dim)
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
    z_k = MX.sym('y_', shape_)

    numerator = times(lambda_k, z_k)
    denominator = numerator + times((1 - lambda_k), (1 - z_k))
    output = numerator / denominator

    return Function('bayes_func', [lambda_k, z_k], [output])


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


import os, csv

def find_latest_folder(base_dir):
    """Finds the latest dated folder in the base directory."""
    # List all subdirectories in the base_dir
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Sort the directories by their modified time in reverse (latest first)
    latest_folder = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    return os.path.join(base_dir, latest_folder)

# Utility Function
def load_surrogate_database(polar, fixed_view, n_input, test_size=0.0, augmented_data=True):
    polar = polar

    # Paths
    base_dir = os. getcwd()+'/datasets/'
    print(base_dir)
    base_dir = os.path.join(base_dir, "polar" if polar else "cartesian")
    base_dir = os.path.join(base_dir, "fixed_view" if fixed_view else "variable_view")
    output_csv_filename = 'SurrogateDatasetCNN_Filtered.csv'

    latest_dataset_folder = find_latest_folder(base_dir=base_dir)
    # Step 4: Create output CSV file inside the latest folder
    DATA_PATH = os.path.join(latest_dataset_folder, output_csv_filename)
    print(DATA_PATH)
    print(DATA_PATH)
    X = []  
    Y = []
    with open(DATA_PATH, 'r') as infile:
        data = csv.reader(infile)
        next(data)  # Skip the header
        for row in data:
            a, b, c, value = map(float, row)
            X.append([a,b, (np.deg2rad(c) % 2* np.pi + np.pi % 2* np.pi) - np.pi ] if n_input == 3 else [a,b,c])
            Y.append(value)

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    x_data = torch.zeros_like(X[:,:2])
    x_data[:,0] = X[:,0] * np.cos(X[:,1])
    x_data[:,1] = X[:,0] * np.sin(X[:,1])
    X = x_data.type(torch.Tensor) 
    if not augmented_data: return X, Y

    augmented_X, augmented_Y = generate_surrogate_augmented_data(X, int(len(X) *0.2),  False, n_input)
    # Concatenate augmented data to the original data
    X = torch.cat((X, augmented_X), dim=0)
    Y = torch.cat((Y, augmented_Y), dim=0)
    return X, Y

import random

def generate_surrogate_augmented_data(X, num_samples, polar, n_input):
    x_min = np.abs(X[:, 0]).min() if polar else (np.sqrt(X[:, 0]**2 + X[:, 1]**2)).min()
    x_max = np.abs(X[:, 0]).max() if polar else None
    yaw_values = [30, -30, 60, -60, 90, -90, 180, -180]
    
    synthetic_X = []
    synthetic_Y = []
    
    for _ in range(num_samples):
        #choice = random.choice([1, 2])
        #if choice == 1:
        # Generate synthetic data in the range (0, rho_min) and (0, phi_min)
        theta = random.uniform(-np.pi, np.pi)
        r = random.uniform(0, x_min)
        a = r       if  polar else r * np.cos(theta) 
        b = theta   if  polar else r * np.sin(theta)
        #random.uniform(-2*np.pi, 2*np.pi)
        #else:
        #    # Generate synthetic data in the range (rho_max, rho_max+some_value) and (5, phi_max)
        #    a = random.uniform(x_max, x_max + 5)    if polar else random.uniform(x_max, x_max + 5) * np.cos(random.uniform(-np.pi, np.pi))
        #    b = random.uniform(-np.pi, np.pi)       if polar else random.uniform(x_max, x_max + 5) * np.sin(random.uniform(-np.pi, np.pi))
        #    
        c = np.radians(random.choice(yaw_values))
        value = 0.5
        synthetic_X.append([a, b, c] if n_input ==3 else [a, b])
        synthetic_Y.append(value)

    return torch.FloatTensor(synthetic_X), torch.FloatTensor(synthetic_Y)

# Define the Radial Basis Function Layer
class RBFLayer(torch.nn.Module):
    def __init__(self, input_dim, num_centers):
        """
        Radial Basis Function Layer.
        Parameters:
            input_dim (int): Dimension of the input.
            num_centers (int): Number of RBF centers.
        """
        super(RBFLayer, self).__init__()
        self.centers = torch.nn.Parameter(torch.rand(num_centers, input_dim))
        self.log_sigmas = torch.nn.Parameter(torch.zeros(num_centers))  # Log scale for stability

    def forward(self, x):
        """
        Computes the RBF activation manually.
        """
        x = x.unsqueeze(1)  # Add a dimension to allow broadcasting
        centers = self.centers.unsqueeze(0)  # Add a dimension for broadcasting
        distances = torch.sum((x - centers) ** 2, dim=-1)  # Squared Euclidean distance
        
        sigmas = torch.exp(self.log_sigmas)  # Convert log_sigma to sigma
        return torch.exp(-distances / (2 * sigmas ** 2))

# Define the RBFNN Model
class RBFNN(torch.nn.Module):
    def __init__(self, input_dim, num_centers, output_dim=1):
        """
        Radial Basis Function Neural Network.
        Parameters:
            input_dim (int): Dimension of the input (2 or 3).
            num_centers (int): Number of RBF centers.
            output_dim (int): Dimension of the output.
        """
        super(RBFNN, self).__init__()
        self.input_dim = input_dim
        self.rbf_layer = RBFLayer(2, num_centers)
        self.linear_layer = torch.nn.Linear(num_centers if input_dim<3 else num_centers + 2, output_dim)
        #self.linear_layer1 = torch.nn.Linear(64, output_dim)


    def forward(self, x):
        if x.shape[-1] == 3:  # Check if the input has 3 dimensions
            # Replace the angle with its sin and cos values
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
        rbf_output = self.rbf_layer(x) if self.input_dim < 3 else torch.cat([self.rbf_layer(x[..., :-1]), sin_cos])
        return self.linear_layer(rbf_output)


class RBFNN3d(torch.nn.Module):
    def __init__(self, input_dim, num_centers, output_dim=1):
        """
        Radial Basis Function Neural Network.
        Parameters:
            input_dim (int): Dimension of the input (2 or 3).
            num_centers (int): Number of RBF centers.
            output_dim (int): Dimension of the output.
        """
        super(RBFNN3d, self).__init__()
        self.input_dim = input_dim
        self.rbf_layer = RBFLayer(input_dim if input_dim < 3 else input_dim - 1, num_centers)
        self.linear_layer = torch.nn.Linear(num_centers + (2 if input_dim == 3 else 0), 64)
        self.linear_layer1 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        if x.shape[-1] == 3:  # Check if the input has 3 dimensions
            # Replace the angle with its sin and cos values
            sin_cos = torch.cat([torch.sin(x[..., -1:]), torch.cos(x[..., -1:])], dim=-1)
            rbf_output = torch.cat([self.rbf_layer(x[..., :-1]), sin_cos], dim=-1)
        else:
            rbf_output = self.rbf_layer(x)
        
        x = self.linear_layer(rbf_output)
        x = torch.sigmoid(x)  # Apply Sigmoid activation here
        return self.linear_layer1(x)


def smooth_transition(x, transition_point=10, steepness=0.5):
    return 1 / (1 + exp(-steepness * (x - transition_point)))