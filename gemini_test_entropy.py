import casadi as ca
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal

# Define symbolic variables (same as before)
x = ca.SX.sym('x')
y = ca.SX.sym('y')
mu = ca.SX.sym('mu', 2)
sigma = ca.SX.sym('sigma', 2, 2)
amplitude = ca.SX.sym('amplitude')

# Gaussian function (same as before)
def gaussian_2d(x, y, mu, sigma, amplitude):
    x_vec = ca.vertcat(x, y)
    exponent = -0.5 * ca.transpose(x_vec - mu) @ ca.inv(sigma) @ (x_vec - mu)
    return 0.5 + amplitude * ca.exp(exponent)

# Set fixed parameters (same as before)
mu_val = np.array([0.0, 0.0])
sigma_val = np.array([[1.0, 0.0], [0.0, 1.0]])

# Create Casadi function (same as before)
gaussian_func = ca.Function('gaussian', [x, y, amplitude], [gaussian_2d(x, y, mu_val, sigma_val, amplitude)])

# Bayesian Update and Entropy (same as before)
prior = 0.5
def bayesian_update(prior, likelihood):
    posterior = (likelihood * prior) / (likelihood * prior + (1-likelihood)*(1-prior))
    return posterior

def entropy(p):
    if p <= 0 or p>=1:
        return 0
    return -p * ca.log10(p)/ca.log10(2) - (1 - p) * ca.log10(1 - p)/ca.log10(2)

# Create meshgrid (same as before)
x_range = np.linspace(-3, 3, 50)
y_range = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x_range, y_range)

# --- Plotting with Plotly ---
def plot_entropy_surface(X, Y, amplitude_val):
    entropy_values = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            likelihood = gaussian_func(X[i, j], Y[i, j], amplitude_val).full().flatten()
            updated_posterior = bayesian_update(prior, likelihood)
            entropy_values[i, j] = entropy(likelihood).flatten()

    fig = go.Figure(data=[go.Surface(z=entropy_values, x=X, y=Y)])
    fig.update_layout(title=f'Entropy Surface (Amplitude = {amplitude_val})',
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Entropy'),
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

# Plot for fixed amplitude (1.0)
# Plot for varying amplitudes

plot_entropy_surface(X, Y, 0.5)