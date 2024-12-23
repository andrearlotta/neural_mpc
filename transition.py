import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def quadratic(x):
    return x**2

def smooth_transition(x, transition_point=36, steepness=0.5):
    transition_factor = 1 / (1 + np.exp(-steepness * (x - transition_point)))
    return (1 - transition_factor) * gaussian(x) + transition_factor * quadratic(x)

# Define the range for x
x = np.linspace(-10, 10, 500)

# Compute the functions
gaussian_values = gaussian(x)
quadratic_values = quadratic(x)
smooth_values = smooth_transition(quadratic(x))

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x, gaussian_values, label="Gaussian", linestyle="--")
plt.plot(x, quadratic_values, label="Quadratic", linestyle="--")
plt.plot(x, smooth_values, label="Smooth Transition", linewidth=2)

plt.title("Smooth Transition Between Gaussian and Quadratic Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.axvline(6, color='gray', linestyle=':', label="Transition Point (x=6)")
plt.legend()
plt.grid()
plt.show()
