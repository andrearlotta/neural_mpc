from tools import *
import plotly.graph_objects as go
import numpy as np
from casadi import MX, DM, Opti, rootfinder, logsumexp
from casadi import *

# Parameters
n_trees = 20
lbx = -100
ubx = 100
trees_p = np.linspace(lbx, ubx, n_trees)
l4c_nn_f = create_l4c_nn_f(n_trees,dev='cpu', input_dim=1, model_name="models/rbfnn_model_1d.pth")

# Analytical Function Rootfinding Example
X = MX.sym('x')  # Define symbolic variable
options = {'print_iteration': True, 'error_on_fail': False}

# Rootfinder for 1 - Gaussian
rf = rootfinder('solver', 'newton', {'x': X, 'g': 1 - gaussian_np(X, mu=2.0)}, options)
print("Root for 1 - Gaussian:")
print(rf([1.], []))
print(rf.stats())

# Rootfinder for Neural Network-based LogSumExp
T = MX.sym('trees', len(trees_p))  # Symbolic tree positions
rf_nn = rootfinder(
    'solver', 'newton',
    {'x': X, 'g':1/logsumexp(0.5-l4c_nn_f(X, trees_p), 0.01)**2},
    options
)

print("Rootfinder with Neural Network LogSumExp:")
print("Root:", rf_nn(DM([1.5]), []))
print(rf_nn.stats())

# Optimization Problem with Opti
opti = Opti()
x = opti.variable()  # Define optimization variable
opti.set_initial(x, 0)
opti.subject_to(x <= ubx + 10)  # Upper bound
opti.subject_to(x >= lbx - 10)  # Lower bound
opti.minimize( - mmax(-0.5+l4c_nn_f(x, trees_p)))# 1/logsumexp(-0.5+l4c_nn_f(x, trees_p), 0.01)**.5)  # Objective function

# Solver options
options = {"ipopt": {"hessian_approximation": "limited-memory"}}
opti.solver('ipopt', options)

# Solve optimization
sol = opti.solve()
print("Optimal solution (x):", sol.value(x))
print("Optimal NN output:", sol.value(l4c_nn_f(x, trees_p)))

# LogSumExp Measurements for Plotting
res_range = np.linspace(lbx-20, ubx+20, 1000)  # Define range for `res`
measurement_list = []

# Compute measurements
for res in res_range:
    measurement =1/logsumexp(-0.5+l4c_nn_f(res, trees_p), 0.01)
    measurement_list.append(measurement)

measurement_list = hcat(measurement_list).full().flatten()
sumsqr_measurement = np.sum(np.square(measurement_list)**.5)


# Plot A: LogSumExp-based Measurements with Tree Positions
fig_a = go.Figure()

# LogSumExp Measurements
fig_a.add_trace(go.Scatter(
    x=res_range, y=measurement_list,
    mode='lines', name='LogSumExp Measurements'
))

# Add markers for tree positions
for tree_pos in trees_p:
    fig_a.add_trace(go.Scatter(
        x=[tree_pos], y=[0],
        mode='markers', name=f'Tree at {tree_pos}',
        marker=dict(size=8, symbol='x', color='red')
    ))

fig_a.update_layout(
    title="Plot A: LogSumExp-based Measurements with Tree Positions",
    xaxis_title="res (varying between -10 and 10)",
    yaxis_title="Measurement Value",
    template="plotly",
    showlegend=True
)
fig_a.show()
