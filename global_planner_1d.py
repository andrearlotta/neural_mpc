from tools import *
import plotly.graph_objects as go
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def generate_max_value_function(gaussians_func, dim):
    """
    Wraps the Gaussian function to output the maximum value among all outputs.
    Args:
    - gaussians_func: CasADi function that computes Gaussian values for a given x.

    Returns:
    - CasADi function that computes the maximum value among all Gaussian values.
    """
    x = MX.sym('x')
    weights = MX.sym('weights', dim)
    centers = MX.sym('centers', dim)
    gaussian_values =  weights * gaussians_func(x, centers)
    max_value = mmax(gaussian_values)
    max_func = Function('max_func', [x, centers, weights], [max_value])
    return max_func

def maximize_with_opti(max_func, l4c_nn_f, centers, weights, sigma, lb, ub, steps=5):
    """
    Uses CasADi's Opti to maximize the output of the max_func.
    Args:
    - max_func: CasADi function that computes the maximum Gaussian value.
    - centers: The Gaussian centers.

    Returns:
    - The optimal x value and the corresponding maximum value.
    """
    opti = Opti()
    b = bayes_func(len(centers))
    # Generate the Gaussian function with dynamic x and weights
    
    # Decision variable
    x = opti.variable()
    X = []
    x = 0
    Obj = []

    w = opti.variable(len(centers))
    opti.subject_to(w ==  weights)
    W = [w]
    for i in range(steps):
        w = opti.variable(len(centers))
        opti.subject_to(w ==  b(l4c_nn_f(x, centers) + 0.5, W[-1]))
        W.append(w)
        Obj.append(-max_func(x,centers, 1 - W[-1]))
        X.append(x)
        x = opti.variable()
        opti.subject_to((x >= lb -1)<= ub + 1)
    
    w = opti.variable(len(centers))
    opti.subject_to(w ==  b(l4c_nn_f(x, centers) + 0.5, W[-1]))
    W.append(w)
    Obj.append(-max_func(x,centers, 1 - W[-1]))
    X.append(x)
    # Objective: Maximize the output of the max_func
    Obj = vcat(Obj)
    X = hcat(X)
    opti.minimize(sumsqr(W[-1]-1))
    W = hcat(W)


    # Solver options
    options = {"ipopt": {"hessian_approximation": "limited-memory"}}
    opti.solver('ipopt', options)

    hessian(opti.f,opti.x)[0].sparsity().spy()
    jacobian(opti.g,opti.x).sparsity().spy()
 
    sol = opti.solve()

    optimal_x = sol.value(X)
    weights = sol.value(W)
    max_value = []
    acquired_data = []
    print(optimal_x.shape)
    print(weights.shape)
    
    print(optimal_x)
    print(1 - weights.T)
    print(opti.value(sumsqr(W-1)))
    for i in range(steps):
        max_value.append(0.5 +max_func(optimal_x[i], centers, 1 - weights[:,i]).full().flatten()[0])
        #acquired_data.append(l4c_nn_f(optimal_x[i], centers).full().flatten())
    return optimal_x.T, 1 - weights.T, max_value

def main():
    # File: gaussian_dynamic_input_with_optimization.py

    # Parameters
    n_points = int(input("Enter the number of Gaussian centers (e.g., 5): "))  # User input for the number of centers
    sigma = 1.0  # Standard deviation
    x_vals = np.linspace(-10, 10, 300)  # Range of x values for evaluation
    
    lb = -10 
    ub = 10
    # Centers (x) and weights
    centers = np.random.uniform(lb, ub, n_points)  # Random centers
    weights = np.ones(n_points) * 0.5  # Random weights

    # Display the generated centers and weights
    print(f"Generated centers: {centers}")
    print(f"Generated weights: {weights}")

    # Generate the Gaussian function with dynamic x and weights
    l4c_nn_f = create_l4c_nn_f(n_points, dev='cpu', input_dim=1, model_name='models/rbfnn_model_1d.pth')

    # Generate the maximum value function
    max_func = generate_max_value_function(l4c_nn_f, len(weights))

    b = bayes_func(len(centers))

    optimal_x_list, optimal_weights_list, max_value_list = maximize_with_opti(max_func,l4c_nn_f, centers, 1 - weights, sigma, lb, ub)
    
    # Create a subplot structure with n rows
    fig = make_subplots(
        rows=len(optimal_x_list),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"Plot {i+1}" for i in range(len(optimal_x_list))]
    )

    # Iterate over the data to create each subplot
    for idx, (optimal_x, optimal_weights, max_value) in enumerate(zip(optimal_x_list, optimal_weights_list, max_value_list)):
        # Evaluate the maximum values for all x values
        print(optimal_weights)
        max_vals = [0.5+max_func(xi, centers, optimal_weights) for xi in x_vals]
        
        # Evaluate the individual Gaussian values
        gaussian_outputs = [0.5 + l4c_nn_f(xi, centers) for xi in x_vals]
        
        # Overlay individual Gaussians on the subplot
        for i, (mu, weight) in enumerate(zip(range(-n_points // 2, n_points // 2 + 1), optimal_weights)):
            
            gaussian_values = [y[i].full().flatten()[0] for y in gaussian_outputs]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=gaussian_values,
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    name=f"Gaussian μ={mu:.2f} (Weight={weight:.2f})",
                    showlegend=False
                ),
                row=idx + 1,
                col=1
            )
        
        # Plot the maximum values
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.array(max_vals).flatten(),
                mode='lines',
                line=dict(color='black', width=2),
                name="Maximum Values",
                showlegend=False
            ),
            row=idx + 1,
            col=1
        )
        
        # Highlight the optimal point
        fig.add_trace(
            go.Scatter(
                x=[optimal_x],
                y=[max_value],
                mode='markers',
                marker=dict(color='red', size=8),
                name="Optimal Point",
                showlegend=False
            ),
            row=idx + 1,
            col=1
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Maximum Value Among Weighted Gaussians",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=300 * len(optimal_x_list),  # Adjust height dynamically based on the number of plots
        template="plotly_white"
    )

    # Show the plot
    fig.show()

if __name__ == "__main__":
    main()
