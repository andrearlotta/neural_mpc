import torch
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import torch.nn.functional as F

# Define the Gaussian function for ground truth
def gaussian(x, mean, std):
    """
    Computes the Gaussian function.
    Parameters:
        x (torch.Tensor): Input tensor.
        mean (float): Mean of the Gaussian.
        std (float): Standard deviation of the Gaussian.
    Returns:
        torch.Tensor: Gaussian values.
    """
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * (2 * torch.pi) ** 0.5)

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
        self.centers = torch.nn.Parameter(torch.randn(num_centers, input_dim))
        self.log_sigmas = torch.nn.Parameter(torch.zeros(num_centers))  # Log scale for stability

    def forward(self, x):
        """
        Computes the RBF activation manually.
        """
        # Calculate squared Euclidean distances manually
        x = x.unsqueeze(1)  # Add a dimension to allow broadcasting
        centers = self.centers.unsqueeze(0)  # Add a dimension for broadcasting
        distances = torch.sum((x - centers) ** 2, dim=-1)  # Squared Euclidean distance
        
        sigmas = torch.exp(self.log_sigmas)  # Convert log_sigma to sigma
        # Apply Gaussian kernel
        return torch.exp(-distances / (2 * sigmas ** 2))
    
# Define the RBFNN Model
class RBFNN(torch.nn.Module):
    def __init__(self, input_dim, num_centers, output_dim=1):
        """
        Radial Basis Function Neural Network.
        Parameters:
            input_dim (int): Dimension of the input.
            num_centers (int): Number of RBF centers.
            output_dim (int): Dimension of the output.
        """
        super(RBFNN, self).__init__()
        self.rbf_layer = RBFLayer(input_dim, num_centers)
        self.linear_layer = torch.nn.Linear(num_centers, output_dim)

    def forward(self, x):
        rbf_output = self.rbf_layer(x)
        return self.linear_layer(rbf_output)

# Training function
def train_rbfnn(mean=0, std=1, num_centers=20, lr=1e-3, epochs=20, batch_size=32, save_path="rbfnn_model.pth"):
    # Generate training data
    x = torch.linspace(-1, 1, 1000)  # Range between -1 and 1
    scaled_x = torch.tanh(x)  # Non-linear transformation for density near 0
    final_x = scaled_x * 50  # Scale the range to [-50, 50]
    x_data = final_x.reshape(-1, 1)
    y_label = gaussian(x_data, mean=mean, std=std).reshape(-1, 1)

    # Create a dataset and DataLoader
    dataset = TensorDataset(x_data, y_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, criterion, and optimizer
    model = RBFNN(input_dim=1, num_centers=num_centers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print epoch loss
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(data_loader):.6f}')

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Test the model
    model.eval()
    x_test = torch.linspace(-100, 100, 10000).reshape(-1, 1)
    y_test = model(x_test.to(device)).cpu().detach().numpy()

    # Plot results
    comparison_fig = go.Figure()

    # Ground truth (Gaussian)
    ground_truth = gaussian(x_test, mean=mean, std=std).numpy()
    comparison_fig.add_trace(go.Scatter(
        x=x_test.numpy().flatten(),
        y=ground_truth.flatten(),
        mode='lines',
        name='Ground Truth (Gaussian)',
        line=dict(color='blue')
    ))

    # Predicted values
    comparison_fig.add_trace(go.Scatter(
        x=x_test.numpy().flatten(),
        y=y_test.flatten(),
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    comparison_fig.update_layout(
        title="Ground Truth (Gaussian) vs Predicted Data",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white"
    )

    # Show the plot
    comparison_fig.show()

# Load and Test the Model
def load_and_test_model(model_path, mean=0, std=1, num_centers=20):
    model = RBFNN(input_dim=1, num_centers=num_centers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate test data
    x_test = torch.linspace(-100, 100, 10000).reshape(-1, 1)
    y_test = model(x_test).detach().numpy()

    # Plot results
    comparison_fig = go.Figure()

    # Ground truth (Gaussian)
    ground_truth = gaussian(x_test, mean=mean, std=std).numpy()
    comparison_fig.add_trace(go.Scatter(
        x=x_test.numpy().flatten(),
        y=ground_truth.flatten(),
        mode='lines',
        name='Ground Truth (Gaussian)',
        line=dict(color='blue')
    ))

    # Predicted values
    comparison_fig.add_trace(go.Scatter(
        x=x_test.numpy().flatten(),
        y=y_test.flatten(),
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    comparison_fig.update_layout(
        title="Loaded Model: Ground Truth (Gaussian) vs Predicted Data",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white"
    )

    comparison_fig.show()

if __name__ == "__main__":
    # Train the model and save it
    train_rbfnn(save_path="rbfnn_model.pth")

    # Load and test the saved model
    load_and_test_model(model_path="rbfnn_model.pth")
