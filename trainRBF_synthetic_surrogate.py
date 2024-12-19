import torch
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import torch.nn.functional as F
import numpy as np
from tools import *

# Training function
def train_rbfnn_2d(mean=torch.tensor([0.0, 0.0]), std=1.0, num_centers=20, lr=1e-3, epochs=10, batch_size=32, save_path="rbfnn_model_2d.pth"):
    x_data, y_label = generate_fake_dataset(100, False, n_input=2)

    dataset = TensorDataset(x_data, y_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RBFNN(input_dim=2, num_centers=num_centers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(data_loader):.6f}')

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Test the model and visualize results in 2D
def test_rbfnn_2d(model_path, mean=torch.tensor([0.0, 0.0]), std=1.0, num_centers=20):
    model = RBFNN(input_dim=2, num_centers=num_centers)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x1 = torch.linspace(-8, 8, 100)
    x2 = torch.linspace(-8, 8, 100)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    x_test = torch.stack([grid_x1.flatten(), grid_x2.flatten()], dim=1)
    y_test = model(x_test).detach().numpy().reshape(100, 100)

    grid, ground_truth = generate_fake_dataset(100, False, n_input=2)
    Z_truth = ground_truth.numpy().reshape(100, 100)

    X = grid_x1.numpy()
    Y = grid_x2.numpy()

    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z_truth, x=X, y=Y, colorscale='Viridis', name='Ground Truth'))
    fig.add_trace(go.Surface(z=y_test, x=X, y=Y, colorscale='Blues', name='Predicted'))

    fig.update_layout(
        title="Ground Truth vs Predicted (2D)",
        scene=dict(
            xaxis_title="x1", 
            yaxis_title="x2", 
            zaxis_title="Output"
        ),
    )
    fig.show()

if __name__ == "__main__":
    # Train the model and save it
    train_rbfnn_2d(save_path="models/rbfnn_model_2d_syntehtic.pth")

    # Test the model
    test_rbfnn_2d(model_path="models/rbfnn_model_2d_syntehtic.pth")
