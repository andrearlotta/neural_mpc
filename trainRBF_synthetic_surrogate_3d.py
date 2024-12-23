import torch
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import torch.nn.functional as F
import numpy as np
from tools import *

# Training function
def train_rbfnn_3d(num_centers=5, lr=1e-2, epochs=50, batch_size=16, save_path="rbfnn_model_2d.pth"):
    x_data, y_label = generate_fake_dataset(250, False, n_input=3)

    dataset = TensorDataset(x_data, y_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RBFNN3d(input_dim=3, num_centers=num_centers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print( torch.cuda.is_available())
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
def test_rbfnn_3d(model_path, num_centers=20):
    model = RBFNN3d(input_dim=3, num_centers=num_centers)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    grid, ground_truth = generate_fake_dataset(100, False, n_input=3)
    Z_truth = ground_truth.numpy().reshape(100, 100)
    y_test = model(grid).detach().numpy().reshape(100, 100)
    fig = go.Figure()
    print(grid.shape)
    fig.add_trace(go.Scatter3d(z=Z_truth.flatten(), x=grid[:,0], y=grid[:,1], name='Ground Truth'))
    fig.add_trace(go.Scatter3d(z=y_test.flatten(), x=grid[:,0], y=grid[:,1], name='Predicted'))

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
    train_rbfnn_3d(save_path="models/rbfnn_model_3d_synthetic.pth", num_centers=1)

    # Test the model
    test_rbfnn_3d(model_path="models/rbfnn_model_3d_synthetic.pth", num_centers=1)
