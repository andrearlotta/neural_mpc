import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 1. Classe Dataset con normalizzazione
class ConfidenceDataset(Dataset):
    def __init__(self):
        self.raw_data = []
        self.scaler = StandardScaler()
        self.scaler_fitted = False

    def add_measurement(self, r, theta, confidence):
        self.raw_data.append([r, theta, confidence])
        if len(self.raw_data) > 1:  # Fit scaler solo quando abbiamo abbastanza dati
            data_array = np.array(self.raw_data)[:,:2]
            if not self.scaler_fitted:
                self.scaler.fit(data_array)
                self.scaler_fitted = True

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        r, theta, conf = self.raw_data[idx]
        if self.scaler_fitted:
            scaled = self.scaler.transform([[r, theta]])
            r, theta = scaled[0]
        return torch.tensor([r, theta], dtype=torch.float32), torch.tensor(conf, dtype=torch.float32)

# 2. Modello Neurale avanzato con dropout
class ConfidenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# 3. Funzioni per training e ottimizzazione
def train_model(model, dataset, epochs=50, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def find_optimal_position(model, dataset, current_pos, lr=0.1, max_iter=200):
    if dataset.scaler_fitted:
        scaled_pos = dataset.scaler.transform([current_pos])
        pos = torch.tensor(scaled_pos[0], requires_grad=True, dtype=torch.float32)
    else:
        pos = torch.tensor(current_pos, requires_grad=True, dtype=torch.float32)
    
    optimizer = optim.Adam([pos], lr=lr)
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        confidence = model(pos.unsqueeze(0))
        (-confidence).backward()  # Massimizziamo la confidenza
        optimizer.step()
    
    optimal_scaled = pos.detach().numpy()
    if dataset.scaler_fitted:
        optimal = dataset.scaler.inverse_transform([optimal_scaled])[0]
    else:
        optimal = optimal_scaled
    
    return optimal

# 4. Funzione di visualizzazione 3D
def plot_confidence_map(model, dataset, current_pos=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    r = np.linspace(0.1, 10, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)
    
    input_grid = np.stack([R.ravel(), Theta.ravel()], axis=1)
    if dataset.scaler_fitted:
        input_grid = dataset.scaler.transform(input_grid)
    
    with torch.no_grad():
        Z = model(torch.tensor(input_grid, dtype=torch.float32)).numpy().reshape(R.shape)
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    if current_pos is not None:
        x = current_pos[0] * np.cos(current_pos[1])
        y = current_pos[0] * np.sin(current_pos[1])
        ax.scatter(x, y, 1.0, c='red', s=100)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Confidenza')
    plt.title('Mappa di Confidenza 3D')
    plt.show()

# 5. Simulazione completa
def main():
    # Inizializzazione
    dataset = ConfidenceDataset()
    model = ConfidenceModel()
    
    # Genera dati iniziali simulati (ground truth: massima confidenza a r=5, theta=pi/2)
    for _ in range(200):
        r = np.random.uniform(0.1, 10)
        theta = np.random.uniform(0, 2*np.pi)
        true_conf = np.exp(-((r-5)**2)/(2*2**2)) * (0.9 + 0.1*np.cos(theta - np.pi/2))
        noise = np.random.normal(0, 0.05)
        dataset.add_measurement(r, theta, np.clip(true_conf + noise, 0, 1))
    
    # Addestramento iniziale
    train_model(model, dataset, epochs=100)
    
    # Posizione iniziale simulata
    current_pos = np.array([7.0, np.pi/4])  # r=7m, theta=45°
    print(f"Posizione Iniziale: r={current_pos[0]:.2f}m, theta={np.degrees(current_pos[1]):.2f}°")
    plot_confidence_map(model, dataset, current_pos)
    
    # Ciclo di ottimizzazione
    for i in range(5):
        # Trova nuova posizione ottimale
        optimal_pos = find_optimal_position(model, dataset, current_pos)
        
        # Simula misurazione nella nuova posizione
        new_conf = np.exp(-((optimal_pos[0]-5)**2)/(2*2**2)) * (0.9 + 0.1*np.cos(optimal_pos[1] - np.pi/2))
        new_conf = np.clip(new_conf + np.random.normal(0, 0.03), 0, 1)
        dataset.add_measurement(optimal_pos[0], optimal_pos[1], new_conf)
        
        # Aggiornamento online
        train_model(model, dataset, epochs=20)
        
        print(f"Iterazione {i+1}:")
        print(f"Nuova posizione ottimale: r={optimal_pos[0]:.2f}m, theta={np.degrees(optimal_pos[1]):.2f}°")
        print(f"Confidenza misurata: {new_conf:.2f}\n")
        
        current_pos = optimal_pos
        plot_confidence_map(model, dataset, current_pos)

if __name__ == "__main__":
    main()