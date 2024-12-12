import numpy as np
import plotly.graph_objects as go
from scipy.special import logsumexp
from scipy import ndimage

def potential_function(x, y, targets, influence_radius=0.5):
    """
    Calcola il potenziale di attrazione usando log-sum-exp e tanh per la normalizzazione.

    Args:
        x: Coordinata x del punto di valutazione.
        y: Coordinata y del punto di valutazione.
        targets: Lista di tuple (x, y) che rappresentano le posizioni degli oggetti di interesse.
        influence_radius: Raggio di influenza di ogni target.

    Returns:
        Il valore del potenziale nel punto (x, y) tra 0 e 1 (1 sui target).
    """
    contributions = []
    for tx, ty in targets:
        distance = np.sqrt((x - tx)**2 + (y - ty)**2)
        contributions.append(-(distance**2) / (2 * influence_radius**2))

    log_potential = logsumexp(contributions)

    # Scala log_potential per un uso migliore con tanh
    scaled_log_potential = log_potential / (np.sqrt(2)*100)**2 #scala per il numero di target

    # Usa tanh per ottenere valori tra -1 e 1
    tanh_potential = np.tanh(scaled_log_potential)

    # Scala e trasla per ottenere valori tra 0 e 1
    normalized_potential = (tanh_potential + 1)

    return scaled_log_potential


# Numero di oggetti di interesse
num_targets = 10

# Genera posizioni casuali per gli oggetti di interesse
np.random.seed(42)  # Per riproducibilità
targets = [(np.random.rand()*100, np.random.rand()*100) for _ in range(num_targets)]

# Crea una griglia di punti per la visualizzazione
x_range = np.linspace(0, 100, 50)
y_range = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x_range, y_range)

# Calcola il potenziale per ogni punto della griglia
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = potential_function(X[i, j], Y[i, j], targets)

# Crea il plot 3D con Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=x_range, y=y_range, colorscale='Viridis')])

# Aggiunge i target come scatter points
target_x, target_y = zip(*targets)
fig.add_trace(go.Scatter3d(x=target_x, y=target_y, z=[1.1]*num_targets, mode='markers', marker=dict(size=5, color='red'), name="Targets")) # Z leggermente sopra per visualizzarli meglio

fig.update_layout(title='Campo Potenziale di Attrazione (Log-Sum-Exp)', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Potenziale'), margin=dict(l=0, r=0, b=0, t=0))
fig.show()

# Mostra le coordinate dei targets
print("Coordinate dei targets:", targets)