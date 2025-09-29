import torch
from torch import nn
from torch.nn import functional as f
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from pgvector.psycopg2 import register_vector
from config import HOST, DBNAME, USERNAME, PASSWORD
from collections import defaultdict
from sklearn.decomposition import PCA

model_name = "all-MiniLM-L6-v2"
split = "_test"
dataset_name = "Banking77Classification" + split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = "models/vae_banking77_supervised.pth"
print("[LOG] Using ", device)

class VAE(nn.Module):
    def __init__(self, latent_dim=77, input_dim=384, hidden_dim=192):
        super(VAE, self).__init__()

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid(),
        )

    def encode(self, x):
        temp = self.encoder(x)
        mu = self.mean(temp)
        log_var = self.log_var(temp)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, X):
        mu, log_var = self.encode(X)
        Z = self.reparameterize(mu, log_var)
        return self.decoder(Z), mu, log_var

class VAE_supervised(VAE):
    def __init__(self, latent_dim=77, input_dim=384, hidden_dim=192):
        super().__init__(latent_dim, input_dim, hidden_dim)

        # Decoder isn't used.
        self.classifier = nn.Linear(latent_dim, 77)

    def forward(self, X):
        mu, log_var = self.encode(X)
        Z = self.reparameterize(mu, log_var)
        recon = self.decoder(Z)
        logits = self.classifier(mu)
        return logits, recon, mu, log_var


conn = psycopg2.connect(
    host=HOST,
    dbname=DBNAME,
    user=USERNAME,
    password=PASSWORD,
)
# pgvector query
register_vector(conn)
curr = conn.cursor()
curr.execute(
    """
SELECT * FROM vectors
WHERE dataset_id = (
    SELECT id FROM vector_datasets
    WHERE name = %s
);
""",
    (dataset_name,),
)
rows = curr.fetchall()
curr.close()
conn.close()

embedding = np.array([row[2] for row in rows])  # (3080, 384)
labels = np.array([row[5] for row in rows])  # (3080, ) min 155, max 231
labels -= 78 # offset

tensor_embedding = torch.tensor(embedding, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.long)
dataset = data.TensorDataset(tensor_embedding, tensor_labels)
dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True)

model = VAE_supervised().to(device)
# model = VAE().to(device)
model.load_state_dict(torch.load(state, weights_only=True))
model.eval()

def plot_latent_projection(model: VAE, save_pth: str):
    with torch.no_grad():
        mu, log_var = model.encode(tensor_embedding.to(device))
        latent = mu.cpu().numpy()
    
    reducer = PCA(n_components=2)
    latent_2d = reducer.fit_transform(latent)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab20", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Label ID")
    plt.title(f"Latent space visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig(save_pth, dpi=300)


def plot_latent_slice(model: VAE, save_path: str, label_1=0, label_2=1, scale=3.0, n=20):
    model.eval()
    figure = np.zeros((n, n)) 
    
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)
    
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros((1, 77)).to(device)
                z[0, label_1] = xi
                z[0, label_2] = yi
                x_decoded = model.decoder(z).cpu().numpy()
                
                figure[i, j] = np.linalg.norm(x_decoded)
    
    plt.imshow(figure, cmap="viridis")
    plt.colorbar()
    plt.title(f"Latent slice along label {label_1}, {label_2}")
    plt.savefig(save_path)

plot_latent_slice(model, "temp/latent_slice_super.png")
plot_latent_projection(model, "temp/latent_proj_super.png")