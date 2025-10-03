import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from pgvector.psycopg2 import register_vector
from config import HOST, DBNAME, USERNAME, PASSWORD
from sklearn.decomposition import PCA

model_name = "all-MiniLM-L6-v2"
split = "_test"
dataset_name = "Banking77Classification" + split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = "models/vae_banking77.pth"
# state = "models/vae_banking77_supervised.pth"
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


# class VAE_supervised(VAE):
#     def __init__(self, latent_dim=77, input_dim=384, hidden_dim=192):
#         super().__init__(latent_dim, input_dim, hidden_dim)

#         self.classifier = nn.Linear(latent_dim, 77)

#     def forward(self, X):
#         mu, log_var = self.encode(X)
#         Z = self.reparameterize(mu, log_var)
#         recon = self.decoder(Z)
#         logits = self.classifier(mu)
#         return logits, recon, mu, log_var

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
tests = curr.fetchall()

split = "_train"
dataset_name = "Banking77Classification" + split

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

trains = curr.fetchall()
curr.close()
conn.close()

train_embedding = np.array([row[2] for row in trains])  # (10003, 384)
train_labels = np.array([row[5] for row in trains])  # (10003, ) min 155, max 231
train_labels -= 155
print(f"{train_embedding.shape=}")

test_embedding = np.array([row[2] for row in tests])  # (3080, 384)
test_labels = np.array([row[5] for row in tests])  # (3080, 384)
test_labels -= 78
print(f"{test_embedding.shape=}")


a = input("Label1: ")
a = int(a) if a else -1
b = input("Label2: ")
b = int(b) if b else -1

# if a == -1 and b == -1:
#     mask = None
# else:
#     mask = (labels == a) | (labels == b)

# if mask is not None and np.any(mask):
#     embedding = embedding[mask]
#     labels = labels[mask]

if a == -1 and b == -1:
    train_mask = test_mask = None
else:
    train_mask = (train_labels == a) | (train_labels == b)
    test_mask = (test_labels == a) | (test_labels == b)

if train_mask is not None and np.any(train_mask) and np.any(test_mask):
    train_embedding = train_embedding[train_mask]
    test_embedding = test_embedding[test_mask]
    train_labels = train_labels[train_mask]
    test_labels = test_labels[test_mask]

train_tensor_embedding = torch.tensor(train_embedding, dtype=torch.float32)
train_tensor_labels = torch.tensor(train_labels, dtype=torch.long)

test_tensor_embedding = torch.tensor(test_embedding, dtype=torch.float32)
test_tensor_labels = torch.tensor(test_labels, dtype=torch.long)

# LOAD
model = VAE().to(device)
# model = VAE_supervised().to(device)
model.load_state_dict(torch.load(state, weights_only=True))
model.eval()


def random_proj_3d(model: VAE, split: str, save_pth: str):
    model.eval()

    tensor_embedding = test_tensor_embedding
    labels = test_labels

    if split == "train":
        tensor_embedding = train_tensor_embedding
        labels = train_labels

    with torch.no_grad():
        mu, _ = model.encode(tensor_embedding.to(device))
        latent = mu.cpu().numpy()

    latent_3d = PCA(n_components=3).fit_transform(latent)


def plot_latent_projection(model: VAE, split: str, save_pth: str):
    model.eval()
    tensor_embedding = test_tensor_embedding
    labels = test_labels

    if split == "train":
        tensor_embedding = train_tensor_embedding
        labels = train_labels

    with torch.no_grad():
        mu, _ = model.encode(tensor_embedding.to(device))
        latent = mu.cpu().numpy()

    latent_2d = PCA(n_components=2).fit_transform(latent)
    _, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab20", s=10, alpha=0.7
    )
    # ax.grid(True, linestyle='--', alpha=0.5)
    # ax.spines['left'].set_position('zero')
    # ax.spines['bottom'].set_position('zero')
    plt.colorbar(scatter, label="Label ID")
    plt.xlabel("latend_2d[:, 0]")
    plt.ylabel("latend_2d[:, 1]")
    plt.title("Latent space visualization")
    plt.savefig(save_pth + ".png", dpi=300)

# 2d latent space
# def plot_latent_projection(model: VAE, split: str, save_pth: str):
#     model.eval()
#     tensor_embedding = test_tensor_embedding
#     labels = test_labels

#     if split == "train":
#         tensor_embedding = train_tensor_embedding
#         labels = train_labels

#     with torch.no_grad():
#         mu, _ = model.encode(tensor_embedding.to(device))
#         latent = mu.cpu().numpy()

#     _, ax = plt.subplots(figsize=(8, 6))
#     scatter = ax.scatter(
#         latent[:, 0], latent[:, 1], c=labels, cmap="tab20", s=10, alpha=0.7
#     )
#     plt.colorbar(scatter, label="Label ID")
#     plt.xlabel("latent[:, 0]")
#     plt.ylabel("latent[:, 1]")
#     plt.title("Latent space visualization")
#     plt.savefig(save_pth + ".png", dpi=300)
#     plt.close()

# def plot_idk(model:VAE, save_pth:str="output/latent_idk.png"):
#     with torch.no_grad():
#         x_hat, mu, log_var = model(tensor_embedding.to(device))

#     x_hat = x_hat.cpu().numpy()

#     # plt.figure(figsize=(8, 6))
#     # plt.savefig(save_pth, dpi=300)
#     # pass

plot_latent_projection(model, "train", "output/latent_proj_train")
plot_latent_projection(model, "test", "output/latent_proj_test")
# random_proj_3d(model, "train", "output/3d_proj_train")
# random_proj_3d(model, "test", "output/3d_proj_test")
# plot_idk(model)
