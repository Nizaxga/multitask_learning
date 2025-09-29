import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pgvector.psycopg2 import register_vector
import psycopg2
from config import HOST, DBNAME, USERNAME, PASSWORD
import torch
from torch import nn
from torch.nn import functional as f
import torch.optim as optim
import torch.utils.data as data


model_name = "all-MiniLM-L6-v2"
split = "_train"
dataset_name = "Banking77Classification" + split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[LOG] Using ", device)

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

#    Column   |            Type             | Collation | Nullable |               Default
# ------------+-----------------------------+-----------+----------+-------------------------------------
#  id         | integer                     |           | not null | nextval('vectors_id_seq'::regclass)
#  dataset_id | integer                     |           | not null |
#  embedding  | vector                      |           | not null |
#  metadata   | jsonb                       |           |          |
#  created_at | timestamp without time zone |           |          | now()
#  label_id   | integer                     |           |          |
#  SELECT count(*) FROM labels; == 77

embedding = np.array([row[2] for row in rows])  # (10003, 384)
labels = np.array([row[5] for row in rows])  # (10003, ) min 155, max 231
labels -= 155  # offset

tensor_embedding = torch.tensor(embedding, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.long)
# tensor_dataset = data.TensorDataset(tensor_embedding)
tensor_dataset = data.TensorDataset(tensor_embedding, tensor_labels)
dataloader = data.DataLoader(tensor_dataset, batch_size=128, shuffle=True)


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

#         # Decoder isn't used.
#         self.classifier = nn.Linear(latent_dim, 77)

#     def forward(self, X):
#         mu, log_var = self.encode(X)
#         Z = self.reparameterize(mu, log_var)
#         recon = self.decoder(Z)
#         logits = self.classifier(mu)
#         return logits, recon, mu, log_var

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r, alpha):
#         super().__init__()
#         self.W = nn.Linear(in_features, out_features, bias=False)
#         self.W.weight.requires_grad = False  # Freeze

#         self.A = nn.Linear(in_features, r, bias=False)
#         self.B = nn.Linear(r, out_features, bias=False)

#         self.alpha = alpha
#         self.r = r

#     def forward(self, x):
#         return self.W(x) + self.B(self.A(x)) * (self.alpha / self.r)

def loss_function(X, X_n, mu, log_var):
    recon_loss = f.mse_loss(X_n, X, reduction="sum")
    # l1_loss = f.l1_loss(X_n, X, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + KLD, recon_loss, KLD


def vae_classifier_loss(x, x_hat, mu, log_var, logits, labels):
    vae_total, recon_loss, kld = loss_function(x, x_hat, mu, log_var)
    # Used cross_entropy for classification
    cls_loss = f.cross_entropy(logits, labels)
    total = vae_total + cls_loss
    return total, recon_loss, kld, cls_loss

# Dataset -> TensorDataset -> DataLoader
tensor_data = torch.tensor(embedding, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.long)
dataset = data.TensorDataset(tensor_data, tensor_labels)
dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True)

model = VAE().to(device)
# model = VAE_supervised().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
recon_history = []
kld_history = []
cls_history = []

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss, total_recon, total_kld, total_cls = 0, 0, 0, 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # logits, x_hat, mu, log_var = model(batch_x)
        # loss, recon_loss, kld, cls_loss = vae_classifier_loss(
        #     batch_x, x_hat, mu, log_var, logits, batch_y
        # )
        # total_cls += cls_loss.item()

        x_hat, mu, log_var = model(batch_x)
        loss, recon_loss, kld = loss_function(batch_x, x_hat, mu, log_var)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()

    loss_history.append(total_loss)
    recon_history.append(total_recon)
    kld_history.append(total_kld)
    # cls_history.append(total_cls)

    # print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KLD: {total_kld:.2f} | Cls: {total_cls:.2f}")
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KLD: {total_kld:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Total Loss")
plt.plot(recon_history, label="Reconstruction Loss")
plt.plot(kld_history, label="KLD Loss")
# CLS
# plt.plot(cls_history, label="Classifier Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss over Epochs")
plt.savefig("temp/Loss_map.png")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/vae_banking77.pth")
print(f"Model saved to models/vae_banking77")
