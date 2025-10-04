import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pgvector.psycopg2 import register_vector
import psycopg2
from config import HOST, DBNAME, USERNAME, PASSWORD
import torch
import torch.optim as optim
import torch.utils.data as data
from Tries.Banking77_Classification.model import VAE_model, loss_function

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

# vectors_table
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
labels -= 155  # offset min 0, max 77

tensor_embedding = torch.tensor(embedding, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.long)
# tensor_dataset = data.TensorDataset(tensor_embedding)
tensor_dataset = data.TensorDataset(tensor_embedding, tensor_labels)
dataloader = data.DataLoader(tensor_dataset, batch_size=128, shuffle=True)

# Dataset -> TensorDataset -> DataLoader
tensor_data = torch.tensor(embedding, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.long)
dataset = data.TensorDataset(tensor_data, tensor_labels)
dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True)

model = VAE_model().to(device)
# model = VAE_supervised().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
recon_history = []
kld_history = []
cls_history = []
tri_history = []

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss, total_recon, total_kld, total_cls, total_tri = 0, 0, 0, 0, 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # with MSEloss
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
    # tri_history.append(total_tri)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KLD: {total_kld:.2f}")
    # print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KLD: {total_kld:.2f} | Cls: {total_cls:.2f}")
    # print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KLD: {total_kld:.2f} | Tri: {total_tri:.2f}")

def plot_losses():
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Total Loss")

    # ELBO
    plt.plot(recon_history, label="Recon Loss")
    plt.plot(kld_history, label="KLD Loss")

    # Third Loss
    # plt.plot(cls_history, label="Classifier Loss")
    # plt.plot(tri_history, label="Triplet Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.savefig("output/loss_map.png")

plot_losses()

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/vae_banking77.pth")
print(f"Model saved to models/vae_banking77")
# torch.save(model.state_dict(), "models/vae_banking77_supervised.pth")
# print(f"Model saved to models/vae_banking77_supervised")
