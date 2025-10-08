from typing import overload
import torch
from torch import nn
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from config import HOST, DBNAME, USERNAME, PASSWORD
import torch.utils.data as data
import torch.nn.functional as f
import mteb

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
        # Z = self.reparameterize(mu / mu.norm(p=2, dim=1, keepdim=True), log_var)
        Z = self.reparameterize(mu, log_var)

        # Normalization might work might break who know.
        # Turn out, It make worse. 0.11
        # Z = Z / Z.norm(p=2, dim=1, keepdim=True)

        return self.decoder(Z), mu, log_var

    # def forward(self, X):
    #     mu, log_var = self.encode(X)
    #     Z = self.reparameterize(mu, log_var)
    #     return self.decoder(Z), mu, log_var

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

embedding = np.array([row[2] for row in trains])  # (10003, 384)
labels = np.array([row[5] for row in trains])  # (10003, ) min 155, max 231
labels -= 155  # offset min 0, max 77

tensor_embedding = torch.tensor(embedding, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.long)
tensor_dataset = data.TensorDataset(tensor_embedding, tensor_labels)
dataloader = data.DataLoader(tensor_dataset, batch_size=128, shuffle=True)

test_embedding = np.array([row[2] for row in tests])  # (3080, 384)
test_labels = np.array([row[5] for row in tests])  # (3080, 384)
test_labels -= 78

test_data = torch.tensor(test_embedding, dtype=torch.float32)
test_label = torch.tensor(test_labels, dtype=torch.long)
test_dataset = data.TensorDataset(test_data, test_label)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# def build_text_to_embedding(rows):
#     text_to_emb = {}
#     for row in rows:
#         sentence_text = row[3]["text"]
#         emb = row[2]
#         text_to_emb[sentence_text] = emb
#     return text_to_emb


# text_to_embedding_test = build_text_to_embedding(tests)
# text_to_embedding_train = build_text_to_embedding(trains)
# text_to_embedding = {**text_to_embedding_test, **text_to_embedding_train}

# model = VAE().to(device)
# model.load_state_dict(torch.load(state, weights_only=True))
# model.eval()

# class All_MiniLM():
#     def __init__(self, sentence_to_embedding: dict):
#         self.sentence_to_embedding = sentence_to_embedding

#     def encode(self, sentences, batch_size=128, **kwargs):
#         return np.array([self.sentence_to_embedding[s] for s in sentences])

# class VAE_eval():
#     def __init__(self, sentence_to_embedding: dict):
#         self.sentence_to_embedding = sentence_to_embedding
#         self.model = model
#         self.model.eval()

#     def encode(self, sentences, batch_size=128, **kwargs):
#         embedding = np.array([self.sentence_to_embedding[s] for s in sentences])
#         tensor_embedding = torch.tensor(embedding, dtype=torch.float32)

#         with torch.no_grad():
#             mu, log_var = self.model.encode(tensor_embedding.to(device))
#             # mu = mu / mu.norm(p=2, dim=1, keepdim=True)
#             latent = mu.cpu().numpy()

#         # Normalization before send out to evaluation
#         # Seem to improve from random guessing 0.12 accuracy to 0.38 accuracy
#         # Maybe due to the high-dim ball ended up have small radiaus
#         # And classification evaluation can't tell the difference between each class.
#         latent /= np.linalg.norm(latent, axis=1, keepdims=True)

#         return latent



# model_eval = VAE_eval(text_to_embedding)
# # model_eval = All_MiniLM(text_to_embedding) # Same performance as All-MiniLM-L6-V2
# task = mteb.get_tasks(tasks=["Banking77Classification"])
# evaluator = mteb.MTEB(task)
# evaluator.run(model_eval, output_folder="output/")

# Ways to create classifcation label (might work)
# 1. space partitioning
    # 1. 
# 2. Local Sensitive Hashing
    # 1. Create hash family
    # 2. L copies of hash family
    # 3. 


class ClassiNet(nn.Module):
    def __init__(self, num_label=77):
        super(ClassiNet, self).__init__()
        self.classi = nn.Linear(77, 77)
    
    def forward(self, x):
        return self.classi(x)

model = VAE().to(device)
model.load_state_dict(torch.load(state, weights_only=True))
model.eval()

classi_model = ClassiNet().to(device)
optimizer = torch.optim.Adam(classi_model.parameters(), lr=1e-3)

epochs = 200

def train_eval():
    classi_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                mu, log_var = model.encode(data)
                mu = mu / mu.norm(p=2, dim=1, keepdim=True)
                # Z = model.reparameterize(mu, log_var)

            optimizer.zero_grad()
            output = classi_model(mu)
            loss = f.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# Evaluation function
def evaluate_classifier():
    classi_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            mu, _ = model.encode(batch_x)
            mu = mu / mu.norm(p=2, dim=1, keepdim=True)
            outputs = classi_model(mu)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

train_eval()
evaluate_classifier()