from typing import overload
import torch
from torch import nn
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from config import HOST, DBNAME, USERNAME, PASSWORD
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
        Z = self.reparameterize(mu / mu.norm(p=2, dim=1, keepdim=True), log_var)

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


def build_text_to_embedding(rows):
    text_to_emb = {}
    for row in rows:
        sentence_text = row[3]["text"]
        emb = row[2]
        text_to_emb[sentence_text] = emb
    return text_to_emb


text_to_embedding_test = build_text_to_embedding(tests)
text_to_embedding_train = build_text_to_embedding(trains)
text_to_embedding = {**text_to_embedding_test, **text_to_embedding_train}

model = VAE().to(device)
model.load_state_dict(torch.load(state, weights_only=True))
model.eval()

class All_MiniLM():
    def __init__(self, sentence_to_embedding: dict):
        self.sentence_to_embedding = sentence_to_embedding

    def encode(self, sentences, batch_size=128, **kwargs):
        return np.array([self.sentence_to_embedding[s] for s in sentences])

class VAE_eval():
    def __init__(self, sentence_to_embedding: dict):
        self.sentence_to_embedding = sentence_to_embedding
        self.model = model
        self.model.eval()

    def encode(self, sentences, batch_size=128, **kwargs):
        embedding = np.array([self.sentence_to_embedding[s] for s in sentences])
        tensor_embedding = torch.tensor(embedding, dtype=torch.float32)

        with torch.no_grad():
            mu, log_var = self.model.encode(tensor_embedding.to(device))
            # mu = mu / mu.norm(p=2, dim=1, keepdim=True)
            latent = mu.cpu().numpy()

        # Normalization before send out to evaluation
        # Seem to improve from random guessing 0.12 accuracy to 0.38 accuracy
        # Maybe due to the high-dim ball ended up have small radiaus
        # And classification evaluation can't tell the difference between each class.
        latent /= np.linalg.norm(latent, axis=1, keepdims=True)

        return latent



model_eval = VAE_eval(text_to_embedding)
# model_eval = All_MiniLM(text_to_embedding) # Same performance as All-MiniLM-L6-V2
task = mteb.get_tasks(tasks=["Banking77Classification"])
evaluator = mteb.MTEB(task)
evaluator.run(model_eval, output_folder="output/")


# Ways to create classifcation label (might work)
# 1. space partitioning
    # 1. 
# 2. Local Sensitive Hashing
    # 1. Create hash family
    # 2. L copies of hash family
    # 3. 