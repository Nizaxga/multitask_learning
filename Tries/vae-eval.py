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
import mteb

model_name = "all-MiniLM-L6-v2"
split = "_test"
dataset_name = "Banking77Classification" + split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state = "models/vae_banking77_supervised.pth"
state = "models/vae_banking77.pth"
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
        sentence_text = row[3]["text"]  # assuming JSONB or dict
        emb = row[2]                    # vector
        text_to_emb[sentence_text] = emb
    return text_to_emb

text_to_embedding_test = build_text_to_embedding(tests)
text_to_embedding_train = build_text_to_embedding(trains)
text_to_embedding = {**text_to_embedding_test, **text_to_embedding_train}

model = VAE().to(device)
model.load_state_dict(torch.load(state, weights_only=True))
model.eval()

# with torch.no_grad():
#     x_hat, mu, log_var = model(tensor_embedding.to(device))
# x_hat = x_hat.cpu().numpy()

class VAE_eval():
    def __init__(self, sentence_to_embedding: dict, vae_model: VAE, device=device):
        self.sentence_to_embedding = sentence_to_embedding
        self.vae = vae_model.to(device)
        self.vae.eval()
        self.device = device

    def encode(self, sentences, batch_size=128, **kwargs):
        return np.array([self.sentence_to_embedding[s] for s in sentences])
        inputs = np.array([self.sentence_to_embedding[s] for s in sentences])
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        # outputs = []
        # with torch.no_grad():
        #     for i in range(0, len(inputs), batch_size):
        #         batch = inputs[i:i+batch_size]
        #         mu, log_var = self.vae.encode(batch)
        #         z = self.vae.reparameterize(mu, log_var)
        #         outputs.append(z.cpu())
        # return torch.cat(outputs, dim=0).numpy()


model_eval = VAE_eval(text_to_embedding, model)
task = mteb.get_tasks(tasks=["Banking77Classification"])
evaluator = mteb.MTEB(task)
evaluator.run(model_eval, output_folder="temp/")