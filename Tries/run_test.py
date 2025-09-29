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
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
split = "_test"
dataset_name = "Banking77Classification" + split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state = "models/vae_banking77_supervised.pth"
state = "models/vae_banking77.pth"
print("[LOG] Using ", device)

model = SentenceTransformer(f"sentence-transformers/{model_name}")
task = mteb.get_tasks(tasks=["Banking77Classification"])
evaluator = mteb.MTEB(task)
evaluator.run(model, output_folder="temp/")