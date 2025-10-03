from collections import defaultdict
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import numpy as np

def list_attr(task):
    for attr in dir(task):
        if not attr.startswith("_"): 
            print(attr)

# model_name = "all-MiniLM-L6-v2"
# print(f"[LOG] Load Model {model_name}")
# model = SentenceTransformer(f"sentence-transformers/{model_name}")
# embedding_size = model.get_sentence_embedding_dimension()


dataset = "quora"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

# Download and extract dataset
data_path = util.download_and_unzip(url, "datasets")

# Load corpus, queries, and qrels for test split
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Corpus size: {len(corpus)}")
print(f"Queries size: {len(queries)}")
print(f"Qrels size: {len(qrels)}")
