from collections import defaultdict
import mteb
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from config import HOST, DBNAME, USERNAME, PASSWORD

model_name = "all-MiniLM-L6-v2"
embedding_size = 384

dataset_name = "Banking77Classification"

# vecdb
conn = psycopg2.connect(
    host=HOST,
    dbname=DBNAME,
    user=USERNAME,
    password=PASSWORD,
)
curr = conn.cursor()


conn.commit()
curr.close()
conn.close()
print("[LOG] DONE")