import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pgvector.psycopg2 import register_vector
import psycopg2
from config import HOST, DBNAME, USERNAME, PASSWORD

model_name = "all-MiniLM-L6-v2"
split = "_train"
dataset_name = "Banking77Classification" + split

conn = psycopg2.connect(
    host=HOST,
    dbname=DBNAME,
    user=USERNAME,
    password=PASSWORD,
)

# pgvector query
register_vector(conn)
curr = conn.cursor()
curr.execute("""
SELECT * FROM vectors
WHERE dataset_id = (
    SELECT id FROM vector_datasets
    WHERE name = %s
);
""", (dataset_name,))
rows = curr.fetchall()
curr.close()
conn.close()

def low_rank_approximation(k):
    U, S, Vt = np.linalg.svd(embedding, full_matrices=False)
    embedding_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return embedding_k

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
embedding_k = low_rank_approximation(k=5)
# print(f"{embedding_k.shape=}")

labels = np.array([row[5] for row in rows]) # (10003, ) min 155, max 231
labels -= 155 # offset

# X_svd = U[:, :2] @ np.diag(S[:2])   # project to first 2 singular vectors
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     X_svd[:, 1],
#     X_svd[:, 0],
#     c=labe,
#     alpha=0.6,
#     s=10
# )
# plt.colorbar(scatter, label="Label ID")
# plt.tight_layout()
# plt.savefig("embeddings_svd", dpi=300)

def project_to_class_axis(embedding, labels, class_a, class_b):
    mu_a = embedding[labels == class_a].mean(axis=0)
    mu_b = embedding[labels == class_b].mean(axis=0)

    v = mu_a - mu_b
    v /= np.linalg.norm(v)  

    projection = embedding @ v
    return projection, v

proj, v = project_to_class_axis(embedding, labels, 0, 1)

plt.figure(figsize=(10, 6))
plt.hist(proj[labels == 0], bins=50, alpha=0.6, label="Class 0")
plt.hist(proj[labels == 1], bins=50, alpha=0.6, label="Class 1")
plt.hist(proj[(labels != 0) & (labels != 1)], bins=50, alpha=0.6, label="Others")
plt.legend()
plt.xlabel("Projection value")
plt.ylabel("Count")
plt.savefig("LDA", dpi=300)