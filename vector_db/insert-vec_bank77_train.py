from collections import defaultdict
import mteb
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
from config import HOST, DBNAME, USERNAME, PASSWORD

model_name = "all-MiniLM-L6-v2"

print(f"[LOG] Load Model {model_name}")
model = SentenceTransformer(f"sentence-transformers/{model_name}")
embedding_size = model.get_sentence_embedding_dimension()

dataset_name = "Banking77Classification"
print(f"[LOG] Load dataset {dataset_name}")
task = mteb.get_task(task_name=dataset_name)
task.load_data()

dataset_name = "Banking77Classification_train"
texts = [row["text"] for row in task.dataset["train"]]
labels = [row["label"] for row in task.dataset["train"]]
label_texts = [row["label_text"] for row in task.dataset["train"]]

print("[LOG] Restructure")
label_to_labeltext = {
    label: label_text for label, label_text in zip(labels, label_texts)
}
label_to_texts = defaultdict(list)
for label, text in zip(labels, texts):
    label_to_texts[label].append(text)

assert set(label_to_texts.keys()) <= set(label_to_labeltext.keys()), (
    "Mismatch in label mapping!"
)
print("[LOG] All labels in label_to_texts have corresponding label_text.")

print("[LOG] Create Embedding")
label_to_embedding = {}
cnt = 0
for label, text in label_to_texts.items():
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    label_to_embedding[label] = embedding
    cnt += len(embedding)

print("[LOG] VectorDB, pgvector insertion")
# vecdb
conn = psycopg2.connect(
    host=HOST,
    dbname=DBNAME,
    user=USERNAME,
    password=PASSWORD,
)
curr = conn.cursor()

curr.execute(
    f"""
    INSERT INTO vector_datasets (name, description, dimension)
    VALUES ('{dataset_name}', '{cnt} rows', {embedding_size})
    ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description
    RETURNING id;
    """
)
dataset_id = curr.fetchone()[0]

label_records = [
    (dataset_id, label_text) for _, label_text in label_to_labeltext.items()
]
curr.executemany(
    """
    INSERT INTO labels (dataset_id, name)
    VALUES (%s, %s)
    ON CONFLICT (dataset_id, name) DO UPDATE SET name = EXCLUDED.name;
    """,
    label_records,
)

curr.execute("SELECT id, name FROM labels WHERE dataset_id = %s;", (dataset_id,))
name_to_id = {name: lid for lid, name in curr.fetchall()}

vector_records = []
for label, embeddings in label_to_embedding.items():
    label_text = label_to_labeltext[label]
    label_id = name_to_id[label_text]
    for emb, text in zip(embeddings, label_to_texts[label]):
        vector_records.append(
            (dataset_id, emb.tolist(), Json({"text": text}), label_id)
        )

curr.executemany(
    """
    INSERT INTO vectors (dataset_id, embedding, metadata, label_id)
    VALUES (%s, %s, %s, %s);
    """,
    vector_records,
)

conn.commit()
curr.close()
conn.close()
print("[LOG] DONE")

