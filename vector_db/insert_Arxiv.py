from collections import defaultdict
import mteb
import torch
import psycopg2
from psycopg2.extras import Json

# from sentence_transformers import SentenceTransformer
from transformers import LongformerModel, LongformerTokenizer
from multiprocessing import cpu_count
from config import HOST, DBNAME, USERNAME, PASSWORD
from itertools import islice


def batched(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def list_attr(task):
    for attr in dir(task):
        if not attr.startswith("_"):
            print(attr)


# model_name = "all-MiniLM-L6-v2"
split = "train"
# print(f"[LOG] Load Model {model_name}")
# model = SentenceTransformer(f"sentence-transformers/{model_name}")
# embedding_size = model.get_sentence_embedding_dimension()

# model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# embedding_size = model.config.hidden_size
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# model.eval()


def embed_longformer_batch(texts, tokenizer, model, device="cpu", max_length=4096):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )

            sequence_output = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            )
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            embeddings.append(mean_embedding.squeeze().cpu().numpy())

    return embeddings


def embed_longformer(
    texts, tokenizer, model, device="cpu", batch_size=4, max_length=4096
):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = embed_longformer_batch(
            batch_texts, tokenizer, model, device, max_length
        )
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


dataset_name = "ArxivClassification"
task = mteb.get_task(task_name=dataset_name)
task.load_data()
# print(task.dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 28388
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 2500
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 2500
#     })
# })
dataset_name += "_" + split
texts = [row["text"] for row in task.dataset[split]]
labels = [row["label"] for row in task.dataset[split]]
# No label_text therefor just passing int to str
print("[LOG] Restructure")
label_to_labeltext = {label: str(label) for label in labels}
label_to_texts = defaultdict(list)
for label, text in zip(labels, texts):
    label_to_texts[label].append(text)
assert set(label_to_texts.keys()) <= set(label_to_labeltext.keys()), (
    "Mismatch in label mapping!"
)
print("[LOG] All labels in label_to_texts have corresponding label_text.")

# print("[LOG] Create Embedding")
# label_to_embedding = {}
# cnt = 0
# for label, texts in label_to_texts.items():
#     print(f"[LOG] Embedding label {label} with {len(texts)} texts")
#     embeddings = embed_longformer(texts, tokenizer, model, device=device, batch_size=4)
#     print(f"[LOG] GOT {len(embeddings)} back")
#     label_to_embedding[label] = embeddings
#     cnt += len(embeddings)

from concurrent.futures import ThreadPoolExecutor, as_completed

print("[LOG] Create Embedding with Parallelism")
label_to_embedding = {}
cnt = 0


def embed_label_group(label_text_pair):
    label, texts = label_text_pair
    print(f"[LOG] Embedding label {label} with {len(texts)} texts")
    embeddings = embed_longformer(texts, tokenizer, model, device=device, batch_size=4)
    print(f"[LOG] Done label {label}")
    return label, embeddings


with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(embed_label_group, item): item[0]
        for item in label_to_texts.items()
    }
    for future in as_completed(futures):
        label, embeddings = future.result()
        label_to_embedding[label] = embeddings
        cnt += len(embeddings)

print("[LOG] VectorDB, pgvector insertion")
try:
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

    def batched(iterable, batch_size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch

    for chunk in batched(vector_records, 1000):
        curr.executemany(
            """
            INSERT INTO vectors (dataset_id, embedding, metadata, label_id)
            VALUES (%s, %s, %s, %s);
            """,
            chunk,
        )
    # curr.executemany(
    #     """
    #     INSERT INTO vectors (dataset_id, embedding, metadata, label_id)
    #     VALUES (%s, %s, %s, %s);
    #     """,
    #     vector_records,
    # )

    conn.commit()

except Exception as e:
    print(f"[ERR] {e}")
    conn.rollback()

finally:
    curr.close()
    conn.close()
    print("[LOG] DONE")
