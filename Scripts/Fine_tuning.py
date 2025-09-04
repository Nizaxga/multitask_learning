from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import mteb

task1 = mteb.get_task("Banking77Classification")
task1.load_data()
task2 = mteb.get_task("STSBenchmark")
task2.load_data()

labels = list(set(row["label"] for row in task1.dataset["train"]))
label2id = {label: idx for idx, label in enumerate(labels)}

train_data_banking = [
    InputExample(texts=[row["text"], row["text"]], label=label2id[row["label"]])
    for row in task1.dataset["train"]
]

train_data_sts = [
    InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"] / 5.0)
    for row in task2.dataset["train"]
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

loader_banking = DataLoader(train_data_banking, batch_size=32, shuffle=True)
loader_sts = DataLoader(train_data_sts, batch_size=32, shuffle=True)

loss_banking = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=len(label2id)
)

loss_sts = losses.CosineSimilarityLoss(model=model)

train_objectives = [
    (loader_banking, loss_banking),
    (loader_sts, loss_sts)
]

model.fit(
    train_objectives=train_objectives,
    epochs=1,
    warmup_steps=100,
    output_path="testing_model"
)
