import mteb
from datasets import Dataset, DatasetDict
from typing import Sequence
import ollama
import json


def embedding(x) -> Sequence[Sequence[float]]:
    z = ollama.embed(model="nomic-embed-text", input=x).embeddings
    return z


task = mteb.get_task(task_name="Banking77Classification")
task.load_data()

DATA = task.dataset["train"]  # This is a HuggingFace Dataset object

output_file = "banking77out.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for example in DATA:
        text = str(example["text"])
        label = example["label"]  # Optional, include if needed
        embed = embedding(text)[0]  # get the embedding vector (single item)
        record = {"text": text, "label": label, "embedding": embed}
        f.write(json.dumps(record) + "\n")

print(f"Saved embeddings to {output_file}")
