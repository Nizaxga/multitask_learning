import mteb
from datasets import Dataset, DatasetDict
from typing import Sequence
import ollama


def embedding(x) -> Sequence[Sequence[float]]:
    z = ollama.embed(model="nomic-embed-text", input=x).embeddings
    return z


task = mteb.get_task(task_name="Banking77Classification")

task.load_data()
DATA = task.dataset.data["train"]["text"]
DATA = [str(text) for text in DATA]

# print(DATA.data['train']['text'])
# print(DATA.data['train']['label'])
# print(DATA.data['train']['label_text'])

# for text in DATA.data['train']['text']:
#     text = str(text)
