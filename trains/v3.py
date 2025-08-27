import mteb
import ollama
from typing import Sequence


def embedding(x) -> Sequence[Sequence[float]]:
    return ollama.embed(model="nomic-embed-text", input=x).embeddings


task_names = [
    "Banking77Classification",
    "MSMARCO",
    "STSBenchmark",
]


tasks = mteb.get_tasks(tasks=task_names)
print(tasks.to_dataframe())
benchmark = mteb.MTEB(tasks=tasks)
print(benchmark)
