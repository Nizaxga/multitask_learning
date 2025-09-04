import mteb
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# GLOBAL VARIABLE
model_name = "testing_model"
dataset_name_1 = "Banking77Classification"
dataset_name_2 = "STSBenchmark"

model = SentenceTransformer(model_name)

task1 = mteb.get_task(task_name=dataset_name_1)
task1.load_data()
data_1 = [row["text"] for row in task1.dataset["test"]]

task2 = mteb.get_task(task_name=dataset_name_2)
task2.load_data()
data_2_1 = [row["sentence1"] for row in task2.dataset["test"]]
data_2_2 = [row["sentence2"] for row in task2.dataset["test"]]
data_2 = [text for pair in zip(data_2_1, data_2_2) for text in pair]

embedd_1 = model.encode(data_1, convert_to_numpy=True, show_progress_bar=True)
embedd_2 = model.encode(data_2, convert_to_numpy=True, show_progress_bar=True)

output_1 = os.path.join("embedded", f"{dataset_name_1}_{model_name}.npy")
np.save(output_1, embedd_1)
print("[LOG] Done embedding Banking77Classification")

output_2 = os.path.join("embedded", f"{dataset_name_2}_{model_name}.npy")
np.save(output_2, embedd_2)
print("[LOG] Done embedding STSBenchmark")