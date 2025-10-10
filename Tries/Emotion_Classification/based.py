import mteb
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
print(f"[LOG] Load Model {model_name}")
model = SentenceTransformer(f"sentence-transformers/{model_name}")
dataset_name = "EmotionClassification"

task = mteb.get_tasks(tasks=[dataset_name])
evaluator = mteb.MTEB(task)
evaluator.run(model, output_folder="output/")