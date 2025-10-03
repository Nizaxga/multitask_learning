import mteb
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
print(f"[LOG] Load Model {model_name}")
model = SentenceTransformer(f"sentence-transformers/{model_name}")
embedding_size = model.get_sentence_embedding_dimension()

task = mteb.get_tasks(["SciFact"])
evaluator = mteb.MTEB(task)
evaluator.run(model)