import mteb
from sentence_transformers import SentenceTransformer

def list_attr(task):
    for attr in dir(task):
        if not attr.startswith("_"): 
            print(attr)

model_name = "all-MiniLM-L6-v2"
print(f"[LOG] Load Model {model_name}")
model = SentenceTransformer(f"sentence-transformers/{model_name}")
embedding_size = model.get_sentence_embedding_dimension()

dataset_name = "DBpediaClassification"
# dataset_name = "Banking77Classification"
task = mteb.get_task(task_name=dataset_name)
task.load_data()
print(task.dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['label', 'title', 'text'],
#         num_rows: 2048
#     })
#     test: Dataset({
#         features: ['label', 'title', 'text'],
#         num_rows: 2048
#     })
# })