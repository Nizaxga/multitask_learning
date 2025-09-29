import json

embeddings = []
with open("banking77out.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        embeddings.append(record)