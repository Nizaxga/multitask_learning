from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random

# Load shared encoder
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Freeze encoder if desired
for param in encoder.parameters():
    param.requires_grad = False

# Task-specific heads
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class RetrievalHead(nn.Module):
    def __init__(self, input_dim, projection_dim=256):
        super().__init__()
        self.project = nn.Linear(input_dim, projection_dim)

    def forward(self, x):
        return self.project(x)

cls_head = ClassificationHead(384, num_classes=77)
ret_head = RetrievalHead(384)

# Dummy data (you should use MTEB tasks here)
sample_texts = ["What is my account balance?", "Find ATMs near me", "Transfer money", "How to close account?"]
sample_labels = [0, 1, 2, 3]  # fake intent classes

train_data_cls = [
    InputExample(texts=[text], label=label) for text, label in zip(sample_texts, sample_labels)
]

train_data_ret = [
    InputExample(texts=["Transfer money", "Send money"], label=1.0),
    InputExample(texts=["Check balance", "What is my balance?"], label=1.0),
    InputExample(texts=["Transfer money", "Open account"], label=0.0),
]

loader_cls = DataLoader(train_data_cls, batch_size=2, shuffle=True)
loader_ret = DataLoader(train_data_ret, batch_size=2, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(list(cls_head.parameters()) + list(ret_head.parameters()), lr=2e-5)
cosine = nn.CosineSimilarity(dim=1)
bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):
    for cls_batch, ret_batch in zip(loader_cls, loader_ret):
        optimizer.zero_grad()

        # Classification forward
        embeddings_cls = encoder.encode([ex.texts[0] for ex in cls_batch], convert_to_tensor=True)
        logits = cls_head(embeddings_cls)
        labels_cls = torch.tensor([ex.label for ex in cls_batch])
        loss_cls = ce(logits, labels_cls)

        # Retrieval forward
        sentences1 = [ex.texts[0] for ex in ret_batch]
        sentences2 = [ex.texts[1] for ex in ret_batch]
        emb1 = encoder.encode(sentences1, convert_to_tensor=True)
        emb2 = encoder.encode(sentences2, convert_to_tensor=True)
        proj1 = ret_head(emb1)
        proj2 = ret_head(emb2)
        cos_sim = cosine(proj1, proj2)
        labels_ret = torch.tensor([ex.label for ex in ret_batch])
        loss_ret = bce(cos_sim, labels_ret)

        loss = loss_cls + loss_ret
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
