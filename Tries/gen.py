from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("distilbert-base-uncased")

# 2. Prepare training data (pairs with similarity scores)
train_examples = [
    InputExample(
        texts=["A man is eating food.", "A person is consuming a meal."], label=0.9
    ),
    InputExample(texts=["The weather is sunny.", "It is raining heavily."], label=0.1),
    InputExample(
        texts=["Dogs are running in a park.", "Canines are playing outdoors."],
        label=0.85,
    ),
    # Add more examples here
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

num_epochs = 1
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
)

model.save("output/all-MiniLM-L6-v2-custom")
