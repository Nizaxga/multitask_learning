# demo_disentangled_framework.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ------------------ Model building blocks ------------------
class SmallEncoder(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim)
        )

    def forward(self, x):
        z = self.conv(x)
        z = F.normalize(z, dim=-1)
        return z

class TaskMaskPool(nn.Module):
    """Produce a soft mask for each task from a small task embedding."""
    def __init__(self, num_tasks, z_dim, hidden=128):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, task_id):
        te = self.task_emb(task_id)
        mask_logits = self.mlp(te)
        mask = torch.sigmoid(mask_logits)  # in (0,1)
        return mask

class TaskHead(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, out_dim)
    def forward(self, z):
        return self.fc(z)

class DisentangledMultiTaskModel(nn.Module):
    def __init__(self, encoder, num_tasks, z_dim=256, head_out_dims=None):
        super().__init__()
        self.encoder = encoder
        self.z_dim = z_dim
        self.masker = TaskMaskPool(num_tasks, z_dim)
        self.heads = nn.ModuleList()
        for od in head_out_dims:
            self.heads.append(TaskHead(z_dim, od))

    def forward(self, x, task_id):
        z = self.encoder(x)                      # [B, D]
        mask = self.masker(task_id).unsqueeze(0)  # [1, D]
        z_task = z * mask                        # masked subspace
        out = self.heads[task_id](z_task)
        return out, z, z_task, mask.squeeze(0)

# ------------------ Regularizers & losses ------------------

def cross_covariance_penalty(z_i, z_j):
    # z_i, z_j: [B, D]
    B = z_i.size(0)
    z_i = z_i - z_i.mean(dim=0, keepdim=True)
    z_j = z_j - z_j.mean(dim=0, keepdim=True)
    cov = (z_i.T @ z_j) / (B - 1)  # [D, D]
    # penalize squared off-diagonal energy (we want near-zero cross-cov)
    return (cov**2).sum()

def orthogonality_penalty(mask_a, mask_b):
    # masks in [D], encourage masks to be "different" (soft)
    # penalize dot product
    return (mask_a * mask_b).sum()

# ------------------ Simple contrastive pretrain (NT-Xent simplified) ------------------

def nt_xent_loss(z_a, z_b, temperature=0.5):
    # z_a, z_b: [B, D] normalized
    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    # mask out self-similarity
    diag_mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(diag_mask, -9e15)
    # for each index i, positive is at i^B (paired example)
    positives = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    logits = sim
    loss = F.cross_entropy(logits, positives)
    return loss

# ------------------ Data helpers ------------------

def get_mnist_loaders(batch_size=256):
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('.', train=True, download=True, transform=tf)
    test = datasets.MNIST('.', train=False, download=True, transform=tf)
    return DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(test, batch_size=batch_size)

# Few-shot subset helper

def sample_fewshot(dataset, per_class=5, num_classes=10):
    label_to_idx = {}
    for i in range(len(dataset)):
        _, y = dataset[i]
        label_to_idx.setdefault(y, []).append(i)
    chosen = []
    classes = sorted(label_to_idx.keys())[:num_classes]
    for c in classes:
        chosen += random.sample(label_to_idx[c], min(per_class, len(label_to_idx[c])))
    return Subset(dataset, chosen)

# ------------------ End-to-end demo flows ------------------

def pretrain_encoder_contrastive(encoder, dataloader, device, epochs=1, lr=1e-3):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f'Pretrain ep{ep}')
        for x, _ in pbar:
            # make two augmentations: here we use simple random crop + noise for demo
            xa = x + (torch.randn_like(x) * 0.01)
            xb = x + (torch.randn_like(x) * 0.01)
            xa, xb = xa.to(device), xb.to(device)
            za = encoder(xa)
            zb = encoder(xb)
            loss = nt_xent_loss(za, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    return encoder


def finetune_head_fewshot(model, dataset, task_id, device, shots=5, epochs=10, lr=1e-2):
    # dataset is a torchvision dataset (trainable)
    subset = sample_fewshot(dataset, per_class=shots)
    dl = DataLoader(subset, batch_size=shots*2, shuffle=True)
    # freeze encoder and masker, train only head
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.masker.parameters(): p.requires_grad = False
    for p in model.heads.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device).train()
    for ep in range(epochs):
        pbar = tqdm(dl, desc=f'Finetune task{task_id} ep{ep}')
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            tid = torch.tensor(task_id, dtype=torch.long, device=device)
            out, z, z_task, mask = model(x, tid)
            loss = loss_fn(out, y)
            # small disentanglement regularizer — keep mask concentrated (optional)
            reg = 0.0
            opt.zero_grad(); (loss+reg).backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    return model

# ------------------ Run demo ------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 256
    num_tasks = 2
    head_outs = [10, 10]  # e.g. task0: MNIST classification, task1: MNIST classification with different label mapping

    encoder = SmallEncoder(z_dim=z_dim)
    model = DisentangledMultiTaskModel(encoder, num_tasks, z_dim=z_dim, head_out_dims=head_outs)

    # Pretrain encoder (self-supervised)
    train_dl, test_dl = get_mnist_loaders(batch_size=256)
    print('Pretraining encoder (1 epoch demo)')
    pretrain_encoder_contrastive(model.encoder, train_dl, device, epochs=1)

    # Few-shot finetune head for task 0
    print('Few-shot finetune head for task 0 (5 shots per class)')
    finetune_head_fewshot(model, datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()), task_id=0, device=device, shots=5, epochs=3)

    print('Demo finished — encoder frozen, head trained with few-shot.')
