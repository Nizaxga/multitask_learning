import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class V1(nn.modules):
    def __init__(self, latent_dim=10, hidden_dim=64, image_size=32*32):
        super(V1, self).__init__()

        self.Gen = nn.Sequential(

        )

        self.Dis = nn.Sequential(

        )
    
    def forward():
        pass

v1 = V1().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(v1.parameters(), lr=lr)