# Probably won't work
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
batch_size = 128
lr = 0.0002
num_epochs = 50
image_size = 28 * 28  # 784
hidden_dim = 256

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(latent_dim, hidden_dim, image_size).to(device)
discriminator = Discriminator(image_size, hidden_dim).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size_curr = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size_curr, -1).to(device)

        # Labels
        real = torch.ones(batch_size_curr, 1).to(device)
        fake = torch.zeros(batch_size_curr, 1).to(device)

        ### Train Discriminator ###
        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_imgs = generator(z).detach()  # Avoid training G here
        d_real = discriminator(real_imgs)
        d_fake = discriminator(fake_imgs)

        loss_real = criterion(d_real, real)
        loss_fake = criterion(d_fake, fake)
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ### Train Generator ###
        z = torch.randn(batch_size_curr, latent_dim).to(device)
        gen_imgs = generator(z)
        d_gen = discriminator(gen_imgs)
        loss_G = criterion(d_gen, real)  # We want the generator to fool D

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if i % 200 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Optional: Generate sample images
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        gen_imgs = generator(z).view(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f"Epoch {epoch}")
        plt.axis("off")
        plt.show()
