import torch
from torch import nn
from torch.nn import functional as f

class VAE_model(nn.Module):
    def __init__(self, latent_dim=77, input_dim=384, hidden_dim=192):
        super(VAE_model, self).__init__()

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid(),
        )

    def encode(self, x):
        temp = self.encoder(x)
        mu = self.mean(temp)
        log_var = self.log_var(temp)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, X):
        mu, log_var = self.encode(X)
        # Normalize mu kinda work? 0.35 accuracy.
        # But not as much as just normalize at classification model.
        # Z = self.reparameterize(mu / mu.norm(p=2, dim=1, keepdim=True), log_var)
        Z = self.reparameterize(mu, log_var)

        # Normalization might work might break who know.
        # Turn out, It make worse. 0.11
        # Z = Z / Z.norm(p=2, dim=1, keepdim=True)

        return self.decoder(Z), mu, log_var

# class VAE_supervised(VAE):
#     def __init__(self, latent_dim=77, input_dim=384, hidden_dim=192):
#         super().__init__(latent_dim, input_dim, hidden_dim)

#         self.classifier = nn.Linear(latent_dim, 77)

#     def forward(self, X):
#         mu, log_var = self.encode(X)
#         Z = self.reparameterize(mu, log_var)
#         recon = self.decoder(Z)
#         logits = self.classifier(mu)
#         return logits, recon, mu, log_var

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r, alpha):
#         super().__init__()
#         self.W = nn.Linear(in_features, out_features, bias=False)
#         self.W.weight.requires_grad = False  # Freeze

#         self.A = nn.Linear(in_features, r, bias=False)
#         self.B = nn.Linear(r, out_features, bias=False)

#         self.alpha = alpha
#         self.r = r

#     def forward(self, x):
#         return self.W(x) + self.B(self.A(x)) * (self.alpha / self.r)

def loss_function(X, X_n, mu, log_var):
    recon_loss = f.mse_loss(X_n, X, reduction="sum")
    # l1_loss = f.l1_loss(X_n, X, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + (5 * KLD), recon_loss, KLD


def vae_classifier_loss(x, x_hat, mu, log_var, logits, labels):
    vae_total, recon_loss, kld = loss_function(x, x_hat, mu, log_var)
    # Used cross_entropy for classification
    cls_loss = f.cross_entropy(logits, labels)
    total = vae_total + cls_loss
    return total, recon_loss, kld, cls_loss

# Use Prototypical Networks or Contrastive Learning (as alternatives)
# Alternatives like contrastive loss, InfoNCE, or ProtoNets might work better for adaptability. Let me know if you're interested in those.
# Used triplet_loss for task clustering.

# # Reference
# def triplet_loss(anchor, positive, negative, margin=1.0):
#     pos_dist = f.pairwise_distance(anchor, positive, p=2)
#     neg_dist = f.pairwise_distance(anchor, negative, p=2)
#     loss = f.relu(pos_dist - neg_dist + margin)
#     return loss.mean()

# # Reference
# def vae_triplet_loss(x, x_hat, mu, log_var, anchor, positive, negative, margin=1.0):
#     vae_total, recon_loss, kld = loss_function(x, x_hat, mu, log_var)
#     trip_loss = triplet_loss(anchor, positive, negative, margin)
#     total_loss = vae_total + trip_loss
#     return total_loss, recon_loss, kld, trip_loss

