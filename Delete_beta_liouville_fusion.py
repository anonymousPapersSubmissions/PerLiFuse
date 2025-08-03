import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import timm

# Text Encoder using ModernBERT
class ModernBertEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', emb_dim=32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, emb_dim)

    def forward(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True
        ).to(next(self.bert.parameters()).device)
        outputs = self.bert(**tokens)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token)

# Image Encoder using ViT
class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', emb_dim=32):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        feat_dim = self.vit.num_features
        self.fc = nn.Linear(feat_dim, emb_dim)

    def forward(self, images):
        feats = self.vit.forward_features(images)
        return self.fc(feats)

# Spectral Fusion with Beta–Liouville prior
class SpectralFusion(nn.Module):
    def __init__(self, emb_dim=32, latent_dim=64, hidden_dim=8, alpha=1.0, beta=1.0):
        super().__init__()
        # MLP to predict Beta parameters (a_f,b_f)
        self.ab_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2), nn.Softplus()
        )
        # MLP for residual gate gamma_f
        self.g_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        # Final projection to latent dim
        self.project = nn.Linear(emb_dim, latent_dim)
        # Precompute orthonormal DCT and IDCT
        self.register_buffer('dct_mat', self._build_dct_matrix(emb_dim))
        self.register_buffer('idct_mat', torch.inverse(self.dct_mat))
        # Prior hyperparameters
        self.alpha = alpha
        self.beta = beta

    def _build_dct_matrix(self, N):
        n = torch.arange(N).unsqueeze(1).float()
        k = torch.arange(N).unsqueeze(0).float()
        alpha = torch.ones(N) * (2.0 / N)**0.5
        alpha[0] = (1.0 / N)**0.5
        return alpha.unsqueeze(1) * torch.cos((torch.pi/(2*N)) * (2*n+1) * k)

    def forward(self, x, y):
        # x,y: [batch, emb_dim]
        # 1) DCT
        X = x @ self.dct_mat
        Y = y @ self.dct_mat
        # 2) Build features per frequency
        feats = torch.stack([X.abs(), Y.abs(), X - Y], dim=-1)  # [B, D, 3]
        # 3) Predict Beta parameters
        ab = self.ab_mlp(feats)              # [B, D, 2]
        a, b = ab.unbind(-1)                 # each [B, D]
        # 4) Sample sticks via Kumaraswamy
        u = torch.rand_like(a)
        v = (1 - u.pow(1/(a+1e-6))).pow(1/(b+1e-6))
        # 5) Stick-breaking to get h_f
        padded = torch.cat([torch.ones_like(v[..., :1]), 1 - v[..., :-1]], dim=-1)
        prefix = torch.cumprod(padded, dim=-1)
        h = v * prefix                       # [B, D]
        # 6) Residual gate
        gamma = self.g_mlp(feats).squeeze(-1)  # [B, D]
        # 7) Spectral fusion
        Z = h * X + (1 - h) * Y + gamma * (X - Y)
        # 8) Inverse DCT
        z = Z @ self.idct_mat                # [B, D]
        latent = self.project(z)             # [B, latent_dim]
        # 9) KL regularization term
        kl = self._beta_liouville_kl(a, b)
        return latent, kl

    def _beta_liouville_kl(self, a, b):
        # KL(Beta(a,b) || Beta(alpha,beta)) summed over sticks
        term1 = torch.lgamma(self.alpha + self.beta) - torch.lgamma(self.alpha) - torch.lgamma(self.beta)
        term2 = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        psi_ab = torch.digamma(a + b)
        kl0 = term1 - term2
        kl1 = (a - self.alpha) * (torch.digamma(a) - psi_ab)
        kl2 = (b - self.beta)  * (torch.digamma(b) - psi_ab)
        return (kl0 + kl1 + kl2).sum()

# Full PerLiFuse Model
class PerLiFuseModel(nn.Module):
    def __init__(self,
                 bert_name='bert-base-uncased',
                 vit_name='vit_base_patch16_224',
                 emb_dim=32, latent_dim=64, hidden_dim=8,
                 num_classes=2, kl_weight=1e-3):
        super().__init__()
        self.text_encoder = ModernBertEncoder(bert_name, emb_dim)
        self.img_encoder  = ViTEncoder(vit_name, emb_dim)
        self.fusion       = SpectralFusion(emb_dim, latent_dim, hidden_dim)
        self.classifier   = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )
        self.kl_weight = kl_weight

    def forward(self, texts, images, labels=None):
        x = self.text_encoder(texts)
        y = self.img_encoder(images)
        fused, kl = self.fusion(x, y)
        logits = self.classifier(fused)
        loss = None
        if labels is not None:
            ce = F.cross_entropy(logits, labels)
            loss = ce + self.kl_weight * kl
        return logits, loss

# Dataset and Training Loop
class FakeNewsDataset(Dataset):
    def __init__(self, texts, images, labels):
        self.texts, self.images, self.labels = texts, images, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.images[idx], self.labels[idx]


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for texts, imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        _, loss = model(texts, imgs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Example data
    texts = ['Sample news'] * 50
    imgs  = torch.randn(50, 3, 224, 224)
    lbls  = torch.randint(0, 2, (50,))
    dataset = FakeNewsDataset(texts, imgs, lbls)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)

    model = PerLiFuseModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
