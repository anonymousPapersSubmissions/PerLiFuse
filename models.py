from importlib_metadata import re
import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import timm

device = torch.device("cuda")

from importlib_metadata import re
import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoTokenizer


device = torch.device("cuda")

class CoherenceGuide(nn.Module):
    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        # Cross-attention mechanism for coherence modeling
        self.text_proj = nn.Linear(emb_dim, emb_dim)
        self.img_proj = nn.Linear(emb_dim, emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        # Project features to coherence space
        q = self.text_proj(x).unsqueeze(1)  # [batch, 1, emb_dim]
        k = self.img_proj(y).unsqueeze(1)   # [batch, 1, emb_dim]
        v = y.unsqueeze(1)                  # [batch, 1, emb_dim]
        
        # Cross-attention: text queries to image values
        attn_out, _ = self.attention(q, k, v)
        aligned_img = attn_out.squeeze(1)   # [batch, emb_dim]
        
        # Compute coherence features
        diff = torch.abs(x - aligned_img)
        prod = x * aligned_img
        coh_feats = torch.cat([x, aligned_img, diff, prod], dim=-1)
        
        # Coherence gating vector
        coh_gate = self.mlp(coh_feats)
        
        # Modulate features using coherence information
        x_prime = x * coh_gate + x
        y_prime = y * coh_gate + y
        
        # Coherence score for auxiliary supervision
        coh_score = torch.sigmoid(diff.mean(dim=1, keepdim=True))
        
        return x_prime, y_prime, coh_score

# Text Encoder using ModernBERT
class ModernBertEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', emb_dim=32):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, emb_dim)

    def forward(self, input_ids, attention_mask):
        device = next(self.bert.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
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
        # Extract features from Vision Transformer
        feats = self.vit.forward_features(images)
        # If sequence of token embeddings, take [CLS] token at index 0
        if feats.ndim == 3:
            cls_token = feats[:, 0, :]  # [batch, feat_dim]
        else:
            cls_token = feats        # already pooled
        return self.fc(cls_token)

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
        self.register_buffer('dct_mat', self.build_dct_matrix(emb_dim))
        self.register_buffer('idct_mat', torch.inverse(self.dct_mat))
        # Prior hyperparameters
        self.alpha = alpha
        self.beta = beta
        # Precompute KL constant term
        self.prior_term = (
            torch.lgamma(torch.tensor(alpha)) 
            + torch.lgamma(torch.tensor(beta)) 
            - torch.lgamma(torch.tensor(alpha+beta))
        )
        
    def build_dct_matrix(self, N):
        n = torch.arange(N).unsqueeze(1).float()
        k = torch.arange(N).unsqueeze(0).float()
        alpha = torch.ones(N) * (2.0 / N)**0.5
        alpha[0] = (1.0 / N)**0.5
        return alpha.unsqueeze(1) * torch.cos((torch.pi/(2*N)) * (2*n+1) * k)

    def forward(self, x, y):
        X = x @ self.dct_mat
        Y = y @ self.dct_mat
        feats = torch.stack([X.abs(), Y.abs(), X - Y], dim=-1)
        ab = self.ab_mlp(feats)
        a, b = ab.unbind(-1)
        u = torch.rand_like(a)
        # Correct Kumaraswamy sampling
        v = (1 - u.pow(1/(a + 1e-7))).pow(1/(b + 1e-7))
        padded = torch.cat([torch.ones_like(v[..., :1]), 1 - v[..., :-1]], dim=-1)
        prefix = torch.cumprod(padded, dim=-1)
        h = v * prefix
        gamma = self.g_mlp(feats).squeeze(-1)
        Z = h * X + (1 - h) * Y + gamma * (X - Y)
        z = Z @ self.idct_mat
        latent = self.project(z)
        
        return latent, a, b



class PerLiFuseModel(nn.Module):
    def __init__(self, config):
        """
        config: dict containing keys:
          - bert_name (str)
          - vit_name (str)
          - emb_dim (int)
          - latent_dim (int)
          - hidden_dim (int)
          - num_classes (int)
          - kl_weight (float)
        """
        super().__init__()
        # extract configuration
        bert_name    = config.get('bert_name', 'bert-base-uncased')
        vit_name     = config.get('vit_name', 'vit_base_patch16_224')
        emb_dim      = config.get('emb_dim', 32)
        latent_dim   = config.get('latent_dim', 64)
        hidden_dim   = config.get('hidden_dim', 8)
        num_classes  = config.get('num_classes', 2)


        self.text_encoder = ModernBertEncoder(bert_name, emb_dim)
        self.img_encoder  = ViTEncoder(vit_name, emb_dim)
        self.coherence_guide  = CoherenceGuide(emb_dim)
        self.fusion       = SpectralFusion(emb_dim, latent_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)    # <= one output, not two
        )

    def forward(self, text, image, labels=None):
        input_ids, attention_mask = text
        x = self.text_encoder(input_ids, attention_mask)
        y = self.img_encoder(image)
        x, y, coh_score = self.coherence_guide(x, y)
        fused, a, b = self.fusion(x, y)
        logits = self.classifier(fused).squeeze(-1)
        pred   = torch.sigmoid(logits).squeeze(-1)
        
        return pred, a, b, fused, coh_score



# # Text Encoder using ModernBERT
# class ModernBertEncoder(nn.Module):
#     def __init__(self, model_name='bert-base-uncased', emb_dim=32):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         hidden_size = self.bert.config.hidden_size
#         self.fc = nn.Linear(hidden_size, emb_dim)

#     def forward(self, input_ids, attention_mask):
#         device = next(self.bert.parameters()).device
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_token = outputs.last_hidden_state[:, 0, :]
#         return self.fc(cls_token)

# # Image Encoder using ViT
# class ViTEncoder(nn.Module):
#     def __init__(self, model_name='vit_base_patch16_224', emb_dim=32):
#         super().__init__()
#         self.vit = timm.create_model(model_name, pretrained=True)
#         feat_dim = self.vit.num_features
#         self.fc = nn.Linear(feat_dim, emb_dim)

#     def forward(self, images):
#         # Extract features from Vision Transformer
#         feats = self.vit.forward_features(images)
#         # If sequence of token embeddings, take [CLS] token at index 0
#         if feats.ndim == 3:
#             cls_token = feats[:, 0, :]  # [batch, feat_dim]
#         else:
#             cls_token = feats        # already pooled
#         return self.fc(cls_token)

# # Spectral Fusion with Beta–Liouville prior
# class SpectralFusion(nn.Module):
#     def __init__(self, emb_dim=32, latent_dim=64, hidden_dim=8, alpha=1.0, beta=1.0):
#         super().__init__()
#         # MLP to predict Beta parameters (a_f,b_f)
#         self.ab_mlp = nn.Sequential(
#             nn.Linear(3, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, 2), nn.Softplus()
#         )
#         # MLP for residual gate gamma_f
#         self.g_mlp = nn.Sequential(
#             nn.Linear(3, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, 1), nn.Sigmoid()
#         )
#         # Final projection to latent dim
#         self.project = nn.Linear(emb_dim, latent_dim)
#         # Precompute orthonormal DCT and IDCT
#         self.register_buffer('dct_mat', self.build_dct_matrix(emb_dim))
#         self.register_buffer('idct_mat', torch.inverse(self.dct_mat))
#         # Prior hyperparameters
#         self.alpha = alpha
#         self.beta = beta

#     def build_dct_matrix(self, N):
#         n = torch.arange(N).unsqueeze(1).float()
#         k = torch.arange(N).unsqueeze(0).float()
#         alpha = torch.ones(N) * (2.0 / N)**0.5
#         alpha[0] = (1.0 / N)**0.5
#         return alpha.unsqueeze(1) * torch.cos((torch.pi/(2*N)) * (2*n+1) * k)

#     def forward(self, x, y):
#         # x,y: [batch, emb_dim]
#         # 1) DCT
#         X = x @ self.dct_mat
#         Y = y @ self.dct_mat
#         # 2) Build features per frequency
#         feats = torch.stack([X.abs(), Y.abs(), X - Y], dim=-1)  # [B, D, 3]
#         # 3) Predict Beta parameters
#         ab = self.ab_mlp(feats)              # [B, D, 2]
#         a, b = ab.unbind(-1)                 # each [B, D]
#         # 4) Sample sticks via Kumaraswamy
#         u = torch.rand_like(a)
#         v = (1 - u.pow(1/(a+1e-6))).pow(1/(b+1e-6))
#         # 5) Stick-breaking to get h_f
#         padded = torch.cat([torch.ones_like(v[..., :1]), 1 - v[..., :-1]], dim=-1)
#         prefix = torch.cumprod(padded, dim=-1)
#         h = v * prefix                       # [B, D]
#         # 6) Residual gate
#         gamma = self.g_mlp(feats).squeeze(-1)  # [B, D]
#         # 7) Spectral fusion
#         Z = h * X + (1 - h) * Y + gamma * (X - Y)
#         # 8) Inverse DCT
#         z = Z @ self.idct_mat                # [B, D]
#         latent = self.project(z)             # [B, latent_dim]
#         # 9) KL regularization term
#         return latent, a, b


# class PerLiFuseModel(nn.Module):
#     def __init__(self, config):
#         """
#         config: dict containing keys:
#           - bert_name (str)
#           - vit_name (str)
#           - emb_dim (int)
#           - latent_dim (int)
#           - hidden_dim (int)
#           - num_classes (int)
#           - kl_weight (float)
#         """
#         super().__init__()
#         # extract configuration
#         bert_name    = config.get('bert_name', 'bert-base-uncased')
#         vit_name     = config.get('vit_name', 'vit_base_patch16_224')
#         emb_dim      = config.get('emb_dim', 32)
#         latent_dim   = config.get('latent_dim', 64)
#         hidden_dim   = config.get('hidden_dim', 8)
#         num_classes  = config.get('num_classes', 2)


#         self.text_encoder = ModernBertEncoder(bert_name, emb_dim)
#         self.img_encoder  = ViTEncoder(vit_name, emb_dim)
#         self.fusion       = SpectralFusion(emb_dim, latent_dim, hidden_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, 1)    # <= one output, not two
#         )

#     def forward(self, text, image, labels=None):
#         input_ids, attention_mask = text
#         x = self.text_encoder(input_ids, attention_mask)
#         y = self.img_encoder(image)
#         fused, a, b = self.fusion(x, y)
#         logits = self.classifier(fused)
#         pred  = torch.sigmoid(logits).squeeze(-1) 
        
#         return pred, a, b, fused