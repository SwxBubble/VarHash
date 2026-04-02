import torch
from torch import nn


class VarHashNet(nn.Module):
    def __init__(self, hash_bits=128, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.hash_head = nn.Linear(hidden_dim, hash_bits)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)
        features = self.encoder(x)
        avg_pool = features.mean(dim=-1)
        max_pool = features.amax(dim=-1)
        norm_length = lengths.float().unsqueeze(-1) / lengths.max().clamp(min=1).float()
        embedding = self.proj(torch.cat([avg_pool, max_pool, norm_length], dim=-1))
        hash_logits = self.hash_head(embedding)
        return embedding, torch.tanh(hash_logits)
