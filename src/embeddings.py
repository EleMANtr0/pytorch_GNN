import torch
import torch.nn as nn


class FoldsEmb(nn.Module):
    def __init__(self, emb_dim, hidden, n_out, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(7, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden*2),
            nn.SiLU(),
            nn.LayerNorm(hidden*2),
            nn.Dropout(drop),
            nn.Linear(hidden*2, n_out)
        )

    def forward(self, x):
        m = x > 1
        m = m | ((x == 1) & ~m.any(-1, keepdim=True))
        x = (self.emb(x) * m.unsqueeze(-1)).sum(-2) / m.sum(-1, keepdim=True).clamp(min=1)
        return self.head(x)

class WaveLenEmb(nn.Module):
    def __init__(self, emb_dim, hidden, n_out, drop=0.3):
        super().__init__()
        self.register_buffer("wl_embs", torch.Tensor([5.14, 5.32, 7.8, 7.85]))
        self.emb = nn.Sequential(
            nn.Embedding(4, emb_dim),
            nn.Linear(emb_dim, hidden*2),
            nn.SiLU(),
            nn.LayerNorm(hidden*2),
            nn.Dropout(drop),
            nn.Linear(hidden*2, n_out)
        )

    def forward(self, x):
        idx = (x == self.wl_embs).long().argmax(dim=-1)
        return self.emb(idx)

