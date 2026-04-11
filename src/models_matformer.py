import torch
import torch.nn as nn
from config import n_out
from torch_scatter import scatter

from src.models_raw.matformer import MatformerConfig, MatformerConv, RBFExpansion


class MatFormer(nn.Module):
    def __init__(self, args): 
        super().__init__()
        
        config = MatformerConfig()
        for k, v in args.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.args = {k:v for k,v in args.items() if k in {"hidden_size", "num_heads", "num_layers"}}
        self.atom_embedding = nn.Linear(8, config.hidden_size)
        
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=5.0, bins=config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Softplus(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.att_layers = nn.ModuleList([
            MatformerConv(in_channels=config.hidden_size, out_channels=config.hidden_size, heads=config.num_heads, edge_dim=config.hidden_size)
            for _ in range(config.num_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), 
            nn.SiLU()
        )
        
        self.wl_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        self.fc_out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size + 64, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, n_out)
        )

    def forward(self, data):
        node_features = self.atom_embedding(data.x)
        
        edge_features = self.rbf(data.dist)
        
        for layer in self.att_layers:
            node_features = layer(node_features, data.edge_index, edge_features)

        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        features = self.fc(features)
        
        wl_out = self.wl_emb(data.wl.view(-1, 1))
        combined = torch.cat([features, wl_out], dim=1)

        out = self.fc_out(combined)
        return out
    
    def __str__(self):
        return "matformer"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
    
    def get_args(self):
        return self.args