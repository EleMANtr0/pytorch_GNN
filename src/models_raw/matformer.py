import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from matformer.models.transformer import MatformerConv
from matformer.models.utils import RBFExpansion

def params_to_matrix(params):
    """Convert [a, b, c, alpha, beta, gamma] to a 3x3 lattice matrix."""
    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    alpha, beta, gamma = torch.deg2rad(params[:, 3]), torch.deg2rad(params[:, 4]), torch.deg2rad(params[:, 5])
    
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    
    val = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    res = torch.stack([
        torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=1),
        torch.stack([b * cos_gamma, b * sin_gamma, torch.zeros_like(a)], dim=1),
        torch.stack([c * cos_beta, c * val, c * torch.sqrt(torch.clamp(1 - cos_beta**2 - val**2, min=0))], dim=1)
    ], dim=1)
    return res

class Matformer(nn.Module):
    """
    Consolidated Matformer architecture adapted from pyg_att.py.
    Modified to accept standard PyG DataBatch and handle lattice conditioning internally.
    """
    def __init__(
        self, 
        node_features=128, 
        edge_features=128, 
        conv_layers=5, 
        node_layer_head=4,
        max_z=100,
        cutoff=10.0,
        use_angle_lattice=True
    ):
        super().__init__()
        self.node_features = node_features
        self.use_angle_lattice = use_angle_lattice
        
        # 1. Atom & Distance Embeddings (As per user Turn 17, using atomic numbers z)
        self.atom_embedding = nn.Embedding(max_z, node_features)
        
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=cutoff, bins=edge_features),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features),
        )

        # 2. Lattice Geometry Embedding (The Matformer 'Secret Sauce')
        if self.use_angle_lattice:
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(vmin=1e-8, vmax=cutoff, bins=edge_features),
                nn.Linear(edge_features, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )
            self.lattice_angle = nn.Sequential(
                RBFExpansion(vmin=-1, vmax=1.0, bins=40),
                nn.Linear(40, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )
            self.lattice_emb = nn.Sequential(
                nn.Linear(node_features * 6, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )
            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(node_features * 2, node_features),
                nn.Softplus(),
                nn.Linear(node_features, node_features)
            )

        # 3. Matformer Attention Layers
        self.att_layers = nn.ModuleList([
            MatformerConv(
                in_channels=node_features, 
                out_channels=node_features, 
                heads=node_layer_head, 
                edge_dim=node_features
            ) for _ in range(conv_layers)
        ])

    def forward(self, data):
        # Extract Batch Data
        z, batch, dist, edge_index = data.z, data.batch, data.dist, data.edge_index
        
        # Initial Embeddings
        node_features = self.atom_embedding(z)
        edge_features = self.rbf(dist)
        
        # Lattice Global Conditioning
        lat_emb = None
        if self.use_angle_lattice:
            lattice = params_to_matrix(data.cell_params)
            lat_len = torch.norm(lattice, dim=-1)
            lat_edge = self.lattice_rbf(lat_len.view(-1)).view(-1, 3 * self.node_features)
            
            def get_cos(v1, v2):
                return torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1))
            
            c1 = self.lattice_angle(torch.clamp(get_cos(lattice[:,0,:], lattice[:,1,:]), -1, 1).unsqueeze(-1))
            c2 = self.lattice_angle(torch.clamp(get_cos(lattice[:,0,:], lattice[:,2,:]), -1, 1).unsqueeze(-1))
            c3 = self.lattice_angle(torch.clamp(get_cos(lattice[:,1,:], lattice[:,2,:]), -1, 1).unsqueeze(-1))
            
            lat_emb = self.lattice_emb(torch.cat((lat_edge, c1.squeeze(1), c2.squeeze(1), c3.squeeze(1)), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lat_emb[batch]), dim=-1))

        # Message Passing Blocks
        for layer in self.att_layers:
            node_features = layer(node_features, edge_index, edge_features)

        # Pooling (Readout)
        pooled_features = scatter(node_features, batch, dim=0, reduce="mean")

        # Inject Global Lattice Residual for Heads
        if lat_emb is not None:
            pooled_features = pooled_features + lat_emb
            
        return pooled_features