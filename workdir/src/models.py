from torch.nn import Linear, ReLU, Dropout, Module, SiLU, Sequential
import torch
from schnetpack.representation import PaiNN
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from src.data.dataset import pyg_batch_to_schnetpack
from src.models_ps.painn import PaiNN_WL
import numpy as np
import torch_geometric.nn
from src.models_ps.schnet import SchNet
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torchmdnet.models.torchmd_et import TorchMD_ET
from torchmdnet.models.model import create_model
from config import n_out


class PaiNN_raman0(Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=256,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_out)
        )

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)

        painn_output = self.painn(batch)

        atom_feats = painn_output["scalar_representation"]

        # pooled_feats = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)
        pooled_feats = global_mean_pool(atom_feats, data.batch, dim=0)

        output = self.head(pooled_feats)

        return output

    def __str__(self):
        return "painn"

class PaiNN_raman2(Module):
    def __init__(self):
        super().__init__()
        self.painn = PaiNN(cutoff_fn=CosineCutoff(cutoff=10),n_interactions=3,n_atom_basis=256,radial_basis=GaussianRBF(n_rbf=30,cutoff=10))
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(320, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_out)
        )
        self.wl_embed = Sequential(
            Linear(1, 32),
            SiLU(),
            Dropout(0.3),
            Linear(32, 64)
        )

    def forward(self, data):
        # data = dat.clone()
        # data.x = drop(data.x,7)
        batch = pyg_batch_to_schnetpack(data, cutoff=10.0)
        painn_output = self.painn(batch)
        atom_feats = painn_output["scalar_representation"]
        # pooled = torch_scatter.scatter_mean(atom_feats, data.batch, dim=0)
        pooled = global_mean_pool(atom_feats, data.batch, dim=0)

        if not hasattr(data,"wl"):
            wl = torch.tensor([5.14] * data.num_graphs, device=data.x.device)
        else:
            wl = data.wl.unsqueeze(1)
        wl_emb = self.wl_embed(wl)
        fused = torch.cat([pooled, wl_emb], dim=-1)
        out = self.head(fused)

        return out

    def __str__(self):
        return "painn2"

base_args = {
    "model": "equivariant-transformer",
    "num_layers": 4,
    "num_rbf": 50,
    "rbf_type": "gauss",
    "trainable_rbf": True,
    "activation": "silu",
    "cutoff_lower": 0.0,
    "cutoff_upper": 10.0,
    "max_z": 100,
    "max_num_neighbors": 128,
    "aggr": "add",
    "derivative": False,
    "atom_filter": -1,
    "prior_model": None,
    "output_model": "Scalar",
    "reduce_op": "add",
    "precision": 32,
    "check_errors": False,
    "distance_influence": "keys",
    "vector_cutoff": False,

    "embedding_dimension": 64,
    "attn_activation": "silu",
    "num_heads": 8,
    "neighbor_embedding": True,
}


class MDNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        for k, v in args.items():
            base_args[k] = v
        # temp = create_model(base_args)

        # self.body = temp.representation_model
        args = base_args
        self.body = TorchMD_ET(
            hidden_channels=args["embedding_dimension"],
            num_layers=args["num_layers"],
            num_rbf=args["num_rbf"],
            rbf_type=args["rbf_type"],
            trainable_rbf=args["trainable_rbf"],
            activation=args["activation"],
            attn_activation=args["attn_activation"],
            neighbor_embedding=args["neighbor_embedding"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            cutoff_lower=args["cutoff_lower"],
            cutoff_upper=args["cutoff_upper"],
            max_z=args["max_z"],
            max_num_neighbors=args["max_num_neighbors"]
        )

        self.wl_emb = nn.Sequential(
            nn.Linear(1,32, bias=False),
            nn.ReLU(),
            nn.Linear(32,64)
        )

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(args["embedding_dimension"]+64, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_out)
        )

    def forward(self,x):
        z = x.z
        pos = x.pos + torch.randn_like(x.pos) * 1e-5
        batch = x.batch
        wl = x.wl.view(-1,1)
        
        atom_feats, *_ = self.body(z, pos, batch)
        pooled_feats = global_mean_pool(atom_feats, batch)

        wl_out = self.wl_emb(wl)
        combined_feats = torch.cat([pooled_feats,wl_out],dim=1)
        
        out = self.head(combined_feats)
        return out
    
    def __str__(self):
        return "MDNet"
    
    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
